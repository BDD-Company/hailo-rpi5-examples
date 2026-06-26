#!/usr/bin/env python

import multiprocessing
from pathlib import Path
import setproctitle
import signal
import os
import gi
import threading
import sys
import cv2
import numpy as np
import time
import queue


gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject, GstApp

from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import DETECTION_APP_TITLE, DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME, DETECTION_POSTPROCESS_FUNCTION
# from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import DISPLAY_PIPELINE

from pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE
)


# Based on hailo_app_python/core/gstreamer/gstreamer_app.py


from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    get_source_type,
)


# Absolute imports for your common utilities
from hailo_apps.hailo_app_python.core.common.defines import (
    HAILO_RGB_VIDEO_FORMAT,
    GST_VIDEO_SINK,
    TAPPAS_POSTPROC_PATH_KEY,
    RESOURCES_PATH_KEY,
    RESOURCES_ROOT_PATH_DEFAULT,
    RESOURCES_VIDEOS_DIR_NAME,
    BASIC_PIPELINES_VIDEO_EXAMPLE_NAME,
    USB_CAMERA,
    RPI_NAME_I,
)
from hailo_apps.hailo_app_python.core.common.camera_utils import (
    get_usb_video_devices,
)
from hailo_apps.hailo_app_python.core.common.core import (
    load_environment,
)
from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)


try:
    from picamera2 import Picamera2
    from libcamera import controls as picamera_controls
except ImportError:
    pass # Available only on Pi OS


import logging

logger = logging.getLogger("GSTApp")



# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# A sample class to be used in the callback function
# This example allows to:
# 1. Count the number of frames
# 2. Setup a multiprocessing queue to pass the frame to the main thread
# Additional variables and functions can be added to this class as needed
class app_callback_class:
    def __init__(self):
        self.frame_count = 0
        self.use_frame = False
        self.frame_queue = multiprocessing.Queue(maxsize=3)
        self.running = True

    def increment(self):
        self.frame_count += 1

    def get_count(self):
        return self.frame_count

    def set_frame(self, frame):
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def get_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        else:
            return None

def dummy_callback(pad, info, user_data):
    """
    A minimal dummy callback function that returns immediately.

    Args:
        pad: The GStreamer pad
        info: The probe info
        user_data: User-defined data passed to the callback

    Returns:
        Gst.PadProbeReturn.OK
    """
    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# GStreamerApp class
# -----------------------------------------------------------------------------------------------
class GStreamerApp:
    def __init__(self, args, user_data: app_callback_class):
        # Set the process title
        setproctitle.setproctitle("Hailo Python App")

        # Create options menu
        self.options_menu = args.parse_args()

        # Set up signal handler for SIGINT (Ctrl-C)
        signal.signal(signal.SIGINT, self.shutdown)

        # Load environment variables
        x=os.environ.get("HAILO_ENV_FILE")
        load_environment(x)

        # Initialize variables
        tappas_post_process_dir = Path(os.environ.get(TAPPAS_POSTPROC_PATH_KEY, ''))
        if tappas_post_process_dir == '':
            logger.error("TAPPAS_POST_PROC_DIR environment variable is not set. Please set it by running set-env in cli")
            exit(1)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.postprocess_dir = tappas_post_process_dir
        if self.options_menu.input is None:
            self.video_source = str(Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_VIDEOS_DIR_NAME / BASIC_PIPELINES_VIDEO_EXAMPLE_NAME)
        else:
            self.video_source = self.options_menu.input
        if self.video_source == USB_CAMERA:
            self.video_source = get_usb_video_devices()
            if not self.video_source:
                logger.error('Provided argument "--input" is set to "usb", however no available USB cameras found. Please connect a camera or specifiy different input method.')
                exit(1)
            else:
                self.video_source = self.video_source[0]
        self.source_type = get_source_type(self.video_source)
        self.frame_rate = self.options_menu.frame_rate
        self.user_data = user_data
        self.video_sink = GST_VIDEO_SINK
        self.pipeline = None
        self.loop = None
        self.threads = []
        self.shutdown_callbacks = []
        self.error_occurred = False
        self.pipeline_latency = 0  # milliseconds; 0 = use each element's natural minimum latency; if frames are dropped or Gstreamer is stalled, then set to 50

        # Set Hailo parameters; these parameters should be set based on the model used
        self.batch_size = 1
        self.video_width = 1280
        self.video_height = 720
        self.video_format = HAILO_RGB_VIDEO_FORMAT
        self.hef_path = None
        self.app_callback = None

        # Set user data parameters
        user_data.use_frame = self.options_menu.use_frame

        self.sync = "false" if (self.options_menu.disable_sync or self.source_type != "file") else "true"
        self.show_fps = self.options_menu.show_fps

        if self.options_menu.dump_dot:
            # pass
            os.environ["GST_DEBUG_DUMP_DOT_DIR"] = "/home/bdd/hailo-rpi5-examples/_DEBUG/pipeline/" #os.getcwd()
        # Ensure stall-detection dot dumps land somewhere even without --dump-dot.
        os.environ.setdefault("GST_DEBUG_DUMP_DOT_DIR", "/home/bdd/hailo-rpi5-examples/_DEBUG/pipeline/")
        os.makedirs(os.environ["GST_DEBUG_DUMP_DOT_DIR"], exist_ok=True)

        self.webrtc_frames_queue = None  # for appsink & GUI mode

    def appsink_callback(self, appsink):
        """
        Callback function for the appsink element in the GStreamer pipeline.
        This function is called when a new sample (frame) is available in the appsink (output from the pipeline).
        """
        sample = appsink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            if buffer:
                format, width, height = get_caps_from_pad(appsink.get_static_pad("sink"))
                frame = get_numpy_from_buffer(buffer, format, width, height)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
                try:
                    self.webrtc_frames_queue.put(frame)  # Add the frame to the queue (non-blocking)
                except queue.Full:
                    logger.warning("Frame queue is full. Dropping frame.")  # Drop the frame if the queue is full
        return Gst.FlowReturn.OK

    def on_fps_measurement(self, sink, fps, droprate, avgfps):
        logger.info(f"FPS: {fps:.2f}, Droprate: {droprate:.2f}, Avg FPS: {avgfps:.2f}")
        return True

    def create_pipeline(self):
        # Initialize GStreamer
        Gst.init(None)

        pipeline_string = self.get_pipeline_string()
        try:
            self.pipeline = Gst.parse_launch(pipeline_string)
        except Exception as e:
            logger.error(f"Error creating pipeline", exc_info=True)
            sys.exit(1)

        # Connect to hailo_display fps-measurements
        if self.show_fps:
            logger.info("Showing FPS")
            display_sink = self.pipeline.get_by_name("hailo_display")
            if display_sink is not None:
                display_sink.connect("fps-measurements", self.on_fps_measurement)
            else:
                logger.warning("hailo_display sink is not present in pipiline, can't show FPS")

        # Create a GLib Main Loop
        self.loop = GLib.MainLoop()

    def bus_call(self, bus, message : Gst.Message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"got error from Gstreamer: %s : %s", err, debug)

            self.error_occurred = (err, debug)
            self.shutdown()
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            src_name = message.src.get_name() if message.src else '?'
            logger.warning("GStreamer warning from %s: %s : %s", src_name, err, debug)
        elif t == Gst.MessageType.INFO:
            err, debug = message.parse_info()
            src_name = message.src.get_name() if message.src else '?'
            logger.info("GStreamer info from %s: %s : %s", src_name, err, debug)
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src is self.pipeline:
                old, new, pending = message.parse_state_changed()
                logger.info("pipeline state %s -> %s (pending %s)", old.value_nick, new.value_nick, pending.value_nick)
        elif t == Gst.MessageType.STREAM_STATUS:
            status_type, owner = message.parse_stream_status()
            if not status_type == 'create' and not status_type == 'enter':
                logger.debug("stream-status %s on %s", status_type.value_nick, owner.get_name() if owner else '?')
        # QOS
        elif t == Gst.MessageType.QOS:
            # Handle QoS message here
            # If lots of QoS messages are received, it may indicate that the pipeline is not able to keep up
            if not hasattr(self, 'qos_count'):
                self.qos_count = 0
            self.qos_count += 1
            qos_element = message.src.get_name()
            logger.info("QoS message received from %s", qos_element)

            if self.qos_count > 50 and self.qos_count % 10 == 0:
                logger.warning("Lots of QoS messages received: %s, consider optimizing the pipeline or reducing the pipeline frame rate see '--frame-rate' flag.", self.qos_count)

        return True


    def on_eos(self):
        if self.source_type == "file":
            if self.sync == "false":
                # Pause the pipeline to clear any queued data. It is required when running with sync=false
                # This will produce some warnings, but it's fine
                logger.debug("Pausing pipeline for rewind... some warnings are expected.")
                self.pipeline.set_state(Gst.State.PAUSED)

            # Seek to the beginning (position 0) using a flush seek.
            success = self.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH, 0)
            if success:
                logger.debug("Video rewound successfully. Restarting playback...")
            else:
                logger.error("Error rewinding video.")

            # Resume playback.
            self.pipeline.set_state(Gst.State.PLAYING)
        else:
            self.shutdown()


    def _add_buffer_counter_probe(self, element_name, pad_name='src'):
        """Install a pad probe that increments self.buffer_counters[(element_name, pad_name)]
        for every buffer that crosses pad_name of element_name. Used to identify where
        buffers stop flowing in the pipeline when a stall is suspected."""
        if not hasattr(self, 'buffer_counters'):
            self.buffer_counters = {}
        element = self.pipeline.get_by_name(element_name)
        if element is None:
            logger.warning("Cannot add buffer counter probe: element %r not found", element_name)
            return
        pad = element.get_static_pad(pad_name)
        if pad is None:
            logger.warning("Cannot add buffer counter probe: pad %r of %r not found", pad_name, element_name)
            return
        key = (element_name, pad_name)
        self.buffer_counters[key] = 0
        def _probe_cb(pad, info, _user):
            self.buffer_counters[key] += 1
            return Gst.PadProbeReturn.OK
        pad.add_probe(Gst.PadProbeType.BUFFER, _probe_cb, None)

    def _install_detection_start_probe(self, element_name='inference_wrapper_input_q', pad_name='sink'):
        """Stamp `timestamp/x-unix` reference meta with time.monotonic_ns() on each buffer
        entering the detection portion of the pipeline. The user callback reads it back as
        detection_start to compute detection-only latency.

        No-op when the meta is already present — the picamera2/appsrc producer attaches it
        upstream, so this only fills in the libcamerasrc path where nothing else does."""
        element = self.pipeline.get_by_name(element_name)
        if element is None:
            logger.warning("Cannot install detection-start probe: element %r not found", element_name)
            return
        pad = element.get_static_pad(pad_name)
        if pad is None:
            logger.warning("Cannot install detection-start probe: pad %r of %r not found", pad_name, element_name)
            return
        unix_ts_caps = Gst.Caps.from_string("timestamp/x-unix")
        def _probe_cb(pad, info, _user):
            buf = info.get_buffer()
            if buf is not None and buf.get_reference_timestamp_meta(unix_ts_caps) is None:
                buf.add_reference_timestamp_meta(unix_ts_caps, time.monotonic_ns(), Gst.CLOCK_TIME_NONE)
            return Gst.PadProbeReturn.OK
        pad.add_probe(Gst.PadProbeType.BUFFER, _probe_cb, None)

    def _log_pipeline_health(self):
        """Periodic diagnostic: log per-probe buffer counts (with deltas since last call)
        and current-level-buffers for every named GstQueue. If the deepest probe (last
        registered) hasn't advanced in a while, dump a dot file for offline inspection."""
        now = time.monotonic()
        prev_counts = getattr(self, '_health_prev_counts', {})
        prev_time = getattr(self, '_health_prev_time', now)
        elapsed = max(now - prev_time, 1e-6)

        counter_parts = []
        for key, count in self.buffer_counters.items():
            delta = count - prev_counts.get(key, 0)
            elem_name, pad_name = key
            counter_parts.append(f"{elem_name}:{pad_name}={count}(+{delta},{delta/elapsed:.1f}fps)")
        logger.info("pipeline buffer counters (over %.2fs): %s", elapsed, " | ".join(counter_parts))

        queue_parts = []
        it = self.pipeline.iterate_elements()
        while True:
            ok, element = it.next()
            if ok != Gst.IteratorResult.OK:
                break
            factory = element.get_factory()
            if factory is None or factory.get_name() != 'queue':
                continue
            qname = element.get_name()
            buffers = element.get_property('current-level-buffers')
            max_buffers = element.get_property('max-size-buffers')
            queue_parts.append(f"{qname}={buffers}/{max_buffers}")
        logger.info("pipeline queue levels (current/max buffers): %s", " ".join(queue_parts))

        # Stall detection: track the LAST registered probe (the deepest one — typically identity_callback)
        if self.buffer_counters:
            tail_key = next(reversed(self.buffer_counters))
            tail_delta = self.buffer_counters[tail_key] - prev_counts.get(tail_key, 0)
            stall_ticks = getattr(self, '_health_stall_ticks', 0)
            if tail_delta == 0 and self.buffer_counters[tail_key] > 0:
                stall_ticks += 1
                logger.error(
                    "STALL: %s:%s has not advanced in %.2fs (%d consecutive ticks). count=%d",
                    tail_key[0], tail_key[1], elapsed, stall_ticks, self.buffer_counters[tail_key],
                )
                if stall_ticks == 1:
                    # Dump dot once per stall episode so we don't spam files.
                    Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.VERBOSE,
                                              f"stall_{int(now)}")
                    logger.error("STALL: dumped pipeline dot to GST_DEBUG_DUMP_DOT_DIR/stall_%d.dot", int(now))
            else:
                stall_ticks = 0
            self._health_stall_ticks = stall_ticks

        self._health_prev_counts = dict(self.buffer_counters)
        self._health_prev_time = now
        return True  # keep timeout active

    def add_shutdown_callback(self, callback):
        """Register a callable invoked at the start of shutdown(), before the pipeline is torn down.
        Use it to signal worker threads (e.g. push a STOP sentinel into their input queue)
        so they can exit even if a producer thread (e.g. picamera) is stuck."""
        self.shutdown_callbacks.append(callback)

    def shutdown(self, signum=None, frame=None):
        logger.info("Shutting down... Hit Ctrl-C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        for cb in self.shutdown_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("Error in shutdown callback %s", cb)
        self.pipeline.set_state(Gst.State.PAUSED)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.READY)
        GLib.usleep(100000)  # 0.1 second delay

        self.pipeline.set_state(Gst.State.NULL)
        GLib.idle_add(self.loop.quit)


    def update_fps_caps(self, new_fps=30, source_name='source'):
        """Updates the FPS by setting max-rate on videorate element directly"""
        # Derive the videorate and capsfilter element names based on the source name
        videorate_name = f"{source_name}_videorate"
        capsfilter_name = f"{source_name}_fps_caps"

        # Get the videorate element
        videorate = self.pipeline.get_by_name(videorate_name)
        if videorate is None:
            logger.warning(f"Element {videorate_name} not found in the pipeline.")
            return

        # Print current properties for debugging
        current_max_rate = videorate.get_property("max-rate")
        logger.debug("Current videorate max-rate: %s", current_max_rate)

        # Update the max-rate property directly
        videorate.set_property("max-rate", new_fps)

        # Verify the change
        updated_max_rate = videorate.get_property("max-rate")
        logger.debug("Updated videorate max-rate to: %s", updated_max_rate)

        # Get the capsfilter element
        capsfilter = self.pipeline.get_by_name(capsfilter_name)
        if capsfilter:
            new_caps_str = f"video/x-raw, framerate={new_fps}/1"
            new_caps = Gst.Caps.from_string(new_caps_str)
            capsfilter.set_property("caps", new_caps)
            logger.debug("Updated capsfilter caps to match new rate: %s", new_caps_str)

        # Update frame_rate property
        self.frame_rate = new_fps


    def get_pipeline_string(self) -> str:
        # This is a placeholder function that should be overridden by the child class
        return ""


    def dump_dot_file(self):
        logger.debug("Dumping dot file...")
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.VERBOSE, "/home/bdd/hailo-rpi5-examples/_DEBUG/pipeline")
        return False


    def run(self):
        # Add a watch for messages on the pipeline's bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.bus_call, self.loop)


        # Connect pad probe to the identity element
        if not self.options_menu.disable_callback:
            identity = self.pipeline.get_by_name("identity_callback")
            if identity is None:
                logger.warning("identity_callback element not found, add <identity name=identity_callback> in your pipeline where you want the callback to be called.")
            else:
                identity_pad = identity.get_static_pad("src")
                identity_pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, self.user_data)

        hailo_display = self.pipeline.get_by_name("hailo_display")
        if hailo_display is None and not getattr(self.options_menu, 'ui', False):
            logger.warning("Warning: hailo_display element not found, add <fpsdisplaysink name=hailo_display> to your pipeline to support fps display.")

        # Disable QoS to prevent frame drops
        disable_qos(self.pipeline)

        # Start a subprocess to run the display_user_data_frame function
        if self.options_menu.use_frame:
            display_process = multiprocessing.Process(target=display_user_data_frame, args=(self.user_data,))
            display_process.start()

        # # Buffer-counter probes at strategic points in the pipeline. The order matters:
        # # _log_pipeline_health() treats the LAST registered probe as the "deepest" one
        # # for stall detection. So register them upstream → downstream.
        # for elem_name in (
        #     'app_source',
        #     'inference_wrapper_input_q',
        #     'inference_hailonet',
        #     'inference_hailofilter',
        #     'inference_wrapper_output_q',
        #     'identity_callback',
        # ):
        #     self._add_buffer_counter_probe(elem_name, 'src')

        if self.source_type == 'libcamera':
            # Stamp detection-start timestamp on buffers entering the inference wrapper.
            # libcamerasrc path has nobody to attach this upstream; picamera2 path already does.
            self._install_detection_start_probe()

        # Periodic pipeline health snapshot (buffer counts, queue levels, stall detection).
        # GLib.timeout_add_seconds(2, self._log_pipeline_health)

        if self.source_type == RPI_NAME_I:
            on_picam_failure = lambda: GLib.idle_add(self.shutdown)
            # Multi-camera support: if the application supplied a CameraSwitcher,
            # spawn one capture thread per configured camera; otherwise fall back
            # to the legacy single-camera launch (the producer synthesizes a
            # default CameraConfig internally).
            camera_switcher = getattr(self, 'camera_switcher', None)
            if camera_switcher is not None:
                # L4: set the shared appsrc caps ONCE, from the switcher's
                # single source of truth, before any producer thread starts.
                # Previously each picamera_thread set caps independently — fine
                # while both happened to agree, but a footgun the moment they
                # didn't. Now there's exactly one writer.
                appsrc = self.pipeline.get_by_name("app_source")
                if appsrc is not None:
                    appsrc.set_property("is-live", True)
                    appsrc.set_property("format", Gst.Format.TIME)
                    appsrc.set_property(
                        "caps",
                        Gst.Caps.from_string(
                            f"video/x-raw, format={camera_switcher.video_format}, "
                            f"width={camera_switcher.width}, height={camera_switcher.height}, "
                            f"framerate={camera_switcher.fps}/1, pixel-aspect-ratio=1/1"
                        ),
                    )
                else:
                    logger.warning("camera_switcher provided but 'app_source' appsrc not found; producer will fall back")
                for cam_cfg in camera_switcher.configs():
                    t = threading.Thread(
                        target=picamera_thread,
                        args=(self.pipeline, self.video_width, self.video_height, self.video_format),
                        kwargs={
                            'camera_config': cam_cfg,
                            'camera_switcher': camera_switcher,
                            'on_failure': on_picam_failure,
                        },
                        name=f"picam-{cam_cfg.camera_id}-{cam_cfg.name or 'cam'}",
                    )
                    self.threads.append(t)
                    t.start()
            else:
                picam_thread = threading.Thread(
                    target=picamera_thread,
                    args=(self.pipeline, self.video_width, self.video_height, self.video_format),
                    kwargs={'on_failure': on_picam_failure},
                    name="picam",
                )
                self.threads.append(picam_thread)
                picam_thread.start()

        # Set the pipeline to PAUSED to ensure elements are initialized
        self.pipeline.set_state(Gst.State.PAUSED)

        # Set pipeline latency
        new_latency = self.pipeline_latency * Gst.MSECOND  # Convert milliseconds to nanoseconds
        self.pipeline.set_latency(new_latency)

        # Set pipeline to PLAYING state
        self.pipeline.set_state(Gst.State.PLAYING)

        # Dump dot file
        if self.options_menu.dump_dot:
            GLib.timeout_add_seconds(3, self.dump_dot_file)

        # Run the GLib event loop
        self.loop.run()

        # Clean up
        try:
            self.user_data.running = False
            self.pipeline.set_state(Gst.State.NULL)
            if self.options_menu.use_frame:
                display_process.terminate()
                display_process.join()
            for t in self.threads:
                t.join()
        except Exception as e:
            logger.error("Error during cleanup", exc_info=True)
        finally:
            if self.error_occurred is not False:
                logger.error("Exiting due to error... %s", self.error_occurred)
                sys.exit(1)
            else:
                logger.info("Exiting...")
                sys.exit(0)


# Shared destroy-notify for zero-extra-copy GstBuffers: handed to new_wrapped_full so
# PyGObject keeps the wrapped bytes alive until the buffer is freed, then this no-op
# runs. Sharing one callable across all buffers (PyGObject still tracks per-buffer
# user_data + notify pairs internally, so this is safe).
_buffer_keepalive_noop = lambda *_: None


# Process-wide PTS baseline shared across every picam thread. Each thread used
# to relativize buffer.pts against its OWN first sensor timestamp, so when the
# active camera switched, the new stream's PTS jumped backward and splitmuxsink
# crashed with "Queued GOP time is negative". With one shared monotonic baseline
# the PTS is monotonic across cameras by construction.
_pts_baseline_lock = threading.Lock()
_pts_baseline_ns : int | None = None


def _shared_pts_ns(now_ns : int) -> int:
    """Return a monotonic, shared-across-all-cameras PTS in nanoseconds."""
    global _pts_baseline_ns
    with _pts_baseline_lock:
        if _pts_baseline_ns is None:
            _pts_baseline_ns = now_ns
        return now_ns - _pts_baseline_ns


def picamera_thread(
        pipeline,
        video_width,
        video_height,
        video_format,
        camera_config=None,
        camera_switcher=None,
        picamera_config=None,
        target_fps = 30,  # aligns appsrc caps with the pipeline's 30/1 -> no videorate duplication
        picamera_controls_initial = None,
        picamera_controls_per_frame_callback = None,
        on_failure=None,
        # imx477's first frame after a fresh start regularly takes 0.5-1.5s
        # (especially after a fast restart of the previous process) and 1s
        # was too tight — half the cold starts now fail. 3s gives the slower
        # sensor headroom without masking a truly stalled camera.
        capture_timeout_s = 3,
        slow_capture_warn_s = 0.2,
        alive_log_every_n_frames = 100,
        appsrc_name = "app_source",
    ):
    """Capture frames from one Picamera2 device into a shared GStreamer appsrc.

    Multi-camera operation: when `camera_switcher` is provided, the thread only
    pushes a captured frame into appsrc while `camera_switcher.is_active(camera_id)`.
    The inactive camera keeps capturing+releasing so AGC/AWB stay warm and the
    switch is instant. All threads share a single appsrc (named `appsrc_name`) —
    the appsrc caps are set once by whichever thread is active first; subsequent
    cameras must produce the same caps.
    """
    # Allow the legacy single-camera call site to keep working: synthesize a
    # default CameraConfig when none is provided.
    if camera_config is None:
        from helpers import CameraConfig, DEFAULT_CAMERA_ID
        camera_config = CameraConfig(
            camera_id=DEFAULT_CAMERA_ID,
            name="default",
            sensor_index=0,
        )

    # Single source of truth for resolution/fps/format (L4):
    # when a CameraSwitcher is provided, everyone reads from there so all
    # producers and the shared appsrc agree on caps. Without a switcher we
    # fall back to the args passed into the function (legacy single-cam).
    if camera_switcher is not None:
        capture_width = camera_switcher.width
        capture_height = camera_switcher.height
        capture_target_fps = camera_switcher.fps
        capture_video_format = camera_switcher.video_format
        # Exposure/gain knobs live in an optional values object (Config.Camera.
        # AutoExposure) or None when disabled. Duck-typed: getattr(None, x, d)->d,
        # so a missing section degrades to plain auto-exposure. Config durations are
        # in MILLISECONDS; convert here to picamera2's native microseconds (and to
        # seconds for the warmup timer) so the logic below stays in those units.
        ae = getattr(camera_switcher, 'autoexposure', None)
        exposure_time_us = getattr(ae, 'exposure_time_ms', 0) * 1000
        analogue_gain = getattr(ae, 'analogue_gain', 0.0)
        exposure_auto_pin_s = getattr(ae, 'exposure_auto_pin_ms', 0) / 1000.0
        exposure_min_us = getattr(ae, 'exposure_min_ms', 0) * 1000
        exposure_max_us = getattr(ae, 'exposure_max_ms', 0) * 1000
        gain_max = getattr(ae, 'gain_max', 0.0)
        buffer_count = getattr(camera_switcher, 'buffer_count', 2)
    else:
        capture_width = video_width
        capture_height = video_height
        capture_target_fps = target_fps
        capture_video_format = video_format
        exposure_time_us = 0
        analogue_gain = 0.0
        exposure_auto_pin_s = 0.0
        exposure_min_us = 0
        exposure_max_us = 0
        gain_max = 0.0
        buffer_count = 2

    # Stage-A latency: pin the shutter (disable AE) so it is short and
    # deterministic. Three modes (priority order):
    #   1. exposure_auto_pin_s > 0: let AE converge for that long, then read back
    #      and pin the measured ExposureTime/AnalogueGain (clamped). Scene-adapted
    #      AND low-latency. Done at runtime in the capture loop.
    #   2. exposure_time_us > 0: pin this fixed value now (+ analogue_gain).
    #   3. else: leave AE on (auto).
    # When pinned (either way) we must NOT re-enable AE on camera activation.
    auto_pin_enabled = exposure_auto_pin_s > 0
    exposure_pinned = (not auto_pin_enabled) and exposure_time_us > 0
    gain_pinned = analogue_gain > 0

    def _clamp_exposure(exp_us: float, gain: float):
        """Apply the exposure limits, shifting clipped light into gain to keep
        brightness. Returns (int exposure_us, float gain). gain<=0 -> treated 1.0."""
        e = float(exp_us)
        g = float(gain) if gain and gain > 0 else 1.0
        if exposure_max_us > 0 and e > exposure_max_us:
            g *= e / exposure_max_us          # preserve brightness (~exp*gain)
            e = exposure_max_us
        if exposure_min_us > 0 and e < exposure_min_us:
            g *= e / exposure_min_us
            e = exposure_min_us
        g = max(g, 1.0)
        if gain_max > 0:
            g = min(g, gain_max)
        return int(round(e)), round(g, 3)

    camera_id = camera_config.camera_id
    log_prefix = f"[cam{camera_id}:{camera_config.name or '?'}]"
    if gain_pinned and not (exposure_pinned or auto_pin_enabled):
        logger.warning("%s analogue_gain=%.2f set but exposure is auto (AE on) — gain will be "
                       "ignored by the AGC; set exposure_time_ms>0 to pin gain", log_prefix, analogue_gain)

    appsrc: GstApp.AppSrc = pipeline.get_by_name(appsrc_name)
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    if False and logger.isEnabledFor(logging.DEBUG):
        prop_lines = []
        for pspec in appsrc.list_properties():
            if not (pspec.flags & GObject.ParamFlags.READABLE):
                continue
            try:
                value = appsrc.get_property(pspec.name)
            except Exception as e:  # some props raise when not yet negotiated
                value = f"<unreadable: {e}>"
            if isinstance(value, Gst.Caps):
                value = value.to_string()
            prop_lines.append(f"  {pspec.name} = {value!r}")
        logger.debug("%s appsrc '%s' properties:\n%s", log_prefix, appsrc.get_name(), "\n".join(prop_lines))
    # Initialize Picamera2

    cam_info = Picamera2.global_camera_info()
    sensor_index = camera_config.sensor_index
    if sensor_index >= len(cam_info):
        logger.error("%s sensor_index=%d but only %d cameras visible: %s",
                     log_prefix, sensor_index, len(cam_info), cam_info)
        if on_failure is not None:
            on_failure()
        return
    camera_model = cam_info[sensor_index]['Model']
    tuning_file = camera_config.tuning_file or f"/usr/share/libcamera/ipa/rpi/pisp/{camera_model}_noir.json"
    logger.info("%s opening Picamera2(camera_num=%d, model=%s, tuning=%s)",
                log_prefix, sensor_index, camera_model, tuning_file)

    with Picamera2(camera_num=sensor_index, tuning=tuning_file) as picam2:
        if picamera_config is None:
            # Single ISP output stream. The inference path only ever consumed one
            # 1280x720 RGB frame; the old config also allocated an identical, UNUSED
            # `lores` stream which doubled ISP output bandwidth for nothing. Capture
            # `main` only. buffer_count (default 2, the floor) shortens the camera
            # pipeline depth (fewer frames in flight => lower worst-case capture
            # latency); each request is released immediately after copying. Configurable
            # via Config.Camera.buffer_count; see camera-stage-a-latency.md.
            main = {'size': (capture_width, capture_height), 'format': 'RGB888'}
            controls = {'FrameRate': capture_target_fps}
            logger.info("%s buffer_count=%d", log_prefix, buffer_count)
            config = picam2.create_preview_configuration(main=main, controls=controls, buffer_count=buffer_count)
        else:
            config = picamera_config
        # Configure the camera with the created configuration
        picam2.configure(config)

        def apply_controls(controls_dict : dict):
            # TODO: creck that control is supported first
            picam2.set_controls(controls_dict)

        # L3: ensure AE/AWB are explicitly enabled. picam2 defaults are
        # usually "auto" but we want guaranteed behavior — without these the
        # inactive camera could end up frozen at a startup exposure that
        # doesn't match the active camera's scene (wide-angle outdoor vs
        # tele-on-sky have very different luminance), and the first frame
        # after a switch would be wildly mis-exposed. Per-camera overrides
        # in `initial_controls` (e.g. a fixed ExposureTime for known-bright
        # sky targets) win over this default.
        merged_initial : dict = {
            'AeEnable': True,
            'AwbEnable': True,
        }
        # Stage-A latency: pin a short manual exposure when a FIXED value is set.
        # Disable AE and set ExposureTime (clamped to the limits); AWB stays on.
        # Applied before per-camera initial_controls so a per-camera value wins.
        # (Auto-pin mode keeps AE on here and pins later in the capture loop.)
        if exposure_pinned:
            fixed_exp, fixed_gain = _clamp_exposure(exposure_time_us, analogue_gain)
            merged_initial['AeEnable'] = False
            merged_initial['ExposureTime'] = fixed_exp
            # Pin gain too when configured (or when the clamp pushed it above 1.0),
            # so a short shutter stays bright enough without lengthening exposure.
            if gain_pinned or fixed_gain > 1.0:
                merged_initial['AnalogueGain'] = fixed_gain
        if camera_config.initial_controls:
            merged_initial.update(camera_config.initial_controls)
        if picamera_controls_initial is not None:
            merged_initial.update(picamera_controls_initial)
        if auto_pin_enabled:
            limits = (f"[{exposure_min_us or '-'}..{exposure_max_us or '-'}]us gain<={gain_max or '-'}")
            logger.info("%s exposure: AUTO-PIN after %.2fs warmup, limits %s", log_prefix,
                        exposure_auto_pin_s, limits)
        else:
            logger.info("%s exposure: %s; gain: %s", log_prefix,
                        f"MANUAL {exposure_time_us}us (AE off)" if exposure_pinned else "auto (AE on)",
                        f"MANUAL {analogue_gain:.2f}" if (exposure_pinned and gain_pinned) else "auto")
        apply_controls(merged_initial)

        # GStreamer caps for the shared appsrc.
        # - Multi-camera path (camera_switcher provided): caps are set ONCE by
        #   GStreamerApp.run() from the switcher's shared config, BEFORE any
        #   producer starts. We skip setting them here to avoid two threads
        #   racing on the same property.
        # - Legacy single-camera path: nobody else sets caps, so we do it here,
        #   based on the captured stream's actual size/format.
        capture_stream = config['main']
        format_str = 'RGB' if capture_stream['format'] == 'RGB888' else capture_video_format
        width, height = capture_stream['size']
        logger.debug("%s Picamera2 configuration: width=%s, height=%s, format=%s",
                     log_prefix, width, height, format_str)
        if camera_switcher is None:
            appsrc.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw, format={format_str}, width={width}, height={height}, "
                    f"framerate={capture_target_fps}/1, pixel-aspect-ratio=1/1"
                )
            )

        sensor_timestamp_caps = Gst.Caps.from_string("timestamp/x-picamera2-sensor")
        unix_timestamp_caps = Gst.Caps.from_string("timestamp/x-unix")
        frame_id_caps = Gst.Caps.from_string("frame-id/x-picamera2")
        camera_id_caps = Gst.Caps.from_string("camera-id/x-picamera2")

        picam2.start()
        frame_count = 0

        # Auto-pin runtime state. The warmup is timed from when the camera became
        # ACTIVE, and we re-pin on each (re)activation so a switched-to camera
        # adapts to its own scene (see the activation transition below).
        auto_pin_done = False
        active_since_monotonic = time.monotonic()

        # Kept only to preserve the existing sensor-timestamp reference meta
        # (used downstream for sensor→detection latency math). buffer.pts is
        # NOT derived from this anymore — see _shared_pts_ns above for why.
        first_sensor_timestamp_ns = 0
        prev_frame_timestamp_ns = 0

        # Dual-camera idle throttle. When this camera is INACTIVE (the other
        # one is feeding inference), we still need it captured so AGC/AWB stay
        # converged for an instant switch — but at the full target_fps the
        # idle camera burned ~half the ISP/CPU budget that the active path
        # needed. Set a low FrameRate while inactive; restore on activation.
        #
        # Switch-in latency is bounded by one inactive frame interval (picam2
        # latches a new FrameRate only on the next ISP cycle). 5 fps gave a
        # ~200 ms worst-case wait between set_active() and the first frame of
        # the new active camera reaching inference; 15 fps brings that down to
        # ~67 ms. With L1 above the per-inactive-frame cost is now small
        # (capture+release only — no make_array / cvtColor / push), so we can
        # spend the extra ISP cycles. Bonus: 3x faster AE/AWB convergence on
        # the inactive camera, narrowing the post-switch exposure transient.
        INACTIVE_FPS = 15
        was_active : bool | None = None  # forces the initial control push on first iteration
        logger.debug("picamera_process started")
        last_alive_log_monotonic = time.monotonic()
        last_alive_log_frame_count = 0
        # Timing breakdown to locate the fps bottleneck: capture wait (camera/ISP) vs
        # post-capture loop work (cvtColor/tobytes/appsrc push). Reset each alive window.
        capture_time_accum = 0.0
        processing_time_accum = 0.0
        emit_time_accum = 0.0  # subset of loop_proc spent inside appsrc push-buffer
        while True:
            # TODO(vnemkov): only set if camera actually supports those:
            # Must set AF params AFTER picam2.start() above, otherwise it doesn't work
#            apply_controls({
#                "AfMode": picamera_controls.AfModeEnum.Manual,
#                "AfRange": controls.AfRangeEnum.Full,
#                "LensPosition": 6,
#            })

            if picamera_controls_per_frame_callback is not None:
                picamera_controls = picamera_controls_per_frame_callback(frame_count)
                if picamera_controls:
                    apply_controls(picamera_controls)

            # Apply FrameRate based on active/inactive transition. Cheap call;
            # only invoked when the active flag actually flips. picam2 latches
            # the new rate on the next ISP cycle (~one frame interval), which
            # is fast enough that the first push after a switch is at full
            # rate already (we ramp up just before becoming active).
            if camera_switcher is not None:
                is_active = camera_switcher.is_active(camera_id)
                if is_active != was_active:
                    new_fps = capture_target_fps if is_active else INACTIVE_FPS
                    transition_controls = {'FrameRate': new_fps}
                    # L3: on inactive -> active, also re-assert AE/AWB so the
                    # algorithms reconverge on the active scene under the new
                    # (higher) cadence instead of riding the inactive estimate.
                    # Cheap; only triggered on the flag flip. When exposure is
                    # pinned we keep AE OFF (re-enabling it would unpin the
                    # shutter and bring back the latency/jitter we removed).
                    if is_active and was_active is False:
                        # Re-enable AE on activation UNLESS a fixed value is pinned.
                        # For auto-pin we also want AE on (to warm up) and we
                        # restart the warmup so the camera re-pins to THIS scene.
                        if not exposure_pinned:
                            transition_controls['AeEnable'] = True
                        transition_controls['AwbEnable'] = True
                        if auto_pin_enabled:
                            auto_pin_done = False
                            active_since_monotonic = time.monotonic()
                    apply_controls(transition_controls)
                    logger.info("%s FrameRate -> %d fps (active=%s)", log_prefix, new_fps, is_active)
                    was_active = is_active

            capture_started_monotonic = time.monotonic()
            try:
                request = picam2.capture_request(wait=capture_timeout_s)
            except TimeoutError:
                logger.error(
                    "%s picam2.capture_request timed out after %.2fs at frame #%d — camera appears stalled, exiting picamera_thread",
                    log_prefix, capture_timeout_s, frame_count,
                )
                if on_failure is not None:
                    on_failure()
                break

            capture_elapsed_s = time.monotonic() - capture_started_monotonic
            if capture_elapsed_s > slow_capture_warn_s:
                logger.warning(
                    "%s picam2.capture_request took %.3fs (frame #%d) — exceeds %.3fs threshold; camera may be stalling",
                    log_prefix, capture_elapsed_s, frame_count, slow_capture_warn_s,
                )
            capture_time_accum += capture_elapsed_s
            proc_start_monotonic = time.monotonic()

            frame_data = None
            frame_meta = None
            frame_timestamp_ns = 0
            try:
                # L1: skip the ~2.7 MB make_array copy + metadata fetch entirely
                # when this camera is inactive. The ISP keeps running (so AGC/AWB
                # stay warm); we only need to release the ISP buffer back to the
                # pool. `continue` inside `try` still runs the `finally` below.
                if camera_switcher is not None and not camera_switcher.is_active(camera_id):
                    frame_count += 1
                    processing_time_accum += time.monotonic() - proc_start_monotonic
                    continue
                frame_data = request.make_array("main")
                frame_meta = request.get_metadata()
            finally:
                request.release()

            if frame_data is None:
                logger.error("%s Failed to capture frame #%s.", log_prefix, frame_count)
                if on_failure is not None:
                    on_failure()
                break

            # Auto-pin: once AE has had `exposure_auto_pin_s` to converge on the
            # active scene, read back its ExposureTime/AnalogueGain, clamp to the
            # configured limits (shifting clipped exposure into gain), and pin them
            # — turning AE off for a deterministic, short, scene-adapted shutter.
            if auto_pin_enabled and not auto_pin_done and frame_meta is not None:
                if (time.monotonic() - active_since_monotonic) >= exposure_auto_pin_s:
                    meas_exp = frame_meta.get("ExposureTime")
                    meas_gain = frame_meta.get("AnalogueGain")
                    if meas_exp:
                        pin_exp, pin_gain = _clamp_exposure(meas_exp, meas_gain)
                        apply_controls({'AeEnable': False,
                                        'ExposureTime': pin_exp,
                                        'AnalogueGain': pin_gain})
                        auto_pin_done = True
                        logger.info("%s auto-pin after %.2fs: measured exp=%sus gain=%.2f -> "
                                    "pinned exp=%dus gain=%.2f (AE off)", log_prefix,
                                    exposure_auto_pin_s, meas_exp, float(meas_gain or 0.0),
                                    pin_exp, pin_gain)

            # Convert framontigue data if necessary
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame_bytes = frame.tobytes()

            # Drop one of the two buffer copies. The old new_wrapped(frame.tobytes())
            # copied the bytes a SECOND time into GstMemory; new_wrapped_full instead
            # references frame_bytes directly (transfer none) and keeps it alive by
            # taking it as user_data with a destroy-notify. Measured ~2.5x faster than
            # new_wrapped on dev. Signature: (flags, data, maxsize, offset, user_data,
            # notify) — 6 args (no `size`; inferred from data length). NOTE: true
            # zero-copy straight from the numpy array is NOT possible via PyGObject
            # (it marshals the array param as a sequence of ints) so the one tobytes()
            # copy is unavoidable here.
            buffer = Gst.Buffer.new_wrapped_full(
                Gst.MemoryFlags.READONLY, frame_bytes, len(frame_bytes), 0,
                frame_bytes, _buffer_keepalive_noop,
            )

            # Set buffer PTS and duration
            # buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, target_fps)

            if frame_meta is not None:
                sensor_timestamp_ns = frame_meta.get("SensorTimestamp", None)
                # Multi-camera correctness: derive buffer.pts from a process-
                # wide monotonic clock instead of per-camera sensor TS so the
                # pipeline sees a single monotonic stream even when the active
                # camera switches. The original sensor TS is still attached
                # as ref-meta so downstream sensor-latency math is unchanged.
                now_monotonic_ns = time.monotonic_ns()
                frame_timestamp_ns = _shared_pts_ns(now_monotonic_ns)
                if sensor_timestamp_ns is not None and first_sensor_timestamp_ns == 0:
                    first_sensor_timestamp_ns = sensor_timestamp_ns

                buffer.pts = frame_timestamp_ns
                buffer.dts = frame_timestamp_ns
                buffer.duration = Gst.CLOCK_TIME_NONE ## ?

                if sensor_timestamp_ns is not None:
                    buffer.add_reference_timestamp_meta(sensor_timestamp_caps, sensor_timestamp_ns, Gst.CLOCK_TIME_NONE)
                buffer.add_reference_timestamp_meta(unix_timestamp_caps, now_monotonic_ns, Gst.CLOCK_TIME_NONE)
                buffer.add_reference_timestamp_meta(frame_id_caps, frame_count, Gst.CLOCK_TIME_NONE)
                # Tag each buffer with the producing camera so the user-callback
                # can populate Detections.camera_id (and so downstream caches
                # can be purged the moment a different camera's frame arrives).
                buffer.add_reference_timestamp_meta(camera_id_caps, camera_id, Gst.CLOCK_TIME_NONE)

                buffer.offset = frame_count

            # Push the buffer to appsrc
            emit_start_monotonic = time.monotonic()
            ret = appsrc.emit('push-buffer', buffer)
            emit_time_accum += time.monotonic() - emit_start_monotonic
            if ret == Gst.FlowReturn.FLUSHING:
                break
            if ret != Gst.FlowReturn.OK:
                logger.error("%s Failed to push buffer: %s", log_prefix, ret)
                if on_failure is not None:
                    on_failure()
                break
            processing_time_accum += time.monotonic() - proc_start_monotonic
            frame_count += 1

            if alive_log_every_n_frames and frame_count % alive_log_every_n_frames == 0:
                now_monotonic = time.monotonic()
                window_s = now_monotonic - last_alive_log_monotonic
                window_frames = frame_count - last_alive_log_frame_count
                window_fps = window_frames / window_s if window_s > 0 else 0.0
                avg_capture_ms = (capture_time_accum / window_frames * 1000.0) if window_frames else 0.0
                avg_proc_ms = (processing_time_accum / window_frames * 1000.0) if window_frames else 0.0
                avg_emit_ms = (emit_time_accum / window_frames * 1000.0) if window_frames else 0.0
                exp_us = frame_meta.get("ExposureTime") if frame_meta else None
                gain = frame_meta.get("AnalogueGain") if frame_meta else None
                logger.info(
                    "%s picamera_thread alive: pushed %d frames so far, last %d frames in %.2fs (%.1f fps); "
                    "per-frame avg: capture_wait=%.1fms loop_proc=%.1fms [convert=%.1fms emit=%.1fms] (interval=%.1fms); "
                    "exposure=%sus gain=%s",
                    log_prefix, frame_count, window_frames, window_s, window_fps,
                    avg_capture_ms, avg_proc_ms, avg_proc_ms - avg_emit_ms, avg_emit_ms,
                    (window_s / window_frames * 1000.0) if window_frames else 0.0,
                    exp_us, (f"{gain:.2f}" if isinstance(gain, (int, float)) else gain),
                )
                last_alive_log_monotonic = now_monotonic
                last_alive_log_frame_count = frame_count
                capture_time_accum = 0.0
                processing_time_accum = 0.0
                emit_time_accum = 0.0


def disable_qos(pipeline):
    """
    Iterate through all elements in the given GStreamer pipeline and set the qos property to False
    where applicable.
    When the 'qos' property is set to True, the element will measure the time it takes to process each buffer and will drop frames if latency is too high.
    We are running on long pipelines, so we want to disable this feature to avoid dropping frames.
    :param pipeline: A GStreamer pipeline object
    """
    # Ensure the pipeline is a Gst.Pipeline instance
    if not isinstance(pipeline, Gst.Pipeline):
        logger.warning("The provided object is not a GStreamer Pipeline")
        return

    # Iterate through all elements in the pipeline
    it = pipeline.iterate_elements()
    while True:
        result, element = it.next()
        if result != Gst.IteratorResult.OK:
            break

        # Check if the element has the 'qos' property
        if 'qos' in GObject.list_properties(element):
            # Set the 'qos' property to False
            element.set_property('qos', False)
            logger.debug(f"Set qos to False for %s", element.get_name())

# This function is used to display the user data frame
def display_user_data_frame(user_data: app_callback_class):
    while user_data.running:
        frame = user_data.get_frame()
        if frame is not None:
            cv2.imshow("User Frame", frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data, parser=None, inference=None):
        if parser == None:
            parser = get_default_parser()

        parser.add_argument(
            "--labels-json",
            default=None,
            help="Path to costume labels JSON file",
        )

        # Call the parent class constructor
        super().__init__(parser, user_data)
        # Additional initialization code can be added here
        # Set Hailo parameters these parameters should be set based on the model used
        self.batch_size = 1
        # Model + NMS come from config.inference (config.yaml); CLI --hef-path /
        # --labels-json still override for ad-hoc runs. `inference` is None only
        # when no config is passed (legacy callers) — then fall back to defaults.
        nms_score_threshold = inference.nms_score_threshold if inference is not None else 0.3
        nms_iou_threshold   = inference.nms_iou_threshold   if inference is not None else 0.45


        # Determine the architecture if not specified
        if self.options_menu.arch is None:
            detected_arch = detect_hailo_arch()
            if detected_arch is None:
                raise ValueError("Could not auto-detect Hailo architecture. Please specify --arch manually.")
            self.arch = detected_arch
            logger.info("Auto-detected Hailo architecture: %s", self.arch)
        else:
            self.arch = self.options_menu.arch

        if self.options_menu.hef_path is not None:
            self.hef_path = self.options_menu.hef_path
        elif inference is not None:
            self.hef_path = str(inference.hef_model_path)
        else:
            self.hef_path = get_resource_path(DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME)

            # Set the post-processing shared object file
        self.post_process_so = get_resource_path(
            DETECTION_PIPELINE, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME
        )

        self.post_function_name = DETECTION_POSTPROCESS_FUNCTION
        # User-defined label JSON file: CLI override, else config.inference, else none.
        if self.options_menu.labels_json is not None:
            self.labels_json = self.options_menu.labels_json
        elif inference is not None and inference.labels_json is not None:
            self.labels_json = str(inference.labels_json)
        else:
            self.labels_json = None

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle(DETECTION_APP_TITLE)

        self.create_pipeline()

    def get_output_pipeline_string(self, video_sink : str, sync : str = 'true', show_fps : str = 'true'):
        return DISPLAY_PIPELINE(video_sink=video_sink, sync=sync, show_fps=show_fps)

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(
                video_source=self.video_source,
                video_width=self.video_width,
                video_height=self.video_height,
                frame_rate=self.frame_rate,
                sync=self.sync,
                # do_timestamp=True
        )
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str)
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)

        tracker_pipeline = ''
        if False:
            tracker_pipeline = TRACKER_PIPELINE(
                class_id=1,
                keep_past_metadata='false',
                qos='false',
                # attempt to remove outdated detections
                keep_tracked_frames=0,
                keep_lost_frames=0, # Lost tracks dropped immediately, no shadow frames =0
                keep_new_frames=1, # Unconfirmed new tracks discarded faster with =1
                iou_thr=0.6,
                kalman_dist_thr=0.6
            )
            tracker_pipeline = f'{tracker_pipeline} ! '

        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        if True: #self.source_type == 'rpi':
            # production case == video from camera, use custom pipeline
            display_pipeline = self.get_output_pipeline_string(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)
        else:
            # here custom pipeline might break rewinding of initial source, use default display pipeline
            display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        # Split the post-inference stream with a tee so the (software, CPU-heavy)
        # video recorder can never back-pressure the detection path.
        #
        # Why the isolating queue goes AFTER the tee, not before it:
        # a tee fans every buffer out to all of its src pads *synchronously, on the
        # thread that pushed the buffer in*. A queue placed BEFORE the tee would only
        # decouple the tee from its upstream — both branches would still run on that one
        # post-queue thread, so a slow encoder would still stall detection. Placing the
        # large leaky queue AFTER the tee, on the recording branch, gives the encoder its
        # own streaming thread; the tee's push into a leaky queue returns immediately
        # (dropping the oldest frame when full), so detection never waits on the encoder.
        #
        #   output_tee
        #     ├─ detection branch:  identity (probe) -> fakesink   (terminates ASAP)
        #     └─ recording branch:  BIG leaky queue -> encoder -> splitmuxsink
        #
        # The recording queue is leaky=downstream so a lagging encoder drops the oldest
        # *raw* frame instead of blocking the tee. Sized in buffers (raw RGB ~2.6 MB each
        # at 1280x720) to bound RAM (~30 buffers ≈ 80 MB); raise it to ride out longer
        # encoder stalls without dropping recorded frames, at the cost of RAM only — it
        # adds no detection latency because it is leaky.
        recording_input_queue = (
            'queue name=recording_input_q leaky=downstream '
            'max-size-buffers=120 max-size-bytes=0 max-size-time=0'
        )

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} '
            f'tee name=output_tee '
            f'output_tee. ! {user_callback_pipeline} ! fakesink name=detection_sink sync=false async=false '
            f'output_tee. ! {recording_input_queue} ! {display_pipeline}'
        )

        logger.debug(pipeline_string)
        return pipeline_string