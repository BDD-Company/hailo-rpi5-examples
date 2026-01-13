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
from gi.repository import Gst, GLib, GObject


from hailo_apps.hailo_app_python.core.common.installation_utils import detect_hailo_arch
from hailo_apps.hailo_app_python.core.common.core import get_default_parser, get_resource_path
from hailo_apps.hailo_app_python.core.common.defines import DETECTION_APP_TITLE, DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME, DETECTION_POSTPROCESS_FUNCTION
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, TRACKER_PIPELINE, USER_CALLBACK_PIPELINE, DISPLAY_PIPELINE

# for SOURCE_PIPELINE
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import QUEUE, get_camera_resulotion


# Based on hailo_app_python/core/gstreamer/gstreamer_app.py

# Absolute import for your local helper
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


def SOURCE_PIPELINE(video_source, video_width=640, video_height=640,
                    name='source', no_webcam_compression=False,
                    frame_rate=30, sync=True,
                    video_format='RGB'):
    """
    Creates a GStreamer pipeline string for the video source with a separate fps caps
    for frame rate control.

    Args:
        video_source (str): The path or device name of the video source.
        video_width (int, optional): The width of the video. Defaults to 640.
        video_height (int, optional): The height of the video. Defaults to 640.
        video_format (str, optional): The video format. Defaults to 'RGB'.
        name (str, optional): The prefix name for the pipeline elements. Defaults to 'source'.

    Returns:
        str: A string representing the GStreamer pipeline for the video source.
    """
    source_type = get_source_type(video_source)

    if source_type == 'usb':
        if no_webcam_compression:
            # When using uncompressed format, only low resolution is supported
            source_element = (
                f'v4l2src device={video_source} name={name} ! '
                f'video/x-raw, width=640, height=480 ! '
                'videoflip name=videoflip video-direction=horiz ! '
            )
        else:
            # Use compressed format for webcam
            width, height = get_camera_resulotion(video_width, video_height)
            source_element = (
                f'v4l2src device={video_source} name={name} ! image/jpeg, framerate=30/1, width={width}, height={height} ! '
                f'{QUEUE(name=f"{name}_queue_decode")} ! '
                f'decodebin name={name}_decodebin ! '
                f'videoflip name=videoflip video-direction=horiz ! '
            )
    elif source_type == 'rpi':
        source_element = (
            f'appsrc name=app_source is-live=true leaky-type=downstream max-buffers=3 ! '
            # 'videoflip name=videoflip video-direction=horiz ! '
            f'video/x-raw, format={video_format}, width={video_width}, height={video_height} ! '
        )
    elif source_type == 'libcamera':
        source_element = (
            f'libcamerasrc name={name} ! '
            f'video/x-raw, format={video_format}, width=1536, height=864 ! '
        )
    elif source_type == 'ximage':
        source_element = (
            f'ximagesrc xid={video_source} ! '
            f'{QUEUE(name=f"{name}queue_scale_")} ! '
            f'videoscale ! '
        )
    else:
        source_element = (
            f'filesrc location="{video_source}" name={name} ! '
            f'{QUEUE(name=f"{name}_queue_decode")} ! '
            f'decodebin name={name}_decodebin ! '
        )

    # Set up the fps caps.
    # If sync is True, constrain the rate with the given frame_rate.
    # Otherwise, pass through (no framerate limitation).
    if sync:
        fps_caps = f"video/x-raw, framerate={frame_rate}/1"
    else:
        fps_caps = "video/x-raw"

    source_pipeline = (
        f'{source_element} '
        f'{QUEUE(name=f"{name}_scale_q")} ! '
        f'videoscale name={name}_videoscale n-threads=2 ! '
        f'{QUEUE(name=f"{name}_convert_q")} ! '
        f'videoconvert n-threads=3 name={name}_convert qos=false ! '
        f'video/x-raw, pixel-aspect-ratio=1/1, format={video_format}, '
        f'width={video_width}, height={video_height} '
        # f'videorate name={name}_videorate ! capsfilter name={name}_fps_caps caps="{fps_caps}" '
    )

    return source_pipeline

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
        self.error_occurred = False
        self.pipeline_latency = 300  # milliseconds

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
            os.environ["GST_DEBUG_DUMP_DOT_DIR"] = os.getcwd()

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

    def bus_call(self, bus, message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream")
            self.on_eos()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"got error from Gstreamer: %s : %s", err, debug)

            self.error_occurred = True
            self.shutdown()
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


    def shutdown(self, signum=None, frame=None):
        logger.info("Shutting down... Hit Ctrl-C again to force quit.")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
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
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "pipeline")
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

        if self.source_type == RPI_NAME_I:
            picam_thread = threading.Thread(target=picamera_thread, args=(self.pipeline, self.video_width, self.video_height, self.video_format))
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
            if self.error_occurred:
                logger.error("Exiting with error...")
                sys.exit(1)
            else:
                logger.info("Exiting...")
                sys.exit(0)


def picamera_thread(pipeline, video_width, video_height, video_format, picamera_config=None, target_fps = 30, picamera_controls_initial = None, picamera_controls_per_frame_callback = None):
    appsrc = pipeline.get_by_name("app_source")
    appsrc.set_property("is-live", True)
    appsrc.set_property("format", Gst.Format.TIME)
    logger.debug("appsrc properties: %s", appsrc)
    # Initialize Picamera2

    with Picamera2() as picam2:
        if picamera_config is None:
            # Default configuration
            main = {'size': (1280, 720), 'format': 'RGB888'}
            lores = {'size': (video_width, video_height), 'format': 'RGB888'}
            controls = {'FrameRate': target_fps}
            config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
        else:
            config = picamera_config
        # Configure the camera with the created configuration
        picam2.configure(config)

        def apply_controls(controls_dict : dict):
            # TODO: creck that control is supported first
            picam2.set_controls(controls_dict)

        if picamera_controls_initial is not None:
            apply_controls(picamera_controls_initial)

        # Update GStreamer caps based on 'lores' stream
        lores_stream = config['lores']
        format_str = 'RGB' if lores_stream['format'] == 'RGB888' else video_format
        width, height = lores_stream['size']
        logger.debug("Picamera2 configuration: width=%s, height=%s, format=%s", width, height, format_str)
        appsrc.set_property(
            "caps",
            Gst.Caps.from_string(
                f"video/x-raw, format={format_str}, width={width}, height={height}, "
                f"framerate={target_fps}/1, pixel-aspect-ratio=1/1"
            )
        )

        sensor_timestamp_caps = Gst.Caps.from_string("timestamp/x-picamera2-sensor")
        unix_timestamp_caps = Gst.Caps.from_string("timestamp/x-unix")
        frame_id_caps = Gst.Caps.from_string("frame-id/x-picamera2")

        picam2.start()
        frame_count = 0

        # used to convert from absolute frame time of Picamera2 to relative of Gstreamer (starting from 0)
        first_frame_timestamp_ns = 0
        prev_frame_timestamp_ns = 0
        logger.debug("picamera_process started")
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

            request = picam2.capture_request()

            frame_data = None
            frame_meta = None
            frame_timestamp_ns = 0
            try:
                frame_data = request.make_array("lores")
                frame_meta = request.get_metadata()
            finally:
                request.release()

            frame_data = picam2.capture_array('lores')
            # frame_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            if frame_data is None:
                logger.error("Failed to capture frame #%s.", frame_count)
                break

            # Convert framontigue data if necessary
            frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            frame = np.asarray(frame)

            # Create Gst.Buffer by wrapping the frame data
            buffer = Gst.Buffer.new_wrapped(frame.tobytes())

            # Set buffer PTS and duration
            # buffer_duration = Gst.util_uint64_scale_int(1, Gst.SECOND, target_fps)

            if frame_meta is not None:
                sensor_timestamp_ns = frame_meta.get("SensorTimestamp", None)
                if sensor_timestamp_ns is not None:
                    if first_frame_timestamp_ns == 0:
                        first_frame_timestamp_ns = sensor_timestamp_ns
                    frame_timestamp_ns = sensor_timestamp_ns - first_frame_timestamp_ns
                    logging.debug("frame #%d\ttimestamp from sensor: %s, frame timestamp: %s", frame_count, sensor_timestamp_ns, frame_timestamp_ns)

                buffer.pts = frame_timestamp_ns
                buffer.dts = frame_timestamp_ns
                buffer.duration = Gst.CLOCK_TIME_NONE ## ?

                buffer.add_reference_timestamp_meta(sensor_timestamp_caps, sensor_timestamp_ns, Gst.CLOCK_TIME_NONE)
                buffer.add_reference_timestamp_meta(unix_timestamp_caps, time.monotonic_ns(), Gst.CLOCK_TIME_NONE)
                buffer.add_reference_timestamp_meta(frame_id_caps, frame_count, Gst.CLOCK_TIME_NONE)

                buffer.offset = frame_count

            # Push the buffer to appsrc
            ret = appsrc.emit('push-buffer', buffer)
            if ret == Gst.FlowReturn.FLUSHING:
                break
            if ret != Gst.FlowReturn.OK:
                logger.error("Failed to push buffer: %s", ret)
                break
            frame_count += 1


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
    def __init__(self, app_callback, user_data, parser=None):
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
        self.batch_size = 2
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45


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
        else:
            self.hef_path = get_resource_path(DETECTION_PIPELINE, RESOURCES_MODELS_DIR_NAME)


            # Set the post-processing shared object file
        self.post_process_so = get_resource_path(
            DETECTION_PIPELINE, RESOURCES_SO_DIR_NAME, DETECTION_POSTPROCESS_SO_FILENAME
        )



        self.post_function_name = DETECTION_POSTPROCESS_FUNCTION
        # User-defined label JSON file
        self.labels_json = self.options_menu.labels_json

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
        source_pipeline = SOURCE_PIPELINE(video_source=self.video_source,
                                          video_width=self.video_width, video_height=self.video_height,
                                          frame_rate=self.frame_rate, sync=self.sync)
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str)
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(class_id=1)
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = self.get_output_pipeline_string(video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps)

        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )

        logger.debug(pipeline_string)
        return pipeline_string