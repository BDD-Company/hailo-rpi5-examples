#!/usr/bin/env python3
"""Debug facility that exercises app.py's app_callback with recorded data.

Replays a recorded flight by feeding app_callback with mock GStreamer
buffers built from log-parsed detections and video frames.  This extends
debug_drone_controller.py to also cover the GStreamer callback code path:
ByteTracker, frame deduplication, detection extraction, timestamp handling.

Usage:
    cd BDD
    python debug_app_callback.py path/to/_DEBUG_dir/
    python debug_app_callback.py path/to/logfile.log --video path/to/video.mp4
"""

import argparse
import ast
import sys
import types
import threading
import time as real_time_module
from math import nan
from pathlib import Path

import cv2
import numpy as np


# ===================================================================
# Module stubs for off-device development
# ===================================================================
# Must run BEFORE importing app.py which pulls in hailo, gi, hailo_apps.

def _ensure_module(name, attrs=None):
    """Ensure a module exists in sys.modules with optional attributes."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        # Wire parent -> child references
        parts = name.split('.')
        for i in range(len(parts) - 1):
            parent_name = '.'.join(parts[:i + 1])
            child_key = parts[i + 1]
            parent = sys.modules.get(parent_name)
            child_full = '.'.join(parts[:i + 2])
            if parent and child_full in sys.modules:
                setattr(parent, child_key, sys.modules[child_full])
    mod = sys.modules[name]
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


def _setup_gst_stubs():
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst  # noqa: F401
        return False
    except (ImportError, ValueError):
        pass

    _ensure_module('gi', {'require_version': lambda *a, **kw: None})

    class _Caps:
        def __init__(self, s=''):
            self._s = s
        @staticmethod
        def from_string(s):
            return _Caps(s)

    gst_attrs = {
        'Caps': _Caps,
        'PadProbeReturn': type('PadProbeReturn', (), {'OK': 0, 'DROP': 1}),
        'ReferenceTimestampMeta': type('ReferenceTimestampMeta', (), {'timestamp': 0}),
        'BUFFER_OFFSET_NONE': 0xFFFFFFFFFFFFFFFF,
        'CLOCK_TIME_NONE':    0xFFFFFFFFFFFFFFFF,
        'Buffer': type('Buffer', (), {}),
        'Pad': type('Pad', (), {}),
        'PadProbeInfo': type('PadProbeInfo', (), {}),
        'Pipeline': type('Pipeline', (), {}),
        'init': lambda *a: None,
        'parse_launch': lambda s: None,
        'SECOND': 1_000_000_000,
        'MSECOND': 1_000_000,
        'Format': type('Format', (), {'TIME': 0}),
        'SeekFlags': type('SeekFlags', (), {'FLUSH': 1}),
        'State': type('State', (), {'NULL': 0, 'READY': 1, 'PAUSED': 2, 'PLAYING': 3}),
        'MessageType': type('MessageType', (), {'EOS': 1, 'ERROR': 2, 'QOS': 3}),
        'FlowReturn': type('FlowReturn', (), {'OK': 0, 'FLUSHING': -1}),
        'IteratorResult': type('IteratorResult', (), {'OK': 0}),
        'DebugGraphDetails': type('DebugGraphDetails', (), {'VERBOSE': 0}),
        'debug_bin_to_dot_file': lambda *a: None,
        'util_uint64_scale_int': lambda a, b, c: 0,
    }
    gst_mod = _ensure_module('gi.repository.Gst', gst_attrs)
    glib_mod = _ensure_module('gi.repository.GLib', {
        'MainLoop': type('MainLoop', (), {'run': lambda s: None, 'quit': lambda s: None}),
        'usleep': lambda us: None,
        'idle_add': lambda f: None,
        'timeout_add_seconds': lambda s, f: None,
    })
    _ensure_module('gi.repository.GObject', {'list_properties': lambda e: []})

    repo = _ensure_module('gi.repository')
    repo.Gst = gst_mod
    repo.GLib = glib_mod
    repo.GObject = sys.modules['gi.repository.GObject']
    return True


def _setup_hailo_stubs():
    try:
        import hailo  # noqa: F401
        return False
    except ImportError:
        pass
    _ensure_module('hailo', {
        'HAILO_DETECTION': 'HAILO_DETECTION',
        'get_roi_from_buffer': lambda buf: None,
    })
    return True


def _setup_hailo_apps_stubs():
    try:
        from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad  # noqa: F401
        return False
    except ImportError:
        pass

    import multiprocessing

    for name in [
        'hailo_apps',
        'hailo_apps.hailo_app_python',
        'hailo_apps.hailo_app_python.core',
        'hailo_apps.hailo_app_python.core.common',
        'hailo_apps.hailo_app_python.core.common.buffer_utils',
        'hailo_apps.hailo_app_python.core.common.core',
        'hailo_apps.hailo_app_python.core.common.defines',
        'hailo_apps.hailo_app_python.core.common.installation_utils',
        'hailo_apps.hailo_app_python.core.common.camera_utils',
        'hailo_apps.hailo_app_python.core.gstreamer',
        'hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app',
        'hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines',
    ]:
        _ensure_module(name)

    buf = sys.modules['hailo_apps.hailo_app_python.core.common.buffer_utils']
    buf.get_caps_from_pad = lambda pad: ('RGB', 640, 480)
    buf.get_numpy_from_buffer = lambda buf, fmt, w, h: None

    core = sys.modules['hailo_apps.hailo_app_python.core.common.core']
    core.get_default_parser = lambda: argparse.ArgumentParser()
    core.get_resource_path = lambda *a: ''
    core.load_environment = lambda *a: None

    defines = sys.modules['hailo_apps.hailo_app_python.core.common.defines']
    for n in [
        'DETECTION_APP_TITLE', 'DETECTION_PIPELINE', 'RESOURCES_MODELS_DIR_NAME',
        'RESOURCES_SO_DIR_NAME', 'DETECTION_POSTPROCESS_SO_FILENAME',
        'DETECTION_POSTPROCESS_FUNCTION', 'HAILO_RGB_VIDEO_FORMAT',
        'GST_VIDEO_SINK', 'TAPPAS_POSTPROC_PATH_KEY', 'RESOURCES_PATH_KEY',
        'RESOURCES_ROOT_PATH_DEFAULT', 'RESOURCES_VIDEOS_DIR_NAME',
        'BASIC_PIPELINES_VIDEO_EXAMPLE_NAME', 'USB_CAMERA', 'RPI_NAME_I',
    ]:
        if not hasattr(defines, n):
            setattr(defines, n, n)

    sys.modules['hailo_apps.hailo_app_python.core.common.installation_utils'].detect_hailo_arch = lambda: 'hailo8l'
    sys.modules['hailo_apps.hailo_app_python.core.common.camera_utils'].get_usb_video_devices = lambda: []

    class _app_callback_class:
        def __init__(self):
            self.frame_count = 0
            self.use_frame = False
            self.frame_queue = multiprocessing.Queue(maxsize=3)
            self.running = True
        def increment(self):
            self.frame_count += 1
        def get_count(self):
            return self.frame_count

    sys.modules['hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app'].app_callback_class = _app_callback_class
    sys.modules['hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines'].get_source_type = lambda s: 'file'
    return True


def _setup_other_stubs():
    if 'setproctitle' not in sys.modules:
        _ensure_module('setproctitle', {'setproctitle': lambda s: None})
    if 'pipelines' not in sys.modules:
        # `pipelines` only ever builds GStreamer pipeline STRINGS, and replay builds no
        # pipeline — detections come from the log. So nothing app_base imports from here is
        # ever called, and the stub can answer to any name.
        #
        # It used to enumerate six specific symbols, which silently rotted the moment
        # app_base imported a seventh (`QUEUE`): the stub shadows the real pipelines.py, so
        # the import failed with a baffling "cannot import name 'QUEUE' from 'pipelines'
        # (unknown location)" and the whole tool was unusable. __getattr__ (PEP 562) cannot
        # fall out of date the same way.
        _pipelines = _ensure_module('pipelines')
        _pipelines.__getattr__ = lambda name: (lambda *args, **kwargs: '')
    try:
        import mavsdk  # noqa: F401
    except ImportError:
        _mavsdk = _ensure_module('mavsdk')
        _mavsdk.System = type("System", (), {"__init__": lambda self: None})
        _off = _ensure_module('mavsdk.offboard')
        for n in ("PositionNedYaw", "VelocityBodyYawspeed", "Attitude",
                   "VelocityNedYaw", "AttitudeRate"):
            setattr(_off, n, type(n, (), {}))
        _off.OffboardError = type("OffboardError", (Exception,), {})
        _tel = _ensure_module('mavsdk.telemetry')
        _tel.Telemetry = type("Telemetry", (), {})
        _tel.EulerAngle = type("EulerAngle", (), {})
        _tel.LandedState = type("LandedState", (), {"IN_AIR": 1})


# --- Run stubs before any BDD imports ---
_setup_gst_stubs()
_setup_hailo_stubs()
_setup_hailo_apps_stubs()
_setup_other_stubs()


# ===================================================================
# BDD imports (now safe)
# ===================================================================

import app as app_module
from app import app_callback, user_app_callback_class
from frame_order import FrameOrderGuard

import hailo
from gi.repository import Gst  # noqa: F811
from bytetrack import BYTETracker

from debug_drone_controller import (
    parse_log, find_files_in_dir,
    MockMonotonicNs,
    ReplayQueue, InteractiveDisplaySink,
    REPLAY_CONFIG_DEFAULT, load_replay_config, run_replay,
    warn_replayed_config_is_not_the_flights,
)
from parse_config import ConfigError

from flight_debugger import VideoReader
from OverwriteQueue import OverwriteQueue
from debug_output import debug_output_thread

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(threadName)s] %(name)s <%(levelname)s> : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SingleOrRepeatedPathAction(argparse.Action):
    """Store one Path as Path, repeated occurrences as list[Path]."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = getattr(namespace, self.dest, None)
        if current is None:
            setattr(namespace, self.dest, values)
        elif isinstance(current, list):
            current.append(values)
        else:
            setattr(namespace, self.dest, [current, values])


# ===================================================================
# Mock GStreamer / Hailo objects
# ===================================================================

class MockHailoBBox:
    """Minimal hailo bbox: xmin/ymin/xmax/ymax as methods."""
    __slots__ = ('_xmin', '_ymin', '_xmax', '_ymax')

    def __init__(self, xmin, ymin, xmax, ymax):
        self._xmin, self._ymin = xmin, ymin
        self._xmax, self._ymax = xmax, ymax

    def xmin(self): return self._xmin
    def ymin(self): return self._ymin
    def xmax(self): return self._xmax
    def ymax(self): return self._ymax


class MockHailoDetection:
    """Minimal hailo detection: bbox + confidence."""
    __slots__ = ('_bbox', '_confidence')

    def __init__(self, bbox: MockHailoBBox, confidence: float):
        self._bbox = bbox
        self._confidence = confidence

    def get_bbox(self): return self._bbox
    def get_confidence(self): return self._confidence


class MockHailoROI:
    """Minimal hailo ROI containing a list of detections."""
    __slots__ = ('_detections',)

    def __init__(self, detections: list):
        self._detections = detections

    def get_objects_typed(self, type_):
        return self._detections


class MockGstBuffer:
    """Mock GStreamer buffer carrying frame_id via offset."""
    __slots__ = ('offset', 'pts')

    def __init__(self, frame_id: int, pts_ns: int = 0):
        self.offset = frame_id
        self.pts = pts_ns or frame_id

    def get_reference_timestamp_meta(self, caps):
        # Return None; app_callback falls back to buffer.offset for frame_id
        # and time.monotonic_ns() for timestamps.
        return None


class MockGstPadProbeInfo:
    """Mock probe info wrapping a buffer."""
    __slots__ = ('_buffer',)

    def __init__(self, buffer: MockGstBuffer):
        self._buffer = buffer

    def get_buffer(self):
        return self._buffer


class MockGstPad:
    """Mock pad (caps are provided via monkey-patched get_caps_from_pad)."""
    pass


# ===================================================================
# Monkey-patching: per-frame data for mocked functions
# ===================================================================

_frame_ctx = threading.local()


def _mock_get_caps_from_pad(pad):
    return (
        getattr(_frame_ctx, 'format', 'RGB'),
        getattr(_frame_ctx, 'width', 640),
        getattr(_frame_ctx, 'height', 480),
    )


def _mock_get_numpy_from_buffer(buffer, fmt, width, height):
    return getattr(_frame_ctx, 'frame', np.zeros((height, width, 3), dtype=np.uint8))


def _mock_get_roi_from_buffer(buffer):
    return getattr(_frame_ctx, 'roi', MockHailoROI([]))


def install_mock_patches():
    """Replace GStreamer/Hailo data-extraction functions in app module."""
    app_module.get_caps_from_pad = _mock_get_caps_from_pad
    app_module.get_numpy_from_buffer = _mock_get_numpy_from_buffer
    hailo.get_roi_from_buffer = _mock_get_roi_from_buffer


def set_frame_context(frame: np.ndarray, detections: list):
    """Populate thread-local context before calling app_callback."""
    h, w = frame.shape[:2]
    _frame_ctx.format = 'RGB'
    _frame_ctx.width = w
    _frame_ctx.height = h
    _frame_ctx.frame = frame

    hailo_dets = []
    for det in detections:
        bbox = MockHailoBBox(
            det.bbox.left_edge,
            det.bbox.top_edge,
            det.bbox.right_edge,
            det.bbox.bottom_edge,
        )
        hailo_dets.append(MockHailoDetection(bbox, det.confidence))
    _frame_ctx.roi = MockHailoROI(hailo_dets)


# ===================================================================
# Pre-process: run all frames through app_callback, collect Detections
# ===================================================================

def run_app_callback_on_all_frames(
    frame_data_list: list[dict],
    mock_pad: MockGstPad,
    user_data: user_app_callback_class,
    mock_monotonic: MockMonotonicNs,
) -> list:
    """Run app_callback synchronously for every frame and return Detections list."""
    from queue import Queue as _SyncQueue
    import queue as _queue_mod

    # Swap user_data's queue for a plain Queue so we can get() without blocking
    orig_queue = user_data.detections_queue
    sync_q = _SyncQueue()
    user_data.detections_queue = sync_q

    detections_list = []
    for fdata in frame_data_list:
        fid   = fdata['frame_id']
        ts_ns = fdata['timestamp_ns']
        mock_monotonic.set_frame(ts_ns)
        set_frame_context(fdata['frame'], fdata['detections'])
        mock_buffer = MockGstBuffer(fid, ts_ns)
        mock_info   = MockGstPadProbeInfo(mock_buffer)
        app_callback(mock_pad, mock_info, user_data)
        try:
            detections_list.append(sync_q.get_nowait())
        except _queue_mod.Empty:
            pass  # frame was deduplicated / skipped by app_callback

    user_data.detections_queue = orig_queue
    logger.info(
        "Pre-processed %d frames → %d Detections objects",
        len(frame_data_list), len(detections_list),
    )
    return detections_list


# ===================================================================
# Build per-frame replay data
# ===================================================================

def build_frame_data_list(frames: dict, video_reader: VideoReader | None) -> list[dict]:
    """Convert parsed log frames + video into per-frame dicts for the feeder."""
    result = []
    for fid in sorted(frames.keys()):
        fd = frames[fid]

        frame = None
        if video_reader is not None and video_reader.available:
            frame = video_reader.read(fid)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result.append({
            'frame_id': fid,
            'timestamp_ns': fd.get('timestamp_ns', 0),
            'detections': fd.get('detections', []),
            'telemetry': fd.get('telemetry', {}),
            'frame': frame,
        })

    logger.info("Built %d frame data entries for app_callback replay", len(result))
    return result


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded flight through app_callback + drone_controller.",
    )
    parser.add_argument(
        "path", type=Path,
        help="Path to a log file (.log) or debug directory",
    )
    parser.add_argument(
        "--video", type=Path, action=SingleOrRepeatedPathAction, default=None,
        help="Path to video file(s) or directory with video files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPLAY_CONFIG_DEFAULT,
        help="Config YAML to replay WITH (default: config.yaml next to this script). "
             "Note this is today's config, not the one the flight flew with.",
    )
    parser.add_argument(
        "--params",
        type=lambda x: dict(ast.literal_eval(x)),
        default=None,
        help="Config overrides as DOTTED paths, e.g. \"{'pd_coeff.p': [4, 4], "
             "'bytetrack.track_thresh': 0.5}\". Validated; unknown keys are an error.",
    )
    parser.add_argument(
        "--autoplay", action='store_true', default=False,
        help="Do NOT wait for user input to advance frames",
    )
    parser.add_argument(
        "--headless", action='store_true', default=False,
        help="No display at all: no window, no keypress. Needs no X/Wayland.",
    )
    args = parser.parse_args()

    # -- Resolve log and video paths --
    if args.path.is_dir():
        log_file, video_files = find_files_in_dir(args.path)
        if log_file is None:
            print(f"No .log file found in {args.path}", file=sys.stderr)
            sys.exit(1)
        video_path = video_files if video_files else None
    else:
        log_file = args.path
        video_path = args.video

    logger.info("Log file: %s", log_file)
    logger.info("Video path: %s", video_path)

    # -- Parse log --
    # parse_log also scrapes the log's `!!! Config:` line; we deliberately do NOT build the
    # control config from it (flat pre-refactor dict on old logs, non-eval-able repr on new
    # ones). See load_replay_config.
    _logged_config, frames, base_ns = parse_log(log_file)
    if not frames:
        print("No frame data found in log file", file=sys.stderr)
        sys.exit(1)
    logger.info("Parsed %d frames from log", len(frames))

    # -- Load the config to replay WITH --
    try:
        config = load_replay_config(args.config, args.params)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    warn_replayed_config_is_not_the_flights(log_file, args.config)

    # -- Open video --
    # app_callback reads pixels (optical refinement, tiling), so video matters more here than
    # in debug_drone_controller. Degrade to blank frames rather than dying, but say so.
    video_reader = VideoReader(video_path)
    if video_reader.available:
        logger.info("Video loaded: %d frames", video_reader.total)
    else:
        logger.warning(
            "!!! No usable video at %s — replaying on BLANK frames. app_callback reads pixels, "
            "so anything image-based (optical refinement) will not behave as it did in flight.",
            video_path,
        )

    # -- Build per-frame data --
    frame_data_list = build_frame_data_list(frames, video_reader)

    # -- Mock time.monotonic_ns --
    # app_callback runs in the pre-pass below and drone_controller in run_replay, but both
    # halves of the replay must see the SAME clock, so it is built here and handed over.
    mock_monotonic = MockMonotonicNs(base_ns)
    mock_time = types.ModuleType("mock_time")
    for attr in dir(real_time_module):
        if not attr.startswith("_"):
            setattr(mock_time, attr, getattr(real_time_module, attr))
    mock_time.monotonic_ns = mock_monotonic
    app_module.time = mock_time     # drone_controller's own patch is run_replay's job

    # -- Install mock patches on app module --
    install_mock_patches()

    # -- Clear app.py's frame-ordering state --
    # A replay starts again at frame 0, so a guard left over from a previous run would
    # reject every frame as a reorder. app_callback looks this up as a module global at
    # call time, so rebinding a fresh guard is enough. (Was `seen_frames.clear()`, from
    # before FrameOrderGuard replaced the old last-10 dedup set.)
    app_module.frame_order_guard = FrameOrderGuard()

    # -- Set up ByteTracker and user_data --
    # From the real Config, exactly as app.py does it. This used to be scraped from flat
    # `bytetrack_*` keys in the log, which no longer exist: every one silently fell back to
    # its default, so the replay tracked with parameters that were nobody's.
    bytetracker = (BYTETracker(**config.bytetrack.tracker_kwargs())
                   if config.bytetrack is not None else None)
    if bytetracker is None:
        logger.warning("bytetrack is disabled in %s — replaying without a tracker.", args.config)
    user_data = user_app_callback_class(OverwriteQueue(maxsize=1), bytetracker)
    user_data.use_frame = True

    # -- Pre-process all frames through app_callback --
    mock_pad = MockGstPad()
    detections_list = run_app_callback_on_all_frames(
        frame_data_list, mock_pad, user_data, mock_monotonic,
    )

    # -- Build ReplayQueue — same pattern as debug_drone_controller.py --
    # (run_replay installs the per-frame callback that winds the clock and telemetry.)
    replay_queue = ReplayQueue(detections_list, auto_advance=args.headless)

    # -- Set up output display --
    output_queue = None
    output_thread = None
    if not args.headless:
        output_queue = OverwriteQueue(maxsize=200)
        sink = InteractiveDisplaySink(
            replay_queue,
            autoplay=args.autoplay,
            window_title=f"App Callback Debug: {log_file}  {video_path}",
        )

        output_thread = threading.Thread(
            target=debug_output_thread,
            args=(output_queue, sink),
            name="DEBUG_DISPLAY",
            daemon=True,
        )
        output_thread.start()

    # -- Run drone_controller on the main thread (same as debug_drone_controller.py) --
    logger.info(
        "Pre-processed %d → %d frames. Starting replay (ByteTracker: %s)...",
        len(frame_data_list), len(detections_list),
        config.bytetrack.tracker_kwargs() if config.bytetrack else None,
    )

    try:
        drone = run_replay(
            config, replay_queue, frames, base_ns,
            output_queue=output_queue,
            mock_monotonic=mock_monotonic,
        )
    except Exception:
        logger.exception("drone_controlling_thread finished with error")
        sys.exit(1)

    logger.info("Replay finished: %d commands issued.", len(drone.commands) if drone else 0)
    if output_thread is not None and output_thread.is_alive():
        output_thread.join(timeout=5)


if __name__ == "__main__":
    main()
