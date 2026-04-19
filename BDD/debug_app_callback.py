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
        _ensure_module('pipelines', {
            'SOURCE_PIPELINE': lambda **kw: '',
            'INFERENCE_PIPELINE': lambda **kw: '',
            'INFERENCE_PIPELINE_WRAPPER': lambda s: '',
            'TRACKER_PIPELINE': lambda **kw: '',
            'USER_CALLBACK_PIPELINE': lambda **kw: '',
            'DISPLAY_PIPELINE': lambda **kw: '',
        })
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
from app import app_callback, user_app_callback_class, seen_frames

import hailo
from gi.repository import Gst  # noqa: F811
from bytetrack import BYTETracker

from debug_drone_controller import (
    parse_log, find_files_in_dir,
    MockMonotonicNs, MockDroneMover,
    InteractiveDisplaySink,
)
import drone_controller as drone_controller_module

from flight_debugger import VideoReader
from helpers import XY, STOP
from OverwriteQueue import OverwriteQueue
from debug_output import debug_output_thread

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(threadName)s] %(name)s <%(levelname)s> : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
# Feeder: drives frames through app_callback one at a time
# ===================================================================

class AppCallbackFeeder:
    """Feeds log-parsed frames through app_callback, paced by advance().

    Exposes advance() / stop() so InteractiveDisplaySink can drive it
    the same way it drives ReplayQueue in debug_drone_controller.
    """

    def __init__(
        self,
        frame_data_list: list[dict],
        frames_dict: dict,
        mock_pad: MockGstPad,
        user_data: user_app_callback_class,
        mock_monotonic: MockMonotonicNs,
        mock_drone_ref: list,
        detections_queue: OverwriteQueue,
    ):
        self._frames = frame_data_list
        self._frames_dict = frames_dict
        self._pad = mock_pad
        self._user_data = user_data
        self._monotonic = mock_monotonic
        self._drone_ref = mock_drone_ref
        self._detections_queue = detections_queue
        self._advance = threading.Event()
        self._stopped = threading.Event()
        # Auto-advance so the first frame goes through without user input
        self._advance.set()

    # -- InteractiveDisplaySink protocol --
    def advance(self):
        self._advance.set()

    def stop(self):
        self._stopped.set()
        self._advance.set()  # unblock

    # -- Thread body --
    def run(self):
        for fdata in self._frames:
            self._advance.wait()
            self._advance.clear()

            if self._stopped.is_set():
                break

            fid = fdata['frame_id']
            ts_ns = fdata['timestamp_ns']

            # Advance mock clock
            self._monotonic.set_frame(ts_ns)

            # Feed telemetry to mock drone
            if self._drone_ref[0] is not None:
                self._drone_ref[0].set_frame_data(fid, fdata['telemetry'])

            # Set per-frame context for monkey-patched functions
            set_frame_context(fdata['frame'], fdata['detections'])

            # Build mock GStreamer probe objects
            mock_buffer = MockGstBuffer(fid, ts_ns)
            mock_info = MockGstPadProbeInfo(mock_buffer)

            # >>> This is the line that exercises app_callback <<<
            result = app_callback(self._pad, mock_info, self._user_data)
            logger.debug("app_callback returned %s for frame #%d", result, fid)

        # End of frames: signal downstream to stop
        self._detections_queue.put(STOP)
        logger.info("Feeder finished: %d frames fed through app_callback", len(self._frames))


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
        "--video", type=Path, default=None,
        help="Path to video file(s) or directory with video files",
    )
    parser.add_argument(
        "--params",
        type=lambda x: dict(ast.literal_eval(x)),
        default=None,
        help="Extra config parameters to overwrite",
    )
    parser.add_argument(
        "--autoplay", action='store_true', default=False,
        help="Do NOT wait for user input to advance frames",
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
    config_dict, frames, base_ns = parse_log(log_file)
    if not frames:
        print("No frame data found in log file", file=sys.stderr)
        sys.exit(1)
    logger.info("Parsed %d frames from log", len(frames))

    if config_dict:
        logger.info("Loaded config from log: %s", config_dict)
    else:
        logger.warning("No config found in log, using defaults")
        config_dict = {
            'confidence_min': 0.4,
            'thrust_takeoff': 0.5,
            'thrust_min': 0.5,
            'thrust_max': 0.5,
            'thrust_dynamic': False,
            'thrust_proportional_to_target_size': False,
            'target_lost_fade_per_frame': 0.99,
            'target_estimator_clear_history_after_target_lost_frames': 3,
            'estimation_3d': False,
            'follow_target_position_ned': False,
            'estimation_lookahead_frames': 2,
            'estimation_lookahead_dynamic': False,
            'pd_coeff_p': 3,
            'pd_coeff_d': 0,
            'pd_coeff_p_safe_min': 0.6,
            'pd_coeff_p_min': 0.5,
            'pd_coeff_p_max': 10,
            'pd_coeff_p_dynamic': False,
            'frame_angular_size_deg': XY(107, 85),
            'target_size_m': XY(1.7, 2),
            'safe_takeoff_period_ns': 300_000_000,
            'delay_takeof_until_n_detection_frames': 3,
            'aim_point': XY(0.5, 0.5),
            'DEBUG': True,
        }

    # -- Config overrides --
    config_dict['DEBUG'] = True
    config_dict.update({
        'follow_target_position_ned': True,
        'estimation_3d': True,
        'estimation_3d_method': 'numpy',
        'estimation_lookahead_frames': 1,
        'estimation_lookahead_dynamic': True,
        'estimation_lookahead_dynamic_sqrt': False,
        'estimation_lookahead_dynamic_factor': 0.1,
        'estimation_lookahead_dynamic_frames_near': 0,
        'estimation_lookahead_dynamic_frames_medium': 0,
        'estimation_lookahead_dynamic_frames_far': 0,
    })

    if args.params:
        config_dict.update(args.params)

    # Remove stale config keys
    for k in ['confidence_move', 'inertia_correction_gain',
              'inertia_correction_limits', 'inertia_correction_min_speed_ms',
              'aim_point_max_offset']:
        config_dict.pop(k, None)

    # -- Extract ByteTracker config (consumed by app_callback, not drone_controller) --
    bytetrack_config = {
        'track_thresh':  config_dict.pop('bytetrack_track_thresh', 0.5),
        'det_thresh':    config_dict.pop('bytetrack_det_thresh', 0.6),
        'match_thresh':  config_dict.pop('bytetrack_match_thresh', 0.8),
        'track_buffer':  config_dict.pop('bytetrack_track_buffer', 30),
        'frame_rate':    config_dict.pop('bytetrack_frame_rate', 30),
    }

    # -- Open video --
    video_reader = VideoReader(video_path)
    if video_reader.available:
        logger.info("Video loaded: %d frames", video_reader.total)
    else:
        logger.error("Failed to load video %s", video_path)
        sys.exit(-1)

    # -- Build per-frame data --
    frame_data_list = build_frame_data_list(frames, video_reader)

    # -- Mock time.monotonic_ns --
    mock_monotonic = MockMonotonicNs(base_ns)
    mock_time = types.ModuleType("mock_time")
    for attr in dir(real_time_module):
        if not attr.startswith("_"):
            setattr(mock_time, attr, getattr(real_time_module, attr))
    mock_time.monotonic_ns = mock_monotonic
    # Patch both app and drone_controller modules
    app_module.time = mock_time
    drone_controller_module.time = mock_time

    # -- Mock DroneMover --
    drone_controller_module.DroneMover = MockDroneMover
    mock_drone = [None]

    _original_init = MockDroneMover.__init__
    def _capturing_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        mock_drone[0] = self
        first_fid = sorted(frames.keys())[0]
        self.set_frame_data(first_fid, frames[first_fid].get("telemetry", {}))
    MockDroneMover.__init__ = _capturing_init

    # -- Install mock patches on app module --
    install_mock_patches()

    # -- Clear app.py's frame deduplication state --
    seen_frames.clear()

    # -- Set up ByteTracker and user_data (mirrors app.py main) --
    bytetracker = BYTETracker(**bytetrack_config)
    detections_queue = OverwriteQueue(maxsize=20)
    user_data = user_app_callback_class(detections_queue, bytetracker)
    user_data.use_frame = True

    # -- Set up feeder --
    mock_pad = MockGstPad()
    feeder = AppCallbackFeeder(
        frame_data_list=frame_data_list,
        frames_dict=frames,
        mock_pad=mock_pad,
        user_data=user_data,
        mock_monotonic=mock_monotonic,
        mock_drone_ref=mock_drone,
        detections_queue=detections_queue,
    )

    feeder_thread = threading.Thread(
        target=feeder.run,
        name="AppCallbackFeeder",
        daemon=True,
    )
    feeder_thread.start()

    # -- Set up output display --
    output_queue = OverwriteQueue(maxsize=200)
    sink = InteractiveDisplaySink(
        feeder,  # has advance()/stop() — same protocol as ReplayQueue
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

    # -- Run drone_controller on the main-ish thread --
    logger.info(
        "Starting app_callback replay with %d frames (ByteTracker: %s)...",
        len(frame_data_list), bytetrack_config,
    )

    drone_config = {"upside_down_angle_deg": 130, "upside_down_hold_s": 0.2}
    try:
        drone_controller_module.drone_controlling_thread(
            "replay://mock",
            drone_config,
            detections_queue,
            control_config=dict(config_dict),
            output_queue=output_queue,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("drone_controlling_thread finished with error")

    logger.info("Replay finished.")
    if output_thread.is_alive():
        output_thread.join(timeout=5)


if __name__ == "__main__":
    main()
