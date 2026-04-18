#!/usr/bin/env python3
"""Debug facility for drone_controller.

Replays a recorded flight by feeding drone_controller with video frames
and detection/telemetry data parsed from a log file.  The real DroneMover
and time.monotonic_ns() are monkey-patched so that drone_controller runs
its full logic on historical data, while all drone commands are simply
logged.

Usage:
    cd BDD
    python debug_drone_controller.py path/to/_DEBUG_dir/
    python debug_drone_controller.py path/to/logfile.log --video path/to/video.mp4
"""

import argparse
import os
import re
import sys
import types
import threading
import queue
from math import nan
from pathlib import Path
from datetime import datetime

# До import cv2: Qt + GLib (Gst) на одной машине часто ломают отрисовку; GTK3 стабильнее.
os.environ.setdefault("OPENCV_UI_BACKEND", "GTK3")

import cv2
import numpy as np
import ast

from helpers import XY, Rect, Detection, Detections, FrameMetadata, dotdict, STOP
from OverwriteQueue import OverwriteQueue
from flight_debugger import VideoReader, parse_log_lines
from debug_output import debug_output_thread
from interfaces import FrameSinkInterface

import time as real_time_module

import logging

# Provide stub mavsdk module if not installed (e.g. developing off-drone)
try:
    import mavsdk  # noqa: F401
except ImportError:
    import types as _types

    _mavsdk = _types.ModuleType("mavsdk")
    _mavsdk.System = type("System", (), {"__init__": lambda self: None})

    _offboard = _types.ModuleType("mavsdk.offboard")
    for _name in ("PositionNedYaw", "VelocityBodyYawspeed", "Attitude",
                   "VelocityNedYaw", "AttitudeRate"):
        setattr(_offboard, _name, type(_name, (), {}))
    _offboard.OffboardError = type("OffboardError", (Exception,), {})

    _telemetry = _types.ModuleType("mavsdk.telemetry")
    _telemetry.Telemetry = type("Telemetry", (), {})
    _telemetry.EulerAngle = type("EulerAngle", (), {})
    _telemetry.LandedState = type("LandedState", (), {"IN_AIR": 1})

    sys.modules["mavsdk"] = _mavsdk
    sys.modules["mavsdk.offboard"] = _offboard
    sys.modules["mavsdk.telemetry"] = _telemetry

import drone_controller as drone_controller_module

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(threadName)s] %(name)s <%(levelname)s> : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
from helpers import (
    debug_collect_call_info,
    LoggerWithPrefix
)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Log parsing
# ───────────────────────────────────────────────────────────────────────

TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
)
FRAME_RE = re.compile(r"frame=#(\d+)")

DETECTION_RE = re.compile(
    r"frame=#(\d+)\s+!!! GOT DETECTIONS.*?\((\[.*?\])\),\s+detection delay"
)
TELEMETRY_RE = re.compile(
    r"frame=#(\d+)\s+(?:!+\s+)?telemetry:\s+(.+)$"
)
CONFIG_RE = re.compile(r"!!! Config:\s+(.+)$")


def _eval_namespace():
    """Build a namespace for eval() so log repr strings can be parsed."""
    ns = {"nan": nan, "None": None, "True": True, "False": False}
    ns["XY"] = XY
    # Rect repr is Rect(x=..., y=..., w=..., h=...) but __init__ takes (p1, p2).
    # in repr string in logs x and y are coords of the center *FACEPALM*
    ns["Rect"] = lambda x=0, y=0, w=0, h=0: Rect.from_xywh(x - w/2, y - h/2, w, h)
    ns["Detection"] = Detection
    # Enums that may appear in config/telemetry repr
    try:
        from estimate_distance import DistanceClass
        ns["DistanceClass"] = DistanceClass
    except ImportError:
        pass
    return ns


def parse_log(logfile_path: Path):
    """Parse a BDD log file.

    Returns:
        config_dict: control_config as logged by app.py, or None
        frames: dict[int, FrameData] mapping frame_id to parsed data
        first_timestamp_ns: nanosecond timestamp of the very first log line
    """
    config_dict = None
    # Per-frame data: {frame_id: {timestamp_ns, detections, telemetry_dict}}
    frames: dict[int, dict] = {}
    first_timestamp: datetime | None = None
    ns = _eval_namespace()

    global logger
    default_logger = logger
    with open(logfile_path, "r", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            logger = LoggerWithPrefix(default_logger, prefix=f'{logfile_path}:{line_no}')
            # Extract wall-clock timestamp
            ts_match = TIMESTAMP_RE.match(line)
            if not ts_match:
                continue
            ts = datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
            if first_timestamp is None:
                first_timestamp = ts

            # Config line, parse it only once
            if not config_dict:
                cm = CONFIG_RE.search(line)
                if cm and config_dict is None:
                    try:
                        config_dict = eval(cm.group(1), {"__builtins__": {}}, ns)  # noqa: S307
                        # found config, there could be no Detections nor Telemetry on that line
                        continue
                    except Exception:
                        logger.warning("Failed to parse config from log", exc_info=True)

            # Detection line
            dm = DETECTION_RE.search(line)
            if dm:
                frame_id = int(dm.group(1))
                frame_data = frames.setdefault(frame_id, {})
                if "detections" not in frame_data:
                    try:
                        det_list = eval(dm.group(2), {"__builtins__": {}}, ns)  # noqa: S307
                        frame_data["detections"] = det_list
                    except Exception:
                        logger.warning("Failed to parse detections for frame %d", frame_id, exc_info=True)
                        frame_data["detections"] = []
                # Record timestamp of first log entry for this frame
                if "timestamp" not in frame_data:
                    frame_data["timestamp"] = ts

                # found Detection there could be no Telemetry on that line
                continue

            # Telemetry line
            tm = TELEMETRY_RE.search(line)
            if tm:
                frame_id = int(tm.group(1))
                frame_data = frames.setdefault(frame_id, {})
                if "telemetry" not in frame_data:
                    try:
                        frame_data["telemetry"] = eval(tm.group(2), {"__builtins__": {}}, ns)  # noqa: S307
                    except Exception:
                        logger.warning("Failed to parse telemetry for frame %d", frame_id, exc_info=True)
                if "timestamp" not in frame_data:
                    frame_data["timestamp"] = ts

    # Convert timestamps to nanoseconds relative to first_timestamp
    if first_timestamp is None:
        first_timestamp = datetime.now()
    base_ns = int(first_timestamp.timestamp() * 1_000_000_000)

    for fid, fd in frames.items():
        ts = fd.get("timestamp", first_timestamp)
        fd["timestamp_ns"] = int(ts.timestamp() * 1_000_000_000)

    return config_dict, frames, base_ns


# ───────────────────────────────────────────────────────────────────────
# Mock time.monotonic_ns
# ───────────────────────────────────────────────────────────────────────

class MockMonotonicNs:
    """Replaces time.monotonic_ns() for deterministic replay.

    On each call, returns baseline + small increment.
    baseline is reset when set_frame() is called with a new timestamp.
    """

    def __init__(self, initial_ns: int):
        self._baseline = initial_ns
        self._offset = 0

    def set_frame(self, timestamp_ns: int):
        self._baseline = timestamp_ns
        self._offset = 0

    def __call__(self) -> int:
        result = self._baseline + self._offset
        self._offset += 50  # 50ns increment per call
        return result


# ───────────────────────────────────────────────────────────────────────
# MockDroneMover
# ───────────────────────────────────────────────────────────────────────

class MockDroneMover:
    """Drop-in replacement for drone.DroneMover that returns recorded
    telemetry and logs all flight commands."""

    def __init__(self, drone_connection_string, config: dict | None = None):
        self.config = config or {}
        self.drone_connection_string = drone_connection_string
        self._current_telemetry: dict = {}
        self._current_frame_id = -1
        logger.info("MockDroneMover created (connection_string=%s)", drone_connection_string)

    def set_frame_data(self, frame_id: int, telemetry_dict: dict):
        self._current_frame_id = frame_id
        self._current_telemetry = telemetry_dict

    async def startup_sequence(self, arm_attempts=100, force_arm=False):
        logger.info("MockDroneMover.startup_sequence(arm_attempts=%s, force_arm=%s)", arm_attempts, force_arm)

    async def get_telemetry_dict(self, wait=False) -> dotdict:
        return dotdict(self._current_telemetry)

    def get_telemetry_dict_sync(self, wait=False) -> dotdict:
        return dotdict(self._current_telemetry)

    def get_telemetry_dict_cached(self) -> dotdict:
        return dotdict(self._current_telemetry)

    def _resolve_current_telemetry(self, current_telemetry) -> dotdict:
        if current_telemetry is None:
            return self.get_telemetry_dict_cached()
        return dotdict(current_telemetry)

    @staticmethod
    def _yaw_deg_from_telemetry(current_telemetry) -> float:
        return ((current_telemetry.get("attitude_euler", {}) or {}).get("yaw_deg", 0))

    async def move_to_target_zenith_async(self, roll_degree: float, pitch_degree: float, thrust: float = 0.0, current_telemetry=None):
        current_telemetry = self._resolve_current_telemetry(current_telemetry)
        yaw_deg = self._yaw_deg_from_telemetry(current_telemetry)
        logger.debug(
            "MockDroneMover.move_to_target_zenith_async(roll=%.2f, pitch=%.2f, thrust=%.3f, yaw=%.2f, telemetry=%s)",
            roll_degree,
            pitch_degree,
            thrust,
            yaw_deg,
            current_telemetry,
        )

    async def move_to_target_ned(self, target_position_ned, current_telemetry=None):
        current_telemetry = self._resolve_current_telemetry(current_telemetry)
        yaw_deg = self._yaw_deg_from_telemetry(current_telemetry)
        logger.debug(
            "MockDroneMover.move_to_target_ned(north=%.2f, east=%.2f, down=%.2f, yaw=%.2f, telemetry=%s)",
            target_position_ned.north_m,
            target_position_ned.east_m,
            target_position_ned.down_m,
            yaw_deg,
            current_telemetry,
        )

    async def standstill(self):
        logger.debug("MockDroneMover.standstill()")

    async def idle(self):
        logger.debug("MockDroneMover.idle()")

    def ABORT(self):
        logger.info("MockDroneMover.ABORT()")


# ───────────────────────────────────────────────────────────────────────
# ReplayQueue — feeds Detections objects from parsed log + video
# ───────────────────────────────────────────────────────────────────────

class ReplayQueue:
    """Queue-like object that yields pre-built Detections one at a time.

    get() blocks until advance() is called (from the display sink on
    keypress), giving frame-by-frame stepping.  The first frame is
    auto-advanced so the pipeline can start without user interaction.
    """

    def __init__(self, items: list):
        self._items = items
        self._index = 0
        self._lock = threading.Lock()
        self._on_frame_callback = None
        self._advance_event = threading.Event()
        self._stopped = False
        # Auto-advance so the first frame is processed immediately
        self._advance_event.set()

    def set_on_frame_callback(self, callback):
        """Called with (frame_id, item_index) before returning each item."""
        self._on_frame_callback = callback

    def get(self, timeout=None):
        # Block until the display sink signals "advance"
        self._advance_event.wait()
        self._advance_event.clear()

        if self._stopped:
            return STOP

        with self._lock:
            if self._index >= len(self._items):
                return STOP
            item = self._items[self._index]
            self._index += 1
            if self._on_frame_callback:
                self._on_frame_callback(item.frame_id, self._index - 1)
            return item

    def advance(self):
        """Unblock get() so the next frame is returned."""
        self._advance_event.set()

    def stop(self):
        """Signal the queue to return STOP and unblock any waiting get()."""
        self._stopped = True
        self._advance_event.set()

    def put(self, item):
        pass

    def qsize(self):
        with self._lock:
            return len(self._items) - self._index


# ───────────────────────────────────────────────────────────────────────
# InteractiveDisplaySink — step-through display controlled by keypress
# ───────────────────────────────────────────────────────────────────────

# Key codes returned by cv2.waitKeyEx on Linux
_KEY_RIGHT = 65363
_KEY_LEFT  = 65361

class InteractiveDisplaySink:
    """Frame sink that blocks on each frame until the user presses a key.

    Right arrow  — advance one frame
    ESC / q      — stop replay
    """

    def __init__(self, replay_queue: ReplayQueue, window_title: str = ""):
        self._replay_queue = replay_queue
        self._window_title = window_title or "Debug Drone Controller Replay"
        self._window_name = "debug_replay"

    def start(self, frame_size):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle(self._window_name, self._window_title)
        cv2.resizeWindow(self._window_name, frame_size[0], frame_size[1])

    def process_frame(self, frame):
        # Не портим буфер кадра из очереди (его же читает логика дрона).
        vis = np.ascontiguousarray(frame)
        # BDD: кадр из GStreamer/VideoReader — RGB; imshow ожидает BGR.
        if vis.ndim == 3 and vis.shape[2] == 3:
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow(self._window_name, vis)
        cv2.waitKey(1)

        while True:
            key = cv2.waitKeyEx(0)
            if key == _KEY_RIGHT or key == ord("d"):
                self._replay_queue.advance()
                return
            if key == 27 or key == ord("q"):  # ESC or q
                self._replay_queue.stop()
                return

    def stop(self):
        cv2.destroyWindow(self._window_name)


# ───────────────────────────────────────────────────────────────────────
# Build replay data
# ───────────────────────────────────────────────────────────────────────

def build_detections_list(frames: dict, video_reader: VideoReader | None) -> list[Detections]:
    """Build a list of Detections objects from parsed log data and video frames."""
    result = []
    sorted_frame_ids = sorted(frames.keys())

    for frame_id in sorted_frame_ids:
        fd = frames[frame_id]
        detections = fd.get("detections", [])
        timestamp_ns = fd.get("timestamp_ns", 0)

        # Read video frame
        frame = None
        if video_reader is not None and video_reader.available:
            frame = video_reader.read(frame_id)

        if frame is None:
            # Create a blank frame as placeholder
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det_obj = Detections(
            frame_id=frame_id,
            frame=frame,
            detections=detections,
            meta=FrameMetadata(
                capture_timestamp_ns=timestamp_ns,
                detection_start_timestamp_ns=timestamp_ns,
                detection_end_timestamp_ns=timestamp_ns,
            ),
        )
        result.append(det_obj)

    logger.info("Built %d Detections objects from log", len(result))
    return result


# ───────────────────────────────────────────────────────────────────────
# Discover files from --dir
# ───────────────────────────────────────────────────────────────────────

def find_files_in_dir(dir_path: Path) -> tuple[Path | None, list[Path]]:
    """Find log file and video files in a debug directory."""
    log_files = sorted(dir_path.glob("*.log"))
    log_file = log_files[0] if log_files else None

    # Prefer RAW (unprocessed) video files; fall back to debug_ or any video
    video_files = sorted(dir_path.glob("RAW_*.mkv")) + sorted(dir_path.glob("RAW_*.mp4"))
    if not video_files:
        video_files = sorted(dir_path.glob("debug_*.mkv")) + sorted(dir_path.glob("debug_*.mp4"))
    if not video_files:
        video_files = sorted(dir_path.glob("*.mkv")) + sorted(dir_path.glob("*.mp4"))

    return log_file, video_files


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay a recorded flight through drone_controller for debugging.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a log file (.log) or a debug directory containing logs and videos",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to video file(s) or directory with video files",
    )
    parser.add_argument(
        "--params",
        type=lambda x: dict(ast.literal_eval(x)),
        default=None,
        help="Extra parameters of config to overwrite",
    )
    args = parser.parse_args()

    # Resolve log and video paths
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

    # ── Parse log ──────────────────────────────────────────────────────
    config_dict, frames, base_ns = parse_log(log_file)
    if not frames:
        print("No frame data found in log file", file=sys.stderr)
        sys.exit(1)
    logger.info("Parsed %d frames from log", len(frames))

    if config_dict:
        logger.info("Loaded config from log: %s", config_dict)
    else:
        logger.warning("No config found in log, using defaults from app.py")
        config_dict = {
            'confidence_min': 0.4,
            'thrust_takeoff': 0.5,
            'thrust_min': 0.5,
            'thrust_max': 0.5,
            'thrust_dynamic': False,
            'thrust_proportional_to_target_size': False,
            'target_lost_fade_per_frame': 0.99,
            'target_estimator_clear_history_after_target_lost_frames': 3,
            'estimation_use_3d': False,
            'follow_target_position_ned': False,
            'estimation_lookahead_frames': 5,
            'estimation_lookahead_dynamic': False,
            'estimation_lookahead_dynamic_frames_near': 3,
            'estimation_lookahead_dynamic_frames_medium': 10,
            'estimation_lookahead_dynamic_frames_far': 20,
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

    # =========================================================================
    # !!! OVERWRITE config
    # =========================================================================
    #

    config_dict['DEBUG'] = True
    config_dict.update({
        'follow_target_position_ned': True,
        'estimation_3d' : True,
        'estimation_3d_method' : 'numpy',
        'estimation_lookahead_frames': 5,
        'estimation_lookahead_dynamic': True,
        'estimation_lookahead_dynamic_frames_near': 2,
        'estimation_lookahead_dynamic_frames_medium': 4,
        'estimation_lookahead_dynamic_frames_far': 8,
    })

    if args.params:
        config_dict.update(args.params)

    #
    # =========================================================================

    # Remove config keys not consumed by current drone_controller to avoid warnings
    _stale_keys = [
        'confidence_move', 'inertia_correction_gain',
        'inertia_correction_limits', 'inertia_correction_min_speed_ms',
        'aim_point_max_offset',
    ]
    for k in _stale_keys:
        config_dict.pop(k, None)

    # ── Open video ─────────────────────────────────────────────────────
    video_reader = VideoReader(video_path)
    if video_reader.available:
        logger.info("Video loaded: %d frames", video_reader.total)
    else:
        logger.error("Failed to load video %s", video_path)
        sys.exit(-1)

    # ── Build replay data ──────────────────────────────────────────────
    detections_list = build_detections_list(frames, video_reader)

    # ── Set up mocks ───────────────────────────────────────────────────
    mock_monotonic = MockMonotonicNs(base_ns)
    mock_drone = [None]  # mutable container so callback can access it

    def on_frame_callback(frame_id, item_index):
        """Called by ReplayQueue when a new frame is about to be returned."""
        fd = frames.get(frame_id, {})
        ts_ns = fd.get("timestamp_ns", base_ns)
        mock_monotonic.set_frame(ts_ns)
        if mock_drone[0] is not None:
            telemetry = fd.get("telemetry", {})
            mock_drone[0].set_frame_data(frame_id, telemetry)

    replay_queue = ReplayQueue(detections_list)
    replay_queue.set_on_frame_callback(on_frame_callback)

    # ── Monkey-patch drone_controller ──────────────────────────────────
    # Replace DroneMover with MockDroneMover
    drone_controller_module.DroneMover = MockDroneMover

    # Replace time.monotonic_ns in drone_controller's time module reference
    # Create a wrapper module that delegates everything to real time
    # but overrides monotonic_ns
    mock_time = types.ModuleType("mock_time")
    for attr in dir(real_time_module):
        if not attr.startswith("_"):
            setattr(mock_time, attr, getattr(real_time_module, attr))
    mock_time.monotonic_ns = mock_monotonic
    drone_controller_module.time = mock_time

    # ── Hook into MockDroneMover creation ──────────────────────────────
    # drone_controller creates DroneMover at line 244.  After our patch,
    # it will create MockDroneMover instead.  We need to capture that
    # instance so on_frame_callback can feed it telemetry.
    _original_init = MockDroneMover.__init__

    def _capturing_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        mock_drone[0] = self
        # Pre-load first frame telemetry
        first_frame_id = sorted(frames.keys())[0]
        fd = frames[first_frame_id]
        self.set_frame_data(first_frame_id, fd.get("telemetry", {}))

    MockDroneMover.__init__ = _capturing_init

    # ── Set up output display ──────────────────────────────────────────
    output_queue = OverwriteQueue(maxsize=200)

    sink = InteractiveDisplaySink(replay_queue, window_title=f"Debug Controller {log_file}  {video_path}")

    output_thread = threading.Thread(
        target=debug_output_thread,
        args=(output_queue, sink),
        name="DEBUG_DISPLAY",
        daemon=True,
    )
    output_thread.start()

    # ── Run drone_controlling_thread ───────────────────────────────────
    logger.info("Starting drone_controller replay with %d frames...", len(detections_list))

    drone_config = {"upside_down_angle_deg": 130, "upside_down_hold_s": 0.2}

    try:
        drone_controller_module.drone_controlling_thread(
            "replay://mock",
            drone_config,
            replay_queue,
            control_config=dict(config_dict),
            output_queue=output_queue,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("drone_controlling_thread finished with error")

    logger.info("Replay finished.")

    # Keep display open until user closes it
    if output_thread.is_alive():
        output_thread.join(timeout=5)


if __name__ == "__main__":
    main()
