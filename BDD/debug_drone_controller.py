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
import re
import sys
import types
import threading
import queue
from math import nan
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import ast
import yaml
from dataclasses import asdict, replace

from helpers import XY, Rect, Detection, Detections, FrameMetadata, dotdict, STOP
from OverwriteQueue import OverwriteQueue
from flight_debugger import VideoReader, parse_log_lines
from debug_output import debug_output_thread
from interfaces import FrameSinkInterface
from config import Config
from parse_config import ConfigError, parse_config

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


def logged_config_text(logfile_path: Path) -> str | None:
    """The raw `!!! Config:` line from the log, verbatim — or None.

    Deliberately NOT parsed. Old logs carry a flat pre-refactor dict and new ones carry a
    `Config` repr that isn't even eval-able (it contains `<VelocityMethod.NUMPY_REGRESSION:
    'numpy'>`). Neither is mechanically convertible into today's `Config`, so this exists
    only to be shown to a human comparing what flew against what is being replayed.
    """
    with open(logfile_path, "r", errors="replace") as f:
        for line in f:
            m = CONFIG_RE.search(line)
            if m:
                return m.group(1).strip()
    return None


# ───────────────────────────────────────────────────────────────────────
# Replay config
# ───────────────────────────────────────────────────────────────────────

REPLAY_CONFIG_DEFAULT = Path(__file__).resolve().parent / "config.yaml"


def _set_dotted(data: dict, dotted_key: str, value) -> None:
    """Set `a.b.c` in a nested dict, creating missing sections."""
    *sections, leaf = dotted_key.split(".")
    node = data
    for i, section in enumerate(sections):
        child = node.get(section)
        if child is None:                       # absent, or a disabled (null) section
            child = node[section] = {}
        elif not isinstance(child, dict):
            path = ".".join(sections[:i + 1])
            raise ConfigError([f"{path}: is a value, cannot set {dotted_key!r} inside it"])
        node = child
    node[leaf] = value


def _stub_missing_inference_files(data: dict, config_path: Path) -> None:
    """Point non-existent model paths at a file that does exist.

    `inference.hef_model_path` is an `ExistingFile`, and every config we ship references
    /home/bdd/models/... — present on the Pi, absent on a dev host. Replay never runs
    inference (detections come from the log), so the path is irrelevant to the result;
    without this, though, the config simply refuses to load and the harness is unusable
    off-rig, which is the only place it is ever used.
    """
    inference = data.get("inference")
    if not isinstance(inference, dict):
        return
    for key in ("hef_model_path", "labels_json"):
        value = inference.get(key)
        if value and not Path(value).is_file():
            inference[key] = str(config_path)
            logger.warning(
                "!!! inference.%s (%s) does not exist here; pointing it at %s so the config "
                "loads. Replay uses the detections recorded in the log — no inference runs — "
                "so this cannot affect the replay.",
                key, value, config_path.name,
            )


def load_replay_config(path: Path | str = REPLAY_CONFIG_DEFAULT,
                       overrides: dict | None = None) -> Config:
    """Load a real `Config` for replay, the same way the app does.

    NOT reconstructed from the log: the config a flight flew with is recorded only as a flat
    pre-refactor dict (see `logged_config_text`) and cannot be mechanically converted. You are
    replaying an old flight against TODAY's config — usually exactly what you want, but never
    assume the replay reproduces the original commands.

    `overrides` are dotted paths applied before validation, e.g. {'pd_coeff.p': 4}, so they are
    checked by the real parser instead of being trusted.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError([f"{path}: top level must be a mapping"])

    _stub_missing_inference_files(data, path)
    for dotted_key, value in (overrides or {}).items():
        _set_dotted(data, dotted_key, value)

    config = parse_config(Config, data, source=str(path))
    # DEBUG is runtime-only (never settable from the file) and replay always wants it.
    return replace(config, DEBUG=True)


def warn_replayed_config_is_not_the_flights(log_file: Path, config_path: Path) -> None:
    flew_with = logged_config_text(log_file)
    logger.warning("!!! Replaying with %s — NOT the config this flight flew with.", config_path)
    if flew_with:
        logger.warning("!!! The flight flew with (from the log, for comparison only): %s", flew_with)


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
        # Every command the control loop issued, and every telemetry sample it was fed, in
        # order. The log is the primary evidence; these make a replay assertable in a test.
        self.commands: list[tuple] = []
        self.telemetry_seen: list[dict] = []
        logger.info("MockDroneMover created (connection_string=%s)", drone_connection_string)

    def set_frame_data(self, frame_id: int, telemetry_dict: dict):
        self._current_frame_id = frame_id
        self._current_telemetry = telemetry_dict
        self.telemetry_seen.append(telemetry_dict)

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
        self.commands.append(
            ("move_to_target_zenith", self._current_frame_id, roll_degree, pitch_degree, thrust))
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
        self.commands.append(("move_to_target_ned", self._current_frame_id, target_position_ned))
        logger.debug(
            "MockDroneMover.move_to_target_ned(north=%.2f, east=%.2f, down=%.2f, yaw=%.2f, telemetry=%s)",
            target_position_ned.north_m,
            target_position_ned.east_m,
            target_position_ned.down_m,
            yaw_deg,
            current_telemetry,
        )

    async def standstill(self, thrust):
        self.commands.append(("standstill", self._current_frame_id, thrust))
        logger.debug(f"MockDroneMover.standstill({thrust})")

    async def idle(self):
        self.commands.append(("idle", self._current_frame_id))
        logger.debug("MockDroneMover.idle()")

    def ABORT(self):
        self.commands.append(("ABORT", self._current_frame_id))
        logger.info("MockDroneMover.ABORT()")


# ───────────────────────────────────────────────────────────────────────
# ReplayQueue — feeds Detections objects from parsed log + video
# ───────────────────────────────────────────────────────────────────────

class ReplayQueue:
    """Queue-like object that yields pre-built Detections one at a time.

    get() blocks until advance() is called (from the display sink on
    keypress), giving frame-by-frame stepping.  The first frame is
    auto-advanced so the pipeline can start without user interaction.

    With auto_advance=True (headless), get() never waits: there is no display sink to
    press a key on, so waiting would deadlock on the first frame.
    """

    def __init__(self, items: list, auto_advance: bool = False):
        self._items = items
        self._index = 0
        self._lock = threading.Lock()
        self._on_frame_callback = None
        self._advance_event = threading.Event()
        self._stopped = False
        self._auto_advance = auto_advance
        # Auto-advance so the first frame is processed immediately
        self._advance_event.set()

    def set_on_frame_callback(self, callback):
        """Called with (frame_id, item_index) before returning each item."""
        self._on_frame_callback = callback

    def get(self, timeout=None):
        if not self._auto_advance:
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
_KEY_SPACE = 32

class InteractiveDisplaySink:
    """Frame sink that blocks on each frame until the user presses a key.

    Right arrow  — advance one frame
    ESC / q      — stop replay
    """

    def __init__(self, replay_queue: ReplayQueue, window_title: str = "", autoplay = False):
        self._autoplay = autoplay
        self._replay_queue = replay_queue
        self._window_title = window_title or "Debug Drone Controller Replay"
        self._window_name = "debug_replay"

    def start(self, frame_size):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowTitle(self._window_name, self._window_title)
        cv2.resizeWindow(self._window_name, frame_size[0], frame_size[1])

    def process_frame(self, frame):
        cv2.imshow(self._window_name, frame)
        while True:
            wait_s = 0 if not self._autoplay else 1
            key = cv2.waitKeyEx(wait_s)
            if key == _KEY_SPACE:
               self._autoplay = not self._autoplay

            if self._autoplay or key == _KEY_RIGHT or key == ord("d"):
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
# Run the real control thread against the replayed flight
# ───────────────────────────────────────────────────────────────────────

def run_replay(config: Config,
               replay_queue: ReplayQueue,
               frames: dict,
               base_ns: int,
               *,
               output_queue=None,
               mock_monotonic: MockMonotonicNs | None = None) -> MockDroneMover:
    """Drive the REAL `drone_controlling_thread` with replayed frames. Returns the
    MockDroneMover, whose `.commands` / `.telemetry_seen` are the record of the run.

    Two things are faked, and only these two: the drone (commands are recorded, not flown)
    and the clock (frozen to each frame's logged timestamp, so the loop's timing maths sees
    the flight's real intervals rather than replay wall-clock). Everything in between is the
    production control loop.

    Blocks until the queue is exhausted. Both monkeypatches are undone on the way out.

    `mock_monotonic` lets a caller share its clock (debug_app_callback drives app_callback
    off the same one, so both halves of the replay must see the same time).
    """
    if mock_monotonic is None:
        mock_monotonic = MockMonotonicNs(base_ns)
    mock_drone: list[MockDroneMover | None] = [None]
    first_frame_id = min(frames) if frames else None

    def on_frame_callback(frame_id, item_index):
        """Called by ReplayQueue before each frame: wind the clock and telemetry to it."""
        fd = frames.get(frame_id, {})
        mock_monotonic.set_frame(fd.get("timestamp_ns", base_ns))
        if mock_drone[0] is not None:
            mock_drone[0].set_frame_data(frame_id, fd.get("telemetry", {}))

    replay_queue.set_on_frame_callback(on_frame_callback)

    class _CapturingMockDroneMover(MockDroneMover):
        """Captures the instance drone_controller constructs, so telemetry can be fed to it."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            mock_drone[0] = self
            if first_frame_id is not None:      # pre-load, the loop reads telemetry immediately
                self.set_frame_data(first_frame_id, frames[first_frame_id].get("telemetry", {}))

    # A stand-in `time` module: everything real except monotonic_ns.
    mock_time = types.ModuleType("mock_time")
    for attr in dir(real_time_module):
        if not attr.startswith("_"):
            setattr(mock_time, attr, getattr(real_time_module, attr))
    mock_time.monotonic_ns = mock_monotonic

    original_drone_mover = drone_controller_module.DroneMover
    original_time = drone_controller_module.time
    drone_controller_module.DroneMover = _CapturingMockDroneMover
    drone_controller_module.time = mock_time
    try:
        drone_controller_module.drone_controlling_thread(
            "replay://mock",
            asdict(config.drone.config),        # DroneMover's own config, as app.py passes it
            replay_queue,
            control_config=config,              # the real Config — NOT a flat dict from the log
            output_queue=output_queue,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        drone_controller_module.DroneMover = original_drone_mover
        drone_controller_module.time = original_time

    return mock_drone[0]


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
        help="Config overrides as DOTTED paths, e.g. \"{'pd_coeff.p': 4, 'thrust.max': 0.6}\". "
             "Validated like any config value; unknown keys are an error.",
    )
    parser.add_argument(
        "--autoplay",
        action='store_true',
        default=False,
        help="do NOT wait for user input to advance frames",
    )
    parser.add_argument(
        "--headless",
        action='store_true',
        default=False,
        help="No display at all: no window, no video decode (unless --video is given). "
             "The fastest way to replay a flight; needs no X/Wayland.",
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
    # parse_log also scrapes the log's `!!! Config:` line, but we deliberately do NOT
    # build the control config from it: on old logs it is a flat pre-refactor dict, on new
    # ones a repr that will not even eval. See load_replay_config.
    _logged_config, frames, base_ns = parse_log(log_file)
    if not frames:
        print("No frame data found in log file", file=sys.stderr)
        sys.exit(1)
    logger.info("Parsed %d frames from log", len(frames))

    # ── Load the config to replay WITH ─────────────────────────────────
    try:
        config = load_replay_config(args.config, args.params)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
    warn_replayed_config_is_not_the_flights(log_file, args.config)

    # ── Open video ─────────────────────────────────────────────────────
    # Headless skips the decode entirely unless video was asked for explicitly: the
    # controller is happy with blank frames, and decoding a whole flight is the slow part.
    if args.headless and args.video is None:
        video_path = None

    video_reader = VideoReader(video_path)
    if video_reader.available:
        logger.info("Video loaded: %d frames", video_reader.total)
    else:
        # Degrade rather than die: blank frames replay fine for everything that doesn't
        # read pixels (which is everything except optical refinement).
        if video_path is not None:
            logger.warning("No usable video at %s — replaying on blank frames.", video_path)
        if config.optical_refinement is not None:
            logger.warning(
                "!!! optical_refinement is ENABLED but the frames are blank: that code reads "
                "pixels, so it will NOT behave as it did in flight. Pass --video for real frames."
            )

    # ── Build replay data ──────────────────────────────────────────────
    detections_list = build_detections_list(frames, video_reader)
    replay_queue = ReplayQueue(detections_list, auto_advance=args.headless)

    # ── Set up output display ──────────────────────────────────────────
    output_queue = None
    output_thread = None
    if not args.headless:
        output_queue = OverwriteQueue(maxsize=200)
        sink = InteractiveDisplaySink(
            replay_queue,
            autoplay=args.autoplay,
            window_title=f"Debug Controller {log_file}  {video_path}",
        )
        output_thread = threading.Thread(
            target=debug_output_thread,
            args=(output_queue, sink),
            name="DEBUG_DISPLAY",
            daemon=True,
        )
        output_thread.start()

    # ── Run drone_controlling_thread ───────────────────────────────────
    logger.info("Starting drone_controller replay with %d frames...", len(detections_list))

    try:
        drone = run_replay(config, replay_queue, frames, base_ns, output_queue=output_queue)
    except Exception:
        logger.exception("drone_controlling_thread finished with error")
        sys.exit(1)

    logger.info("Replay finished: %d commands issued.", len(drone.commands) if drone else 0)

    # Keep display open until user closes it
    if output_thread is not None and output_thread.is_alive():
        output_thread.join(timeout=5)


if __name__ == "__main__":
    main()
