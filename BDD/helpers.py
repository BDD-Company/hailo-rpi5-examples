#!/usr/bin/env python3

from dataclasses import dataclass, field
from collections import deque
from functools import wraps
import inspect
import datetime
import math
import sys
import threading

import numpy as np

import logging
import logging.handlers
import queue
import atexit

# Milliseconds = int
# def get_current_time_ms() -> Milliseconds:
#     return int(time.time_ns() / (1000 * 1000))


class FPSCounter:
    def __init__(self, average_over_frames=10):
        self.frame_times = deque(maxlen=average_over_frames)
        self.frames_count = 0

    def recorded_fps(self):
        if len(self.frame_times) < 2:
            return None

        time_delta = self.frame_times[-1] - self.frame_times[0]
        if time_delta == 0:
            return None

        # 1000 since time is in miliseconds
        return int(len(self.frame_times) * 1000 // time_delta)

    @property
    def frames(self):
        return self.frames_count

    def on_frame(self):
        self.frame_times.append(get_current_time_ms())
        self.frames_count += 1

        return self.recorded_fps()


class LoggingFPSCounter(FPSCounter):
    def __init__(self, logger=logging.info, log_message = "FPS: %s", logging_interval = 300, average_over_frames=30):
        super().__init__(average_over_frames)
        self.logger = logger
        self.log_message = log_message if log_message else "FPS: %s"
        self.logging_interval = logging_interval

    def on_frame(self):
        result = super().on_frame()
        if self.logger \
                and result is not None \
                and self.frames_count % self.logging_interval == 0:
            self.logger(self.log_message, result)

        return result


@dataclass(slots=True, order=True)
class XY:
    x : float = 0.0
    y : float = 0.0

    def __add__(self, other : 'XY'):
        return self.__class__(self.x + other.x, self.y + other.y)

    def __iadd__(self, other : 'XY'):
        self.x += other.x
        self.y += other.y

        return self

    def __sub__(self, other : 'XY'):
        return self.__class__(self.x - other.x, self.y - other.y)

    def __isub__(self, other : 'XY'):
        self.x -= other.x
        self.y -= other.y

        return self

    def __mul__(self, scalar):
        return self.__class__(self.x * scalar, self.y * scalar)

    def __imul__(self, scalar):
        self.x *= scalar
        self.y *= scalar

        return self

    def __truediv__(self, scalar):
        return self.__class__(self.x / scalar, self.y / scalar)

    def __itruediv__(self, scalar):
        self.x /= scalar
        self.y /= scalar

        return self

    def clone(self) -> 'XY':
        return self.__class__(self.x, self.y)

    def multiplied_by_XY(self, other_xy: 'XY') -> 'XY':
        return self.__class__(self.x * other_xy.x, self.y * other_xy.y)

    def divided_by_XY(self, other_xy: 'XY') -> 'XY':
        return self.__class__(self.x / other_xy.x, self.y / other_xy.y)

    def abs(self) -> 'XY':
        return self.__class__(abs(self.x), abs(self.y))

    def distance_squared_to(self, other: 'XY') -> float:
        return (self.x - other.x)**2 + (self.y - other.y)**2

    def distance_to(self, other: 'XY') -> float:
        return math.sqrt(self.distance_squared_to(other))

    def to_intXY(self) -> 'XY':
        return self.__class__(round(self.x), round(self.y))

    def to_tuple(self, to = lambda x: x):
        return (to(self.x), to(self.y))

    def max_val(self):
        return max(self.x, self.y)

    def min_val(self):
        return min(self.x, self.y)

    def __str__(self):
        return f"XY({self.x:.3f}, {self.y:.3f})"

    def __format__(self, format_spec):
        # Apply the format spec to x and y individually, e.g. f"{xy:.2f}".
        # An empty spec falls back to the default __str__ representation.
        if not format_spec:
            return self.__str__()
        return f"XY({self.x:{format_spec}}, {self.y:{format_spec}})"


@dataclass(init=True, slots=True, order=True, repr=False)
class Rect:
    p1 : XY = field(default_factory = XY)
    p2 : XY = field(default_factory = XY)

    def __post_init__(self):
        self.normalize()

    # def __str__(self):
    #     return "({}, {})".format(self.p1, self.p2)

    def __repr__(self):
        xywh = self.to_xywh()
        labels = 'xywh'
        return f"{type(self).__name__}({', '.join('%s=%.3f' % (l, v) for l, v in zip(labels, xywh))})"


    def normalize(self):
        new_p1 = XY(
            min(self.p1.x, self.p2.x),
            min(self.p1.y, self.p2.y)
        )
        new_p2 = XY(
            max(self.p1.x, self.p2.x),
            max(self.p1.y, self.p2.y)
        )
        self.p1 = new_p1
        self.p2 = new_p2


    @classmethod
    def from_xyxy(cls, x1 : float, y1 : float, x2 : float, y2 : float) -> 'Rect':
        return cls(XY(float(x1), float(y1)), XY(float(x2), float(y2)))

    @classmethod
    def from_xywh(cls, x : float, y : float, w : float, h : float) -> 'Rect':
        return cls(XY(float(x), float(y)), XY(float(x + w), float(y + h)))

    @classmethod
    def from_p1p2(cls, p1 : XY, p2 : XY) -> 'Rect':
        return cls(p1, p2)

    # @classmethod
    # def from_xywh(cls, x1 : float, y1 : float, w : float, h : float) -> 'Rect':
    #     halfwidth = w / 2
    #     halfheight = h / 2
    #     return cls(XY(x1 - halfwidth, y1 - halfheight), XY(x1 + halfwidth, y1 + halfheight))

    @property
    def width(self) -> float:
        return abs(self.p1.x - self.p2.x)

    @property
    def height(self) -> float:
        return abs(self.p1.y - self.p2.y)

    @property
    def center(self) -> XY:
        self.normalize()
        return XY(x = self.p1.x + self.width / 2, y = self.p1.y + self.height / 2)

    @property
    def left_edge(self) -> float:
        return min(self.p1.x, self.p2.x)

    @property
    def right_edge(self) -> float:
        return max(self.p1.x, self.p2.x)

    @property
    def top_edge(self) -> float:
        return min(self.p1.y, self.p2.y)

    @property
    def bottom_edge(self) -> float:
        return max(self.p1.y, self.p2.y)

    @property
    def size(self) -> XY:
        return XY(x = self.width, y = self.height)

    @property
    def min_point(self) -> XY:
        return XY(self.left_edge, self.top_edge)

    @property
    def max_point(self) -> XY:
        return XY(self.right_edge, self.bottom_edge)

    @property
    def top_left(self) -> XY:
        return XY(self.left_edge, self.top_edge)

    @property
    def top_right(self) -> XY:
        return XY(self.right_edge, self.top_edge)

    @property
    def bottom_left(self) -> XY:
        return XY(self.left_edge, self.bottom_edge)

    @property
    def bottom_right(self) -> XY:
        return XY(self.right_edge, self.bottom_edge)

    def moved_by(self, d : XY) -> 'Rect':
        return self.__class__(self.p1 + d, self.p2 + d)

    def to_xywh(self):
        c = self.center
        return (c.x, c.y, self.width, self.height)

    def to_xyxy(self):
        return (self.p1.x, self.p1.y, self.p2.x, self.p2.y)

    def multiplied_by_XY(self, other_xy: 'XY'):
        return self.__class__(self.p1.multiplied_by_XY(other_xy), self.p2.multiplied_by_XY(other_xy))

    def divided_by_XY(self, other_xy: 'XY'):
        div_xy = XY(1 / other_xy.x, 1 / other_xy.y)
        return self.multiplied_by_XY(div_xy)

    def is_point_inside(self, point : XY) -> bool:
        return point >= self.min_point and point <= self.max_point

    def clone(self) -> 'Rect':
        return self.__class__(self.p1.clone(), self.p2.clone())

    def to_intRect(self) -> 'Rect':
        return self.__class__(self.p1.to_intXY(), self.p2.to_intXY())

    def to_tuple(self, to = lambda x: x):
        return (*self.p1.to_tuple(to=to), *self.p2.to_tuple(to=to))

    def corners(self):
        return (self.p1, self.p2, XY(self.p1.x, self.p2.y), XY(self.p2.x, self.p1.y))

    def area(self):
        return self.width * self.height


    def intersection(self, other: 'Rect') -> 'Rect':
        """
        Compute the intersection of self and other rect.

        Args:
        other (Rect): The other rectangle to intersect with.

        Returns:
        Rect: A new rectangle representing the intersection of self and other.
        """
        x1 = max(self.left_edge, other.left_edge)
        y1 = max(self.top_edge, other.top_edge)
        x2 = min(self.right_edge, other.right_edge)
        y2 = min(self.bottom_edge, other.bottom_edge)

        # Check if the intersection is valid (i.e., not empty)
        if x1 < x2 and y1 < y2:
            return Rect.from_xyxy(x1, y1, x2, y2)
        else:
            # If the intersection is empty, return None or a special "empty" rectangle
            return Rect()


# @dataclass(init=True, slots=True, order=True)
class RelativeRect(Rect):
    """
    Rect with every coordinate of every point is inside [0..1] range
    i.e. rect relative to width/height of some image
    """

    def normalize(self):
        super().normalize()

        self.p1.x = max(0.0, self.p1.x)
        self.p1.y = max(0.0, self.p1.y)
        self.p2.x = min(1.0, self.p2.x)
        self.p2.y = min(1.0, self.p2.y)

    @classmethod
    def from_wh_at_center(cls, width: float, height: float) -> 'RelativeRect':
        """ Constructs Rect at center (0.5, 0.5) with given width and height.
        """
        half_width = width / 2
        half_height = height / 2
        return cls.from_xyxy(0.5 - half_width, 0.5 - half_height, 0.5 + half_width, 0.5 + half_height)

    # needed since Rect has no limitations and can be operated in int
    def to_rect(self):
        return Rect(self.p1, self.p2)


# based on https://docs.python.org/3/library/typing.html#typing.Annotated
@dataclass(frozen=True)
class ValueRange:
    lowest_value: int # lowest possible value (including)
    highest_value: int # highest possible value (including)


def to_timedelta(seconds : float):
    whole_seconds = int(seconds)
    microseconds= (seconds - whole_seconds) * 1000000
    return datetime.timedelta(seconds=whole_seconds, microseconds=int(microseconds))


def log_execution_time(logger=logging.debug, threshold = datetime.timedelta(microseconds=1), prefix='', log_with_arguments=True):
    if not isinstance(threshold, datetime.timedelta):
        threshold = to_timedelta(threshold)

    def wrapper_outer(f):
        @wraps(f)
        def wrapper_inner(*args, **kwargs):
            start = datetime.datetime.now()

            result = f(*args, **kwargs)

            end = datetime.datetime.now()
            duration = end - start

            if threshold.total_seconds() == 0 or duration >= threshold:
                if log_with_arguments:
                    args_tuple = tuple(args)
                    #TODO: less than ideal because it converts kwargs to strings... let's fix it someday
                    args_tuple = args_tuple + tuple(f"{k}={v!r}" for k, v in kwargs.items())

                    logger('%s%s%r\ttimed %s', prefix, f.__qualname__, args_tuple, duration, stacklevel=2)
                else:
                    logger('%s%s(-omitted-)\ttimed %s', prefix, f.__qualname__, duration, stacklevel=2)

            return result

        return wrapper_inner

    return wrapper_outer


# Keeps the async log listener alive for the lifetime of the process.
_log_queue_listener = None
_log_listener_atexit_registered = False


def _stop_log_listener():
    """Flush and stop the async log listener if it is running. Guarded because
    QueueListener.stop() is not idempotent (it sets _thread=None then joins it), so a
    second call — e.g. atexit after a manual/reconfigure stop — would raise."""
    listener = _log_queue_listener
    if listener is not None and getattr(listener, "_thread", None) is not None:
        listener.stop()


def configure_logging(level=logging.NOTSET, process_prefix="", log_file_name=""):
    """Configure root logging with an ASYNC pipeline so log I/O never blocks the
    calling thread (critical for the real-time control loop).

    The only handler on the root logger is a QueueHandler: its emit() resolves the
    record's message args and drops the record on an unbounded in-memory queue (no I/O,
    no blocking). A background QueueListener thread drains the queue and runs the real
    StreamHandler — so the asctime/prefix assembly and the write+flush syscall all
    happen off the hot path.

    NOTE: the message %-args ARE resolved on the calling thread (inside QueueHandler),
    so logging a large/mutable object (e.g. the telemetry dict) still pays its repr
    cost here — that is intentional: deferring it to the listener would let later
    mutations of that object corrupt the logged value. Keep such lines cheap/guarded.
    """
    global _log_queue_listener, _log_listener_atexit_registered

    class _ExcludeGrpcCallInitFilter(logging.Filter):
        def filter(self, record):
            return not (record.module == "_call")

    process_prefix = f"{process_prefix}-" if process_prefix else ""
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [" + process_prefix + "%(threadName)s] @ { %(filename)s:%(lineno)s : %(funcName)20s() } <%(levelname)s> :\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Real handler: runs on the listener thread; does the heavy formatting + I/O.
    stream_handler = logging.StreamHandler(sys.stdout)  # Writes to standard output
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_ExcludeGrpcCallInitFilter())

    # Unbounded queue: put_nowait() never blocks the producer (the control thread).
    log_queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)

    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(queue_handler)

    # (Re)start the background listener that owns the real handler. Stop any previous
    # one first so reconfiguration doesn't leak threads or duplicate output.
    _stop_log_listener()
    _log_queue_listener = logging.handlers.QueueListener(
        log_queue, stream_handler, respect_handler_level=True
    )
    _log_queue_listener.start()
    if not _log_listener_atexit_registered:
        atexit.register(_stop_log_listener)
        _log_listener_atexit_registered = True

    # based on https://stackoverflow.com/a/7995762
    logging.addLevelName(logging.INFO, "\033[1;34m%s\033[1;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.CRITICAL, "\033[1;101m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL))
    logging.addLevelName(logging.FATAL, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.FATAL))


# based on https://stackoverflow.com/a/23689767
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



@dataclass(slots=True, order=True, frozen=True)
class Detection:
    bbox : Rect = field(default_factory=Rect)
    confidence : float = 0.0
    track_id : int|None = 0
    class_id : int|None = None

@dataclass(slots=True, frozen=True)
class FrameMetadata:
    capture_timestamp_ns : int = 0
    detection_start_timestamp_ns : int = 0
    detection_end_timestamp_ns : int = 0


# Default camera id used when only a single camera is configured (the original
# single-camera path). Frames carrying this id behave like the legacy behaviour:
# downstream code that does not yet know about per-camera configs should still
# work via the single CameraConfig that the producer registers under this id.
DEFAULT_CAMERA_ID: int = 0


@dataclass(slots=True, frozen=True)
class CameraConfig:
    """Per-camera configuration that travels with every frame.

    `frame_angular_size_deg` is the per-camera horizontal/vertical FOV used by
    drone/platform controllers to convert a normalized bbox center into a
    steering angle. Different physical cameras (wide vs tele) have different
    FOVs and zoom — the consumer must look this up by `camera_id` so it does
    not apply the wide-lens FOV to a tele-lens frame.

    `zoom_factor` is NOT provided by the user; CameraSwitcher fills it in at
    construction time as the linear magnification relative to the widest
    camera in the set (so the widest camera always has zoom_factor = 1.0).

    Resolution / framerate / video_format are NOT here on purpose: all
    cameras feed a single shared appsrc, so they must produce identical
    caps. CameraSwitcher holds those values once and all producers + the
    pipeline read them from there.
    """
    camera_id : int = DEFAULT_CAMERA_ID
    name : str = ""                         # human-readable, for logs/debug
    sensor_index : int = 0                  # Picamera2 camera_num
    frame_angular_size_deg : XY = field(default_factory=lambda: XY(107, 85))
    # Linear magnification relative to the widest camera (1.0 = widest).
    # Auto-filled by CameraSwitcher; do not set in app config.
    zoom_factor : float = 1.0
    # libcamera/picamera2 tuning override; None lets the producer pick a default.
    tuning_file : str | None = None
    # Optional picam2 controls applied at startup (FrameRate, ExposureTime, ...).
    initial_controls : dict | None = None


@dataclass(slots=True, frozen=True)
class Detections:
    frame_id : int
    frame : np.ndarray | None = None
    detections : list[Detection] = field(default_factory=list)
    meta : FrameMetadata = field(default_factory=FrameMetadata)
    # Identifies which physical camera produced this frame. Consumers must use
    # it to look up the matching CameraConfig (FOV/zoom) and to detect camera
    # switches so they can purge per-camera caches (target trajectory, ByteTrack
    # lock, PD history, etc.).
    camera_id : int = DEFAULT_CAMERA_ID


@dataclass(slots=True)
class MoveCommand:
    # X - yaw
    adjust_attitude : XY = field(default_factory=XY)
    move_speed_ms : float = 0.0


def _fill_zoom_factor(cam : CameraConfig, peers : list[CameraConfig]) -> CameraConfig:
    """Return `cam` with zoom_factor set to its linear magnification relative
    to the widest camera in `peers` (so the widest camera ends up as 1.0).
    Uses the tan-based formula so it is exact for rectilinear lenses at any
    FOV; under small-angle approx it reduces to the FOV ratio."""
    wide_fov_x = max(p.frame_angular_size_deg.x for p in peers)
    if cam.frame_angular_size_deg.x <= 0 or wide_fov_x <= 0:
        return cam
    tan_wide = math.tan(math.radians(wide_fov_x / 2.0))
    tan_self = math.tan(math.radians(cam.frame_angular_size_deg.x / 2.0))
    if tan_self <= 0:
        return cam
    from dataclasses import replace
    return replace(cam, zoom_factor=tan_wide / tan_self)


class CameraSwitcher:
    """Shared, thread-safe selector that controls which camera_id is currently
    feeding the inference pipeline.

    Picam producer threads call `is_active(camera_id)` before pushing a frame
    into the (shared) appsrc, so the inactive camera keeps capturing (to stay
    AGC/AWB-warm for instant switch) but does not waste GStreamer/inference
    bandwidth. The controller calls `set_active(camera_id)` to switch.

    `get_config(camera_id)` returns the matching CameraConfig — the consumer
    (drone_controller/platform_controller) uses this to swap FOV/zoom when a
    new camera_id appears on a Detections object.
    """
    def __init__(self,
                 configs : list[CameraConfig],
                 *,
                 width : int = 1280,
                 height : int = 720,
                 fps : int = 30,
                 video_format : str = 'RGB',
                 active_id : int | None = None,
                 switch_to_wide_size : float = 0.25,
                 switch_to_zoom_size : float = 0.015,
                 autoexposure = None,
                 buffer_count : int = 2):
        if not configs:
            raise ValueError("CameraSwitcher requires at least one CameraConfig")
        # All cameras share one appsrc → one set of caps. Hold them here so
        # there is exactly one source of truth; the producer threads and the
        # pipeline both read width/height/fps/video_format from this object
        # instead of from per-camera fields.
        self.width = width
        self.height = height
        self.fps = fps
        self.video_format = video_format
        # Exposure/gain control for the picamera producer: a values-only object
        # with the Config.Camera.AutoExposure fields (exposure_time_us,
        # analogue_gain, exposure_auto_pin_s, exposure_min_us, exposure_max_us,
        # gain_max), or None to leave plain auto-exposure on. Read field-by-field
        # by picamera_thread (duck-typed, no config import here).
        self.autoexposure = autoexposure
        # picamera2 DMA pool depth (frames in flight); 2 = floor. See Config.Camera.
        self.buffer_count = buffer_count
        # Switch policy thresholds, EMA-tested in the controller:
        #   zoom→wide fires when EMA(max(w,h)) >= switch_to_wide_size
        #   wide→zoom fires when EMA(max(w,h)) <= switch_to_zoom_size
        # Held here so the consumer reads one policy object instead of
        # duplicating values across control_config / per-controller code.
        self.switch_to_wide_size = switch_to_wide_size
        self.switch_to_zoom_size = switch_to_zoom_size
        # Derive each camera's zoom factor relative to the widest camera in the
        # set. We use the tan-based linear magnification (correct for a pinhole
        # / rectilinear projection), normalized so the widest horizontal FOV
        # maps to zoom_factor = 1.0. Horizontal FOV is used as the reference;
        # vertical FOV usually has the same aspect ratio.
        configs = [_fill_zoom_factor(c, configs) for c in configs]
        self._configs : dict[int, CameraConfig] = {c.camera_id: c for c in configs}
        if len(self._configs) != len(configs):
            raise ValueError("CameraConfig.camera_id must be unique")
        if active_id is None:
            active_id = configs[0].camera_id
        if active_id not in self._configs:
            raise ValueError(f"active_id {active_id} not in configs")
        self._active_id = active_id
        self._lock = threading.Lock()

    def configs(self) -> list[CameraConfig]:
        return list(self._configs.values())

    def num_cameras(self):
        return len(self._configs)

    def get_config(self, camera_id : int) -> CameraConfig | None:
        return self._configs.get(camera_id)

    def active_id(self) -> int:
        with self._lock:
            return self._active_id

    def active_config(self) -> CameraConfig:
        with self._lock:
            return self._configs[self._active_id]

    def is_active(self, camera_id : int) -> bool:
        with self._lock:
            return self._active_id == camera_id

    def set_active(self, camera_id : int) -> bool:
        """Set the active camera. Returns True if it changed."""
        with self._lock:
            if camera_id not in self._configs:
                raise ValueError(f"unknown camera_id {camera_id}")
            if self._active_id == camera_id:
                return False
            self._active_id = camera_id
            return True

    def toggle(self, from_id = None) -> int | None:
        """Convenience: switch to the next camera_id in the configured order.
        Optional: if `from_id` is NOT None, then switch only if current `active_id == from_id`
        Returns: None if NOT switched, otherise active_id
        """
        with self._lock:
            if from_id is not None and self._active_id != from_id:
                return None

            ids = list(self._configs.keys())
            idx = ids.index(self._active_id)
            self._active_id = ids[(idx + 1) % len(ids)]
            return self._active_id



# =============================================================================
# DEBUG STUFF
# =============================================================================
#
def full_classname(obj):
    # based on https://gist.github.com/clbarnes/edd28ea32010eb159b34b075687bb49e#file-classname-py
    cls = type(obj)
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name

def DEBUG_dump(prefix, obj):
    print(prefix, obj, full_classname(obj), dir(obj))


# Source - https://stackoverflow.com/a/70397050
# Posted by timfjord
# Retrieved 2026-03-02, License - CC BY-SA 4.0
class LoggerWithPrefix(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger, prefix = '') -> None:
        super().__init__(logger, None)
        self.prefix = prefix

    def process(self, msg, kwargs):
        return f"{self.prefix} {msg}", kwargs



#
# =============================================================================


def _safe_repr(value, max_length: int = 160) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class _DebugCallProxy:
    """
    Transparent proxy that records the last invoked method call and forwards
    all operations to the wrapped object.
    """

    def __init__(self, wrapped, object_name: str, history_max_size = 3):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_object_name", object_name)
        object.__setattr__(self, "_last_command", "")
        object.__setattr__(self, "_command_history", deque(maxlen = history_max_size))

    def _format_call(self, method_name: str, callable_obj, args, kwargs) -> str:
        try:
            signature = inspect.signature(callable_obj)
            bound = signature.bind_partial(*args, **kwargs)
            params = [f"{name}={_safe_repr(value)}" for name, value in bound.arguments.items()]
        except Exception:
            positional = [f"arg{i + 1}={_safe_repr(value)}" for i, value in enumerate(args)]
            keyword = [f"{name}={_safe_repr(value)}" for name, value in kwargs.items()]
            params = positional + keyword

        return f"{self._object_name}.{method_name}({', '.join(params)})"

    def _record_call(self, method_name: str, callable_obj, args, kwargs):
        command = self._format_call(method_name, callable_obj, args, kwargs)
        object.__setattr__(self, "_last_command", command)
        self._command_history.append(command)

    def last_command(self) -> str:
        return self._last_command

    def command_history(self) -> list[str]:
        return list(self._command_history)

    def clear_command_history(self) -> None:
        self._command_history.clear()
        object.__setattr__(self, "_last_command", "")

    def unwrap(self):
        return self._wrapped

    def __getattr__(self, name: str):
        attr = getattr(self._wrapped, name)
        if not callable(attr):
            return attr

        method_name = getattr(attr, "__name__", name)

        if inspect.iscoroutinefunction(attr):
            @wraps(attr)
            async def async_wrapper(*args, **kwargs):
                self._record_call(method_name, attr, args, kwargs)
                return await attr(*args, **kwargs)

            return async_wrapper

        @wraps(attr)
        def sync_wrapper(*args, **kwargs):
            self._record_call(method_name, attr, args, kwargs)
            return attr(*args, **kwargs)

        return sync_wrapper

    def __setattr__(self, name, value):
        if name.startswith("_") or hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)

    def __delattr__(self, name):
        if name.startswith("_") or hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self._wrapped, name)

    def __dir__(self):
        return sorted(set(dir(self._wrapped) + list(type(self).__dict__.keys())))

    def __repr__(self):
        return f"{type(self).__name__}(wrapped={self._wrapped!r}, object_name={self._object_name!r})"


def debug_collect_call_info(obj, object_name: str | None = None, history_max_size = 0):
    """
    Wrap any object so method calls are logged as strings like:
    `drone.move_to_target_zenith_async(roll_degree=1.0, pitch_degree=2.0, thrust=0.4)`
    """
    if isinstance(obj, _DebugCallProxy):
        return obj

    if object_name is None:
        object_name = type(obj).__name__
        if "drone" in object_name.lower():
            object_name = "drone"

    return _DebugCallProxy(obj, object_name=object_name, history_max_size = history_max_size)

@dataclass(frozen=True, slots=True)
class StopSignal:
    # __slots__ = ()
    pass

STOP = StopSignal()
