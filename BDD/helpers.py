#!/usr/bin/env python3

from dataclasses import dataclass, field
from collections import deque
from functools import wraps
import inspect
import datetime
import math

import numpy as np

import logging

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

    def __str__(self):
        return f"XY({self.x:.2f}, {self.y:.2f})"


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


def configure_logging(level=logging.NOTSET, process_prefix=""):
    class _ExcludeGrpcCallInitFilter(logging.Filter):
        def filter(self, record):
            return not (record.module == "_call")
            # return not (
            #     record.filename == "_call.py"
            #     and record.lineno == 562
            #     and record.funcName == "__init__"
            # )

    process_prefix = f"{process_prefix}-" if process_prefix else ""
    logging.basicConfig(level=level,
        format="%(asctime)s.%(msecs)03d [" + process_prefix + "%(threadName)s] @ { %(filename)s:%(lineno)s : %(funcName)20s() } <%(levelname)s> :\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    exclude_grpc_call_init_filter = _ExcludeGrpcCallInitFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(exclude_grpc_call_init_filter)

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

@dataclass(slots=True, frozen=True)
class FrameMetadata:
    capture_timestamp_ns : int = 0
    detection_start_timestamp_ns : int = 0
    detection_end_timestamp_ns : int = 0

@dataclass(slots=True, frozen=True)
class Detections:
    frame_id : int
    frame : np.ndarray | None = None
    detections : list[Detection] = field(default_factory=list)
    meta : FrameMetadata = field(default_factory=FrameMetadata)


@dataclass(slots=True)
class MoveCommand:
    # X - yaw
    adjust_attitude : XY = field(default_factory=XY)
    move_speed_ms : float = 0.0



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

#
# =============================================================================


def _safe_repr(value, max_length: int = 160) -> str:
    text = repr(value)
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
