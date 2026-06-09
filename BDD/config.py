#!/usr/bin/env python3
"""Typed, validated runtime configuration.

The application configuration lives in a YAML file (see ``config.yaml``) and is
parsed into the :class:`Config` dataclass below. The dataclass IS the schema:
field types and the constraint metadata attached to them (``Range``,
``Choices``, ``MinItems``) drive both type checking and bound checking while
parsing.

Design notes:
  * Top-level subsections that used to be nested dicts (``camera``, ``drone``,
    ``bytetrack``) are now nested dataclass instances.
  * The parser accumulates *all* type/bound problems and reports them in bulk
    instead of failing on the first one.
  * Any key present in the file that does not map to a dataclass field is
    reported as an error too (so typos in the config are caught loudly).
"""

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union, get_args, get_origin
import inspect
import types

from helpers import XY

# VelocityMethod is used directly as the choice set for `estimation_3d_method`.
from TargetEstimator import VelocityMethod


# ---------------------------------------------------------------------------
# Constraint types. A constraint instance IS the field's type annotation and
# carries the underlying base type, e.g.
#     confidence_min:  Range(float, 0.0, 1.0)
#     video_format:    Choices('RGB', 'BGR', ...)
#     estimation_method: Choices(VelocityMethod)        # enum -> its values
#     cameras:         MinItems(CameraEntry, 1)         # -> list[CameraEntry]
# Each `validate` appends human-readable problems to `errors` and never raises.
# ---------------------------------------------------------------------------
class _Constraint:
    #: The underlying type the parser coerces the value to before validating.
    base: Any = Any

    def validate(self, value: Any, path: str, errors: list[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Range(_Constraint):
    """A numeric (or XY) `base` with an inclusive-by-default bound.

    Usage: ``Range(float, 0.0, 1.0)``, ``Range(int, min=1)``,
    ``Range(XY, min=0.0, max=360.0)``, ``Range(Optional[float], min=0.0)``.
    """
    def __init__(self, base, min=None, max=None, *,
                 min_inclusive: bool = True, max_inclusive: bool = True):
        self.base = base
        self.min = min
        self.max = max
        self.min_inclusive = min_inclusive
        self.max_inclusive = max_inclusive

    def _check_scalar(self, value: float, path: str, errors: list[str]) -> None:
        if self.min is not None:
            if self.min_inclusive and value < self.min:
                errors.append(f"{path}: {value} is below minimum {self.min}")
            elif not self.min_inclusive and value <= self.min:
                errors.append(f"{path}: {value} must be greater than {self.min}")
        if self.max is not None:
            if self.max_inclusive and value > self.max:
                errors.append(f"{path}: {value} is above maximum {self.max}")
            elif not self.max_inclusive and value >= self.max:
                errors.append(f"{path}: {value} must be less than {self.max}")

    def validate(self, value: Any, path: str, errors: list[str]) -> None:
        # Numeric scalars are checked directly; dataclasses (e.g. XY) have the
        # bound applied to each of their numeric fields individually.
        if is_dataclass(value):
            for f in fields(value):
                self.validate(getattr(value, f.name), f"{path}.{f.name}", errors)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self._check_scalar(value, path, errors)


class Choices(_Constraint):
    """Value must be one of a fixed set.

    Pass an Enum class to use its member values (``Choices(VelocityMethod)``),
    or pass the allowed values directly (``Choices('RGB', 'BGR')``). The base
    type is inferred from the values.
    """
    def __init__(self, *allowed):
        if len(allowed) == 1 and isinstance(allowed[0], type) and issubclass(allowed[0], Enum):
            self.allowed = tuple(m.value for m in allowed[0])
        else:
            self.allowed = tuple(allowed)
        self.base = type(self.allowed[0]) if self.allowed else str

    def validate(self, value: Any, path: str, errors: list[str]) -> None:
        if value not in self.allowed:
            errors.append(f"{path}: {value!r} is not one of {list(self.allowed)}")


class MinItems(_Constraint):
    """A ``list[item_type]`` that must contain at least `count` items.

    Usage: ``MinItems(CameraEntry, 1)``.
    """
    def __init__(self, item_type, count: int):
        self.base = list[item_type]
        self.count = count

    def validate(self, value: Any, path: str, errors: list[str]) -> None:
        if isinstance(value, list) and len(value) < self.count:
            errors.append(f"{path}: needs at least {self.count} item(s), got {len(value)}")


def _xy_factory(x: float, y: float):
    return lambda: XY(x, y)


# ---------------------------------------------------------------------------
# Nested sections
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ByteTrackSection:
    """Parameters for the ByteTrack multi-object tracker.

    Everything except `target_lock` is forwarded verbatim to BYTETracker; use
    `tracker_kwargs()` to get exactly that subset.
    """
    track_thresh:      Range(float, 0.0, 1.0) = 0.5
    det_thresh:        Range(float, 0.0, 1.0) = 0.6
    match_thresh:      Range(float, 0.0, 1.0) = 0.8
    track_buffer:      Range(int, min=1)      = 30
    frame_rate:        Range(float, min=0.0, min_inclusive=False) = 30.0
    match_max_dist:    Range(Optional[float], min=0.0) = None
    recovery_max_dist: Range(Optional[float], min=0.0) = None
    nms_thresh:        Range(Optional[float], 0.0, 1.0) = None
    nms_dist_thresh:   Range(Optional[float], min=0.0) = None
    # Consumed by the controller (not BYTETracker): keep the locked target id.
    target_lock:       bool = True

    def tracker_kwargs(self) -> dict:
        return {
            'track_thresh':      self.track_thresh,
            'det_thresh':        self.det_thresh,
            'match_thresh':      self.match_thresh,
            'track_buffer':      self.track_buffer,
            'frame_rate':        self.frame_rate,
            'match_max_dist':    self.match_max_dist,
            'recovery_max_dist': self.recovery_max_dist,
            'nms_thresh':        self.nms_thresh,
            'nms_dist_thresh':   self.nms_dist_thresh,
        }


@dataclass(slots=True)
class CameraEntry:
    """One physical camera. Maps onto helpers.CameraConfig.

    Resolution/fps/video_format are intentionally NOT here: all cameras share a
    single appsrc and must produce identical caps (those live in CameraSection).
    """
    camera_id:              Range(int, min=0) = 0
    name:                   str = ""
    sensor_index:           Range(int, min=0) = 0
    # Per-camera horizontal/vertical FOV in degrees, (0, 360].
    frame_angular_size_deg: Range(XY, min=0.0, min_inclusive=False, max=360.0) = \
        field(default_factory=_xy_factory(107, 85))


@dataclass(slots=True)
class CameraSection:
    """Shared camera caps + the list of cameras.

    The shared caps apply to every camera (single appsrc, identical caps); the
    per-camera differences live in `cameras`.
    """
    width:                Range(int, min=1) = 1280
    height:               Range(int, min=1) = 720
    fps:                  Range(int, min=1) = 30
    video_format:         Choices('RGB', 'BGR', 'RGBA', 'BGRA',
                                  'XRGB', 'XBGR', 'YUV420', 'NV12') = 'RGB'
    active_id:            Range(int, min=0) = 0
    # Relative target size (max(w,h)) that triggers switching wide/zoom. Both in (0,1].
    switch_to_wide_size:  Range(float, 0.0, 1.0, min_inclusive=False) = 0.25
    switch_to_zoom_size:  Range(float, 0.0, 1.0, min_inclusive=False) = 0.015
    # Platform-only initial position (normalized).
    platform_initial_pos: XY = field(default_factory=_xy_factory(0, 0))
    # At least one camera must be configured explicitly (no default).
    cameras: MinItems(CameraEntry, 1) = field(kw_only=True)


@dataclass(slots=True)
class DroneControlConfig:
    """The `drone.config` block — consumed by DroneMover."""
    upside_down_angle_deg:           Range(float, 0.0, 360.0) = 130.0
    upside_down_hold_s:              Range(float, min=0.0) = 0.2
    use_set_attitude:                bool = False
    min_lift_fraction:               Range(float, 0.0, 1.0) = 0.1
    # Upward velocity headroom (m/s) when tilt restrictions are relaxed.
    lift_velocity_headroom_ms:       Range(float, min=0.0) = 3.0
    # Upward acceleration headroom (m/s^2) when tilt restrictions are relaxed.
    lift_accel_headroom_mss:         Range(float, min=0.0) = 5.0
    belly_down_yaw:                  bool = True
    belly_down_yaw_kp:               Range(float, min=0.0) = 1.5
    belly_down_yaw_max_rate_deg_s:   Range(float, min=0.0) = 90.0
    belly_down_min_horizontal_g_mss: Range(float, min=0.0) = 2.0


@dataclass(slots=True)
class DroneSection:
    connection_string: str = 'usb'
    # The control block must be present explicitly (no default).
    config: DroneControlConfig = field(kw_only=True)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Config:
    """Validated runtime configuration.

    Scalar/XY fields are flat at the top level; the former nested dicts
    (`camera`, `drone`, `bytetrack`) are nested dataclasses.
    """
    confidence_min:  Range(float, 0.0, 1.0) = 0.4
    confidence_move: Range(float, 0.0, 1.0) = 0.3

    # PX4 normalized thrust, 0..1.
    thrust_takeoff: Range(float, 0.0, 1.0) = 1.0
    thrust_cruise:  Range(float, 0.0, 1.0) = 0.53
    thrust_hover:   Range(float, 0.0, 1.0) = 0.4
    thrust_min:     Range(float, 0.0, 1.0) = 0.4
    thrust_max:     Range(float, 0.0, 1.0) = 0.9

    thrust_dynamic:                  bool = False
    thrust_proportional_to_distance: bool = True
    thrust_proportional_to_distance_far_coeff:        Range(float, min=0.0) = 1.0
    thrust_proportional_to_distance_medium_distance_m: Range(float, min=0.0) = 20.0
    thrust_proportional_to_distance_medium_coeff:     Range(float, min=0.0) = 0.9
    thrust_proportional_to_distance_near_distance_m:   Range(float, min=0.0) = 10.0
    thrust_proportional_to_distance_near_coeff:       Range(float, min=0.0) = 1.1

    # Per-frame multiplicative fade of target confidence after loss, 0..1.
    target_lost_fade_per_frame: Range(float, 0.0, 1.0) = 0.99
    target_estimator_clear_history_after_target_lost_frames: Range(int, min=0) = 3

    estimation_3d:                      bool = True
    estimation_3d_method:               Choices(VelocityMethod) = 'numpy'
    estimation_3d_use_initial_velocity: bool = False

    estimation_lookahead_frames:                Range(int, min=0) = 1
    estimation_lookahead_dynamic:               bool = True
    estimation_lookahead_dynamic_frames_max:    Range(int, min=0) = 5
    estimation_lookahead_dynamic_sqrt:          bool = False
    estimation_lookahead_dynamic_factor:        Range(float, min=0.0) = 0.1
    estimation_lookahead_dynamic_frames_near:   Range(int, min=0) = 0
    estimation_lookahead_dynamic_frames_medium: Range(int, min=0) = 0
    estimation_lookahead_dynamic_frames_far:    Range(int, min=0) = 0

    optical_methods_to_refine_target_size_and_center: bool = True
    adjust_aim_point_at_edge_of_frame:                bool = True
    adjust_aim_point_at_edge_of_frame_threshold:      Range(float, 0.0, 1.0) = 0.01
    # w*h, so a normalized area in 0..1.
    adjust_aim_point_at_edge_of_frame_max_size:       Range(float, 0.0, 1.0) = 0.25

    # Per-axis P gain (x, y); non-negative. These must be set explicitly (no
    # default) — they directly shape the control response and silently
    # defaulting them is dangerous.
    pd_coeff_p:          Range(XY, min=0.0) = field(kw_only=True)
    pd_coeff_d:          float = 0.0
    pd_coeff_p_safe_min: Range(XY, min=0.0) = field(kw_only=True)
    pd_coeff_p_min:      Range(XY, min=0.0) = field(kw_only=True)
    pd_coeff_p_max:      Range(XY, min=0.0) = field(kw_only=True)

    pd_coeff_p_dynamic:               bool = False
    pd_coeff_p_dynamic_use_piecewise: bool = False
    # Normalized target size w*h (both in 0..1), so the product is in 0..1.
    pd_coeff_p_dynamic_min_target_size: Range(float, 0.0, 1.0) = 0.0005
    pd_coeff_p_dynamic_min:             Range(float, min=0.0) = 0.6
    pd_coeff_p_dynamic_max_target_size: Range(float, 0.0, 1.0) = 0.0120
    pd_coeff_p_dynamic_max:             Range(float, min=0.0) = 6.0

    # Normalized size thresholds s in 0..1.
    pd_coeff_p_dynamic_stage_1_threshold: Range(float, 0.0, 1.0) = 0.01
    pd_coeff_p_dynamic_stage_2_threshold: Range(float, 0.0, 1.0) = 0.05
    # Relative P ratios inside [min, max], 0..1.
    pd_coeff_p_dynamic_stage_1_ratio: Range(float, 0.0, 1.0) = 1.0
    pd_coeff_p_dynamic_stage_2_ratio: Range(float, 0.0, 1.0) = 1.0
    pd_coeff_p_dynamic_stage_3_ratio: Range(float, 0.0, 1.0) = 1.0

    # Physical target size in meters; positive.
    target_size_m: Range(XY, min=0.0, min_inclusive=False) = field(default_factory=_xy_factory(2, 2))
    # Legacy / single-camera FOV fallback in degrees, (0, 360].
    frame_angular_size_deg: Range(XY, min=0.0, min_inclusive=False, max=360.0) = \
        field(default_factory=_xy_factory(120, 90))

    safe_takeoff_period_ns:                Range(int, min=0) = 1_000_000_000
    delay_takeof_until_n_detection_frames: Range(int, min=0) = 10

    # Normalized image coordinates, 0..1.
    aim_point:            Range(XY, 0.0, 1.0) = field(default_factory=_xy_factory(0.5, 0.5))
    aim_point_max_offset: Range(XY, 0.0, 1.0) = field(default_factory=_xy_factory(0.5, 0.6))

    follow_target_position_ned: bool = False

    # EMA smoothing factor for camera-switch target size, 0..1.
    camera_switch_size_ema_alpha: Range(float, 0.0, 1.0) = 0.3

    # Complex sub-sections must be present explicitly (no default) so an
    # incomplete config fails loudly instead of silently using a stand-in.
    camera:    CameraSection    = field(kw_only=True)
    drone:     DroneSection     = field(kw_only=True)
    bytetrack: ByteTrackSection = field(kw_only=True)

    # Runtime-only fields: set programmatically, NEVER read from the config
    # file (providing them in the file is reported as an unknown key).
    DEBUG:                bool = field(default=False, metadata={'runtime': True})
    debug_telemetry_dict: Optional[dict] = field(default=None, metadata={'runtime': True})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class ConfigError(ValueError):
    """Raised with every accumulated type/bound/unknown-key problem at once."""
    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        joined = "\n  - ".join(self.problems)
        super().__init__(f"Invalid configuration ({len(self.problems)} problem(s)):\n  - {joined}")


# Sentinel signalling a value failed to coerce; the field keeps its default.
_INVALID = object()
_NONE_TYPE = type(None)


def _split_constraint(ann):
    """Return (base_type, [constraints]) for a field annotation.

    A constraint instance (Range/Choices/MinItems) IS the annotation and
    carries its own base type; anything else is a plain type with no
    constraints.
    """
    if isinstance(ann, _Constraint):
        return ann.base, [ann]
    return ann, []


def _is_optional(hint):
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        return _NONE_TYPE in get_args(hint)
    return False


def _non_none_arg(hint):
    return next(a for a in get_args(hint) if a is not _NONE_TYPE)


def _coerce_xy(value, path, errors):
    if isinstance(value, XY):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
            errors.append(f"{path}: expected [x, y] of two numbers, got {value!r}")
            return _INVALID
        return XY(float(value[0]), float(value[1]))
    if isinstance(value, dict):
        if set(value) != {'x', 'y'} or any(
                isinstance(value[k], bool) or not isinstance(value[k], (int, float)) for k in ('x', 'y')):
            errors.append(f"{path}: expected mapping with numeric x and y, got {value!r}")
            return _INVALID
        return XY(float(value['x']), float(value['y']))
    errors.append(f"{path}: expected an XY ([x, y] or {{x, y}}), got {type(value).__name__}")
    return _INVALID


def _coerce(value, ann, path, errors):
    base, constraints = _split_constraint(ann)

    # Optional[...] / X | None
    if _is_optional(base):
        if value is None:
            return None
        base, inner_constraints = _split_constraint(_non_none_arg(base))
        constraints = constraints + inner_constraints

    coerced = _INVALID
    if base is bool:
        if not isinstance(value, bool):
            errors.append(f"{path}: expected a boolean, got {type(value).__name__}")
        else:
            coerced = value
    elif base is int:
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path}: expected an integer, got {type(value).__name__}")
        else:
            coerced = value
    elif base is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{path}: expected a number, got {type(value).__name__}")
        else:
            coerced = float(value)
    elif base is str:
        if not isinstance(value, str):
            errors.append(f"{path}: expected a string, got {type(value).__name__}")
        else:
            coerced = value
    elif base is XY:
        coerced = _coerce_xy(value, path, errors)
    elif is_dataclass(base):
        coerced = _parse_dataclass(base, value, path, errors)
    elif get_origin(base) in (list, types.GenericAlias) or base is list:
        item_type = get_args(base)[0] if get_args(base) else Any
        if not isinstance(value, list):
            errors.append(f"{path}: expected a list, got {type(value).__name__}")
        else:
            items = []
            for i, item in enumerate(value):
                r = _coerce(item, item_type, f"{path}[{i}]", errors)
                items.append(r if r is not _INVALID else None)
            coerced = items
    else:
        errors.append(f"{path}: unsupported field type {base!r}")

    if coerced is not _INVALID:
        for c in constraints:
            c.validate(coerced, path, errors)
    return coerced


def _required_names(cls) -> set:
    """Field names that have no default (neither default nor default_factory)."""
    return {f.name for f in fields(cls)
            if f.default is MISSING and f.default_factory is MISSING
            and not f.metadata.get('runtime')}


def _blank(cls):
    """Construct an instance, passing None for every required field so it never
    raises. Only used on the error path, where the result is discarded."""
    return cls(**{name: None for name in _required_names(cls)})


def _parse_dataclass(cls, data, path, errors):
    prefix = f"{path}." if path else ""
    required = _required_names(cls)
    if not isinstance(data, dict):
        errors.append(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")
        return _blank(cls)

    anns = inspect.get_annotations(cls)
    valid_names = set()
    kwargs = {}
    for f in fields(cls):
        valid_names.add(f.name)
        if f.metadata.get('runtime'):
            # Never read runtime fields from the file; leave the default.
            continue
        if f.name in data:
            r = _coerce(data[f.name], anns[f.name], f"{prefix}{f.name}", errors)
            if r is not _INVALID:
                kwargs[f.name] = r
        elif f.name in required:
            errors.append(f"{prefix}{f.name}: missing required configuration key")

    for key in data:
        if key not in valid_names or (key in valid_names and _runtime_field(cls, key)):
            errors.append(f"{prefix}{key}: unknown configuration key")

    # Placeholder for any required field still missing, so construction (used to
    # return *something*) never raises; on the error path the object is discarded.
    for name in required:
        kwargs.setdefault(name, None)
    try:
        return cls(**kwargs)
    except Exception as exc:  # pragma: no cover - kwargs are pre-validated
        errors.append(f"{path or '<root>'}: could not build {cls.__name__}: {exc}")
        return _blank(cls)


def _runtime_field(cls, name) -> bool:
    for f in fields(cls):
        if f.name == name:
            return bool(f.metadata.get('runtime'))
    return False


def _collect_node_lines(node, prefix: str, out: dict[str, int]) -> None:
    """Walk a yaml.compose() node tree, recording 1-based file lines per key path."""
    import yaml
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            path = f"{prefix}.{key_node.value}" if prefix else str(key_node.value)
            out[path] = key_node.start_mark.line + 1
            _collect_node_lines(value_node, path, out)
    elif isinstance(node, yaml.SequenceNode):
        for i, item in enumerate(node.value):
            path = f"{prefix}[{i}]"
            out[path] = item.start_mark.line + 1
            _collect_node_lines(item, path, out)


def _line_for_path(path: str, line_map: dict[str, int]) -> int | None:
    """Resolve an error path to a file line, falling back to the nearest parent.

    Synthetic sub-paths that have no YAML node of their own (an XY component
    like `aim_point.y`, or a list element's missing key) resolve to the line of
    their closest enclosing key/element.
    """
    p = path
    while p:
        if p in line_map:
            return line_map[p]
        cut = max(p.rfind('.'), p.rfind('['))
        if cut <= 0:
            break
        p = p[:cut]
    return line_map.get(p)


def _with_line_numbers(errors: list[str], line_map: dict[str, int], source: str) -> list[str]:
    """Append `(<source> line N)` to each error using its leading path token."""
    enriched = []
    for err in errors:
        path = err.split(":", 1)[0].strip()
        line = _line_for_path(path, line_map) if path and path != "<root>" else None
        enriched.append(f"{err} ({source} line {line})" if line else err)
    return enriched


def parse_config(data: dict | None, line_map: dict[str, int] | None = None,
                 source: str = "config file") -> Config:
    """Validate a parsed-YAML mapping and return a Config, or raise ConfigError.

    If `line_map` (path -> file line) is supplied, every reported problem is
    annotated with the offending line in `source`.
    """
    errors: list[str] = []
    cfg = _parse_dataclass(Config, data or {}, "", errors)
    if errors:
        raise ConfigError(_with_line_numbers(errors, line_map, source) if line_map else errors)
    return cfg


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated Config (raises ConfigError).

    Errors are reported in bulk and annotated with the offending file line.
    """
    import yaml
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError([f"<root>: top level must be a mapping, got {type(data).__name__}"])
    line_map: dict[str, int] = {}
    _collect_node_lines(yaml.compose(text), "", line_map)
    return parse_config(data, line_map, source=Path(path).name)
