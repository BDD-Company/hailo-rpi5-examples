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

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Optional, Union, get_args, get_origin, get_type_hints
import types

from helpers import XY

# VelocityMethod's string values are the allowed `estimation_3d_method` choices.
from TargetEstimator import VelocityMethod


# ---------------------------------------------------------------------------
# Constraint metadata. Attached to a field type via typing.Annotated, e.g.
#     confidence_min: Annotated[float, Range(0.0, 1.0)]
# Each validator appends human-readable problems to `errors` and never raises.
# ---------------------------------------------------------------------------
class _Constraint:
    def validate(self, value: Any, path: str, errors: list[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class Range(_Constraint):
    """Inclusive-by-default numeric bound. Applied per-component for XY."""
    min: float | None = None
    max: float | None = None
    min_inclusive: bool = True
    max_inclusive: bool = True

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
        # TODO: rework to work for any dataclass (inpsect fields and chech those individually)
        if isinstance(value, XY):
            self._check_scalar(value.x, f"{path}.x", errors)
            self._check_scalar(value.y, f"{path}.y", errors)
        elif isinstance(value, (int, float)):
            self._check_scalar(value, path, errors)


@dataclass(frozen=True)
class Choices(_Constraint):
    """Value must be one of a fixed set."""
    allowed: tuple

    def __init__(self, *allowed):
        object.__setattr__(self, "allowed", tuple(allowed))

    def validate(self, value: Any, path: str, errors: list[str]) -> None:
        if value not in self.allowed:
            errors.append(f"{path}: {value!r} is not one of {list(self.allowed)}")


@dataclass(frozen=True)
class MinItems(_Constraint):
    """A list must contain at least `count` items."""
    count: int

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
    track_thresh:      Annotated[float, Range(0.0, 1.0)] = 0.5
    det_thresh:        Annotated[float, Range(0.0, 1.0)] = 0.6
    match_thresh:      Annotated[float, Range(0.0, 1.0)] = 0.8
    track_buffer:      Annotated[int,   Range(min=1)]    = 30
    frame_rate:        Annotated[float, Range(min=0.0, min_inclusive=False)] = 30.0
    match_max_dist:    Annotated[Optional[float], Range(min=0.0)] = None
    recovery_max_dist: Annotated[Optional[float], Range(min=0.0)] = None
    nms_thresh:        Annotated[Optional[float], Range(0.0, 1.0)] = None
    nms_dist_thresh:   Annotated[Optional[float], Range(min=0.0)] = None
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
    camera_id:              Annotated[int, Range(min=0)] = 0
    name:                   str = ""
    sensor_index:           Annotated[int, Range(min=0)] = 0
    # Per-camera horizontal/vertical FOV in degrees, (0, 360].
    frame_angular_size_deg: Annotated[XY, Range(min=0.0, min_inclusive=False, max=360.0)] = \
        field(default_factory=_xy_factory(107, 85))


@dataclass(slots=True)
class CameraSection:
    """Shared camera caps + the list of cameras.

    The shared caps apply to every camera (single appsrc, identical caps); the
    per-camera differences live in `cameras`.
    """
    width:                Annotated[int, Range(min=1)] = 1280
    height:               Annotated[int, Range(min=1)] = 720
    fps:                  Annotated[int, Range(min=1)] = 30
    video_format:         Annotated[str, Choices('RGB', 'BGR', 'RGBA', 'BGRA',
                                                 'XRGB', 'XBGR', 'YUV420', 'NV12')] = 'RGB'
    active_id:            Annotated[int, Range(min=0)] = 0
    # Relative target size (max(w,h)) that triggers switching wide/zoom. Both in (0,1].
    switch_to_wide_size:  Annotated[float, Range(0.0, 1.0, min_inclusive=False)] = 0.25
    switch_to_zoom_size:  Annotated[float, Range(0.0, 1.0, min_inclusive=False)] = 0.015
    # NOTE: must be at least 1 camera.
    cameras: Annotated[list[CameraEntry], MinItems(1)] = field(
        default_factory=lambda: [CameraEntry(camera_id=0, name='wide', sensor_index=0,
                                             frame_angular_size_deg=XY(107, 85))])


@dataclass(slots=True)
class DroneControlConfig:
    """The `drone.config` block — consumed by DroneMover."""
    upside_down_angle_deg:           Annotated[float, Range(0.0, 360.0)] = 130.0
    upside_down_hold_s:              Annotated[float, Range(min=0.0)] = 0.2
    use_set_attitude:                bool = False
    min_lift_fraction:               Annotated[float, Range(0.0, 1.0)] = 0.1
    # Upward velocity headroom (m/s) when tilt restrictions are relaxed.
    lift_velocity_headroom_ms:       Annotated[float, Range(min=0.0)] = 3.0
    # Upward acceleration headroom (m/s^2) when tilt restrictions are relaxed.
    lift_accel_headroom_mss:         Annotated[float, Range(min=0.0)] = 5.0
    belly_down_yaw:                  bool = True
    belly_down_yaw_kp:               Annotated[float, Range(min=0.0)] = 1.5
    belly_down_yaw_max_rate_deg_s:   Annotated[float, Range(min=0.0)] = 90.0
    belly_down_min_horizontal_g_mss: Annotated[float, Range(min=0.0)] = 2.0


@dataclass(slots=True)
class DroneSection:
    connection_string: str = 'usb'
    config: DroneControlConfig = field(default_factory=DroneControlConfig)


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Config:
    """Validated runtime configuration.

    Scalar/XY fields are flat at the top level; the former nested dicts
    (`camera`, `drone`, `bytetrack`) are nested dataclasses.
    """
    confidence_min:  Annotated[float, Range(0.0, 1.0)] = 0.4
    confidence_move: Annotated[float, Range(0.0, 1.0)] = 0.3

    # PX4 normalized thrust, 0..1.
    thrust_takeoff: Annotated[float, Range(0.0, 1.0)] = 1.0
    thrust_cruise:  Annotated[float, Range(0.0, 1.0)] = 0.53
    thrust_hover:   Annotated[float, Range(0.0, 1.0)] = 0.4
    thrust_min:     Annotated[float, Range(0.0, 1.0)] = 0.4
    thrust_max:     Annotated[float, Range(0.0, 1.0)] = 0.9

    thrust_dynamic:                  bool = False
    thrust_proportional_to_distance: bool = True
    thrust_proportional_to_distance_far_coeff:        Annotated[float, Range(min=0.0)] = 1.0
    thrust_proportional_to_distance_medium_distance_m: Annotated[float, Range(min=0.0)] = 20.0
    thrust_proportional_to_distance_medium_coeff:     Annotated[float, Range(min=0.0)] = 0.9
    thrust_proportional_to_distance_near_distance_m:   Annotated[float, Range(min=0.0)] = 10.0
    thrust_proportional_to_distance_near_coeff:       Annotated[float, Range(min=0.0)] = 1.1

    # Per-frame multiplicative fade of target confidence after loss, 0..1.
    target_lost_fade_per_frame: Annotated[float, Range(0.0, 1.0)] = 0.99
    target_estimator_clear_history_after_target_lost_frames: Annotated[int, Range(min=0)] = 3

    estimation_3d:                      bool = True
    estimation_3d_method:               Annotated[str, Choices(*[m.value for m in VelocityMethod])] = 'numpy'
    estimation_3d_use_initial_velocity: bool = False

    estimation_lookahead_frames:                Annotated[int, Range(min=0)] = 1
    estimation_lookahead_dynamic:               bool = True
    estimation_lookahead_dynamic_frames_max:    Annotated[int, Range(min=0)] = 5
    estimation_lookahead_dynamic_sqrt:          bool = False
    estimation_lookahead_dynamic_factor:        Annotated[float, Range(min=0.0)] = 0.1
    estimation_lookahead_dynamic_frames_near:   Annotated[int, Range(min=0)] = 0
    estimation_lookahead_dynamic_frames_medium: Annotated[int, Range(min=0)] = 0
    estimation_lookahead_dynamic_frames_far:    Annotated[int, Range(min=0)] = 0

    optical_methods_to_refine_target_size_and_center: bool = True
    adjust_aim_point_at_edge_of_frame:                bool = True
    adjust_aim_point_at_edge_of_frame_threshold:      Annotated[float, Range(0.0, 1.0)] = 0.01
    # w*h, so a normalized area in 0..1.
    adjust_aim_point_at_edge_of_frame_max_size:       Annotated[float, Range(0.0, 1.0)] = 0.25

    # Per-axis P gain (x, y); non-negative.
    pd_coeff_p:          Annotated[XY, Range(min=0.0)] = field(default_factory=_xy_factory(8, 2))
    pd_coeff_d:          float = 0.0
    pd_coeff_p_safe_min: Annotated[XY, Range(min=0.0)] = field(default_factory=_xy_factory(0.1, 0.1))
    pd_coeff_p_min:      Annotated[XY, Range(min=0.0)] = field(default_factory=_xy_factory(0.5, 0.5))
    pd_coeff_p_max:      Annotated[XY, Range(min=0.0)] = field(default_factory=_xy_factory(4, 2))

    pd_coeff_p_dynamic:               bool = False
    pd_coeff_p_dynamic_use_piecewise: bool = False
    # Normalized target size w*h (both in 0..1), so the product is in 0..1.
    pd_coeff_p_dynamic_min_target_size: Annotated[float, Range(0.0, 1.0)] = 0.0005
    pd_coeff_p_dynamic_min:             Annotated[float, Range(min=0.0)] = 0.6
    pd_coeff_p_dynamic_max_target_size: Annotated[float, Range(0.0, 1.0)] = 0.0120
    pd_coeff_p_dynamic_max:             Annotated[float, Range(min=0.0)] = 6.0

    # Normalized size thresholds s in 0..1.
    pd_coeff_p_dynamic_stage_1_threshold: Annotated[float, Range(0.0, 1.0)] = 0.01
    pd_coeff_p_dynamic_stage_2_threshold: Annotated[float, Range(0.0, 1.0)] = 0.05
    # Relative P ratios inside [min, max], 0..1.
    pd_coeff_p_dynamic_stage_1_ratio: Annotated[float, Range(0.0, 1.0)] = 1.0
    pd_coeff_p_dynamic_stage_2_ratio: Annotated[float, Range(0.0, 1.0)] = 1.0
    pd_coeff_p_dynamic_stage_3_ratio: Annotated[float, Range(0.0, 1.0)] = 1.0

    # Physical target size in meters; positive.
    target_size_m: Annotated[XY, Range(min=0.0, min_inclusive=False)] = field(default_factory=_xy_factory(2, 2))
    # Legacy / single-camera FOV fallback in degrees, (0, 360].
    frame_angular_size_deg: Annotated[XY, Range(min=0.0, min_inclusive=False, max=360.0)] = \
        field(default_factory=_xy_factory(120, 90))

    safe_takeoff_period_ns:                Annotated[int, Range(min=0)] = 1_000_000_000
    delay_takeof_until_n_detection_frames: Annotated[int, Range(min=0)] = 10

    # Normalized image coordinates, 0..1.
    aim_point:            Annotated[XY, Range(0.0, 1.0)] = field(default_factory=_xy_factory(0.5, 0.5))
    aim_point_max_offset: Annotated[XY, Range(0.0, 1.0)] = field(default_factory=_xy_factory(0.5, 0.6))

    follow_target_position_ned: bool = False

    # EMA smoothing factor for camera-switch target size, 0..1.
    camera_switch_size_ema_alpha: Annotated[float, Range(0.0, 1.0)] = 0.3

    # Platform-only initial position (normalized).
    platform_initial_pos: XY = field(default_factory=_xy_factory(0, 0))

    camera:    CameraSection    = field(default_factory=CameraSection)
    drone:     DroneSection     = field(default_factory=DroneSection)
    bytetrack: ByteTrackSection = field(default_factory=ByteTrackSection)

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


def _unwrap_annotated(hint):
    """Return (base_type, [constraints]) splitting off typing.Annotated extras."""
    if get_origin(hint) is Annotated:
        args = get_args(hint)
        return args[0], [m for m in args[1:] if isinstance(m, _Constraint)]
    return hint, []


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


def _coerce(value, hint, path, errors):
    base, constraints = _unwrap_annotated(hint)

    # Optional[...] / X | None
    if _is_optional(base):
        if value is None:
            return None
        base, inner_constraints = _unwrap_annotated(_non_none_arg(base))
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


def _parse_dataclass(cls, data, path, errors):
    prefix = f"{path}." if path else ""
    if not isinstance(data, dict):
        errors.append(f"{path or '<root>'}: expected a mapping, got {type(data).__name__}")
        return cls()

    hints = get_type_hints(cls, include_extras=True)
    valid_names = set()
    kwargs = {}
    for f in fields(cls):
        valid_names.add(f.name)
        if f.metadata.get('runtime'):
            # Never read runtime fields from the file; leave the default.
            continue
        if f.name in data:
            r = _coerce(data[f.name], hints[f.name], f"{prefix}{f.name}", errors)
            if r is not _INVALID:
                kwargs[f.name] = r

    for key in data:
        if key not in valid_names or (key in valid_names and _runtime_field(cls, key)):
            errors.append(f"{prefix}{key}: unknown configuration key")

    try:
        return cls(**kwargs)
    except Exception as exc:  # pragma: no cover - kwargs are pre-validated
        errors.append(f"{path or '<root>'}: could not build {cls.__name__}: {exc}")
        return cls()


def _runtime_field(cls, name) -> bool:
    for f in fields(cls):
        if f.name == name:
            return bool(f.metadata.get('runtime'))
    return False


def parse_config(data: dict | None) -> Config:
    """Validate a parsed-YAML mapping and return a Config, or raise ConfigError."""
    errors: list[str] = []
    cfg = _parse_dataclass(Config, data or {}, "", errors)
    if errors:
        raise ConfigError(errors)
    return cfg


def load_config(path: str | Path) -> Config:
    """Read a YAML file and return a validated Config (raises ConfigError)."""
    import yaml
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError([f"<root>: top level must be a mapping, got {type(data).__name__}"])
    return parse_config(data)
