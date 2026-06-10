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

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Annotated, Any, Optional

from helpers import XY

# VelocityMethod is used directly as the choice set for `estimation_3d_method`.
from TargetEstimator import VelocityMethod


# ---------------------------------------------------------------------------
# Constraint metadata, used as the second argument of typing.Annotated:
#     confidence_min:    Annotated[float, Range(0.0, 1.0)]
#     video_format:      Annotated[str, Choices('RGB', 'BGR', ...)]
#     estimation_method: Annotated[str, Choices(VelocityMethod)]   # enum -> values
#     cameras:           Annotated[list[CameraEntry], MinItems(1)]
# The field keeps a real, introspectable type (get_type_hints sees the base
# type) and the validator rides along as Annotated metadata. Each `validate`
# appends human-readable problems to `errors` and never raises.
# ---------------------------------------------------------------------------
class _Constraint:
    """Base for the Annotated metadata that validates a parsed value."""
    def validate(self, value: Any, path: str, errors: list[str]) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class Range(_Constraint):
    """Inclusive-by-default numeric bound; applied per-field for dataclasses (XY).

    Usage: ``Annotated[float, Range(0.0, 1.0)]``, ``Annotated[int, Range(min=1)]``,
    ``Annotated[XY, Range(min=0.0, max=360.0)]``.
    """
    def __init__(self, min=None, max=None, *,
                 min_inclusive: bool = True, max_inclusive: bool = True):
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
    or pass the allowed values directly (``Choices('RGB', 'BGR')``).
    """
    def __init__(self, *allowed):
        if len(allowed) == 1 and isinstance(allowed[0], type) and issubclass(allowed[0], Enum):
            self.allowed = tuple(m.value for m in allowed[0])
        else:
            self.allowed = tuple(allowed)

    def validate(self, value: Any, path: str, errors: list[str]) -> None:
        if value not in self.allowed:
            errors.append(f"{path}: {value!r} is not one of {list(self.allowed)}")


class MinItems(_Constraint):
    """A list must contain at least `count` items.

    Usage: ``Annotated[list[CameraEntry], MinItems(1)]``.
    """
    def __init__(self, count: int):
        self.count = count

    def validate(self, value: Any, path: str, errors: list[str]) -> None:
        if isinstance(value, list) and len(value) < self.count:
            errors.append(f"{path}: needs at least {self.count} item(s), got {len(value)}")


# ---------------------------------------------------------------------------
# Nested sections
# ---------------------------------------------------------------------------
@dataclass(slots=True, kw_only=True, frozen=True)
class ByteTrackSection:
    """Parameters for the ByteTrack multi-object tracker.

    Everything except `target_lock` is forwarded verbatim to BYTETracker; use
    `tracker_kwargs()` to get exactly that subset.
    """
    track_thresh:      Annotated[float, Range(0.0, 1.0)] = 0.5
    det_thresh:        Annotated[float, Range(0.0, 1.0)] = 0.6
    match_thresh:      Annotated[float, Range(0.0, 1.0)] = 0.8
    track_buffer:      Annotated[int, Range(min=1)]      = 30
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


@dataclass(slots=True, kw_only=True, frozen=True)
class CameraEntry:
    """One physical camera. Maps onto helpers.CameraConfig.

    Resolution/fps/video_format are intentionally NOT here: all cameras share a
    single appsrc and must produce identical caps (those live in Camera).
    """
    camera_id:              Annotated[int, Range(min=0)] = 0
    name:                   str = ""
    sensor_index:           Annotated[int, Range(min=0)] = 0
    # Per-camera horizontal/vertical FOV in degrees, (0, 360].
    frame_angular_size_deg: Annotated[XY, Range(min=0.0, min_inclusive=False, max=360.0)]


@dataclass(slots=True, kw_only=True, frozen=True)
class Camera:
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

    # EMA smoothing factor for camera-switch target size, 0..1.
    switch_size_ema_alpha: Annotated[float, Range(0.0, 1.0)] = 0.3

    # At least one camera must be configured explicitly (no default).
    cameras: Annotated[list[CameraEntry], MinItems(1)]

@dataclass(slots=True, kw_only=True, frozen=True)
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


@dataclass(slots=True, kw_only=True, frozen=True)
class Drone:
    connection_string: str = 'usb'
    # The control block must be present explicitly (no default).
    config: DroneControlConfig

# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass(slots=True, kw_only=True, frozen=True)
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
    estimation_3d_method:               Annotated[str, Choices(VelocityMethod)] = 'numpy'
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

    # Per-axis P gain (x, y); non-negative. These must be set explicitly (no
    # default) — they directly shape the control response and silently
    # defaulting them is dangerous.
    pd_coeff_p:          Annotated[XY, Range(min=0.0)]
    pd_coeff_d:          float = 0.0
    pd_coeff_p_safe_min: Annotated[XY, Range(min=0.0)]
    pd_coeff_p_min:      Annotated[XY, Range(min=0.0)]
    pd_coeff_p_max:      Annotated[XY, Range(min=0.0)]
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
    target_size_m: Annotated[XY, Range(min=0.0, min_inclusive=False)]

    safe_takeoff_period_ns:                Annotated[int, Range(min=0)] = 1_000_000_000
    delay_takeof_until_n_detection_frames: Annotated[int, Range(min=0)] = 10

    # Normalized image coordinates, 0..1.
    aim_point:            Annotated[XY, Range(0.0, 1.0)] = field(default_factory=lambda: XY(0.5, 0.5))
    aim_point_max_offset: Annotated[XY, Range(0.0, 1.0)] = field(default_factory=lambda: XY(0.5, 0.6))

    follow_target_position_ned: bool = False

    # Complex sub-sections must be present explicitly (no default) so an
    # incomplete config fails loudly instead of silently using a stand-in.
    camera:    Camera
    drone:     Drone
    bytetrack: ByteTrackSection
    # Runtime-only fields: set programmatically, NEVER read from the config
    # file (providing them in the file is reported as an unknown key).
    DEBUG:                bool = field(default=False, metadata={'runtime': True})
    debug_telemetry_dict: Optional[dict] = field(default=None, metadata={'runtime': True})
