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

from dataclasses import dataclass, field, fields, is_dataclass, asdict
from enum import Enum
from typing import Annotated, Any, Optional
from pathlib import Path

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
        # An enum-typed field is coerced to a member upstream; compare by value.
        v = value.value if isinstance(value, Enum) else value
        if v not in self.allowed:
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

# Used as a field type (``hef_model_path: ExistingFile``): coerces a config
# string to a pathlib.Path and requires the file to exist. Implemented as a
# factory that RETURNS a plain Path rather than subclassing Path — subclassing
# Path is fragile across versions (the Pi runs 3.11, the dev host 3.12: 3.11
# parses in __new__, 3.12 in __init__), so a factory is the portable choice.
class ExistingFile:
    def __new__(cls, value):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"{path} does not point to an existing file")
        return path



# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------
@dataclass(slots=True, kw_only=True, frozen=True)
class Config:
    confidence_min:  Annotated[float, Range(0.0, 1.0)] = 0.4
    confidence_move: Annotated[float, Range(0.0, 1.0)] = 0.3

    # Normalized image coordinates, 0..1.
    aim_point:            Annotated[XY, Range(0.0, 1.0)] = field(default_factory=lambda: XY(0.5, 0.5))
    aim_point_max_offset: Annotated[XY, Range(0.0, 1.0)] = field(default_factory=lambda: XY(0.5, 0.6))

    # Physical target size in meters; positive.
    target_size_m: Annotated[XY, Range(min=0.0, min_inclusive=False)]

    follow_target_position_ned: bool = False

    # Record the camera feed to ./_DEBUG/RAW_*.mkv (x264enc, software). Costs ~1
    # CPU core; disable to free CPU for inference/tiling (the recorded video is a
    # debug/analysis aid, not required for control). The --no-record CLI flag
    # forces this off regardless of the config value.
    record_videos: bool = True

    @dataclass(slots=True, kw_only=True, frozen=True)
    class Inference:
        hef_model_path:      ExistingFile
        nms_score_threshold: Annotated[float, Range(0.0, 1.0)] = 0.3
        nms_iou_threshold:   Annotated[float, Range(0.0, 1.0)] = 0.45
        labels_json:         Optional[ExistingFile] = None
    inference: Inference

    @dataclass(slots=True, kw_only=True, frozen=True)
    class Tiling:
        # Inference tile grid. 1×1 = whole-frame (default, lowest latency). >1
        # enables hailotilecropper: tiles_x*tiles_y inferences/frame, raising
        # small-object recall at the cost of latency (~N×15 ms on Hailo-8).
        tiles_x: Annotated[int, Range(min=1)] = 1
        tiles_y: Annotated[int, Range(min=1)] = 1
        # Fractional tile overlap on both axes; 0 = abutting tiles. Strictly < 1:
        # at 1.0 every tile would cover the whole frame (degenerate grid).
        overlap: Annotated[float, Range(0.0, 1.0, max_inclusive=False)] = 0.0
        # IoU above which the hailotileaggregator merges two detections coming from
        # different tiles — i.e. how aggressively objects straddling a tile seam get
        # deduped. Only used on the tiled path. Lower = merge more eagerly.
        tile_iou_threshold: Annotated[float, Range(0.0, 1.0)] = 0.4
        # Runtime-switchable tiling: build BOTH a whole-frame branch and a
        # tiles_x×tiles_y branch behind valves + an input-selector, and hot-switch
        # between them at runtime (whole-frame active at startup). Lets a policy
        # (e.g. tile-to-reacquire when the target is lost) trade latency for recall
        # on demand. When false, tiling is static (whole-frame if 1×1, else fixed).
        switchable: bool = False
        # Automatic detection-state policy (implies switchable): switch to tiling
        # after `lost_frames_to_tile` consecutive frames with no confident target
        # (reacquire small/distant objects), and back to whole-frame after
        # `locked_frames_to_whole` consecutive confident frames (restore low
        # latency for control). A target counts as "confident" at >= switch_conf.
        auto_switch: bool = False
        lost_frames_to_tile:    Annotated[int, Range(min=1)] = 10
        locked_frames_to_whole: Annotated[int, Range(min=1)] = 5
        switch_conf: Annotated[float, Range(0.0, 1.0)] = 0.4
    tiling: Tiling = field(default_factory=Tiling)

    @dataclass(slots=True, kw_only=True, frozen=True)
    class Thrust:
        # PX4 normalized thrust, 0..1.
        cruise:  Annotated[float, Range(0.0, 1.0)] = 0.53
        hover:   Annotated[float, Range(0.0, 1.0)] = 0.4
        min:     Annotated[float, Range(0.0, 1.0)] = 0.4
        max:     Annotated[float, Range(0.0, 1.0)] = 0.9

        dynamic:                  bool = False
        @dataclass(slots=True, kw_only=True, frozen=True)
        class ProportionalToDistance:
            far_coeff:        Annotated[float, Range(min=0.0)] = 1.0
            medium_distance_m: Annotated[float, Range(min=0.0)] = 20.0
            medium_coeff:     Annotated[float, Range(min=0.0)] = 0.9
            near_distance_m:   Annotated[float, Range(min=0.0)] = 10.0
            near_coeff:       Annotated[float, Range(min=0.0)] = 1.1
        proportional_to_distance : Optional[ProportionalToDistance] = None
    thrust: Thrust = field(default_factory=Thrust)

    @dataclass(slots=True, kw_only=True, frozen=True)
    class TargetLost:
        # Keep emitting same command for given number of frames before fading.
        repeat_same_commands_frames: Annotated[int, Range(min=0)] = 1
        # Per-frame multiplicative fade of target confidence after loss, 0..1.
        fade_per_frame: Annotated[float, Range(0.0, 1.0)] = 0.99
        clear_estimator_history_after_frames: Annotated[int, Range(min=0)] = 3
    target_lost : TargetLost

    @dataclass(slots=True, kw_only=True, frozen=True)
    class Estimation:
        @dataclass(slots=True, kw_only=True, frozen=True)
        class Estimation3D:
            method:               VelocityMethod = VelocityMethod.NUMPY_REGRESSION
            use_initial_velocity: bool = False
        estimation_3d : Optional[Estimation3D]

        @dataclass(slots=True, kw_only=True, frozen=True)
        class Lookahead:
            frames:                Annotated[int, Range(min=0)] = 1
            dynamic:               bool = True
            dynamic_frames_max:    Annotated[int, Range(min=0)] = 5
            dynamic_sqrt:          bool = False
            dynamic_factor:        Annotated[float, Range(min=0.0)] = 0.1
            dynamic_frames_near:   Annotated[int, Range(min=0)] = 0
            dynamic_frames_medium: Annotated[int, Range(min=0)] = 0
            dynamic_frames_far:    Annotated[int, Range(min=0)] = 0
        lookahead : Lookahead
    estimation : Estimation

    @dataclass(slots=True, kw_only=True, frozen=True)
    class OpticalRefinement:
        # enabled=False disables the whole feature (parser returns None for the
        # section); check `config.optical_refinement is not None`, not .enabled.
        # enabled: bool = True
        adjust_aim_point_at_edge_of_frame:                bool = True
        adjust_aim_point_at_edge_of_frame_threshold:      Annotated[float, Range(0.0, 1.0)] = 0.01
        # w*h, so a normalized area in 0..1.
        adjust_aim_point_at_edge_of_frame_max_size:       Annotated[float, Range(0.0, 1.0)] = 0.25
    optical_refinement: Optional[OpticalRefinement]

    @dataclass(slots=True, kw_only=True, frozen=True)
    class PDCoeff:
        # Per-axis P gain (x, y); non-negative. These must be set explicitly (no
        # default) — they directly shape the control response and silently
        # defaulting them is dangerous.
        p:          Annotated[XY, Range(min=0.0)]
        d:          float = 0.0
        p_min:      Annotated[XY, Range(min=0.0)]
        p_max:      Annotated[XY, Range(min=0.0)]
        p_dynamic:               bool = False
        p_dynamic_use_piecewise: bool = False
        # Normalized target size w*h (both in 0..1), so the product is in 0..1.
        p_dynamic_min_target_size: Annotated[float, Range(0.0, 1.0)] = 0.0005
        p_dynamic_min:             Annotated[float, Range(min=0.0)] = 0.6
        p_dynamic_max_target_size: Annotated[float, Range(0.0, 1.0)] = 0.0120
        p_dynamic_max:             Annotated[float, Range(min=0.0)] = 6.0

        # Normalized size thresholds s in 0..1.
        p_dynamic_stage_1_threshold: Annotated[float, Range(0.0, 1.0)] = 0.01
        p_dynamic_stage_2_threshold: Annotated[float, Range(0.0, 1.0)] = 0.05
        # Relative P ratios inside [min, max], 0..1.
        p_dynamic_stage_1_ratio: Annotated[float, Range(0.0, 1.0)] = 1.0
        p_dynamic_stage_2_ratio: Annotated[float, Range(0.0, 1.0)] = 1.0
        p_dynamic_stage_3_ratio: Annotated[float, Range(0.0, 1.0)] = 1.0
    pd_coeff: PDCoeff

    @dataclass(slots=True, kw_only=True, frozen=True)
    class TakeOff:
        delay_until_n_detection_frames: Annotated[int, Range(min=0)] = 10
        duration_ns:                    Annotated[int, Range(min=0)] = 1_000_000_000
        pd_coeff_p:                     Annotated[XY, Range(min=0.0)]
        thrust:                         Annotated[float, Range(0.0, 1.0)] = 1.0
    takeoff: TakeOff

    @dataclass(slots=True, kw_only=True, frozen=True)
    class Camera:
        """Shared camera caps + the list of cameras.

        The shared caps apply to every camera (single appsrc, identical caps); the
        per-camera differences live in `cameras`.
        """
        width:                Annotated[int, Range(min=1)] = 1280
        height:               Annotated[int, Range(min=1)] = 720
        fps:                  Annotated[int, Range(min=1)] = 30
        # TODO: infer video format based on hef file input
        video_format:         Annotated[str, Choices('RGB', 'BGR', 'RGBA', 'BGRA',
                                    'XRGB', 'XBGR', 'YUV420', 'NV12')] = 'RGB'
        active_id:            Annotated[int, Range(min=0)] = 0
        # Relative target size (max(w,h)) that triggers switching wide/zoom. Both in (0,1].
        switch_to_wide_size:  Annotated[float, Range(0.0, 1.0, min_inclusive=False)] = 0.25
        switch_to_zoom_size:  Annotated[float, Range(0.0, 1.0, min_inclusive=False)] = 0.015

        # EMA smoothing factor for camera-switch target size, 0..1.
        switch_size_ema_alpha: Annotated[float, Range(0.0, 1.0)] = 0.3

        @dataclass(slots=True, kw_only=True, frozen=True)
        class AutoExposure:
            # Manual exposure pin (Stage-A latency). 0 = leave auto-exposure on (AE
            # picks the integration time). >0 = disable AE and pin the shutter to this
            # many milliseconds. A short shutter (~8 ms) both enables a steady 30 fps
            # in lower light AND cuts/​de-jitters Stage-A latency (centres the captured
            # "moment" instead of smearing it across a long integration). Needs enough
            # light; see experiments/camera-stage-a-latency.md.
            exposure_time_ms:     Annotated[int, Range(min=0)] = 0
            # Auto-estimate-then-pin exposure. >0 = run auto-exposure for this many
            # milliseconds at startup (and again whenever a camera becomes active),
            # then READ BACK the AE-converged ExposureTime/AnalogueGain and pin them —
            # so the shutter is scene-adapted AND then deterministic/short (no ongoing
            # AE jitter). The AE estimate is a GUIDE, clamped by the limits below. This
            # supersedes the fixed exposure_time_ms pin when set. 0 = disabled.
            exposure_auto_pin_ms: Annotated[int, Range(min=0)] = 0
            # Limits applied to the auto-pinned (or fixed) exposure, in ms. 0 = no
            # limit. exposure_max_ms is the important one: it caps the shutter to
            # protect the frame rate and Stage-A latency. When AE wants a LONGER
            # exposure than the cap, the excess light is shifted into AnalogueGain (up
            # to gain_max) so brightness is preserved instead of the frame going dark.
            exposure_min_ms:      Annotated[int, Range(min=0)] = 0
            exposure_max_ms:      Annotated[int, Range(min=0)] = 0

            # Upper limit on the (auto-pinned/compensated) gain; 0 = no cap. Bounds the
            # noise the brightness-compensation above is allowed to add.
            gain_max:             Annotated[float, Range(min=0.0)] = 0.0
            # Manual analogue (sensor) gain, paired with the exposure pin above.
            # 0 = let the AGC choose the gain; >0 = pin AnalogueGain to this value
            # (sensor minimum is 1.0). This is what rescues a SHORT pinned exposure in
            # dimmer light: with AE off the gain would otherwise stay at 1.0 and the
            # frame goes dark, so raise gain instead of lengthening the shutter (which
            # would bring back the Stage-A latency/jitter). Only takes effect with AE
            # off, i.e. when exposure_time_ms > 0. Higher gain = more sensor noise.
            analogue_gain:        Annotated[float, Range(min=0.0)] = 0.0
        autoexposure:         AutoExposure

        # Size of the picamera2/PiSP DMA buffer pool (frames in flight between the
        # sensor/ISP and the app). It caps worst-case Stage-A staleness: fewer
        # buffers => lower worst-case capture latency. 2 is the FLOOR — it keeps
        # double-buffering (the sensor fills #2 while the app holds #1); 1 breaks
        # that and stalls/drops, so the minimum is enforced.
        buffer_count:         Annotated[int, Range(min=2)] = 2

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
        cameras: Annotated[list[CameraEntry], MinItems(1)]
    camera:    Camera

    @dataclass(slots=True, kw_only=True, frozen=True)
    class Drone:
        connection_string: str = 'usb'

        @dataclass(slots=True, kw_only=True, frozen=True)
        class DroneControlConfig:
            """consumed by DroneMover."""
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
        config: DroneControlConfig
    drone:     Drone

    @dataclass(slots=True, kw_only=True, frozen=True)
    class ByteTrack:
        """Parameters for the ByteTrack multi-object tracker.

        The controller-only flags (`enabled`, `target_lock`) are excluded from
        `tracker_kwargs()`; everything else is forwarded verbatim to BYTETracker.

        enabled=False disables tracking (parser returns None for the section);
        check `config.bytetrack is not None`, not .enabled.
        """
        # enabled:           bool = False
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
            # target_lock is consumed by the controller, NOT a valid BYTETracker
            # constructor kwarg. (`enabled` is a file-only toggle, never a field.)
            controller_only = ('target_lock',)
            return {k: v for k, v in asdict(self).items() if k not in controller_only}
    bytetrack: Optional[ByteTrack]

    # Runtime-only fields: set programmatically, NEVER read from the config
    # file (providing them in the file is reported as an unknown key).
    DEBUG:                bool = field(default=False, metadata={'runtime': True})
    debug_telemetry_dict: Optional[dict] = field(default=None, metadata={'runtime': True})

    # Convenience entry points: parse/validate a YAML file or an already-parsed
    # mapping into a Config. The parser is generic over the root type; these
    # forward `Config` so callers don't repeat it. (lazy import avoids a cycle:
    # parse_config imports config for _Constraint.)
    @staticmethod
    def load(path) -> "Config":
        from parse_config import load_config
        return load_config(Config, path)

    @staticmethod
    def parse(data, **kwargs) -> "Config":
        from parse_config import parse_config
        return parse_config(Config, data, **kwargs)
