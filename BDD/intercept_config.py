"""Single source of truth for the interception controller config.

Defaults live in intercept_config.yaml (grouped + commented). Per-run overrides
arrive as a flat dict (from --control-config JSON). from_overrides() raises on an
unknown key, so a typo or a stale key fails loudly instead of silently defaulting.

XY-typed values (frame_angular_size_deg, target_size_m, aim_point, etc.) are
stored as [x, y] lists here; app.py converts them to XY objects after loading.
"""
import os
from dataclasses import dataclass, fields, asdict
from typing import Optional

import yaml

_YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intercept_config.yaml")


def _load_yaml_defaults():
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class InterceptConfig:
    # --- detection confidence ---
    confidence_min: float = 0.4
    confidence_move: float = 0.3

    # --- thrust ---
    thrust_takeoff: float = 0.5
    thrust_min: float = 0.7
    thrust_max: float = 1.0
    thrust_dynamic: bool = False
    thrust_proportional_to_target_size: bool = False

    # --- target tracking ---
    target_lost_fade_per_frame: float = 0.99
    target_estimator_clear_history_after_target_lost_frames: int = 6

    # --- estimation ---
    estimation_3d: bool = True
    estimation_3d_method: str = "cluster"
    estimation_3d_use_initial_velocity: bool = True
    estimation_3d_max_distance_m: int = 25

    # --- estimation lookahead ---
    estimation_lookahead_frames: int = 2
    estimation_lookahead_dynamic: bool = True
    estimation_lookahead_dynamic_sqrt: bool = True
    estimation_lookahead_dynamic_factor: int = 0
    estimation_lookahead_dynamic_frames_near: int = 1
    estimation_lookahead_dynamic_frames_medium: int = 1
    estimation_lookahead_dynamic_frames_far: int = 1

    # --- PD regulator ---
    pd_coeff_p: int = 3
    pd_coeff_d: int = 30
    pd_coeff_p_safe_min: float = 0.6
    pd_coeff_p_min: float = 0.5
    pd_coeff_p_max: int = 10

    # --- PD dynamic P ---
    pd_coeff_p_dynamic: bool = False
    pd_coeff_p_dynamic_use_piecewise: bool = False
    pd_coeff_p_dynamic_min_target_size: float = 0.0005
    pd_coeff_p_dynamic_min: float = 0.6
    pd_coeff_p_dynamic_max_target_size: float = 0.012
    pd_coeff_p_dynamic_max: int = 6
    pd_coeff_p_dynamic_stage_1_threshold: float = 0.01
    pd_coeff_p_dynamic_stage_2_threshold: float = 0.05
    pd_coeff_p_dynamic_stage_1_ratio: int = 1
    pd_coeff_p_dynamic_stage_2_ratio: int = 1
    pd_coeff_p_dynamic_stage_3_ratio: int = 1

    # --- optics / geometry (XY stored as [x, y] list) ---
    frame_angular_size_deg: list = None
    target_size_m: list = None
    distance_scale: float = 1.0
    size_measure_contour: bool = True

    # --- inertia correction ---
    inertia_correction_gain: int = 0
    inertia_correction_limits: list = None
    inertia_correction_min_speed_ms: int = 5

    # --- takeoff ---
    safe_takeoff_period_ns: int = 300000000
    delay_takeof_until_n_detection_frames: int = 12

    # --- aim point (XY stored as [x, y] list) ---
    aim_point: list = None
    aim_point_max_offset: list = None

    # --- guidance: mode flags ---
    follow_target_position_ned: bool = False

    # --- guidance: pronav ---
    guidance_pronav: bool = False
    pronav_closing_speed: float = 15.0
    pronav_n: float = 1.0
    pronav_v_max: float = 25.0
    pronav_vz_max: float = 10.0
    pronav_use_kalman: bool = False
    pronav_kalman_q: float = 1.0
    pronav_kalman_r: float = 2.0

    # --- guidance: visual ---
    guidance_visual: bool = False
    visual_v_far: float = 12.0
    visual_v_close: float = 14.0
    visual_n_gain: float = 8.0
    visual_term_gain: float = 16.0
    visual_mid_thresh: float = 0.06
    visual_near_thresh: float = 0.20
    visual_v_max: float = 30.0
    visual_climb_min: float = 3.0
    visual_term_perp_cap: float = 1.0

    # --- guidance: lead ---
    guidance_lead: bool = False
    lead_speed: float = 12.0
    lead_t_max: float = 4.0
    lead_alt_offset: float = 0.0
    lead_max_lat: float = 60.0
    lead_max_alt_m: float = 70.0
    lead_visual_terminal: bool = False
    lead_visual_dist: float = 12.0
    lead_far_visual: bool = False
    lead_far_dist: float = 30.0

    # --- drone controller params ---
    drone_use_set_attitude: bool = False
    drone_min_lift_fraction: float = 0.1
    drone_lift_velocity_headroom_ms: float = 3.0
    drone_lift_accel_headroom_mss: float = 5.0
    drone_max_attitude_rate_deg_s: int = 120

    # --- misc ---
    DEBUG: bool = False

    # --- bytetrack ---
    bytetrack_track_thresh: float = 0.3
    bytetrack_det_thresh: float = 0.35
    bytetrack_match_thresh: float = 0.3
    bytetrack_track_buffer: int = 30
    bytetrack_frame_rate: int = 30
    bytetrack_match_max_dist: float = 0.2
    bytetrack_recovery_max_dist: Optional[float] = None
    bytetrack_nms_thresh: float = 0.3
    bytetrack_nms_dist_thresh: float = 0.06
    bytetrack_target_lock: bool = True

    # guidance: phased
    guidance_phased: bool = False
    phased_mid_dist: float = 35.0
    phased_far_vmax: float = 25.0
    phased_far_speed: float = 25.0
    phased_far_vz_max: float = 10.0
    phased_adaptive: bool = False

    # estimation: parallax range
    parallax_range: bool = False
    parallax_buffer: int = 15
    parallax_min_baseline_m: float = 1.0
    parallax_min_sin2: float = 0.02
    parallax_max_miss_m: float = 8.0
    parallax_max_w: float = 0.7

    def __post_init__(self):
        # Replace None sentinel defaults with actual list defaults from YAML
        if self.frame_angular_size_deg is None:
            self.frame_angular_size_deg = [107, 85]
        if self.target_size_m is None:
            self.target_size_m = [3.0, 3.0]
        if self.inertia_correction_limits is None:
            self.inertia_correction_limits = [1, 1]
        if self.aim_point is None:
            self.aim_point = [0.5, 0.5]
        if self.aim_point_max_offset is None:
            self.aim_point_max_offset = [0.5, 0.6]

    @classmethod
    def load_defaults(cls):
        return cls.from_overrides({})

    @classmethod
    def from_overrides(cls, overrides):
        data = dict(_load_yaml_defaults())
        valid = {f.name for f in fields(cls)}
        for k, v in overrides.items():
            if k not in valid:
                raise KeyError(f"unknown intercept config key: {k!r}")
            data[k] = v
        unknown_yaml = set(data) - valid
        if unknown_yaml:
            raise KeyError(f"intercept_config.yaml has keys not in InterceptConfig: {sorted(unknown_yaml)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
