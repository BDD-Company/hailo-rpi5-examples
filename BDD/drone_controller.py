#! /usr/bin/env python3

import asyncio
import math
import time
from  copy import copy
from queue import Empty, Queue

from helpers import XY

from drone import DroneMover
from CommandRegulator import CommandRegulator
from TargetEstimator import TargetEstimator, TargetEstimator3D, VelocityMethod
# from telemetry_position import PositionNED, VelocityNED
from estimate_distance import estimate_distance_class, DistanceClass, OpticalObjectInfo
from telemetry_position import (
    # get_position_ned,
    # get_orientation_quaternion,
    project_camera_to_ned,
    get_pose,
    project_ned_to_camera
)
# from drone_killswitch import kill_on_rc_switch_on_channel
from helpers import Detection, Detections, MoveCommand, STOP, CameraSwitcher, DEFAULT_CAMERA_ID, CameraConfig

try:
    from gpiozero import CPUTemperature
except:
    class _MockCPUTemperature:
        def __init__(self):
            self.temperature = '--mocked--'

    CPUTemperature = _MockCPUTemperature



from helpers import (
    debug_collect_call_info,
    LoggerWithPrefix
)

import logging
logger = logging.getLogger(__name__)
global_logger = logger

DEBUG = False


def drone_controlling_thread(*args, **kwargs):
    # Exceptions propagate out of run() so threading.excepthook in app.py
    # can tear the GStreamer pipeline down. Used to be caught and logged
    # here, which left the pipeline running behind a dead drone.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(drone_controlling_thread_async(*args, **kwargs))
    finally:
        loop.close()

def get_position_from_telemetry(telemetry_dict) -> XY|None:
    pos = telemetry_dict.get('odometry', {}).get('position_body', None)
    if pos:
        return XY(pos['x_m'], pos['y_m'])
    else:
        return None

def is_drone_moving(telemetry_dict):
    velocity = telemetry_dict.get('odometry', {}).get('velocity_body', None)
    if velocity:
        return abs(velocity["x_m_s"]) > 0.01 \
            and abs(velocity["y_m_s"]) > 0.01 \
            and abs(velocity["z_m_s"]) > 0.01


def clamp(min_val, val, max_val):
    typeof_val = type(val)
    if typeof_val == XY:
        return clamp_xy(min_val, val, max_val)

    return typeof_val(max(min_val, min(max_val, val)))


def clamp_xy(min_val, val: XY, max_val) -> XY:
    """Clamp each component of an XY independently.

    Bounds may be scalars (same bound on both axes) or XY (per-axis bounds);
    in either case x is clamped against the x bound and y against the y bound.
    """
    min_x = min_val.x if isinstance(min_val, XY) else min_val
    min_y = min_val.y if isinstance(min_val, XY) else min_val
    max_x = max_val.x if isinstance(max_val, XY) else max_val
    max_y = max_val.y if isinstance(max_val, XY) else max_val

    return XY(clamp(min_x, val.x, max_x), clamp(min_y, val.y, max_y))


def _pick_target_detection(
    detections: list,
    confidence_min: float,
    locked_track_id: int | None,
    use_track_lock: bool,
) -> Detection | None:
    """
    Выбор одной детекции для сопровождения. При use_track_lock и заданном
    locked_track_id возвращает только детекции с этим ByteTrack track_id,
    чтобы не перескакивать на случайные высоко-уверенные ложные срабатывания.
    Если заблокированный трек в кадре отсутствует — None (ведём себя как
    при потере цели, оценщик/затухание сами продолжают движение).
    """
    pool = [d for d in detections if d is not None and d.confidence >= confidence_min]
    if not pool:
        return None
    effective_lock = locked_track_id if use_track_lock else None
    if effective_lock is not None:
        locked = [d for d in pool if d.track_id == effective_lock]
        if locked:
            return max(locked, key=lambda d: d.confidence)
        return None
    return max(pool, key=lambda d: d.confidence)


def compute_inertia_correction(telemetry_dict, target_relative_pos, gain, min_speed_ms=0.3):
    """
    Inertia correction computed entirely in FRD reference frame.

    Compares actual body velocity (FRD, from telemetry) against the desired
    velocity direction (derived from target position in camera frame).
    Returns correction in camera frame to ADD to target_relative_pos.

    Camera frame: x>0 = target LEFT of centre, y>0 = target above centre (fwd).
    FRD frame:    x = forward, y = right, z = down.

    Frame conversions (consistent with move_to_target_zenith_async mapping:
      roll_deg_s = -angle.x, pitch_deg_s = angle.y):
        camera -> FRD:  frd = XY( cam.y, -cam.x)
        FRD -> camera:  cam = XY(-frd.y,  frd.x)
    """
    if gain == 0 or target_relative_pos is None:
        return XY(0.0, 0.0)

    odometry = telemetry_dict.get('odometry') or None
    if odometry is None:
        return XY(0.0, 0.0)

    vel = odometry.get('velocity_body', None)
    if vel is None:
        return XY(0.0, 0.0)

    # velocity_body is in FRD frame
    v_frd_x = vel['x_m_s']  # forward
    v_frd_y = vel['y_m_s']  # right
    v_frd_z = vel['z_m_s']  # down

    # NOTE: not sure if should include z into full speed computation?
    speed = math.sqrt(v_frd_x ** 2 + v_frd_y ** 2 + v_frd_z ** 2)
    horiz_speed = math.sqrt(v_frd_x ** 2 + v_frd_y ** 2)
    if horiz_speed < min_speed_ms: # or speed < min_speed_ms ?
        return XY(0.0, 0.0)

    # Convert target_relative_pos from camera frame to FRD
    target_frd_x = target_relative_pos.y    # forward = camera up
    target_frd_y = -target_relative_pos.x   # right   = negative camera left

    # Build 3D ray direction from 2D angular offsets on the unit sphere.
    # Camera optical axis = -Z in FRD (looking up from belly).
    # target_frd_x, target_frd_y are angular offsets; Z completes the unit sphere.
    horiz_sq = target_frd_x ** 2 + target_frd_y ** 2
    if horiz_sq < 1e-12:
        return XY(0.0, 0.0)
    if horiz_sq >= 1.0:
        # Target at extreme edge — clamp to horizontal ray
        scale = 1.0 / math.sqrt(horiz_sq)
        ray_x = target_frd_x * scale
        ray_y = target_frd_y * scale
        ray_z = 0.0
    else:
        ray_x = target_frd_x
        ray_y = target_frd_y
        ray_z = -math.sqrt(1.0 - horiz_sq)  # upward = -Z in FRD

    # Desired speed in FRD (3D): ray direction * actual 3D speed
    desired_frd_x = ray_x * speed
    desired_frd_y = ray_y * speed
    desired_frd_z = ray_z * speed

    # Correction in FRD (3D) = desired - actual, then take only x,y for roll/pitch
    correction_frd_x = (desired_frd_x - v_frd_x) * gain
    correction_frd_y = (desired_frd_y - v_frd_y) * gain

    # Convert correction from FRD back to camera frame
    return XY(
        -correction_frd_y,  # cam.x = -frd.y
        correction_frd_x,   # cam.y =  frd.x
    )


async def drone_controlling_thread_async(
        drone_connection_string,
        drone_config,
        detections_queue,
        control_config = {},
        output_queue = None,
        signal_event_when_ready = None,
        camera_switcher : CameraSwitcher | None = None
    ):
    # from math import radians

    # will owerwrite logger here many times, make sure that rest of the systems are not affected
    global global_logger
    logger = global_logger

    START_TIME_MS = time.monotonic_ns() / 1000_000

    global DEBUG
    DEBUG                = control_config.pop('DEBUG', False)
    DEBUG_TELEMETRY_DICT = control_config.pop('debug_telemetry_dict', None)
    logger.debug("!!!!! DEBUG state: %s", DEBUG)

    CONFIDENCE_MIN  = control_config.pop('confidence_min', 0.1)
    # MOVE_CONFIDENCE = control_config.get('confidence_move', 0.4)

    THRUST_MAX      = control_config.pop('thrust_max', 0.5)
    THRUST_MIN      = control_config.pop('thrust_min', 0.4)
    THRUST_CRUISE   = control_config.pop('thrust_cruise', 0.4)
    THRUST_TAKEOFF  = control_config.pop('thrust_takeoff', 0.5)
    THRUST_HOVER    = control_config.pop('thrust_hover', 0.5)

    THRUST_DYNAMIC  = control_config.pop('thrust_dynamic', False)

    THRUST_PROPORTIONAL_TO_DISTANCE                   = control_config.pop('thrust_proportional_to_distance', False)
    THRUST_PROPORTIONAL_TO_DISTANCE_NEAR_COEFF        = control_config.pop('thrust_proportional_to_distance_near_coeff', 1.0)
    THRUST_PROPORTIONAL_TO_DISTANCE_MEDIUM_COEFF      = control_config.pop('thrust_proportional_to_distance_medium_coeff', 1.0)
    THRUST_PROPORTIONAL_TO_DISTANCE_FAR_COEFF         = control_config.pop('thrust_proportional_to_distance_far_coeff', 1.0)
    THRUST_PROPORTIONAL_TO_DISTANCE_MEDIUM_DISTANCE_M = control_config.pop('thrust_proportional_to_distance_medium_distance_m', 15)
    THRUST_PROPORTIONAL_TO_DISTANCE_NEAR_DISTANCE_M   = control_config.pop('thrust_proportional_to_distance_near_distance_m', 7)


    FADE_COEFF      = control_config.pop('target_lost_fade_per_frame', 0.9)
    TARGET_ESTIMATOR_CLEAR_HISTORY_AFTER_TARGET_LOST_FRAMES = control_config.pop('target_estimator_clear_history_after_target_lost_frames', 3)

    PD_COEFF_P                      = control_config.pop('pd_coeff_p', XY(1, 1))
    # P may be different along each axis, so it is carried around as an XY.
    # Accept a plain scalar from config too and apply it to both axes.
    if not isinstance(PD_COEFF_P, XY):
        PD_COEFF_P = XY(PD_COEFF_P, PD_COEFF_P)
    PD_COEFF_D                      = control_config.pop('pd_coeff_d', 0)

    PD_COEFF_P_DYNAMIC               = control_config.pop('pd_coeff_p_dynamic', False)
    PD_COEFF_P_DYNAMIC_USE_PIECEWISE = control_config.pop('pd_coeff_p_dynamic_use_piecewise', False)
    PD_COEFF_P_MIN_TARGET_SIZE       = control_config.pop('pd_coeff_p_dynamic_min_target_size', 0.003)
    PD_COEFF_P_MAX_TARGET_SIZE       = control_config.pop('pd_coeff_p_dynamic_max_target_size', 0.005)
    PD_COEFF_P_DYNAMIC_MIN           = control_config.pop('pd_coeff_p_dynamic_min', 0.5)
    PD_COEFF_P_DYNAMIC_MAX           = control_config.pop('pd_coeff_p_dynamic_max', 2)

    PD_COEFF_P_SAFE_MIN              = control_config.pop('pd_coeff_p_safe_min', XY(0.5, 0.5))
    if not isinstance(PD_COEFF_P_SAFE_MIN, XY):
        PD_COEFF_P_SAFE_MIN = XY(PD_COEFF_P_SAFE_MIN, PD_COEFF_P_SAFE_MIN)
    PD_COEFF_P_MIN                   = control_config.pop('pd_coeff_p_min', XY(0.5, 0.5))
    PD_COEFF_P_MAX                   = control_config.pop('pd_coeff_p_max', XY(5, 5))
    # P bounds may be different along each axis, carried around as XY.
    # Accept a plain scalar from config too and apply it to both axes.
    if not isinstance(PD_COEFF_P_MIN, XY):
        PD_COEFF_P_MIN = XY(PD_COEFF_P_MIN, PD_COEFF_P_MIN)
    if not isinstance(PD_COEFF_P_MAX, XY):
        PD_COEFF_P_MAX = XY(PD_COEFF_P_MAX, PD_COEFF_P_MAX)

    OPTICAL_METHODS_TO_REFINE_TARGET_SIZE_AND_CENTER  = control_config.pop('optical_methods_to_refine_target_size_and_center', False)
    ADJUST_AIM_POINT_AT_EDGE_OF_FRAME = control_config.pop('adjust_aim_point_at_edge_of_frame', False)
    ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD = control_config.pop('adjust_aim_point_at_edge_of_frame_threshold', 0.01)
    ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_MAX_SIZE = control_config.pop('adjust_aim_point_at_edge_of_frame_max_size', 0.3)


    # Normalized target size thresholds for dynamic P profile:
    # s = 0.0 means target is at or below PD_COEFF_P_MIN_TARGET_SIZE
    # s = 1.0 means target is at or above PD_COEFF_P_MAX_TARGET_SIZE
    #
    # Below STAGE_1_THRESHOLD:
    #   target is considered small / far, P grows quickly from minimum.
    PD_COEFF_P_STAGE_1_THRESHOLD = control_config.pop('pd_coeff_p_dynamic_stage_1_threshold', 0.2)

    # Between STAGE_1_THRESHOLD and STAGE_2_THRESHOLD:
    #   target is in the working mid-range, P continues growing up to maximum.
    # Above STAGE_2_THRESHOLD:
    #   target is considered large / near, P starts decreasing to avoid overshoot
    #   and overly aggressive control close to the target.
    PD_COEFF_P_STAGE_2_THRESHOLD = control_config.pop('pd_coeff_p_dynamic_stage_2_threshold', 0.6)


    # Relative P ratios inside [PD_COEFF_P_MIN, PD_COEFF_P_MAX]:
    #
    # Ratio reached at STAGE_1_THRESHOLD.
    # Example: 0.60 means that by s = 0.2, P reaches 60% of the full range
    # between PD_COEFF_P_MIN and PD_COEFF_P_MAX.
    PD_COEFF_P_STAGE_1_RATIO = control_config.pop('pd_coeff_p_dynamic_stage_1_ratio', 0.60)

    # Ratio reached at STAGE_2_THRESHOLD.
    # Usually 1.00, meaning the maximum P is reached in the mid-range.
    PD_COEFF_P_STAGE_2_RATIO = control_config.pop('pd_coeff_p_dynamic_stage_2_ratio', 1.00)

    # Ratio used when target is very large / very near (s -> 1.0).
    # This reduces P near the target to make control softer and reduce oscillation.
    PD_COEFF_P_STAGE_3_RATIO = control_config.pop('pd_coeff_p_dynamic_stage_3_ratio', 0.35)

    TARGET_SIZE_M = control_config.pop('target_size_m', XY(1, 0.5))
    # Legacy / single-camera fallback. When camera_switcher is provided, FOV
    # is looked up per-frame from the matching CameraConfig instead, so two
    # cameras with different FOVs (wide vs tele) yield correct steering angles.
    FRAME_ANGLUAR_SIZE_DEG_DEFAULT = control_config.pop('frame_angular_size_deg', XY(120, 90))

    def fov_for_camera(cam_id : int) -> XY:
        if camera_switcher is not None:
            cfg = camera_switcher.get_config(cam_id)
            if cfg is not None:
                return cfg.frame_angular_size_deg
        return FRAME_ANGLUAR_SIZE_DEG_DEFAULT

    # INERTIA_CORRECTION_GAIN = control_config.pop('inertia_correction_gain', 0.0)
    # INERTIA_CORRECTION_LIMITS : XY = control_config.pop('inertia_correction_limits', XY(1, 1))
    # INERTIA_CORRECTION_MIN_SPEED_MS = control_config.pop('inertia_correction_min_speed_ms', 0.3)

    ESTIMATION_3D = control_config.pop('estimation_3d', None)
    # if ESTIMATION_3D is None:
    #     ESTIMATION_3D = control_config.pop('estimation_use_3d', False)
    # else:
    #     control_config.pop('estimation_use_3d', None)

    ESTIMATION_3D_METHOD = VelocityMethod(control_config.pop('estimation_3d_method', None))
    ESTIMATION_3D_USE_INITIAL_VELOCITY         = control_config.pop('estimation_3d_use_initial_velocity', True)

    ESTIMATION_LOOKAHEAD_FRAMES                = control_config.pop('estimation_lookahead_frames', 2)
    ESTIMATION_LOOKAHEAD_DYNAMIC               = control_config.pop('estimation_lookahead_dynamic', False)
    ESTIMATION_LOOKAHEAD_DYNAMIC_SQRT          = control_config.pop('estimation_lookahead_dynamic_sqrt', True)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FACTOR        = control_config.pop('estimation_lookahead_dynamic_factor', 1)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_NEAR   = control_config.pop('estimation_lookahead_dynamic_frames_near', 2)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_MEDIUM = control_config.pop('estimation_lookahead_dynamic_frames_medium', 4)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_FAR    = control_config.pop('estimation_lookahead_dynamic_frames_far', 8)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_MAX    = control_config.pop('estimation_lookahead_dynamic_frames_max', 8)

    FOLLOW_TARGET_POSITION_NED                 = control_config.pop('follow_target_position_ned', False)


    DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES = control_config.pop('delay_takeof_until_n_detection_frames', 3)

    BYTETRACK_TARGET_LOCK = control_config.pop('bytetrack_target_lock', True)

    AIM_POINT = control_config.pop('aim_point', XY(0.5, 0.5))
    aim_point = AIM_POINT
    # AIM_POINT = XY(0.5, 0.5)

    SAFE_TAKEOFF_PERIOD_NS = control_config.pop('safe_takeoff_period_ns', 300_000_000)
    if FOLLOW_TARGET_POSITION_NED and not ESTIMATION_3D:
        logger.warning("follow_target_position_ned requires 3D estimation, enabling it automatically")
        ESTIMATION_3D = True

    # Switch policy thresholds: read from camera_switcher (the single source
    # of truth) when one is provided, else use defaults. The thresholds also
    # determine when EMA-smoothed target size triggers a switch.
    if camera_switcher is not None:
        CAMERA_SWITCH_TO_WIDE_SIZE = camera_switcher.switch_to_wide_size
        CAMERA_SWITCH_TO_ZOOM_SIZE = camera_switcher.switch_to_zoom_size
    else:
        CAMERA_SWITCH_TO_WIDE_SIZE = 0.25
        CAMERA_SWITCH_TO_ZOOM_SIZE = 0.015
    # S2: EMA smoothing of target size for switching decisions.
    # Raw per-frame bbox size jitters enough to flap the switch near a
    # threshold; an EMA cuts that without much lag. alpha=0.3 → effective
    # window ~3-4 frames at the controller's loop rate.
    CAMERA_SWITCH_SIZE_EMA_ALPHA = control_config.pop('camera_switch_size_ema_alpha', 0.3)


    # DRONE_CONFIG_PREFIX = 'drone_'
    # for drone_config_key in [k for k in control_config.keys() if k.startswith(DRONE_CONFIG_PREFIX)]:
    #     drone_config_key_stripped = drone_config_key.removeprefix(DRONE_CONFIG_PREFIX)
    #     drone_config[drone_config_key_stripped]=control_config.pop(drone_config_key)

    if len(control_config) > 0:
        logger.warning("Unknown/unused config parameters: %s", control_config)


    distance_r = 0.1
    distance_r *= distance_r
    seen_target = False
    last_seen_target_at_frame = 0
    # Monotonic per-iteration counter. Incremented for every frame dequeued
    # from detections_queue, so it does NOT reset across camera switches —
    # unlike detections_obj.frame_id, which is the camera-local index and
    # jumps when the active camera changes. All internal "frames since X"
    # logic (target-lost history clearing, periodic logging, idle throttling)
    # must use this counter to stay consistent across switches.
    frame_id = 0
    pd_coeff_p_dynamic_stage = None

    cpu = CPUTemperature()
    logger.info("PRE START CPU Temperature: %s°C", cpu.temperature)

    drone = DroneMover(drone_connection_string, drone_config)
    logger.debug("starting up drone... with %s, config: %s", drone_connection_string, drone_config)

    logger.info("POST START CPU Temperature: %s°C", cpu.temperature)

    # udp_port = 14560
    # killdrone_thread = threading.Thread(
    #     target = kill_on_rc_switch_on_channel,
    #     args = (udp_port, 6, drone)
    # )
    # killdrone_thread.start()

    if DEBUG:
        await drone.startup_sequence(1, force_arm=True)
    else:
        await drone.startup_sequence(100_000)

    logger.debug("drone started")

    # logger.debug("raw telemetry (NO-WAIT): %s", await drone.get_telemetry_dict(False))

    # logger.debug("!!! getting telemetry")
    telemetry_dict = await drone.get_telemetry_dict(False)
    # current_attitude : EulerAngle = await drone.get_cached_attitude(wait_for_first=False)
    # logger.debug("GOT telemetry: %s and attitude: %s", telemetry_dict, current_attitude)

    #logger.debug("!!! detections_queue: %s (%s items)", detections_queue, detections_queue.qsize())

    if signal_event_when_ready:
        signal_event_when_ready.set()

    #logger.debug("!!! detections_queue: %s (%s items)", detections_queue, detections_queue.qsize())

    # debug wrapper to collect executed commands
    if True: # allow logging of commands send to the drone
        drone = debug_collect_call_info(drone, history_max_size=3)
        # pass
    else:
        # just to keep existing code itact, remove when no longer needed
        drone.clear_command_history = lambda : None
        drone.last_command = lambda : "--"

    moving = False
    flight_time_ns = 0
    # Tracks which camera last produced a frame, so we can detect a camera
    # switch and clear all per-camera caches (estimators, ByteTrack lock,
    # PD history) — geometry/FOV change at the switch makes old samples
    # meaningless and would otherwise inject a bogus velocity spike.
    last_camera_id : int | None = None
    # FOV in degrees of the camera that produced the CURRENT frame. Initialized
    # to the default; refreshed every iteration from the incoming Detections.
    FRAME_ANGLUAR_SIZE_DEG : XY = FRAME_ANGLUAR_SIZE_DEG_DEFAULT
    # S2: smoothed target size, used by maybe_switch_to_another_camera to
    # avoid flapping near the hysteresis edge. Reset on camera switch (the
    # normalized size scale changes by ~zoom_factor) and on prolonged target
    # loss (the previous trajectory is no longer relevant).
    target_size_ema : XY | None = None
    takeoff_time_ns = None
    prev_angle_to_target = XY()
    skipped_detetions = 0
    prev_detection_timestamp_ns = time.monotonic_ns()
    current_detection_timestamp_ns = 0
    prev_frame_timestamp_ns = time.monotonic_ns()
    current_frame_timestamp_ns = time.monotonic_ns()
    target_position_ned = None
    target_position_prev_ned = None
    estimated_velocity_ned = None
    locked_track_id: int | None = None

    # NOTE: HUGE age to avoid purging prev positions, since it doesn't work as expected RN
    max_target_age_ns = 5_000_000_000_000
    target_estimator_2d = TargetEstimator(max_positions=60, max_age_ns=max_target_age_ns)
    target_estimator_3d = TargetEstimator3D(max_positions=60, max_age_ns=max_target_age_ns)

    for t in (target_estimator_2d, target_estimator_3d):
        proper_max_history_size = DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES * 2
        assert t.max_history_size() >= proper_max_history_size, f"{t} max history size is insufficient, must be at least {proper_max_history_size}"

        target_fps = 30
        proper_max_age_ns = (DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES * 1_000_000_000 / (target_fps + 2)) * 2
        assert t.max_age_ns() >= proper_max_age_ns#, f"{t.name()} max_age_ns is insufficient, must be at least {proper_max_age_ns}"


    def update_timestamps():
        nonlocal prev_frame_timestamp_ns
        nonlocal current_frame_timestamp_ns
        new_frame_timestamp = time.monotonic_ns()
        prev_frame_timestamp_ns = current_frame_timestamp_ns
        current_frame_timestamp_ns = new_frame_timestamp

    def update_timestamps_on_detection():
        nonlocal prev_detection_timestamp_ns
        nonlocal current_detection_timestamp_ns

        new_detection_timestamp = current_frame_timestamp_ns
        prev_detection_timestamp_ns = current_detection_timestamp_ns
        current_detection_timestamp_ns = new_detection_timestamp

        return current_detection_timestamp_ns - prev_detection_timestamp_ns

    def piecewise_p_ratio(s: float) -> float:
        nonlocal pd_coeff_p_dynamic_stage
        s = clamp(0.0, s, 1.0)

        if s < PD_COEFF_P_STAGE_1_THRESHOLD:
            pd_coeff_p_dynamic_stage = 1
            return PD_COEFF_P_STAGE_1_RATIO * (s / PD_COEFF_P_STAGE_1_THRESHOLD)

        if s < PD_COEFF_P_STAGE_2_THRESHOLD:
            pd_coeff_p_dynamic_stage = 2
            return PD_COEFF_P_STAGE_1_RATIO + (
                (PD_COEFF_P_STAGE_2_RATIO - PD_COEFF_P_STAGE_1_RATIO)
                * ((s - PD_COEFF_P_STAGE_1_THRESHOLD) / (PD_COEFF_P_STAGE_2_THRESHOLD - PD_COEFF_P_STAGE_1_THRESHOLD))
            )

        pd_coeff_p_dynamic_stage = 3
        return PD_COEFF_P_STAGE_2_RATIO + (
            (PD_COEFF_P_STAGE_3_RATIO - PD_COEFF_P_STAGE_2_RATIO)
            * ((s - PD_COEFF_P_STAGE_2_THRESHOLD) / (1.0 - PD_COEFF_P_STAGE_2_THRESHOLD))
        )

    def pd_coeff_p_for_target_size(target_size) -> XY:
        if isinstance(target_size, XY):
            target_size = target_size.x * target_size.y

        def compute_p(target_size) -> XY:
            # avoid tipping over on hallucinations while close to the ground
            if flight_time_ns <= SAFE_TAKEOFF_PERIOD_NS:
                logger.warning("Initial stage of flight, reducing P to %s", PD_COEFF_P_SAFE_MIN)
                return copy(PD_COEFF_P_SAFE_MIN)

            if not PD_COEFF_P_DYNAMIC:
                return copy(PD_COEFF_P)

            min_size = PD_COEFF_P_MIN_TARGET_SIZE
            max_size = PD_COEFF_P_MAX_TARGET_SIZE
            p_min = PD_COEFF_P_DYNAMIC_MIN
            p_max = PD_COEFF_P_DYNAMIC_MAX

            if max_size <= min_size:
                logger.warning("Invalid target size range: min=%s max=%s", min_size, max_size)
                return XY(p_min, p_min)

            s = (target_size - min_size) / (max_size - min_size)
            s = clamp(0.0, s, 1.0)

            # Dynamic profile produces a single size-based gain; apply it to both axes.
            if PD_COEFF_P_DYNAMIC_USE_PIECEWISE:
                ratio = piecewise_p_ratio(s)
                p = clamp(p_min, p_min + ratio * (p_max - p_min), p_max)
                return XY(p, p)

            result = p_min + s * (p_max - p_min)

            p = clamp(p_min, result, p_max)
            return XY(p, p)

        p = compute_p(target_size)
        p = copy(clamp_xy(PD_COEFF_P_MIN, p, PD_COEFF_P_MAX))
        return p

    def _nearest_peer(current_cfg : CameraConfig, *, wider : bool) -> CameraConfig | None:
        """Closest peer to `current_cfg` in the zoom_factor ordering.

        wider=True  → camera with the largest zoom_factor that is still
                       smaller than current's (i.e. one step toward wider FOV)
        wider=False → camera with the smallest zoom_factor that is still
                       larger than current's (i.e. one step toward more zoom)
        """
        best = None
        for cfg in camera_switcher.configs():
            if cfg.camera_id == current_cfg.camera_id:
                continue
            if wider:
                if cfg.zoom_factor < current_cfg.zoom_factor and (
                    best is None or cfg.zoom_factor > best.zoom_factor
                ):
                    best = cfg
            else:
                if cfg.zoom_factor > current_cfg.zoom_factor and (
                    best is None or cfg.zoom_factor < best.zoom_factor
                ):
                    best = cfg
        return best

    def maybe_switch_to_another_camera(target_size : XY, frame_camera_id : int):
        nonlocal target_size_ema
        if camera_switcher is None or camera_switcher.num_cameras() <= 1:
            return

        current_camera_id = camera_switcher.active_id()
        if frame_camera_id != current_camera_id:
            # Switch already requested; this frame is a delayed one from the
            # old camera. Don't re-toggle and don't pollute the EMA with a
            # value measured under the OLD camera's geometry.
            return

        # S2: update the EMA with this frame's raw target_size, then test
        # thresholds against the smoothed value. alpha is small enough to
        # stay responsive (a few-frame window) but removes single-frame
        # bbox jitter that previously caused flapping near the threshold.
        a = CAMERA_SWITCH_SIZE_EMA_ALPHA
        if target_size_ema is None:
            target_size_ema = target_size
        else:
            target_size_ema = XY(
                target_size_ema.x * (1 - a) + target_size.x * a,
                target_size_ema.y * (1 - a) + target_size.y * a,
            )

        # S1: decide via the auto-computed zoom_factor instead of free-text
        # `name` strings. Old code did `name == 'zoom'` / `name == 'wide'`,
        # which silently no-op'd the moment someone renamed a camera in app
        # config. Now the policy is: target growing past the wide threshold
        # → step toward a wider peer; target shrinking past the zoom
        # threshold → step toward a more-zoomed peer. Also generalizes to
        # >2 cameras (it will only ever step one peer at a time).
        #
        # Asymmetric axis test:
        # - zoom→wide uses MAX(w,h): if either axis is about to clip the
        #   frame edge we lose the target, so switch when any dimension
        #   crosses the threshold. The old min(w,h) rule was too tolerant
        #   of elongated bboxes (aspect ~1.3 is normal for non-square
        #   objects) and never fired on real bench sweeps.
        # - wide→zoom keeps MAX(w,h) ≤ threshold: BOTH axes must be small
        #   for us to confidently say "target is far away, zoom in safely".
        #   Using min() here would let a long thin false-positive flip us
        #   into zoom on a frame that actually has a large target.
        current_cfg : CameraConfig = camera_switcher.get_config(current_camera_id)
        target_cfg = None
        if target_size_ema.max_val() >= CAMERA_SWITCH_TO_WIDE_SIZE:
            target_cfg = _nearest_peer(current_cfg, wider=True)
        elif target_size_ema.max_val() <= CAMERA_SWITCH_TO_ZOOM_SIZE:
            target_cfg = _nearest_peer(current_cfg, wider=False)

        if target_cfg is None:
            return
        if not camera_switcher.set_active(target_cfg.camera_id):
            return

        logger.warning(
            "Switching cameras: %s (zoom=%.2fx) -> %s (zoom=%.2fx), target_size_ema=%s (raw=%s)",
            current_cfg.name, current_cfg.zoom_factor,
            target_cfg.name, target_cfg.zoom_factor, target_size_ema, target_size,
        )
        # After we trigger a switch, the next frames will be under the new
        # camera's geometry; the EMA's prior history is meaningless there.
        target_size_ema = None


    command_regulator = CommandRegulator(Pk = PD_COEFF_P, Dk = PD_COEFF_D)
    while True:
        extra = ''
        try:
            detections_obj = Detections(-1)
            distance_to_center : float = float('NaN')
            angle_to_target = XY()
            move_command = MoveCommand()
            thrust = THRUST_CRUISE

            logger = global_logger
            # logger.debug("!!! awaiting detection... ")
            try:
                # OverwriteQueue.get is a synchronous threading.Condition.wait — calling it
                # directly from this asyncio coroutine would block the event loop and starve
                # the mavsdk telemetry consumer tasks (50 Hz attitude/odometry/imu streams),
                # eventually overflowing mavsdk_server's user-callback queue and freezing
                # cached telemetry. Offload the blocking wait to the default thread executor
                # so the loop stays free to drain telemetry while we wait.
                r : Detections = await asyncio.to_thread(detections_queue.get, 0.01)
                if r is STOP or r is None:
                    logger.info("stopping")
                    break
                detections_obj = r

            except Empty:
                # No detections, not even frame with ID
                skipped_detetions += 1

                # It is OK to have occasionally no frames from the queue
                # however, long streaks of no frames could cause a crash.
                # 30 is arbitrary, with the wait timeout of .01 of detections_queue.get above
                # that constitutes 0.3 seconds without any commands.
                if moving and skipped_detetions > 30:
                    # % 10 is to limit the number of commands sent to drone per second
                    if skipped_detetions % 10 == 0:
                        # hover to allow drone to iether recover detection and pursuit
                        # OR operator to take over and land it safely.
                        await drone.standstill(THRUST_HOVER)

                # 53 is arbitrary to reduce log noise
                if skipped_detetions % 53 == 0:
                    logger_log_to = logger.warning
                    if skipped_detetions > 500:
                        logger_log_to = logger.error

                    logger_log_to(
                            "No frames (%d times), no detections, input queue empty? prev action: %s",
                            skipped_detetions,
                            drone.last_command()
                    )
                continue

            except:
                logger.exception("Serious error getting next detection from a queue", exc_info=True)
                break

            update_timestamps()
            frame_id += 1
            logger = LoggerWithPrefix(logger, prefix=f'frame=#{frame_id:04}')
            logger.debug(f"frame cam{detections_obj.camera_id}#{detections_obj.frame_id:04}")

            # Camera switch detection. On switch, refresh the per-frame FOV
            # used for all geometry below AND drop the camera-frame caches:
            #   - target_estimator_2d holds normalized bbox positions under
            #     the OLD camera's pinhole; reusing them after a switch would
            #     compute a wildly wrong velocity at the discontinuity.
            #   - ByteTrack track ids are camera-local; the locked id must
            #     be released so the next frame can establish a new lock.
            #   - PD regulator history is from the old camera's angular
            #     resolution; carrying it across yields one bogus large step.
            #
            # Deliberately KEPT across switches:
            #   - target_estimator_3d (NED, world-frame): immune to camera
            #     swap since project_camera_to_ned uses the per-frame FOV.
            #     Preserving it means switch-time velocity estimate stays
            #     continuous and the controller can keep flying the target
            #     trajectory through the optical discontinuity. Assumes both
            #     cameras are coaxial (no extrinsic offset); we can add a
            #     per-camera angular offset to CameraConfig if needed.
            #   - target_position_ned / _prev_ned / estimated_velocity_ned:
            #     world-frame too; kept for the same reason.
            if last_camera_id is None or last_camera_id != detections_obj.camera_id:
                if last_camera_id is not None:
                    logger.warning(
                        "!!! CAMERA SWITCH: %s -> %s, purging 2D estimator / track lock / PD history; keeping 3D NED trajectory",
                        last_camera_id, detections_obj.camera_id,
                    )
                    target_estimator_2d.clear_history()
                    locked_track_id = None
                    prev_angle_to_target = XY()
                    # S2: drop the smoothed target-size; the new camera's
                    # geometry rescales every dimension by ~zoom_factor, so
                    # carrying the old value across would mis-fire the
                    # symmetric threshold immediately.
                    target_size_ema = None
                    # CommandRegulator stores last input internally; reinstating
                    # the configured P/D forces it to drop its previous-sample
                    # state (next_command() initializes on first call).
                    command_regulator = CommandRegulator(Pk = PD_COEFF_P, Dk = PD_COEFF_D)
                last_camera_id = detections_obj.camera_id
                FRAME_ANGLUAR_SIZE_DEG = fov_for_camera(detections_obj.camera_id)

            # if DEBUG:
            #     # NOTE: injecting fake detections to debug
            #     import math
            #     __tmp_delta_confidence = math.sin(detections_obj.frame_id / 100) / 10
            #     __tmp_delta_x = math.sin(detections_obj.frame_id / 100) / 4
            #     __tmp_delta_y = math.cos(detections_obj.frame_id / 100) / 4
            #     detections_obj.detections.append(
            #                 Detection(
            #                     bbox = Rect.from_xywh(0.2 + __tmp_delta_x, 0.2 + __tmp_delta_y, 0.05, 0.05),
            #                     confidence = 0.3 + __tmp_delta_confidence,
            #                     track_id = 1
            #                 )
            #     )

            logger.debug("!!! GOT DETECTIONS, objects detected: %s (%s), detection delay: %sms, total delay: %sms",
                    len(detections_obj.detections),
                    detections_obj.detections,
                    (detections_obj.meta.detection_end_timestamp_ns - detections_obj.meta.detection_start_timestamp_ns) / 1000_000,
                    (detections_obj.meta.detection_end_timestamp_ns - detections_obj.meta.capture_timestamp_ns) / 1000_000
                )
            skipped_detetions = 0
            frame_capture_timestampt_ns = detections_obj.meta.capture_timestamp_ns or None

            telemetry_dict : dict = drone.get_telemetry_dict_cached()
            if DEBUG and not all(telemetry_dict.values()) and DEBUG_TELEMETRY_DICT:
                telemetry_dict = DEBUG_TELEMETRY_DICT
                logger.warning("!!! USING DEBUG TELEMETRY data !!!")

            # Throttle: the full telemetry dict is ~1869 chars including two
            # 21-element covariance matrices full of NaN; str()-ing it costs
            # ~30 ms on the hot thread. %-style is "lazy" only when the level
            # check fails, but DEBUG is enabled, so the format runs every
            # iteration. Once every 10 controller frames (~1.2 s at ~8.5 fps)
            # keeps the dump useful for offline analysis without spending
            # ~30 ms per frame on the hot path.
            if logger.isEnabledFor(logging.DEBUG) and frame_id % 10 == 0:
                logger.debug("telemetry: %s", telemetry_dict)
            debug_info = telemetry_dict

            ## Check if take off
            if takeoff_time_ns is None:
                if moving:
                    takeoff_time_ns = time.monotonic_ns()
                    logger.info("!!! TAKEOFF AT: %s", takeoff_time_ns)
            else:
                flight_time_ns = time.monotonic_ns() - takeoff_time_ns
                # logger.info("!!! flight time: %ss", flight_time_ns / 1000_000_000)

            debug_info['start_time_ms'] = START_TIME_MS
            debug_info['flight_time_ms'] = flight_time_ns / 1000_000

            detections = detections_obj.detections
            detection = None

            # so telemetry action doesn't get into the logs
            drone.clear_command_history()

            # filter out accidential Nones
            detections = [d for d in detections if d is not None]
            target_relative_pos = None
            target_relative_pos_uncorrected = None
            target_position_ned = None
            target_estimator = None
            target_center = None
            target_size = None

            picked = _pick_target_detection(
                detections, CONFIDENCE_MIN, locked_track_id, BYTETRACK_TARGET_LOCK
            )
            detection = picked if picked is not None else Detection()
            # Guard the read: cpu.temperature is a sysfs access, so even %-style would
            # evaluate it every frame. Skip it entirely when INFO is disabled.
            if logger.isEnabledFor(logging.INFO) and frame_id % 10 == 0:
                logger.info("!!!!! CPU Temperature: %s°C", cpu.temperature)

            if detection.confidence >= CONFIDENCE_MIN:
            #     await drone.move_to_target_zenith_async(roll_degree=0, pitch_degree=0, thrust=0.2, current_telemetry=telemetry_dict)
            # elif True:
            #     await drone.move_to_target_zenith_async(roll_degree=0, pitch_degree=0, thrust=0.01, current_telemetry=telemetry_dict)

                if BYTETRACK_TARGET_LOCK and detection.track_id is not None:
                    locked_track_id = detection.track_id

                seen_target = True
                last_seen_target_at_frame = frame_id
                delay_between_detections_ns = update_timestamps_on_detection()

                aim_point = copy(AIM_POINT)

                target_size = detection.bbox.size
                target_center = detection.bbox.center

                if OPTICAL_METHODS_TO_REFINE_TARGET_SIZE_AND_CENTER:
                    optical_object_info = OpticalObjectInfo(detections_obj.frame, detection.bbox)
                    target_size = optical_object_info.object_size() or target_size
                    target_center = optical_object_info.object_circle_center() or target_center

                if ADJUST_AIM_POINT_AT_EDGE_OF_FRAME and \
                        target_size.x * target_size.y < ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_MAX_SIZE:

                    # bbox coord are in [0..1] range here
                    min_p : XY = detection.bbox.min_point
                    max_p : XY = detection.bbox.max_point

                    if min_p.x <= ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD:
                        target_center.x = max(0, min_p.x) + ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD
                    elif max_p.x >= 1 - ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD:
                        target_center.x = min(1, max_p.x) - ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD

                    if min_p.y <= ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD:
                        target_center.y = max(0, min_p.y) + ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD
                    elif max_p.y >= 1 - ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD:
                        target_center.y = min(1, max_p.y) - ADJUST_AIM_POINT_AT_EDGE_OF_FRAME_THRESHOLD

                logger.debug("!!! %s, %s, visual size: %s/%s, visual center: %s/%s",
                        TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, target_size, detection.bbox.size, target_center, detection.bbox.center)
                estimated_distance_class, estimated_distance_m = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, target_size)

                logger.debug("!!! RAW estimated_distance: %s %s",
                        estimated_distance_class, estimated_distance_m)

                maybe_switch_to_another_camera(target_size, detections_obj.camera_id)

                # if flight_time_ns <= SAFE_TAKEOFF_PERIOD_NS and estimated_distance_m < 10:
                #     # HACK: we have a distance estimation issue here, distance is at least 50m
                #     estimated_distance_m = 80

                estimated_distance_m = estimated_distance_m if estimated_distance_m else 1
                # logger.debug(f"!!! estimated_distance: {estimated_distance_m}")
                # NOTE: At large distances estimation higly undershoots, formula corrects it to be good enough
                # estimated_distance_m *= math.e #(math.log(estimated_distance_m, 10) + 0.5)
                logger.debug("!!! estimated_distance: %s", estimated_distance_m)

                try:
                    drone_pose = get_pose(telemetry_dict)
                except:
                    logger.warning("Can't get dron pose from telemetry")
                    drone_pose = None

                mode = 'follow'

                # target_size = target_size #detection.bbox.area()
                pd_coeff_p = pd_coeff_p_for_target_size(target_size.x * target_size.y)

                target_relative_pos = aim_point - target_center
                logger.debug("!!! target : %s, size: %s, pd_coeff_p: %s", target_relative_pos, target_size, pd_coeff_p)

                # TODO maybe use frame capture time?
                target_estimator_2d.add_target_pos(
                    target_relative_pos,
                    # estimation is too far off, when using frame capture time.
                    current_frame_timestamp_ns #frame_capture_timestampt_ns if frame_capture_timestampt_ns else current_frame_timestamp_ns
                )

                estimate_lookeahead_frames = ESTIMATION_LOOKAHEAD_FRAMES
                if ESTIMATION_LOOKAHEAD_DYNAMIC:
                    distance = estimated_distance_m if estimated_distance_m else 1
                    if ESTIMATION_LOOKAHEAD_DYNAMIC_SQRT:
                        estimate_lookeahead_frames = int(math.sqrt(distance))
                    elif ESTIMATION_LOOKAHEAD_DYNAMIC_FACTOR is not None:
                        estimate_lookeahead_frames = distance * ESTIMATION_LOOKAHEAD_DYNAMIC_FACTOR

                    if estimated_distance_class == DistanceClass.FAR:
                        estimate_lookeahead_frames += ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_FAR
                    elif estimated_distance_class == DistanceClass.MEDIUM:
                        estimate_lookeahead_frames += ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_MEDIUM
                    elif estimated_distance_class == DistanceClass.NEAR:
                        estimate_lookeahead_frames += ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_NEAR

                    estimate_lookeahead_frames = int(clamp(0, estimate_lookeahead_frames, ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_MAX)) * 1.0

                estimate_delta_ns = int((current_frame_timestamp_ns - prev_frame_timestamp_ns) * estimate_lookeahead_frames)
                estimate_at_ns = current_frame_timestamp_ns + estimate_delta_ns
                estimate_mode = ''
                target_relative_pos_old = target_relative_pos

                if ESTIMATION_3D and estimated_distance_m is not None and drone_pose:
                    target_estimator = target_estimator_3d
                    # --- 3-D world-frame position estimation ---
                    try:
                        # _quat = get_orientation_quaternion(telemetry_dict)
                        # _drone_pos = get_position_ned(telemetry_dict)
                        target_pos_ned = project_camera_to_ned(
                            target_center.x,
                            target_center.y,
                            AIM_POINT.x,
                            AIM_POINT.y,
                            FRAME_ANGLUAR_SIZE_DEG.x,
                            FRAME_ANGLUAR_SIZE_DEG.y,
                            estimated_distance_m,
                            drone_pose.quaternion,
                            drone_pose.position,
                        )
                        target_estimator_3d.add(target_pos_ned, current_frame_timestamp_ns)
                        logger.debug("!!! drone pos NED: N=%.2f E=%.2f D=%.2f\n\ttarget NED: N=%.2f E=%.2f D=%.2f (distance=%.1fm)",
                                drone_pose.position.north_m, drone_pose.position.east_m, drone_pose.position.down_m,
                                target_pos_ned.north_m, target_pos_ned.east_m, target_pos_ned.down_m,
                                estimated_distance_m)

                        if ESTIMATION_3D_USE_INITIAL_VELOCITY and target_estimator_3d.history_size() < DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES:
                            # still accumulating trajectory estimation
                            estimated_velocity_ned = target_estimator_3d.estimate_velocity(
                                estimate_at_ns,
                                None,
                                method=ESTIMATION_3D_METHOD
                            )
                            logger.debug("!!! estimated velocity: %s", estimated_velocity_ned)

                        if estimated_velocity_ned:
                            estimated_pos_ned = target_estimator_3d.estimate_based_on_velocity(estimate_at_ns, None, estimated_velocity_ned)
                        else:
                            estimated_pos_ned = target_estimator_3d.estimate(
                                estimate_at_ns,
                                None,
                                method=ESTIMATION_3D_METHOD
                            )

                        if estimated_pos_ned is None:
                            logger.warning('3D estimation fallback to: %s, target_estimator_3d has %s items',
                                target_pos_ned, target_estimator_3d.history_size())
                            target_relative_pos = target_relative_pos_old
                            target_position_ned = target_position_prev_ned
                        else:
                            target_position_prev_ned = target_position_ned
                            target_position_ned = estimated_pos_ned
                            # third one is distane, which we don't need
                            estimated_x, estimated_y, _ = project_ned_to_camera(
                                estimated_pos_ned,
                                AIM_POINT.x,
                                AIM_POINT.y,
                                FRAME_ANGLUAR_SIZE_DEG.x,
                                FRAME_ANGLUAR_SIZE_DEG.y,
                                drone_pose.quaternion,
                                drone_pose.position
                            )
                            target_relative_pos = XY(estimated_x, estimated_y)
                        target_relative_pos = AIM_POINT - target_relative_pos

                        # estimate_mode = f'3D={ESTIMATION_3D_METHOD}'
                    except Exception:
                        logger.debug("3D estimation failed", exc_info=True)

                # NOTE: ??? maybe use as fallback if 3d estimation is not available
                else:
                    if ESTIMATION_3D:
                        logger.warning("!!! USING 2D estimator because either POSE or DISTANCE are unavailable")

                    target_estimator = target_estimator_2d
                    target_relative_pos = target_estimator.estimate_target_pos(estimate_at_ns, target_relative_pos)

                estimate_mode = target_estimator.describe_prev_estimation() if target_estimator else None
                if estimate_mode:
                    mode += f' *{estimate_mode}:{estimate_lookeahead_frames/1.0:.2f}f '

                    logger.debug("!!! %s estimated new target pos %s (was %s), for +%sms (%.2f frames)",
                            estimate_mode,
                            target_relative_pos,
                            target_relative_pos_old,
                            estimate_delta_ns / 1000_000,
                            estimate_lookeahead_frames/1.0 # just to force it to be float
                        )

                target_relative_pos_uncorrected = target_relative_pos
                # Inertia correction: feedforward from actual velocity in FRD frame
                # if INERTIA_CORRECTION_GAIN != 0 and target_relative_pos is not None:
                #     inertia_correction = compute_inertia_correction(
                #         telemetry_dict,
                #         target_relative_pos,
                #         INERTIA_CORRECTION_GAIN,
                #         INERTIA_CORRECTION_MIN_SPEED_MS
                #     )

                #     logger.info("inertia correction before clamping: %s", inertia_correction)
                #     # clamping to the limits
                #     inertia_correction = XY(
                #         clamp(-INERTIA_CORRECTION_LIMITS.x, inertia_correction.x, INERTIA_CORRECTION_LIMITS.x),
                #         clamp(-INERTIA_CORRECTION_LIMITS.y, inertia_correction.y, INERTIA_CORRECTION_LIMITS.y)
                #     )
                #     extra += f'inertia correction gain: {INERTIA_CORRECTION_GAIN:.2f} val: {inertia_correction}'
                #     target_relative_pos = target_relative_pos + inertia_correction
                #     logger.debug("inertia correction: %s, adjusted target: %s", inertia_correction, target_relative_pos)

                # Note target_relative_pos is already offset from AIM_POINT
                distance_to_center = target_relative_pos.distance_to(XY(0, 0))
                thrust = THRUST_CRUISE
                if flight_time_ns <= SAFE_TAKEOFF_PERIOD_NS:
                    thrust = THRUST_TAKEOFF
                    logger.warning('takeoff low thrust mode: %s', thrust)
                else:
                    if THRUST_DYNAMIC:
                        if distance_to_center < 0.1:
                            thrust= THRUST_MAX
                            mode += " GREEN "
                            # pd_coeff_p /= 3
                        elif distance_to_center < 0.2:
                            thrust= THRUST_MIN + (THRUST_MAX - THRUST_MIN) / 2
                            mode += " YELLOW "
                            # pd_coeff_p /= 1.5
                        else:
                            thrust= THRUST_MIN
                            mode += " RED "
                            #pd_coeff_p

                    if THRUST_PROPORTIONAL_TO_DISTANCE:
                        if estimated_distance_m > THRUST_PROPORTIONAL_TO_DISTANCE_MEDIUM_DISTANCE_M:
                            thrust *= THRUST_PROPORTIONAL_TO_DISTANCE_FAR_COEFF
                            # pd_coeff_p *= 1
                            extra += ' FAR'

                        # NEAR
                        if estimated_distance_m < THRUST_PROPORTIONAL_TO_DISTANCE_NEAR_DISTANCE_M:
                            thrust *= THRUST_PROPORTIONAL_TO_DISTANCE_NEAR_COEFF
                            pd_coeff_p *= 1.1
                            extra += ' NEAR'
                            pass
                        # MEDIUM
                        elif estimated_distance_m < THRUST_PROPORTIONAL_TO_DISTANCE_MEDIUM_DISTANCE_M:
                            thrust *= THRUST_PROPORTIONAL_TO_DISTANCE_MEDIUM_COEFF
                            pd_coeff_p *= 1.1
                            extra += ' MEDIUM'

                        extra += f' changing thrust to: {thrust}, p to: {pd_coeff_p} '


                logger.info("Setting new command regulator coeffs P=%s D=%s", pd_coeff_p, PD_COEFF_D)
                command_regulator.set_coeffs(Pk = pd_coeff_p, Dk = PD_COEFF_D)
                target_relative_pos_pd = target_relative_pos
                USE_SET_ATTITUDE = drone_config.get('use_set_attitude', False)
                if USE_SET_ATTITUDE:
                    logger.warning('NOT APPLYING P D since flying via set_attitude API')
                    target_relative_pos_pd = target_relative_pos
                elif target_relative_pos is not None:
                    logger.debug("!!! target before PD: %s", target_relative_pos)
                    target_relative_pos_pd = command_regulator.next_command(target_relative_pos, delay_between_detections_ns / 1000_000)
                    logger.debug("!!! target after PD: %s, regulator coeffs: %s", target_relative_pos_pd, command_regulator.get_coeffs())

                angle_to_target  = target_relative_pos_pd.multiplied_by_XY(FRAME_ANGLUAR_SIZE_DEG)
                prev_angle_to_target = angle_to_target

                logger.debug("angle to target: %s", angle_to_target)

                mode += f'size: {target_size}, estimated distance: ({estimated_distance_class} @ {estimated_distance_m:.1f}m), p: {pd_coeff_p:.2f} '

                # while still taking off, avoid dangerous moves
                # if flight_time_ns < SAFE_TAKEOFF_PERIOD_NS:
                #     MAX_CLOSE_TO_GROUND_ANGLES = XY(60, 60)
                #     new_angle_to_target = angle_to_target
                #     if abs(angle_to_target.x) > MAX_CLOSE_TO_GROUND_ANGLES.x:
                #         sign = angle_to_target.x / abs(angle_to_target.x)
                #         new_angle_to_target.x = min(MAX_CLOSE_TO_GROUND_ANGLES.x, abs(angle_to_target.x)) * sign

                #     if abs(angle_to_target.y) > MAX_CLOSE_TO_GROUND_ANGLES.y:
                #         sign = angle_to_target.y / abs(angle_to_target.y)
                #         new_angle_to_target.y = min(MAX_CLOSE_TO_GROUND_ANGLES.y, abs(angle_to_target.y)) * sign

                #     if new_angle_to_target != angle_to_target:
                #         logger.warning("Too steep atack close to the ground %s, clamping to %s ", angle_to_target, new_angle_to_target)
                #         angle_to_target = new_angle_to_target

                debug_info["mode"] = mode
                thrust = clamp(THRUST_MIN, thrust, THRUST_MAX)
                if not takeoff_time_ns and target_estimator.history_size() < DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES:
                    logger.warning("Delaying takeoff for %s frames (now have %s)", DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES, target_estimator.history_size())
                    pass
                else:
                    if FOLLOW_TARGET_POSITION_NED and target_position_ned:
                        mode += " NED "
                        debug_info["mode"] = mode
                        await drone.move_to_target_ned(target_position_ned, telemetry_dict)
                        moving = True
                    else:
                        # NOTE: perfroming conversion from camera referene frame to done's FRD
                        await drone.move_to_target_zenith_async(roll_degree=-angle_to_target.x, pitch_degree=angle_to_target.y, thrust=thrust, current_telemetry=telemetry_dict)
                        moving = True

                    if moving and takeoff_time_ns is None:
                        takeoff_time_ns = time.monotonic_ns()
                        # logger.info("!!! IN AIR since: %s", takeoff_time_ns)

            else:
                # IF no detection or NONE of the detections has big confidence
                if abs(frame_id - last_seen_target_at_frame) > TARGET_ESTIMATOR_CLEAR_HISTORY_AFTER_TARGET_LOST_FRAMES and target_estimator_2d.history_size() > 0:
                    logger.warning("!!! CLEARING HISTORY")
                    target_estimator_2d.clear_history()
                    target_estimator_3d.clear_history()
                    locked_track_id = None
                    # S2: prolonged target loss → smoothed size is stale.
                    target_size_ema = None

                if seen_target:
                    if FOLLOW_TARGET_POSITION_NED:
                        if target_position_prev_ned is not None:
                            await drone.move_to_target_ned(target_position_prev_ned, telemetry_dict)
                            moving = True
                            debug_info["mode"] = "follow last-ned"
                        else:
                            moving = False
                            debug_info["mode"] = "hover"
                    else:
                        prev_angle_to_target *= FADE_COEFF
                        # NOTE: perfroming conversion from camera referene frame to done's FRD
                        await drone.move_to_target_zenith_async(roll_degree=-prev_angle_to_target.x, pitch_degree=prev_angle_to_target.y, thrust=thrust, current_telemetry=telemetry_dict)
                        # Just t visualize the point we are moving to
                        target_relative_pos = prev_angle_to_target.divided_by_XY(FRAME_ANGLUAR_SIZE_DEG)
                        # await drone.standstill()
                        # moving = False
                        debug_info["mode"] = "hover"
                else:
                    debug_info["mode"] = "idle"
                    if not FOLLOW_TARGET_POSITION_NED and frame_id % 30 == 0:
                        # moving = False
                        await drone.idle()

            # Stage C latency: stamp the moment the control command has actually been
            # sent to the drone (the awaited move/idle/hover above has returned), then
            # split the capture→command path into its segments:
            #   sensor→command (e2e)  = the true end-to-end number
            #   callback→command (C)  = queue_wait + processing
            #     queue_wait = callback (frame enqueued) → this iteration dequeued it
            #     processing = dequeue → command sent (telemetry + estimation + PD + await)
            # detection_end (callback) and capture (sensor) come from the frame meta;
            # current_frame_timestamp_ns is when this iteration dequeued the frame.
            command_sent_ns = time.monotonic_ns()
            raw_last_command = drone.last_command()
            last_command = raw_last_command or '<<== NO ==>>'
            debug_info["action"] = last_command
            if raw_last_command:  # a control command was actually issued this frame
                _meta = detections_obj.meta
                logger.warning(
                    "!!! LATENCY sensor→command(e2e): %.1fms | callback→command(stageC): %.1fms = queue_wait %.1fms + processing %.1fms",
                    (command_sent_ns - _meta.capture_timestamp_ns) / 1000_000,
                    (command_sent_ns - _meta.detection_end_timestamp_ns) / 1000_000,
                    (current_frame_timestamp_ns - _meta.detection_end_timestamp_ns) / 1000_000,
                    (command_sent_ns - current_frame_timestamp_ns) / 1000_000,
                )
            mode = debug_info.get('mode', '')
            logger.info("MODE: %s, ACTION: %s", mode, last_command)

            if PD_COEFF_P_DYNAMIC:
                extra += (f"p_stage={pd_coeff_p_dynamic_stage} "
                    f"stage1_thr={PD_COEFF_P_STAGE_1_THRESHOLD:.3f} "
                    f"stage2_thr={PD_COEFF_P_STAGE_2_THRESHOLD:.3f} "
                    f"stage1_r={PD_COEFF_P_STAGE_1_RATIO:.2f} "
                    f"stage2_r={PD_COEFF_P_STAGE_2_RATIO:.2f} "
                    f"stage3_r={PD_COEFF_P_STAGE_3_RATIO:.2f} "
                    f"p_d_min={PD_COEFF_P_DYNAMIC_MIN:.2f} "
                    f"p_d_max={PD_COEFF_P_DYNAMIC_MAX:.2f} "
                    f"p_min={PD_COEFF_P_MIN:.2f} "
                    f"p_max={PD_COEFF_P_MAX:.2f} ")

            debug_info['extra'] = extra
            # Surface frame identity to the visualization. frame_id is the
            # controller's monotonic counter (unaffected by camera switches);
            # camera_id + camera_frame_id locate the underlying source frame.
            # debug_info['frame_id'] = (frame_id, detections_obj.camera_id, detections_obj.frame_id)
            debug_info['camera_id'] = detections_obj.camera_id
            debug_info['camera_frame_id'] = detections_obj.frame_id

            # -1 means that there was no frame and no detections
            if output_queue is not None:
                output = {
                    'detections' : detections_obj,
                    'aim_point'  : aim_point,
                    'selected' : detection,
                    'selected_aim_point' : target_center,
                    'telemetry': debug_info,
                    'selected_detection_projected_pos' : target_relative_pos_uncorrected,
                    # 'target_pos_ned' : target_estimator_3d.latest,
                    # 'target_pos_ned_estimated' : target_estimator_3d.estimate(current_frame_timestamp_ns),
                    # 'inertia_accumuated' : target_relative_pos,
                    'move_goal' : target_relative_pos,
                    'frame_id' : frame_id,
                    'camera_id' : detections_obj.camera_id,
                    'camera_frame_id' : detections_obj.frame_id,
                    'frame_angular_size_deg' : FRAME_ANGLUAR_SIZE_DEG,
                }
                output_queue.put(output)

        except:
            logger.exception("Got exception: %s %s COMMAND: %s",
                    detections_obj, distance_to_center, move_command, exc_info=True)
