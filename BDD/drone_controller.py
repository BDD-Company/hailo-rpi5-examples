#! /usr/bin/env python3

import asyncio
import math
import time
from queue import Empty, Queue

from helpers import XY

from drone import DroneMover
from CommandRegulator import CommandRegulator
from TargetEstimator import TargetEstimator, TargetEstimator3D, VelocityMethod
from estimate_distance import estimate_distance_class, DistanceClass
from telemetry_position import (
    # get_position_ned,
    # get_orientation_quaternion,
    project_camera_to_ned,
    get_pose,
    project_ned_to_camera
)
# from drone_killswitch import kill_on_rc_switch_on_channel
from helpers import Detection, Detections, MoveCommand, STOP

from helpers import (
    debug_collect_call_info,
    LoggerWithPrefix
)

import logging
logger = logging.getLogger(__name__)
global_logger = logger

DEBUG = False


def drone_controlling_thread(*args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(drone_controlling_thread_async(*args, **kwargs))
    except Exception as e:
        logger.error("in drone event loop", exc_info=True, stack_info=True)
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
    return typeof_val(max(min_val, min(max_val, val)))


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


async def drone_controlling_thread_async(drone_connection_string, drone_config, detections_queue, control_config = {}, output_queue = None, signal_event_when_ready = None):
    # from math import radians

    # will owerwrite logger here many times, make sure that rest of the systems are not affected
    global global_logger
    logger = global_logger

    START_TIME_MS = time.monotonic_ns() / 1000_000

    global DEBUG
    DEBUG           = control_config.pop('DEBUG', False)
    logger.debug("!!!!! DEBUG state: %s", DEBUG)
    CONFIDENCE_MIN  = control_config.pop('confidence_min', 0.1)
    # MOVE_CONFIDENCE = control_config.get('confidence_move', 0.4)

    THRUST_MAX      = control_config.pop('thrust_max', 0.5)
    THRUST_MIN      = control_config.pop('thrust_min', 0.5)
    THRUST_TAKEOFF  = control_config.pop('thrust_takeoff', 0.5)
    THRUST_DYNAMIC  = control_config.pop('thrust_dynamic', False)
    THRUST_PROPORTIONAL_TO_TARGET_SIZE = control_config.pop('thrust_proportional_to_target_size', False)

    FADE_COEFF      = control_config.pop('target_lost_fade_per_frame', 0.9)
    TARGET_ESTIMATOR_CLEAR_HISTORY_AFTER_TARGET_LOST_FRAMES = control_config.pop('target_estimator_clear_history_after_target_lost_frames', 3)

    PD_COEFF_P      = control_config.pop('pd_coeff_p', 1)
    PD_COEFF_D      = control_config.pop('pd_coeff_d', 0)

    PD_COEFF_P_DYNAMIC = control_config.pop('pd_coeff_p_dynamic', False)
    PD_COEFF_P_DYNAMIC_USE_PIECEWISE = control_config.pop('pd_coeff_p_dynamic_use_piecewise', False)
    PD_COEFF_P_MIN_TARGET_SIZE = control_config.pop('pd_coeff_p_dynamic_min_target_size', 0.003)
    PD_COEFF_P_MAX_TARGET_SIZE = control_config.pop('pd_coeff_p_dynamic_max_target_size', 0.005)
    PD_COEFF_P_DYNAMIC_MIN  = control_config.pop('pd_coeff_p_dynamic_min', 0.5)
    PD_COEFF_P_DYNAMIC_MAX  = control_config.pop('pd_coeff_p_dynamic_max', 2)

    PD_COEFF_P_SAFE_MIN  = control_config.pop('pd_coeff_p_safe_min', 0.5)
    PD_COEFF_P_MIN  = control_config.pop('pd_coeff_p_min', 0.5)
    PD_COEFF_P_MAX  = control_config.pop('pd_coeff_p_max', 5)


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
    FRAME_ANGLUAR_SIZE_DEG = control_config.pop('frame_angular_size_deg', XY(120, 90))

    # INERTIA_CORRECTION_GAIN = control_config.pop('inertia_correction_gain', 0.0)
    # INERTIA_CORRECTION_LIMITS : XY = control_config.pop('inertia_correction_limits', XY(1, 1))
    # INERTIA_CORRECTION_MIN_SPEED_MS = control_config.pop('inertia_correction_min_speed_ms', 0.3)

    ESTIMATION_3D                   = control_config.pop('estimation_3d', False)
    ESTIMATION_3D_METHOD                 = VelocityMethod(control_config.pop('estimation_3d_method', 'numpy')) # OR any VelocityMethod 'wls'
    ESTIMATION_LOOKAHEAD_FRAMES         = control_config.pop('estimation_lookahead_frames', 2)
    ESTIMATION_LOOKAHEAD_DYNAMIC        = control_config.pop('estimation_lookahead_dynamic', False)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_NEAR   = control_config.pop('estimation_lookahead_dynamic_frames_near', 2)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_MEDIUM = control_config.pop('estimation_lookahead_dynamic_frames_medium', 4)
    ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_FAR    = control_config.pop('estimation_lookahead_dynamic_frames_far', 8)

    DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES = control_config.pop('delay_takeof_until_n_detection_frames', 3)

    AIM_POINT = control_config.pop('aim_point', XY(0.5, 0.5))
    aim_point = AIM_POINT
    # AIM_POINT = XY(0.5, 0.5)

    SAFE_TAKEOFF_PERIOD_NS = control_config.pop('safe_takeoff_period_ns', 300_000_000)
    if len(control_config) > 0:
        logger.warning("Unknonw/unused config parameters: %s", control_config)


    distance_r = 0.1
    distance_r *= distance_r
    seen_target = False
    last_seen_target_at_frame = 0
    frame_id = 0
    pd_coeff_p_dynamic_stage = None


    drone = DroneMover(drone_connection_string, drone_config)
    logger.debug("starting up drone... with %s, config: %s", drone_connection_string, drone_config)

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
    drone = debug_collect_call_info(drone, history_max_size=3)

    moving = False
    flight_time_ns = 0
    takeoff_time_ns = None
    prev_angle_to_target = XY()
    skipped_detetions = 0
    prev_detection_timestamp_ns = time.monotonic_ns()
    current_detection_timestamp_ns = 0
    prev_frame_timestamp_ns = time.monotonic_ns()
    current_frame_timestamp_ns = time.monotonic_ns()

    # NOTE: HUGE age to avoid purging prev positions, since it doesn't work as expected RN
    target_estimator = TargetEstimator(
        max_target_positions=max(DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES + 5, 20),
        max_target_position_age_nanoseconds=500_000_000_000,
    )
    target_estimator_3d = TargetEstimator3D(max_positions=60, max_age_ns=500_000_000_000)

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

    def pd_coeff_p_for_target_size(target_size):
        def compute_p(target_size):
            # avoid tipping over on hallucinations while close to the ground
            if flight_time_ns <= SAFE_TAKEOFF_PERIOD_NS:
                logger.warning("Initial stage of flight, reducing P to %s", PD_COEFF_P_MIN)
                return PD_COEFF_P_SAFE_MIN

            if not PD_COEFF_P_DYNAMIC:
                return PD_COEFF_P

            min_size = PD_COEFF_P_MIN_TARGET_SIZE
            max_size = PD_COEFF_P_MAX_TARGET_SIZE
            p_min = PD_COEFF_P_DYNAMIC_MIN
            p_max = PD_COEFF_P_DYNAMIC_MAX

            if max_size <= min_size:
                logger.warning("Invalid target size range: min=%s max=%s", min_size, max_size)
                return p_min

            s = (target_size - min_size) / (max_size - min_size)
            s = clamp(0.0, s, 1.0)

            if PD_COEFF_P_DYNAMIC_USE_PIECEWISE:
                ratio = piecewise_p_ratio(s)
                return clamp(p_min, p_min + ratio * (p_max - p_min), p_max)

            result = p_min + s * (p_max - p_min)

            return clamp(p_min, result, p_max)

        p = compute_p(target_size)
        p = clamp(PD_COEFF_P_MIN, p, PD_COEFF_P_MAX)
        return p


    command_regulator = CommandRegulator(Pk = PD_COEFF_P, Dk = PD_COEFF_D)
    while True:
        extra = ''
        try:
            detections_obj = Detections(-1)
            distance_to_center : float = float('NaN')
            angle_to_target = XY()
            move_command = MoveCommand()

            logger = global_logger
            # logger.debug("!!! awaiting detection... ")
            try:
                # Keep the asyncio loop responsive while waiting for a queue item.
                r : Detections = detections_queue.get(0.02)
                if r is STOP:
                    logger.info("stopping")
                    break
                detections_obj = r

            except Empty:
                # No detections, not even frame with ID and image
                skipped_detetions += 1
                drone.clear_command_history()
                if skipped_detetions > 20 and skipped_detetions < 30:
                    await drone.standstill()
                elif skipped_detetions > 30:
                    await drone.idle()

                if skipped_detetions > 4:
                    logger.warning("No frames (%d times), no detections, input queue empty? prev action: %s", skipped_detetions, drone.last_command())

                continue
            except:
                logger.exception("Serious error getting next detection from a queue", exc_info=True)
                break

            update_timestamps()
            logger = LoggerWithPrefix(logger, prefix=f'frame=#{detections_obj.frame_id:04}')

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

            detections, frame = detections_obj.detections, detections_obj.frame
            detection = None

            # so telemetry action doesn't get into the logs
            drone.clear_command_history()

            # filter out accidential Nones
            detections = [d for d in detections if d is not None]
            # detections.sort(reverse=True, key = lambda d : d.track_id)
            detections.sort(reverse=True, key = lambda d : d.confidence)
            target_relative_pos = None
            target_relative_pos_uncorrected = None
            frame_id = detections_obj.frame_id

            detection = detections[0] if len(detections) > 0 else Detection()
            if detection.confidence >= CONFIDENCE_MIN:
                last_seen_target_at_frame = detections_obj.frame_id
                delay_between_detections_ns = update_timestamps_on_detection()
                estimated_distance_class, estimated_distance_m = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, detection.bbox.size)

                drone_pose = get_pose(telemetry_dict)


                mode = 'follow'

                # distance_to_center = detection.bbox.center.distance_to(center)
                target_size = detection.bbox.area()
                pd_coeff_p = pd_coeff_p_for_target_size(target_size)

                target_relative_pos = AIM_POINT - detection.bbox.center
                logger.debug("!!! target : %s, size: %s, pd_coeff_p: %s", target_relative_pos, target_size, pd_coeff_p)

                # TODO maybe use frame capture time?
                target_estimator.add_target_pos(
                    target_relative_pos,
                    # estimation is too far off, when using frame capture time.
                    current_frame_timestamp_ns #frame_capture_timestampt_ns if frame_capture_timestampt_ns else current_frame_timestamp_ns
                )

                estimate_lookeahead_frames = ESTIMATION_LOOKAHEAD_FRAMES
                if ESTIMATION_LOOKAHEAD_DYNAMIC:
                    distance = estimated_distance_m if estimated_distance_m else 1
                    if estimated_distance_class == DistanceClass.FAR:
                        estimate_lookeahead_frames = ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_FAR + int(math.sqrt(distance))
                    elif estimated_distance_class == DistanceClass.MEDIUM:
                        estimate_lookeahead_frames = ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_MEDIUM + int(math.sqrt(distance))
                    elif estimated_distance_class == DistanceClass.NEAR:
                        estimate_lookeahead_frames = ESTIMATION_LOOKAHEAD_DYNAMIC_FRAMES_NEAR

                estimate_delta_ns = (current_frame_timestamp_ns - prev_frame_timestamp_ns) * estimate_lookeahead_frames
                estimate_at_ns = current_frame_timestamp_ns + estimate_delta_ns
                estimate_mode = ''
                target_relative_pos_old = target_relative_pos

                if ESTIMATION_3D:
                    # --- 3-D world-frame position estimation ---
                    if estimated_distance_m is not None and drone_pose:
                        try:
                            # _quat = get_orientation_quaternion(telemetry_dict)
                            # _drone_pos = get_position_ned(telemetry_dict)
                            target_pos_ned = project_camera_to_ned(
                                detection.bbox.center.x,
                                detection.bbox.center.y,
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

                            estimated_pos = target_estimator_3d.estimate(
                                estimate_at_ns,
                                None,
                                method=ESTIMATION_3D_METHOD
                            )
                            if estimated_pos is None:
                                logger.warning('3D estimation fallback to: %s, target_estimator_3d has %s items',
                                    target_pos_ned, target_estimator_3d.history_size())
                                target_relative_pos = target_relative_pos_old
                            else:
                                # third one is distane, which we don't need
                                estimated_x, estimated_y, _ = project_ned_to_camera(
                                    estimated_pos,
                                    AIM_POINT.x,
                                    AIM_POINT.y,
                                    FRAME_ANGLUAR_SIZE_DEG.x,
                                    FRAME_ANGLUAR_SIZE_DEG.y,
                                    drone_pose.quaternion,
                                    drone_pose.position
                                )
                                target_relative_pos = XY(estimated_x, estimated_y)
                            target_relative_pos = AIM_POINT - target_relative_pos

                            estimate_mode = f'3D={ESTIMATION_3D_METHOD}'
                        except Exception:
                            logger.debug("3D estimation failed", exc_info=True)

                # NOTE: ??? maybe use as fallback if 3d estimation is not available
                else:
                    if target_estimator.history_size() >= 2:
                        # estimate target based on previous positions
                        estimate_mode = '2D'
                        target_relative_pos = target_estimator.estimate_target_pos(estimate_at_ns, target_relative_pos)

                if estimate_mode:
                    mode += f' *{estimate_mode}:{estimate_lookeahead_frames}f '

                    logger.debug("!!! %s estimated new target pos %s (was %s), for +%sms (%d frames)",
                            estimate_mode,
                            target_relative_pos,
                            target_relative_pos_old,
                            estimate_delta_ns / 1000_000,
                            estimate_lookeahead_frames)

                seen_target = True

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

                # Note target_relative_pos is already an offset from AIM_POINT
                distance_to_center = target_relative_pos.distance_to(XY(0, 0))
                thrust = THRUST_MIN
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

                    if THRUST_PROPORTIONAL_TO_TARGET_SIZE:
                        if estimated_distance_m < 7:
                            thrust *= 1.1
                            pd_coeff_p *= 1.1

                            if estimated_distance_m < 5:
                                thrust *= 1.1
                                pd_coeff_p *= 1.1

                            if estimated_distance_m < 3:
                                thrust *= 1.1
                                pd_coeff_p *= 1.1
                                pass

                            extra += f' WE ARE SOOOO CLOSE, BOOSTING thrust to: {thrust}, p to: {pd_coeff_p} '


                logger.info("Setting new command regulator coeffs P=%s D=%s", pd_coeff_p, PD_COEFF_D)
                command_regulator.set_coeffs(Pk = pd_coeff_p, Dk = PD_COEFF_D)
                target_relative_pos_pd = target_relative_pos
                if target_relative_pos is not None:
                    logger.debug("!!! target before PD: %s", target_relative_pos)
                    target_relative_pos_pd = command_regulator.next_command(target_relative_pos, delay_between_detections_ns / 1000_000)
                    logger.debug("!!! target after PD: %s, regulator coeffs: %s", target_relative_pos_pd, command_regulator.get_coeffs())

                angle_to_target  = target_relative_pos_pd.multiplied_by_XY(FRAME_ANGLUAR_SIZE_DEG)
                prev_angle_to_target = angle_to_target

                logger.debug("angle to target: %s", angle_to_target)

                mode += f'size: {target_size:.3}, estimated distance: ({estimated_distance_class} @ {estimated_distance_m:.1f}m), p: {pd_coeff_p * 1.0 : .3} '

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
                if not moving and target_estimator.history_size() < DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES:
                    logger.warning("Delaying takeoff for %s frames (now have %s)", DELAY_TAKEOF_UNTIL_N_DETECTION_FRAMES, target_estimator.history_size())
                    pass
                else:
                    # NOTE: perfroming conversion from camera referene frame to done's FRD
                    await drone.move_to_target_zenith_async(roll_degree=-angle_to_target.x, pitch_degree=angle_to_target.y, thrust=thrust)
                    moving = True

                    if takeoff_time_ns is None:
                        takeoff_time_ns = time.monotonic_ns()
                        # logger.info("!!! IN AIR since: %s", takeoff_time_ns)

            else:
                if abs(frame_id - last_seen_target_at_frame) > TARGET_ESTIMATOR_CLEAR_HISTORY_AFTER_TARGET_LOST_FRAMES and target_estimator.history_size() > 0:
                    logger.warning("!!! CLEARING HISTORY")
                    target_estimator.clear_history()
                    target_estimator_3d.clear_history()

                if seen_target:
                    prev_angle_to_target *= FADE_COEFF
                    # NOTE: perfroming conversion from camera referene frame to done's FRD
                    await drone.move_to_target_zenith_async(roll_degree=-prev_angle_to_target.x, pitch_degree=prev_angle_to_target.y, thrust=thrust)
                    # Just t visualize the point we are moving to
                    target_relative_pos = prev_angle_to_target.divided_by_XY(FRAME_ANGLUAR_SIZE_DEG)
                    # await drone.standstill()
                    moving = False
                    debug_info["mode"] = "hover"
                else:
                    debug_info["mode"] = "idle"
                    if detections_obj.frame_id % 30 == 0:
                        moving = False
                        await drone.idle()

            last_command = drone.last_command() or '<<== NO ==>>'
            debug_info["action"] = last_command
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

            # -1 means that there was no frame and no detections
            if output_queue is not None:
                output = {
                    'detections' : detections_obj,
                    'aim_point'  : aim_point,
                    'selected' : detection,
                    'telemetry': debug_info,
                    'selected_detection_projected_pos' : target_relative_pos_uncorrected,
                    # 'target_pos_ned' : target_estimator_3d.latest,
                    # 'target_pos_ned_estimated' : target_estimator_3d.estimate(current_frame_timestamp_ns),
                    # 'inertia_accumuated' : target_relative_pos,
                    'move_goal' : target_relative_pos
                }
                output_queue.put(output)

        except:
            logging.exception(f"Got exception: %s %s COMMAND: %s", detections_obj, distance_to_center, move_command, exc_info=True)
