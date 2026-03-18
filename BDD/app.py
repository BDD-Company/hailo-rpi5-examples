#!/usr/bin/env python

from math import nan
from pathlib import Path
import asyncio

from queue import Empty, Queue
from dataclasses import dataclass, field
from collections import deque
import threading

import os
import sys
import datetime
import time

import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from app_base import GStreamerDetectionApp

# from mavsdk.telemetry import EulerAngle

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from helpers import FrameMetadata, Rect, XY,  Detection, Detections, MoveCommand
from CommandRegulator import CommandRegulator
from OverwriteQueue import OverwriteQueue
from TargetEstimator import TargetEstimator
from estimate_distance import estimate_distance_class, DistanceClass
from debug_output import debug_output_thread
from video_sink_gstreamer import RecorderSink
from video_sink_multi import MultiSink
from opencv_show_image_sink import OpenCVShowImageSink
from drone_killswitch import kill_on_rc_switch_on_channel


# logging and debugging stuff
from helpers import (
    configure_logging,
    debug_collect_call_info,
    LoggerWithPrefix
)
import logging
logger = logging.getLogger(__name__)
global_logger = logger # a hack
DEBUG = False

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self, detections_queue):
        super().__init__()
        self.detections_queue = detections_queue


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

sensor_timestamp_caps = Gst.Caps.from_string("timestamp/x-picamera2-sensor")
unix_timestamp_caps = Gst.Caps.from_string("timestamp/x-unix")
frame_id_caps = Gst.Caps.from_string("frame-id/x-picamera2")


def normalized_timestamp(ts):
    if ts is not None:
        if isinstance(ts, Gst.ReferenceTimestampMeta):
            return ts.timestamp
        else:
            return int(ts)
    else:
        return 0

seen_frames = deque(maxlen=10)

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad: Gst.Pad, info: Gst.PadProbeInfo, user_data : user_app_callback_class):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    string_to_print = f"" #Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    sensor_timestamp_ns  = normalized_timestamp(buffer.get_reference_timestamp_meta(sensor_timestamp_caps))
    detection_start_timestamp_ns  = normalized_timestamp(buffer.get_reference_timestamp_meta(unix_timestamp_caps))
    detection_end_timestamp_ns  = time.monotonic_ns()

    frame_id = buffer.get_reference_timestamp_meta(frame_id_caps)
    if frame_id is not None:
        frame_id = frame_id.timestamp
    else:
        frame_id = time.monotonic_ns()

    if frame_id in seen_frames:
        # logger.warning("!!!!!!!!!!!! Skipped duplicated frame %s", frame_id)
        return Gst.PadProbeReturn.OK
    seen_frames.append(frame_id)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    # if user_data.use_frame and format is not None and width is not None and height is not None:
    #     # Get video frame
    frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # logger.debug("frame #%d \t pipeline delay: %sms \t detections %s (%s), frame object: %s (%s)",
    #         frame_id,
    #         (detection_end_timestamp_ns - detection_start_timestamp_ns)/1000000,
    #         len(detections),
    #         detections,
    #         id(frame), hash(frame.data.tobytes())
    # )

    # Parse the detections
    detection_count = 0
    detections_list = []
    for detection in detections:
        detection : hailo.HailoDetection = detection

        # DEBUG_dump('detecion: ', detection)

        # label = detection.get_label()
        bbox = detection.get_bbox()

        confidence = detection.get_confidence()
        # if label == "person":
        # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        else:
            track_id = None

        detection_count += 1
        detections_list.append(Detection(
            bbox = Rect.from_xyxy(bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()),
            confidence = confidence,
            track_id = track_id,
        ))

    # if len(detections) != 0:
    user_data.detections_queue.put(
        Detections(
            frame_id,
            frame,
            detections_list,
            meta = FrameMetadata(
                capture_timestamp_ns=sensor_timestamp_ns,
                detection_start_timestamp_ns = detection_start_timestamp_ns,
                detection_end_timestamp_ns=detection_end_timestamp_ns)
        )
    )

    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK


class StopSignal:
    pass
STOP = StopSignal()


def drone_controlling_tread(*args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(drone_controlling_tread_async(*args, **kwargs))
    except Exception as e:
        logger.error("in drone event loop", exc_info=True, stack_info=True)
    finally:
        loop.close()

def get_position_from_telemetry(telemetry_dict) -> XY:
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
            and abs(velocity["y_m_s"]) > 0.01


async def drone_controlling_tread_async(drone_connection_string, drone_config, detections_queue, control_config = {}, output_queue = None, signal_event_when_ready = None):
    from math import radians

    # will owerwrite logger here many times, make sure that rest of the systems are not affected
    global global_logger
    logger = global_logger

    MIN_CONFIDENCE  = control_config.get('confidence_min', 0.1)
    # MOVE_CONFIDENCE = control_config.get('confidence_move', 0.4)
    MAX_THRUST      = control_config.get('thrust_max', 0.5)
    MIN_THRUST      = control_config.get('thrust_min', 0.4)
    FADE_COEFF      = control_config.get('target_lost_fade_per_frame', 0.9)

    PD_COEFF_P      = control_config.get('pd_coeff_p', 12)
    PD_COEFF_D      = control_config.get('pd_coeff_d', 1)

    PD_COEFF_P_DYNAMIC = control_config.get('pd_coeff_p_dynamic', False)
    PD_COEFF_P_MIN_TARGET_SIZE = control_config.get('pd_coeff_p_dynamic_min_target_size', 0.003)
    PD_COEFF_P_MAX_TARGET_SIZE = control_config.get('pd_coeff_p_dynamic_max_target_size', 0.005)
    PD_COEFF_P_MIN  = control_config.get('pd_coeff_p_dynamic_min', 0.5)
    PD_COEFF_P_MAX  = control_config.get('pd_coeff_p_dynamic_max', 2)

    TARGET_SIZE_M = control_config.get('target_size_m', XY(1, 0.5))
    FRAME_ANGLUAR_SIZE_DEG = control_config.get('frame_angular_size_deg', XY(120, 90))

    SAFE_TAKEOFF_PERIOD_NS = control_config.get('safe_takeoff_period_ns', 300_000_000)

    center = XY(0.5, 0.5)
    distance_r = 0.1
    distance_r *= distance_r
    seen_target = False

    from drone import DroneMover, is_in_air
    drone = DroneMover(drone_connection_string, drone_config)
    logger.debug("starting up drone...")

    udp_port = 14560
    killdrone_thread = threading.Thread(
        target = kill_on_rc_switch_on_channel,
        args = (udp_port, 6, drone)
    )
    killdrone_thread.start()

    global DEBUG
    if DEBUG:
        await drone.startup_sequence(1, force_arm=True)
    else:
        await drone.startup_sequence(100)

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
    target_estimator = TargetEstimator(max_target_position_age_nanoseconds=500_000_000_000)

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

    def pd_coeff_p_for_target_size(target_size):
        if not PD_COEFF_P_DYNAMIC:
            return PD_COEFF_P

        # avoid tipping over on hallucinations while close to the ground
        if flight_time_ns <= SAFE_TAKEOFF_PERIOD_NS:
            return PD_COEFF_P_MIN

        min_target_size = PD_COEFF_P_MIN_TARGET_SIZE
        max_target_size = PD_COEFF_P_MAX_TARGET_SIZE
        min_pd_coeff_p = PD_COEFF_P_MIN
        max_pd_coeff_p = PD_COEFF_P_MAX

        target_size_ratio = (target_size - min_target_size) / (max_target_size - min_target_size)
        result = min_pd_coeff_p + target_size_ratio * (max_pd_coeff_p - min_pd_coeff_p)

        deviance_coeff = 1 # could be 2
        return min(PD_COEFF_P_MAX * deviance_coeff, max(PD_COEFF_P_MIN / deviance_coeff, result))


    command_regulator = CommandRegulator(Pk = PD_COEFF_P, Dk = PD_COEFF_D)

    while True:
        try:
            detections_obj = Detections(-1)
            distance_to_center : float = float('NaN')
            angle_to_target = XY()
            forward_speed = 0
            move_command = MoveCommand()

            logger = global_logger
            # logger.debug("!!! awaiting detection... ")
            try:
                # Keep the asyncio loop responsive while waiting for a queue item.
                r : Detections = await asyncio.to_thread(detections_queue.get, 0.1)
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

                logger.warning("No frames (%d times), no detections, input queue empty? ACTION: %s", skipped_detetions, drone.last_command())
                continue
            except:
                logger.exception("Serious error getting next detection from a queue", exc_info=True)
                break

            update_timestamps()
            logger.debug("!!!")
            logger = LoggerWithPrefix(logger, prefix=f'frame=#{detections_obj.frame_id:04}')

            if DEBUG:
                # NOTE: injecting fake detections to debug
                import math
                __tmp_delta_confidence = math.sin(detections_obj.frame_id / 100) / 10
                __tmp_delta_x = math.sin(detections_obj.frame_id / 100) / 4
                __tmp_delta_y = math.cos(detections_obj.frame_id / 100) / 4
                detections_obj.detections.append(
                            Detection(
                                bbox = Rect.from_xywh(0.2 + __tmp_delta_x, 0.2 + __tmp_delta_y, 0.05, 0.05),
                                confidence = 0.3 + __tmp_delta_confidence,
                                track_id = 1
                            )
                )

            logger.debug("!!! GOT DETECTIONS, objects detected: %s (%s), detection delay: %sms, total delay: %sms",
                    len(detections_obj.detections),
                    detections_obj.detections,
                    (detections_obj.meta.detection_end_timestamp_ns - detections_obj.meta.detection_start_timestamp_ns) / 1000_000,
                    (detections_obj.meta.detection_end_timestamp_ns - detections_obj.meta.capture_timestamp_ns) / 1000_000
                )
            skipped_detetions = 0
            frame_capture_timestampt_ns = detections_obj.meta.capture_timestamp_ns or None

            # telemetry_dict = await drone.get_telemetry_dict()
            # logger.debug("telemetry: %s", telemetry_dict)
            debug_info = telemetry_dict

            ## Check if take off
            if takeoff_time_ns == None:
                if moving:
                    takeoff_time_ns = time.monotonic_ns()
                    # logger.info("!!! IN AIR: %s", takeoff_time_ns)
            else:
                flight_time_ns = time.monotonic_ns() - takeoff_time_ns
                # logger.info("!!! flight time: %ss", flight_time_ns / 1000_000_000)

            detections, frame = detections_obj.detections, detections_obj.frame
            detection = None

            current_attitude = await drone.get_cached_attitude(wait_for_first=False)
            # so telemetry action doesn't get into the logs
            drone.clear_command_history()

            # filter out accidential Nones
            detections = [d for d in detections if d is not None]
            # detections.sort(reverse=True, key = lambda d : d.track_id)
            detections.sort(reverse=True, key = lambda d : d.confidence)
            target_relative_pos = None

            detection = detections[0] if len(detections) > 0 else Detection()
            if detection.confidence >= MIN_CONFIDENCE:
                delay_between_detections_ns = update_timestamps_on_detection()
                estimated_distance = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, detection.bbox.size)
                # logger.debug("!!! Detection: %s", detection)

                # drone_attitude = telemetry_dict.get('attitude_euler', 0)
                # drone_pitch = drone_attitude['pitch_deg']
                # drone_roll = drone_attitude['roll_deg']
                # logger.debug("drone attitude: %s", drone_attitude)
                mode = 'follow'

                distance_to_center = detection.bbox.center.distance_to(center)
                target_size = detection.bbox.area()
                pd_coeff_p = pd_coeff_p_for_target_size(target_size)

                target_relative_pos = center - detection.bbox.center
                logger.debug("!!! target : %s, size: %s, pd_coeff_p: %s", target_relative_pos, target_size, pd_coeff_p)

                # TODO maybe use frame capture time?
                target_estimator.add_target_pos(
                    target_relative_pos,
                    # estimation is too fra off, when using frame capture time.
                    current_frame_timestamp_ns #frame_capture_timestampt_ns if frame_capture_timestampt_ns else current_frame_timestamp_ns
                )

                if True and target_estimator.history_size() >= 2:
                    # estimate target based on previous positions
                    mode = 'follow* '

                    number_of_frames_to_estimate_pos = 2
                    # if estimated_distance is not None:
                    #     estimated_distance_class, estimated_distance_meters = estimated_distance
                    #     if estimated_distance_class == DistanceClass.FAR:
                    #         number_of_frames_to_estimate_pos = 10
                    #     elif estimated_distance_class == DistanceClass.MEDIUM:
                    #         number_of_frames_to_estimate_pos = 5
                    #     elif estimated_distance_class == DistanceClass.NEAR:
                    #         number_of_frames_to_estimate_pos = 2

                    estimation_delta_ns = (current_frame_timestamp_ns - prev_frame_timestamp_ns) * number_of_frames_to_estimate_pos
                    target_relative_pos_old = target_relative_pos
                    target_relative_pos = target_estimator.estimate_target_pos(current_frame_timestamp_ns + estimation_delta_ns, target_relative_pos)
                    logger.debug("!!! estimated new target pos %s (was %s), for +%sms (%d frames)",
                            target_relative_pos,
                            target_relative_pos_old,
                            estimation_delta_ns / 1000_1000,
                            number_of_frames_to_estimate_pos)

                odometry = telemetry_dict.get('odometry', {}) or {}
                flight_altitude = -1 * odometry.get('position_body', {}).get("z_m", 0)

                # max_angle_divisor = 4
                # # # Adjusting how much drone can pitch or roll based on distance to target
                # if flight_altitude > 4: # detection.bbox.width > 0.3 or detection.bbox.height > 0.3:
                #     # Drone is close
                #     max_angle_divisor = 1
                #     mode += " FLIGHT  "
                # elif flight_altitude > 3: #detection.bbox.width > 0.15 or detection.bbox.height > 0.15:
                #     # Drone is mid-range
                #     max_angle_divisor = 2
                #     mode += " SPEEDUP "
                # else:
                #     # Drone is far
                #     max_angle_divisor = 4
                #     mode += " TAKEOFF "

                # if True: # move sideways more
                #     roll_pitch_adjust = XY(
                #         0.75, # roll
                #         1.5   # pitch
                #     )
                #     angle_to_target = angle_to_target.multiplied_by_XY(roll_pitch_adjust)
                #     logger.debug("angle to target adjusted: %s", angle_to_target)


                # # logger.debug('!!!! max_angle_divisor: %s', max_angle_divisor)
                # logger.debug("angle to target adjusted for mode: %s", angle_to_target)

                seen_target = True

                thrust = MIN_THRUST
                if distance_to_center < 0.2:
                    thrust= MAX_THRUST
                    mode += " GREEN "
                    # pd_coeff_p /= 3
                elif distance_to_center < 0.4:
                    thrust= MAX_THRUST + (MAX_THRUST - MIN_THRUST) / 2
                    mode += " YELLOW "
                    # pd_coeff_p /= 1.5
                else:
                    thrust= MIN_THRUST
                    mode += " RED "
                    #pd_coeff_p

                # command_regulator.set_coeffs(Pk = pd_coeff_p, Dk = PD_COEFF_D)
                target_relative_pos_pd = target_relative_pos
                if target_relative_pos is not None:
                    logger.debug("!!! target before PD: %s", target_relative_pos)
                    target_relative_pos_pd = command_regulator.next_command(target_relative_pos, delay_between_detections_ns / 1000_000)
                    logger.debug("!!! target after PD: %s", target_relative_pos)

                angle_to_target  = target_relative_pos_pd.multiplied_by_XY(FRAME_ANGLUAR_SIZE_DEG)
                prev_angle_to_target = angle_to_target

                logger.debug("angle to target: %s", angle_to_target)

                mode += f'size: {target_size:.3}, estimated distance: {estimated_distance}, p: {pd_coeff_p * 1.0 : .3} '


                # while still taking off, avoid dangerous moves
                if flight_time_ns < SAFE_TAKEOFF_PERIOD_NS:
                    MAX_CLOSE_TO_GROUND_ANGLES = XY(60, 60)
                    new_angle_to_target = angle_to_target
                    if abs(angle_to_target.x) > MAX_CLOSE_TO_GROUND_ANGLES.x:
                        sign = angle_to_target.x / abs(angle_to_target.x)
                        new_angle_to_target.x = min(MAX_CLOSE_TO_GROUND_ANGLES.x, abs(angle_to_target.x)) * sign

                    if abs(angle_to_target.y) > MAX_CLOSE_TO_GROUND_ANGLES.y:
                        sign = angle_to_target.y / abs(angle_to_target.y)
                        new_angle_to_target.y = min(MAX_CLOSE_TO_GROUND_ANGLES.y, abs(angle_to_target.y)) * sign

                    if new_angle_to_target != angle_to_target:
                        logger.warning("Too steep atack close to the ground %s, clamping to %s ", angle_to_target, new_angle_to_target)
                        angle_to_target = new_angle_to_target

                # await drone.move_to_target_zenith_async(roll_degree=-45, pitch_degree=0, thrust=thrust)
                await drone.move_to_target_zenith_async(roll_degree=-angle_to_target.x, pitch_degree=angle_to_target.y, thrust=thrust)
                debug_info["mode"] = mode

                moving = True
                if takeoff_time_ns == None:
                    takeoff_time_ns = time.monotonic_ns()
                    # logger.info("!!! IN AIR since: %s", takeoff_time_ns)

            else:
                if seen_target:
                    prev_angle_to_target *= FADE_COEFF
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

            # -1 means that there was no frame and no detections
            if output_queue is not None:
                output = {
                    'detections' : detections_obj,
                    'selected' : detection,
                    # 'move_command': move_command,
                    'telemetry': debug_info,
                    'target' : target_relative_pos, #detection.bbox.center + XY(0.1, 0.1) if detection else XY(), # target_relative_pos
                }
                output_queue.put(output)

        except:
            logging.exception(f"Got exception: %s %s COMMAND: %s", detections_obj, distance_to_center, move_command, exc_info=True)


class App(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data, parser=None, video_output_path = None, video_output_chunk_length_s = 30, video_filename_base=None, record_videos=True):
        self.video_output_directory = video_output_path or '.'
        self.video_output_chunk_length_s = video_output_chunk_length_s or 30
        self.video_filename_base = video_filename_base
        self.record_videos = record_videos
        super().__init__(app_callback, user_data, parser)

        #NOTE: unfortunatelly that has to be string, rest of the HAILO python code depends on it
        self.sync = 'false'


    def get_output_pipeline_string(self, video_sink: str, sync: str = 'true', show_fps: str = 'true'):
        if not self.record_videos:
            return "fakesink"

        record_start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        video_file_name = Path(self.video_filename_base if self.video_filename_base else f"RAW_{record_start_time_str}.mkv")

        # add "_%05d" so we get multiple files w/o overwriting anything
        video_file_name = video_file_name.stem + "_%05d" + (video_file_name.suffix if video_file_name.suffix else '.mkv')

        video_output_chunk_length_ns = self.video_output_chunk_length_s * 1000 * 1000 * 1000
        return f'''
            videoconvert \
            ! x264enc \
                key-int-max=30 \
                bframes=0 \
                tune=zerolatency \
                speed-preset=ultrafast \
            ! h264parse config-interval=1 \
            ! queue name=raw_video_output_queue \
                leaky=downstream \
                max-size-buffers=300 \
                max-size-bytes=0 \
                max-size-time=10000000000 \
            ! splitmuxsink \
                muxer-factory=matroskamux \
                muxer-properties="properties,streamable=true" \
                max-size-time={video_output_chunk_length_ns} async-finalize=true \
                location="{self.video_output_directory}/{video_file_name}"
        '''

    def run(self, wait_event_before_starting=None):
        if wait_event_before_starting:
            wait_event_before_starting.wait()
        logger.info("!!! Starting the application (and generating frames with detections)")

        super().run()


def main():
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # Create an instance of the user app callback class

    configure_logging(level = logging.DEBUG)
    # shushing verbose loggers
    logging.getLogger("picamera2").setLevel(logging.WARNING)
    logging.getLogger("mavsdk_server").setLevel(logging.ERROR)

    global DEBUG
    if "--DEBUG" in sys.argv:
        DEBUG=True
        sys.argv.remove('--DEBUG')

    if DEBUG:
        logger.error('')
        logger.error("!!! ============================================================== !!!")
        logger.error("!!! Will run in DEBUG mode, behaviour might differ from production !!!")
        logger.error("!!! ============================================================== !!!")
        logger.error('')

    detections_queue = OverwriteQueue(maxsize=20)
    output_queue = OverwriteQueue(maxsize=200)

    drone_config = {}

    start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    event = threading.Event()
    user_data = user_app_callback_class(detections_queue)
    user_data.use_frame = True
    app = App(
        app_callback,
        user_data,
        video_output_chunk_length_s=10,
        video_output_path='./_DEBUG',
        video_filename_base=f"RAW_{start_time_str}",
        record_videos=True)

    control_config = {
        'confidence_min': 0.2,
        'confidence_move': 0.4,
        'thrust_max': 0.5,
        'thrust_min': 0.4,

        'target_lost_fade_per_frame': 0.75,

        'pd_coeff_p': 3, #12.5
        'pd_coeff_d': 0,

        # Dynamically adjust P coeff based on target size.
        # P is approximated linearly between min and max, NEVER exceeding min nor max
        'pd_coeff_p_dynamic': False,
        'pd_coeff_p_dynamic_min_target_size' : 0.0005, # normalized target size w * h, where both w and are in range (0..1)
        'pd_coeff_p_dynamic_min' : 0.6,
        'pd_coeff_p_dynamic_max_target_size' : 0.001,  # normalized target size
        'pd_coeff_p_dynamic_max' : 4,

        'target_size_m' : XY(0.2, 0.2),

        'safe_takeoff_period_ns': 200_000_000
    }

    drone_thread = threading.Thread(
        target = drone_controlling_tread,
        args = ('udp://:14550', drone_config, detections_queue),
        kwargs= dict(
            control_config= control_config,
            output_queue= output_queue,
            signal_event_when_ready= event,
        ),
        name = "Drone"
    )
    drone_thread.start()

    sink = MultiSink([
        # RtspStreamerSink(30, 8554),
        RecorderSink(30,
            "./_DEBUG",
            segment_seconds=10,
            filename_base=f"debug_{start_time_str}",
        ),
        # OpenCVShowImageSink(window_title='DEBUG IMAGE')
    ])

    output_thread = threading.Thread(
        target = debug_output_thread,
        args = (output_queue, sink),
        name="DEBUG"
    )
    output_thread.start()

    if DEBUG:
        for i in range(3):
            detections_queue.put(
                Detections(-1,
                    frame = None,
                    detections = [
                        Detection(
                            bbox = Rect.from_xyxy(0.1, 0.1, 0.2, 0.2),
                            confidence = 0.1,
                            track_id = 1
                        ),
                        Detection(
                            bbox = Rect.from_xyxy(0.1, 0.1, 0.2, 0.2),
                            confidence = 0.9,
                            track_id = 2
                        ),
                        Detection(
                            bbox = Rect.from_xyxy(0.1, 0.1, 0.2, 0.2),
                            confidence = 0.7,
                            track_id = 3
                        ),
                    ],
                )
            )

    app.run(event)
    print("Done !!!")
    detections_queue.put(STOP)
    drone_thread.join()

if __name__ == "__main__":
    main()
