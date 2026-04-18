#!/usr/bin/env python

from math import nan
from pathlib import Path

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
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from app_base import GStreamerDetectionApp
from drone_controller import drone_controlling_thread
from platform_controller import platform_controlling_thread

# from mavsdk.telemetry import EulerAngle

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from bytetrack import BYTETracker

from helpers import FrameMetadata, Rect, XY,  Detection, Detections, MoveCommand
from OverwriteQueue import OverwriteQueue
from debug_output import debug_output_thread
from video_sink_gstreamer import RecorderSink
from video_sink_multi import MultiSink
from opencv_show_image_sink import OpenCVShowImageSink


# logging and debugging stuff
from helpers import (
    configure_logging,
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
    def __init__(self, detections_queue, tracker: BYTETracker):
        super().__init__()
        self.detections_queue = detections_queue
        self.tracker = tracker


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

def normalized_frame_id(buffer: Gst.Buffer, frame_meta) -> int:
    """Return a stable per-frame identifier suitable for deduplication.

    Priority:
      1. Picamera2 frame-id reference timestamp meta (appsrc / rpi path)
      2. buffer.offset — libcamerasrc sets this to the frame sequence number
      3. buffer.pts   — always set by libcamerasrc; unique per frame
      4. time.monotonic_ns() — last resort; unique per call, dedup won't fire
    """
    if frame_meta is not None:
        return frame_meta.timestamp

    offset = buffer.offset
    if offset != Gst.BUFFER_OFFSET_NONE:
        return int(offset)

    pts = buffer.pts
    if pts != Gst.CLOCK_TIME_NONE:
        return int(pts)

    return time.monotonic_ns()


seen_frames = deque(maxlen=10)

_MIN_MATCH_IOU = 0.1


def _match_track_to_detection(
    track_det_bbox: np.ndarray, rects: list
) -> int | None:
    """Return index of rect in `rects` with highest IoU against track_det_bbox."""
    best_idx, best_iou = None, _MIN_MATCH_IOU
    x1, y1, x2, y2 = track_det_bbox
    for i, b in enumerate(rects):
        ix1 = max(x1, b.left_edge)
        iy1 = max(y1, b.top_edge)
        ix2 = min(x2, b.right_edge)
        iy2 = min(y2, b.bottom_edge)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (x2 - x1) * (y2 - y1)
        area_b = b.width * b.height
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad: Gst.Pad, info: Gst.PadProbeInfo, user_data : user_app_callback_class):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    sensor_timestamp_ns  = normalized_timestamp(buffer.get_reference_timestamp_meta(sensor_timestamp_caps))
    detection_start_timestamp_ns  = normalized_timestamp(buffer.get_reference_timestamp_meta(unix_timestamp_caps))
    detection_end_timestamp_ns  = time.monotonic_ns()

    frame_id = normalized_frame_id(buffer, buffer.get_reference_timestamp_meta(frame_id_caps))

    # Picamera2 metadata absent when using libcamerasrc; fall back to wall-clock time
    if detection_start_timestamp_ns == 0:
        detection_start_timestamp_ns = detection_end_timestamp_ns
    if sensor_timestamp_ns == 0:
        sensor_timestamp_ns = detection_start_timestamp_ns

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

    # Extract raw detection data before constructing frozen Detection objects
    raw_dets = []
    for detection in detections:
        bbox = detection.get_bbox()
        raw_dets.append((
            Rect.from_xyxy(bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()),
            detection.get_confidence(),
        ))

    # Run ByteTracker to assign stable track IDs
    if raw_dets:
        dets_array = np.array([
            [r.left_edge, r.top_edge, r.right_edge, r.bottom_edge, c]
            for r, c in raw_dets
        ])
    else:
        dets_array = np.empty((0, 5))

    # BYTETracker is not thread-safe; safe here because GStreamer uses a single streaming thread.
    active_tracks = user_data.tracker.update(dets_array, frame_id)

    # Build index → track_id map before constructing Detection objects
    track_id_map: dict[int, int] = {}
    temp_rects = [r for r, _ in raw_dets]
    for track in active_tracks:
        idx = _match_track_to_detection(track.det_bbox, temp_rects)
        if idx is not None:
            track_id_map[idx] = track.track_id

    # Construct immutable Detection objects with track_id set at creation time
    detections_list = [
        Detection(
            bbox=rect,
            confidence=conf,
            track_id=track_id_map.get(i),
        )
        for i, (rect, conf) in enumerate(raw_dets)
    ]

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

    return Gst.PadProbeReturn.OK


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

        video_output_chunk_length_ns = self.video_output_chunk_length_s * 1000_000_000
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

    start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    event = threading.Event()

    arg_parser = get_default_parser()
    arg_parser.add_argument('--action', type=str, choices=["platform", "drone"])

    control_config = {
        'confidence_min': 0.4,
        'confidence_move': 0.3,

        'thrust_takeoff' : 0.5,
        'thrust_min': 0.7,
        'thrust_max': 0.9,
        'thrust_dynamic': False,
        'thrust_proportional_to_target_size' : False,

        'target_lost_fade_per_frame': 0.99,
        'target_estimator_clear_history_after_target_lost_frames' : 3,

        'estimation_3d': True,
        'estimation_3d_mode': 'cluster',
        'estimation_3d_use_initial_velocity' : True,

        'estimation_lookahead_frames': 2,
        'estimation_lookahead_dynamic': True,
        'estimation_lookahead_dynamic_sqrt': True,
        'estimation_lookahead_dynamic_frames_near':   1,
        'estimation_lookahead_dynamic_frames_medium': 1,
        'estimation_lookahead_dynamic_frames_far':    1, # can't be too big -- estimation will be too FAAR away.

        'pd_coeff_p': 3,
        'pd_coeff_d': 0, #-1, # -1
        'pd_coeff_p_safe_min': 0.6,
        'pd_coeff_p_min' : 0.5,
        'pd_coeff_p_max' : 10,

        # Dynamically adjust P coeff based on target size.
        # Old mode: linear interpolation between min and max.
        # New mode: piecewise profile controlled by stage thresholds and ratios.
        'pd_coeff_p_dynamic': False,
        'pd_coeff_p_dynamic_use_piecewise': False,
        'pd_coeff_p_dynamic_min_target_size' : 0.0005, # normalized target size w * h, where both w and are in range (0..1)

        'pd_coeff_p_dynamic_min' : 0.6,
        'pd_coeff_p_dynamic_max_target_size' : 0.0120,  # normalized target size
        'pd_coeff_p_dynamic_max' : 6,

        'pd_coeff_p_dynamic_stage_1_threshold': 0.01,
        'pd_coeff_p_dynamic_stage_2_threshold': 0.05,
        'pd_coeff_p_dynamic_stage_1_ratio': 1,
        'pd_coeff_p_dynamic_stage_2_ratio': 1,
        'pd_coeff_p_dynamic_stage_3_ratio': 1,

        'frame_angular_size_deg' : XY(107, 85),

        # 'target_size_m' : XY(0.2, 0.2),             # baloon
        'target_size_m' : XY(1.8, 1.8),             # shahed small
        # 'target_size_m' : XY(3.5, 2.5),             # shahed large
        # 'target_size_m' : XY(1_000_000, 1_000_000), # SUN

        'inertia_correction_gain' : 0, #-0.02, # 0.01 #, 1.0, etc
        'inertia_correction_limits': XY(1, 1),
        'inertia_correction_min_speed_ms': 5,

        'safe_takeoff_period_ns': 300_000_000,
        'delay_takeof_until_n_detection_frames' : 30,

        'aim_point': XY(0.5, 0.5),
        'aim_point_max_offset': XY(0.5, 0.6),

        'follow_target_position_ned' : False,

        # params to go to the drone config ("drone_" prefix is stripped then)
        'drone_use_set_attitude': False,
        'drone_min_lift_fraction': 0.1,
        'drone_lift_velocity_headroom_ms': 3.0, # upward velocity when tilt angle restirctions are relaxed significantly
        'drone_lift_accel_headroom_mss': 5.0, # upward acceleration when tilt angle restirctions are relaxed significantly

        'DEBUG': DEBUG,

        'bytetrack_track_thresh': 0.5,
        'bytetrack_det_thresh':   0.6,
        'bytetrack_match_thresh': 0.8,
        'bytetrack_track_buffer': 30,
        'bytetrack_frame_rate':   30,
    }

    bytetracker = BYTETracker(
        track_thresh=control_config['bytetrack_track_thresh'],
        det_thresh=control_config['bytetrack_det_thresh'],
        match_thresh=control_config['bytetrack_match_thresh'],
        track_buffer=control_config['bytetrack_track_buffer'],
        frame_rate=control_config['bytetrack_frame_rate'],
    )
    user_data = user_app_callback_class(detections_queue, bytetracker)
    user_data.use_frame = True

    app = App(
        app_callback,
        user_data,
        parser=arg_parser,
        video_output_chunk_length_s=10,
        video_output_path='./_DEBUG',
        video_filename_base=f"RAW_{start_time_str}",
        record_videos=True)

    logger.info("!!! Config: %s", control_config)
    if DEBUG:
        import math
        nan = math.nan
        control_config['debug_telemetry_dict'] = {'attitude_euler': {'pitch_deg': 1.1012928485870361, 'roll_deg': -2.5803990364074707, 'timestamp_us': 4597491000, 'yaw_deg': -139.22280883789062}, 'odometry': {'angular_velocity_body': {'pitch_rad_s': 0.005480111576616764, 'roll_rad_s': -0.004354139324277639, 'yaw_rad_s': 0.00451350212097168}, 'child_frame_id': '1 (BODY_NED)', 'frame_id': '1 (BODY_NED)', 'pose_covariance': {'covariance_matrix': (0.0006513984990306199, nan, nan, nan, nan, nan, 0.0006680359947495162, nan, nan, nan, nan, 0.0795387253165245, nan, nan, nan, 0.00014700590691063553, nan, nan, 0.0001532444730401039, nan, 0.0037478541489690542)}, 'position_body': {'x_m': 10155.28515625, 'y_m': 1922.0908203125, 'z_m': 0.21114209294319153}, 'q': {'timestamp_us': 0, 'w': -0.3483775854110718, 'x': -0.0011684682685881853, 'y': -0.024444160982966423, 'z': 0.9370348453521729}, 'time_usec': 4597481426, 'velocity_body': {'x_m_s': 0.014301709830760956, 'y_m_s': -0.007258167490363121, 'z_m_s': 0.04777885600924492}, 'velocity_covariance': {'covariance_matrix': (0.002916615456342697, nan, nan, nan, nan, nan, 0.0030234858859330416, nan, nan, nan, nan, 0.005942340008914471, nan, nan, nan, nan, nan, nan, nan, nan, nan)}}, 'landed_state': None, 'imu': {'acceleration_frd': {'down_m_s2': -10.367236137390137, 'forward_m_s2': -0.11148512363433838, 'right_m_s2': 0.5069471597671509}, 'angular_velocity_frd': {'down_rad_s': -0.0006937360158190131, 'forward_rad_s': 0.004322248511016369, 'right_rad_s': 0.0015081189339980483}, 'magnetic_field_frd': {'down_gauss': 0.2559056580066681, 'forward_gauss': -0.3339233696460724, 'right_gauss': 0.3074171245098114}, 'temperature_degc': 15.0, 'timestamp_us': 4597496423}}

    action_thread = None
    if app.options_menu.action == 'platform':
        action_thread = threading.Thread(
            target = platform_controlling_thread,
            args = (
                '/dev/ttyUSB0',
                dict(
                    speed_adjustments=XY(1, -1),
                    # speed=0, #
                    # acceleration=0
                    ),
                detections_queue),
            kwargs = dict(
                control_config= control_config,
                output_queue= output_queue,
                signal_event_when_ready= event,
            )
        )
    else:

        action_thread = threading.Thread(
            target = drone_controlling_thread,
            args = (
                'udp://:14550',
                {
                    'upside_down_angle_deg': 130,
                    'upside_down_hold_s': 0.2,
                },
                detections_queue
            ),
            kwargs= dict(
                control_config= control_config,
                output_queue= output_queue,
                signal_event_when_ready= event,
            ),
            name = "Drone"
        )
    action_thread.start()

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

    # if DEBUG:
    #     for i in range(3):
    #         detections_queue.put(
    #             Detections(-1,
    #                 frame = None,
    #                 detections = [
    #                     Detection(
    #                         bbox = Rect.from_xyxy(0.1, 0.1, 0.2, 0.2),
    #                         confidence = 0.1,
    #                         track_id = 1
    #                     ),
    #                     Detection(
    #                         bbox = Rect.from_xyxy(0.1, 0.1, 0.2, 0.2),
    #                         confidence = 0.9,
    #                         track_id = 2
    #                     ),
    #                     Detection(
    #                         bbox = Rect.from_xyxy(0.1, 0.1, 0.2, 0.2),
    #                         confidence = 0.7,
    #                         track_id = 3
    #                     ),
    #                 ],
    #             )
    #         )

    app.run(event)
    print("Done !!!")
    detections_queue.put(STOP)
    action_thread.join()

if __name__ == "__main__":
    main()
