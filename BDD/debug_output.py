#! /usr/bin/env python

import subprocess
import time
from queue import Queue
import json
import pprint
import dataclasses

import cv2
import numpy as np

from helpers import Detections, Detection, XY, MoveCommand, Rect
from interfaces import FrameSinkInterface

import logging
logger = logging.getLogger(__name__)


DETECTED_OBJECT_COLOR = (255, 0, 0)    # blue
SELECTED_OBJECT_COLOR = (255, 0, 255)  # magenta
NEUTRAL_RECT_COLOR    = (0, 255, 0)    # green
CROSSHAIR_COLOR = SELECTED_OBJECT_COLOR
SPEED_COLOR           = SELECTED_OBJECT_COLOR #(255, 255, 0) # cyan
FRAME_METADATA_COLOR = (0, 255, 0)    # green
FRAME_METADATA_COLOR_BG = (80, 80, 80) #(120, 120, 0)
MOVE_COMMAND_COLOR = (0, 0, 255) # red


def change_color(color, diff = (0, 0, 0), factor : float = 1):
    assert len(color) == len(diff)

    def normalize(c):
        return max(0, min(255, int(c)))

    return tuple(normalize((c + d) * factor) for c, d in zip(color, diff))


def shadow_color(color):
    diff = -100
    if sum(color) / len(color) < 128:
        # dark color, make shadow light
        diff *= -1

    return change_color(color, diff=(diff,)*3)

def draw_rect(frame, rect: Rect, color, line_thickness = 1):
    frame_size = XY(frame.shape[1], frame.shape[0])

    cv2.rectangle(
        frame,
        rect.min_point.to_tuple(to = int),
        rect.max_point.to_tuple(to = int),
        color,
        line_thickness
    )


def draw_text(frame : np.ndarray, text : str, pos : XY, font = cv2.FONT_HERSHEY_SIMPLEX, font_scale : float = 1, color = (255, 255, 255), bg_color = None, line_width = 1):
    if isinstance(pos, XY):
        frame_size = XY(frame.shape[1], frame.shape[0])
        pos_tuple = pos.to_tuple(to = int)
    else:
        pos_tuple = pos

    if bg_color is not None:
        bg_line_width_diff = 4 * font_scale
        if bg_line_width_diff > 4:
            bg_line_width_diff = 4
        if bg_line_width_diff < 1:
            bg_line_width_diff = 1
        bg_line_width = int(line_width + bg_line_width_diff)

        cv2.putText(frame,
            text,
            pos_tuple,
            font,
            font_scale,
            bg_color,
            bg_line_width,
            cv2.LINE_AA)

    cv2.putText(frame,
        text,
        pos_tuple,
        font,
        font_scale,
        color,
        line_width,
        cv2.LINE_AA)

def filterdict(dirty: dict, filter_func) -> dict:
    clean = {}
    for k, v in dirty.items():
        if filter_func(k, v):
            if isinstance(v, dict):
                clean.update({k: filterdict(v, filter_func)})
            else:
                clean.update({k: v})

    return clean

# Based on  - https://stackoverflow.com/a/44356856
# Retrieved 2026-02-11, License - CC BY-SA 3.0
class FormatPrinter(pprint.PrettyPrinter):

    def __init__(self, formats, **kwrags):
        super().__init__(**kwrags)
        self.formats = formats

    def format(self, object, context, maxlevels, level):
        if type(object) in self.formats:
            return self.formats[type(object)] % object, True, False

        return super().format(object, context, maxlevels, level)

# ceanup_json_re= re.compile(r'\s*[{}],?|"')
def make_debug_info_dict(frame_id, telemetry_dict, frame_metadata):
    result = dict(**telemetry_dict)
    # remove 'covariance_matrix' which is too verbose
    result = filterdict(result, lambda k, v :  k != 'covariance_matrix')
    # remove keys with empty values entirely
    result = filterdict(result, lambda k, v :  not (hasattr(v, '__len__') and len(v) == 0))
    # remove various timestamps
    result = filterdict(result, lambda k, v :  not (isinstance(k, str) and 'time' in k))
    result['frame'] = frame_id

    detection_delay = (frame_metadata.detection_end_timestamp_ns - frame_metadata.detection_start_timestamp_ns) / 1000000
    result['time'] = {
        'captured at': frame_metadata.capture_timestamp_ns,
        'detection delay': detection_delay
    }
    result['state'] = telemetry_dict.get('landed_state', '')

    return result


def draw_detection(frame, detection : Detection, color, line_thickness = 1):
    frame_size = XY(frame.shape[1], frame.shape[0])
    bbox = detection.bbox.multiplied_by_XY(frame_size)
    # bbox
    draw_rect(frame, bbox, color, line_thickness=line_thickness)

    # object center
    circle_size = max(1, min(bbox.height, detection.bbox.width) / 4)
    cv2.circle(frame, bbox.center.to_tuple(to = int), circle_size, color, line_thickness, cv2.FILLED)

    text = f"{float(detection.confidence):.2f}"
    if detection.track_id is not None:
        text =  f"# {detection.track_id} : {text}"

    # confidence label
    draw_text(frame,
        text,
        bbox.min_point + XY(0, -4),
        font_scale = 0.5,
        color = color,
        # bg_color = shadow_color(color),
        line_width = 1
    )

    # # track ID label
    # if detection.track_id is not None:
    #     track_id = str(detection.track_id)
    #     draw_text(frame,
    #         track_id,
    #         bbox.min_point - XY(0, 10),
    #         font_scale = 0.5,
    #         color = color,
    #         bg_color = shadow_color(color),
    #         line_width = line_thickness
    #     )

def annotate_frame_with_detection_info(detection_dict) -> np.ndarray:
    # output = {
    #                 'detections' : detections_obj,
    #                 'selected' : selected_detection,
    #                 'move_command': command,
    #                 'telemetry': await drone.get_telemetry_async(),
    #             }
    if detection_dict is None:
        return None

    detections : Detections = detection_dict.get('detections', None)
    selected : Detection|None = detection_dict.get('selected', None)
    move_command : MoveCommand | None = detection_dict.get('move_command', None)
    telemetry : dict | None = detection_dict.get('telemetry', {})

    frame = detections.frame if detections else None

    if frame is None:
        return None

    # telemetry['_frame'] = {
    #     'id' : detections.frame_id,
    #     'meta' : {
    #         '0 capture ts' : detections.meta.capture_timestamp_ns,
    #         '1 detection delta ms' : (detections.meta.detection_end_timestamp_ns - detections.meta.detection_start_timestamp_ns) / 1000000,
    #         '2 display ms' : (time.monotonic_ns() - detections.meta.capture_timestamp_ns) / 1000000
    #     }
    # }

    # logger.debug("frame #%d \t delay: %sms \tdetections %s ms, frame object: %s (%s)",
    #         detections.frame_id,
    #         (time.monotonic_ns() - detections.meta.detection_start_timestamp_ns)/1000000,
    #         len(detections.detections),
    #         id(frame), hash(frame.data.tobytes())
    # )

    frame_size = XY(frame.shape[1], frame.shape[0])
    frame_center = frame_size / 2

    if telemetry is not None:
        debug_info = make_debug_info_dict(detections.frame_id, telemetry, detections.meta)

        lines = []
        printer = FormatPrinter({float: "%.2f"}, indent=0, sort_dicts=True, compact=True)
        def add_line(key : str, dict_with_values, default=''):
            value = dict_with_values.get(key.strip(), default)
            value_str : str = printer.pformat(value)
            value_str = value_str.replace(":", "=")
            value_str = value_str.replace("'", "")
            value_str = value_str.replace("{", "")
            value_str = value_str.replace("}", "")
            value_str = value_str.replace("\n", "")

            lines.append(f'{key}: {value_str}')

        odo_dict = debug_info.get("odometry", {})
        add_line('frame', debug_info)
        add_line('time  ',  debug_info)
        add_line('state ', debug_info)
        add_line('position_body', odo_dict)
        add_line('velocity_body ', odo_dict)
        add_line('angular_velocity_body', odo_dict)
        add_line('attitude_euler        ', debug_info)
        add_line('mode', debug_info)
        add_line('action', debug_info)


        for line_no, line in enumerate(lines):
            font_scale = 0.4
            draw_text(frame, line, XY(0, 20 + 40 * line_no * font_scale), font_scale=font_scale, color=FRAME_METADATA_COLOR, bg_color=FRAME_METADATA_COLOR_BG, line_width=1)

    for detection in detections.detections:
        draw_detection(frame, detection, DETECTED_OBJECT_COLOR, 2)

    if selected is not None:
        draw_detection(frame, selected, SELECTED_OBJECT_COLOR, 1)

    if move_command is not None:
        # move command in degrees here, but we don't care
        move_command_end = frame_center + move_command.adjust_attitude# * 3
        cv2.arrowedLine(frame, frame_center.to_tuple(to = int), move_command_end.to_tuple(to = int), MOVE_COMMAND_COLOR, 2, cv2.LINE_AA)

    cv2.drawMarker(
        frame,
        frame_center.to_tuple(int),
        CROSSHAIR_COLOR,
        cv2.MARKER_CROSS,
        30,
        2
    )

    return frame


def debug_output_thread(frame_queue : Queue, sink : FrameSinkInterface = None):

    # Get the first frame and figure out image dimensions
    frame = None
    detection_dict = None
    while frame is None:
        detection_dict = frame_queue.get()
        frame : np.ndarray = detection_dict['detections'].frame
    frame_h, frame_w, _ = frame.shape
    logger.debug("Got first frame of size W:%u, H:%u", frame_w, frame_h)

    sink.start((frame_w, frame_h))
    try:
        while True:
            try:
                if detection_dict is None:
                    detection_dict = frame_queue.get()
                annotated_frame = annotate_frame_with_detection_info(detection_dict)

                if annotated_frame is not None:
                    sink.process_frame(annotated_frame)

            except:
                frame_id = -1 if detection_dict is None else detection_dict['detections'].frame_id
                logger.exception("exception while processing frame %d", frame_id, exc_info=True, stack_info=True)
                break

            finally:
                detection_dict = None
    finally:
        sink.stop()


if __name__ == '__main__':
    import threading
    import time
    import math

    from video_sink_gstreamer import RtspStreamerSink, RecorderSink
    from video_sink_multi import MultiSink
    from opencv_show_image_sink import OpenCVShowImageSink
    from helpers import configure_logging

    output_queue = Queue()

    configure_logging(logging.DEBUG)

    # def generate_frames():
    #     from picamera2 import Picamera2
    #     from libcamera import controls as picamera_controls

    #     picamera_config = None
    #     picamera_controls_initial = None
    #     video_format = 'RGB'
    #     video_width = 800
    #     video_height = 600
    #     target_fps = 30

    #     with Picamera2() as picam2:
    #         if picamera_config is None:
    #             # Default configuration
    #             main = {'size': (1280, 720), 'format': 'RGB888'}
    #             lores = {'size': (video_width, video_height), 'format': 'RGB888'}
    #             controls = {'FrameRate': target_fps}
    #             config = picam2.create_preview_configuration(main=main, lores=lores, controls=controls)
    #         else:
    #             config = picamera_config
    #         # Configure the camera with the created configuration
    #         picam2.configure(config)

    #         def apply_controls(controls_dict : dict):
    #             # TODO: creck that control is supported first
    #             picam2.set_controls(controls_dict)

    #         if picamera_controls_initial is not None:
    #             apply_controls(picamera_controls_initial)

    #         # Update GStreamer caps based on 'lores' stream
    #         lores_stream = config['lores']
    #         format_str = 'RGB' if lores_stream['format'] == 'RGB888' else video_format
    #         width, height = lores_stream['size']
    #         logger.debug("Picamera2 configuration: width=%s, height=%s, format=%s", width, height, format_str)

    #         picam2.start()
    #         frame_count = 0

    #         # used to convert from absolute frame time of Picamera2 to relative of Gstreamer (starting from 0)
    #         first_frame_timestamp_ns = 0
    #         prev_frame_timestamp_ns = 0
    #         logger.debug("picamera_process started")
    #         while True:
    #             request = picam2.capture_request()

    #             frame_data = None
    #             frame_meta = None
    #             frame_timestamp_ns = 0
    #             try:
    #                 frame_data = request.make_array("lores")
    #                 frame_meta = request.get_metadata()
    #             finally:
    #                 request.release()

    #             frame_data = picam2.capture_array('lores')
    #             # frame_data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    #             if frame_data is None:
    #                 logger.error("Failed to capture frame #%s.", frame_count)
    #                 break

    #             # Convert framontigue data if necessary
    #             frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
    #             frame = np.asarray(frame)
    #             yield frame

    #             frame_count += 1
    def generate_frames():
        video_capture = cv2.VideoCapture("/home/bdd/hailo-rpi5-examples/TEST_DATA/sample.mp4")

        if not video_capture.isOpened():
            raise RuntimeError("Cannot open video")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break   # end of video
            yield frame

        video_capture.release()



    def producer_thread_func(n_frames = 1000, delay_between_frames_ms=10):

        # frame = cv2.imread('/home/bdd/hailo-rpi5-examples/TEST_DATA/sample.mp4') # np.zeros(shape=[800, 600, 3], dtype=np.uint8)
        detections = [
            Detection(
                bbox = Rect.from_xyxy(0.1, 0.15, 0.2, 0.25),
                confidence = 0.1,
                track_id = 1
            ),
            Detection(
                bbox = Rect.from_xyxy(0.2, 0.25, 0.3, 0.35),
                confidence = 0.9,
                track_id = 2
            ),
            Detection(
                bbox = Rect.from_xyxy(0.4, 0.45, 0.5, 0.55),
                confidence = 0.7,
                track_id = 3
            ),
        ]
        detections_template = Detections(0, frame=None, detections=detections)
        telemetry = json.loads('{"odometry": {"angular_velocity_body": {"pitch_rad_s": -0.0010726579930633307, "roll_rad_s": 0.0010208315216004848, "yaw_rad_s": 0.0006282856920734048}, "child_frame_id": "1 (BODY_NED)", "frame_id": "1 (BODY_NED)", "pose_covariance": {"covariance_matrix": [0.00013371351815294474, NaN, NaN, NaN, NaN, NaN, 0.00013370705710258335, NaN, NaN, NaN, NaN, 0.07561597973108292, NaN, NaN, NaN, 1.736115154926665e-05, NaN, NaN, 1.3997990208736155e-05, NaN, 0.0010160470847040415]}, "position_body": {"x_m": 0.0010958998464047909, "y_m": -0.00166346225887537, "z_m": -1.66695237159729}, "q": {"timestamp_us": 0, "w": 0.7084895968437195, "x": 0.02082471176981926, "y": 0.019742604345083237, "z": -0.7051376104354858}, "time_usec": 1102239786672, "velocity_body": {"x_m_s": 0.0006445666076615453, "y_m_s": 0.001224401406943798, "z_m_s": 0.005852778907865286}, "velocity_covariance": {"covariance_matrix": [0.0014366698451340199, NaN, NaN, NaN, NaN, NaN, 0.0014359699562191963, NaN, NaN, NaN, NaN, 0.004651403985917568, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]}}, "scaled_pressure": {"absolute_pressure_hpa": 1008.449951171875, "differential_pressure_hpa": 0.0, "differential_pressure_temperature_deg": 0.0, "temperature_deg": 36.290000915527344, "timestamp_us": 1102240195000}, "attitude_euler": {"pitch_deg": 3.287358045578003, "roll_deg": 0.0956246480345726, "timestamp_us": 1102239771000, "yaw_deg": -89.72563171386719}, "flight_mode": "7 (OFFBOARD)"}')

        for frame, i in zip(generate_frames(), range(0, n_frames)):
            logger.debug("got frame: %s", i)
            # _, frame = camera.read()

            # Just to keep numbers rolling
            # telemetry['a_frameid'] = i
            odo = telemetry['odometry']
            odo_avb = odo['angular_velocity_body']
            odo_avb['pitch_rad_s'] += 0.01
            odo_avb['yaw_rad_s'] += 0.01
            odo_pb = odo['position_body']
            odo_pb['x_m'] += 0.05
            odo_pb['y_m'] += 0.05
            odo_pb['z_m'] += 0.05


            d = dataclasses.replace(detections_template, frame_id = i, frame=frame.copy())
            q = i / 10
            output_queue.put({
                'detections': d,
                'selected' : d.detections[0],
                'move_command': MoveCommand(adjust_attitude=XY(math.cos(q) * 20, math.sin(q)* 20), move_speed_ms=10),
                'telemetry' : telemetry
            })
            time.sleep(delay_between_frames_ms / 1000)
        output_queue.put(None) # Just to terminate by exception

    producer_thread = threading.Thread(
        target=producer_thread_func,
        name="Producer",
        args=(1000000, 100)
    )
    producer_thread.start()

    sink = MultiSink([
        RtspStreamerSink(30, 8554),
        RecorderSink(10, "_TMP/test_recordings"),
        OpenCVShowImageSink(window_title='DEBUG IMAGE', fps_hint=500)
    ])

    debug_output_thread(frame_queue=output_queue, sink=sink)
