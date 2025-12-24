#! /usr/bin/env python

import subprocess
import datetime
from queue import Queue
import logging
import json
# import re
import dataclasses

import cv2
import numpy as np

from helpers import Detections, Detection, XY, MoveCommand, Rect
from interfaces import FrameSinkInterface

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


# ceanup_json_re= re.compile(r'\s*[{}],?|"')
def telemetry_as_text(telemetry_dict):
    # remove 'covariance_matrix' which is too verbose
    telemetry_dict = filterdict(telemetry_dict, lambda k, v :  k != 'covariance_matrix')
    # remove keys with empty values entirely
    telemetry_dict = filterdict(telemetry_dict, lambda k, v :  not (hasattr(v, '__len__') and len(v) == 0))
    # remove various timestamps
    telemetry_dict = filterdict(telemetry_dict, lambda k, v :  not (isinstance(k, str) and 'time' in k))

    # convert to pretty-ish multi-line text
    return json.dumps(telemetry_dict, sort_keys=True, indent=2)


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


def process_output(output) -> np.ndarray:
    # output = {
    #                 'detections' : detections_obj,
    #                 'selected' : selected_detection,
    #                 'move_command': command,
    #                 'telemetry': await drone.get_telemetry_async(),
    #             }
    if output is None:
        return None

    detections = output.get('detections', Detections(-1))
    selected : Detection|None = output.get('selected', None)
    move_command : MoveCommand | None = output.get('move_command', None)
    telemetry : dict | None = output.get('telemetry', None)

    frame = detections.frame if detections else None

    if frame is None:
        return None

    frame_size = XY(frame.shape[1], frame.shape[0])
    frame_center = frame_size / 2

    if telemetry is not None:
        telemetry_text = telemetry_as_text(telemetry)
        for line_no, line in enumerate(telemetry_text.splitlines()):
            font_scale = 0.3
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
    while frame is None:
        output = frame_queue.get()
        frame : np.ndarray = output['detections'].frame
    frame_h, frame_w, _ = frame.shape
    logger.debug("Got first frame of size W:%u, H:%u", frame_w, frame_h)

    sink.start((frame_w, frame_h))
    try:
        output = None
        while True:
            try:
                output = frame_queue.get()
                annotated_frame = process_output(output)

                if annotated_frame is not None:
                    sink.process_frame(annotated_frame)

            except:
                frame_id = output or -1
                logger.exception("exception while processing frame %d", frame_id, exc_info=True, stack_info=True)
                break
    finally:
        sink.stop()


if __name__ == '__main__':
    import threading
    import time
    import math

    from video_sink_gstreamer import RtspStreamerSink, RecorderSink
    from video_sink_multi import MultiSink
    from opencv_show_image_sink import OpenCVShowImageSink

    output_queue = Queue()

    def producer_thread_func(n_frames = 1000, delay_between_frames_ms=100):
        frame = cv2.imread('/home/bdd/hailo-rpi5-examples/SAMPLE_800x600.jpg') # np.zeros(shape=[800, 600, 3], dtype=np.uint8)
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


        for i in range(0, n_frames):
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
        RecorderSink(10, "/home/bdd/hailo-rpi5-examples/test_recordings"),
        # OpenCVShowImageSink(window_title='DEBUG IMAGE')
    ])

    debug_output_thread(frame_queue=output_queue, sink=sink)
