#!/usr/bin/env python

from math import nan
from pathlib import Path
import asyncio

from queue import Queue, Empty
from collections import deque
from dataclasses import dataclass, field

import os
import numpy as np
import json
import datetime

import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from app_base import GStreamerDetectionApp

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from helpers import Rect, XY, configure_logging, OverwriteQueue, Detection, Detections

import logging
logger = logging.getLogger(__name__)



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


# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data : user_app_callback_class):
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

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Parse the detections
    detection_count = 0
    detections_list = []
    for detection in detections:
        detection : hailo.HailoDetection = detection

        # DEBUG_dump('detecion: ', detection)

        label = detection.get_label()
        bbox = detection.get_bbox()

        confidence = detection.get_confidence()
        if label == "person":
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
                # DEBUG_dump("track: ", track[0])
            else:
                track_id = None

            # string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")
            detection_count += 1
            detections_list.append(Detection(
                bbox = Rect.from_xyxy(bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()),
                confidence = confidence,
                track_id = track_id,
            ))

            # DEBUG_dump("bbox: ", bbox)
    if len(detections) != 0:
        user_data.detections_queue.put(Detections(user_data.frame_count, frame, detections_list))

    # if user_data.use_frame:
    #     # Note: using imshow will not work here, as the callback function is not running in the main thread
    #     # Let's print the detection count to the frame
    #     cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     # Example of how to use the new_variable and new_function from the user_data
    #     # Let's print the new_variable and the result of the new_function to the frame
    #     cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     # Convert the frame to BGR
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     user_data.set_frame(frame)

    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK


import threading
import asyncio

class StopSignal:
    pass
STOP = StopSignal()


def drone_controlling_tread(*args, **kwargs):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(drone_controlling_tread_async(*args, **kwargs))
    finally:
        loop.close()

MIN_CONFIDENCE = 0.4

async def drone_controlling_tread_async(drone_connection_string, drone_config, detections_queue, output_queue = None):
    from math import radians

    center = XY(0.5, 0.5)
    distance_r = 0.1
    distance_r *= distance_r
    frame_angular_size = XY(120, 90)

    from drone import DroneMover
    drone = DroneMover(drone_connection_string, drone_config)

    logger.debug("starting up drone...")
    await drone.startup_sequence()
    logger.debug("drone started")

    logger.debug("Getting telemetry")
    telemetry_data = await drone.get_telemetry_async()
    logger.debug("Drone telemetry: %s", json.dumps(telemetry_data))
    current_attitude = await drone.get_current_attitude_async()

    while True:
        try:
            detections_obj = Detections(-1)
            distance_to_center : float = float('NaN')
            command = XY()

            logger.debug("!!! awaiting detection... ")
            try:
                detections_obj : Detections = detections_queue.get(timeout = 0.02)
            except Empty:
                current_attitude = await drone.get_current_attitude_async()
                logger.debug("attitude: %s", current_attitude)
                continue

            if detections_obj is STOP:
                logger.info("stopping detection loop")
                break

            detections, frame = detections_obj.detections, detections_obj.frame
            selected_detection = None

            # telemetry_data = await drone.get_telemetry_async()
            # logger.debug("Drone telemetry: %s", json.dumps(telemetry_data))
            current_attitude = await drone.get_current_attitude_async()
            logger.debug("attitude: %s", current_attitude)

            detections.sort(reverse=True, key = lambda d : d.confidence)
            if len(detections) > 0:
                best_detection = detections[0]
                if best_detection.confidence >= MIN_CONFIDENCE:
                    selected_detection = best_detection

                    logger.debug("!!! Detection: %s", selected_detection)
                    # TODO: if track id is None and confidence < 0.3 -- ignore target

                    distance_to_center = selected_detection.bbox.center.distance_squared_to(center)
                    logger.debug("distance to center: %s", distance_to_center)

                    diff_xy = center - selected_detection.bbox.center
                    logger.debug("move command: %s, frame: %s", diff_xy, frame_angular_size)
                    command = diff_xy.multiplied_by_XY(frame_angular_size)

                    # if distance_to_center < distance_r:
                    #     logger.debug("drone in the crosshair: move to center")
                    #     asyncio.run(drone.move_to_target_async(command.x, command.y, 0.5))

                    telemetry_data = await drone.get_telemetry_async()

                    if distance_to_center >= distance_r / 2:
                        diff_xy = center - selected_detection.bbox.center
                        logger.debug("move command: %s, frame: %s", diff_xy, frame_angular_size)
                        command = diff_xy.multiplied_by_XY(frame_angular_size)

                        logger.debug("move command: %s", command)
                        command.x -= current_attitude['yaw_deg']
                        logger.debug("move command, adjusted for current yaw: %s", command)
                        await drone.move_relative_async(command.x, command.y)
                        logger.debug("move command done")

            # -1 means that there was no frame and no detections
            if output_queue is not None and detections_obj.frame_id != -1:
                output = {
                    'detections' : detections_obj,
                    'selected' : selected_detection,
                    'move_command': command,
                    'telemetry': await drone.get_telemetry_async(),
                }

        except:
            logging.exception(f"Got exception: %s %s COMMAND: %s", detections_obj, distance_to_center, command, exc_info=True)



class App(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data, parser=None, video_output_path = None, video_output_chunk_length_s = 30):
        self.video_output_directory = video_output_path or '.'
        self.video_output_chunk_length_s = video_output_chunk_length_s or 30
        super().__init__(app_callback, user_data, parser)

    def get_output_pipeline_string(self, video_sink: str, sync: str = 'true', show_fps: str = 'true'):
        record_start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        video_output_chunk_length_ns = self.video_output_chunk_length_s * 1000 * 1000 * 1000
        return f'''
            videoconvert ! x264enc tune=zerolatency speed-preset=ultrafast \
            ! h264parse config-interval=1 \
            ! splitmuxsink name=smx max-size-time={video_output_chunk_length_ns} muxer-factory=mp4mux async-finalize=true location="{self.video_output_directory}/raw_{record_start_time_str}_%03d.mp4"
        '''



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

    detections_queue = OverwriteQueue(maxsize=2)
    output_queue = OverwriteQueue(maxsize=2)

    drone_config = {
        'cruise_altitude' : 1
    }

    drone_thread = threading.Thread(
        target = drone_controlling_tread,
        args = ('udp://:14550', drone_config, detections_queue, output_queue),
        name = "Drone"
    )
    drone_thread.start()

    user_data = user_app_callback_class(detections_queue)
    user_data.use_frame = True
    app = App(app_callback, user_data)

    DEBUG = True
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

    app.run()
    print("Done !!!")
    detections_queue.put(STOP)
    drone_thread.join()

if __name__ == "__main__":
    main()
