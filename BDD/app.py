#!/usr/bin/env python

from pathlib import Path
from attr import dataclass

import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from app_base import GStreamerDetectionApp
# from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp


import asyncio

from queue import Queue
from collections import deque

from helpers import Rect, XY, configure_logging
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

def full_classname(obj):
    # based on https://gist.github.com/clbarnes/edd28ea32010eb159b34b075687bb49e#file-classname-py
    cls = type(obj)
    module = cls.__module__
    name = cls.__qualname__
    if module is not None and module != "__builtin__":
        name = module + "." + name
    return name

def DEBUG_dump(prefix, obj):
    print(prefix, obj, full_classname(obj), dir(obj))


@dataclass(slots=True, order=True, frozen=True)
class Detection:
    bbox : Rect = field(default_factory=Rect)
    confidence : float = 0.0
    track_id : int|None = 0


class OverwriteQueue(Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)
        # to make sure that Queue always stores elements, effectively overwriting some older ones
        self.maxsize = 0

    def _init(self, maxsize):
        self.queue = deque(maxlen=maxsize)

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example

    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

detections_queue = OverwriteQueue(maxsize=2)

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
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
        detections_queue.put(detections_list)

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

async def drone_controlling_tread_async(drone_connection_string, drone_config):
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
    logger.debug("Drone telemetry: %s", telemetry_data)

    while True:
        try:
            detection = None
            distance_to_center : float = 0.0
            command = XY()

            logger.debug("!!! awaiting detection... ")
            detections : list[Detection] = detections_queue.get()
            if detections is STOP:
                logger.info("stopping detection loop")
                break

            if len(detections) == 0:
                continue

            detections.sort(reverse=True, key = lambda d : d.confidence)
            detection = detections[0]

            logger.debug("!!! Detection: %s", detection)
            # TODO: if track id is None and confidence < 0.3 -- ignore target

            distance_to_center = detection.bbox.center.distance_squared_to(center)
            logger.debug("distance to center: %s", distance_to_center)

            diff_xy = center - detection.bbox.center
            logger.debug("move command: %s, frame: %s", diff_xy, frame_angular_size)
            command = diff_xy.multiplied_by_XY(frame_angular_size)

            # if distance_to_center < distance_r:
            #     logger.debug("drone in the crosshair: move to center")
            #     asyncio.run(drone.move_to_target_async(command.x, command.y, 0.5))

            if distance_to_center >= distance_r / 2:
                diff_xy = center - detection.bbox.center
                logger.debug("move command: %s, frame: %s", diff_xy, frame_angular_size)
                command = diff_xy.multiplied_by_XY(frame_angular_size)

                logger.debug("move command: %s", command)
                await drone.move_relative_async(command.x, command.y)
                logger.debug("move command done")

            # logger.debug("Drone telemetry: %s", await drone.get_telemetry_async())

        except:
            logging.exception(f"Got exception: %s %s COMMAND: %s", detection, distance_to_center, command, exc_info=True)


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

    drone_config = {
        'cruise_altitude' : 1
    }

    drone_thread = threading.Thread(
        target = drone_controlling_tread,
        args = ('udp://:14550', drone_config, ),
        name = "Drone"
    )
    drone_thread.start()

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    for i in range(3):
        detections_queue.put([
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
        ]
        )
    app.run()
    print("Done !!!")
    detections_queue.put(STOP)
    drone_thread.join()

if __name__ == "__main__":
    # import nest_asyncio

    # nest_asyncio.apply()
    main()
