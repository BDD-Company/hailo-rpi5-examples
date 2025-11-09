#!/usr/bin/env python

from pathlib import Path
from attr import dataclass

import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

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
            detections_queue.put(Detection(
                bbox = Rect.from_xyxy(bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()),
                confidence = confidence,
                track_id = track_id,
            ))

            # DEBUG_dump("bbox: ", bbox)


    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK


import threading
import asyncio


def drone_controlling_tread(drone):

    from math import radians

    center = XY(0.5, 0.5)
    distance_r = 0.1
    distance_r *= distance_r
    frame_angular_size = XY(120, 90)

    detection = None
    distance_to_center : float = 0.0
    command = XY()
    while True:
        try:
            detection = None
            distance_to_center : float = 0.0
            command = XY()

            logger.debug("!!! awaiting detection... ")
            detection : Detection = detections_queue.get()
            logger.debug("!!! Detection: %s", detection)

            distance_to_center = detection.bbox.center.distance_squared_to(center)
            logger.debug("distance to center: %s", distance_to_center)
            if distance_to_center >= distance_r / 2:
                diff_xy = center - detection.bbox.center
                logger.debug("move command: %s, frame: %s", diff_xy, frame_angular_size)
                command = diff_xy.multiplied_by_XY(frame_angular_size)
                # diff_xy = XY(math.degrees(diff_xy.x), math.degrees(diff_xy.y))

                logger.debug("move command: %s", command)
                drone.move_relative(command.x, command.y)

            if distance_to_center < distance_r:
                logger.debug("drone in the crosshair: move to center")
                drone.move_forward(10.0)
            sleep(0.5)

        except:
            logging.exception(f"Got exception: %s %s COMMAND: %s", detection, distance_to_center, command, exc_info=True)


async def main():
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # Create an instance of the user app callback class

    configure_logging(level = logging.DEBUG)
    # # otherwise too much verbose
    # def logger_filter(r : logging.LogRecord):
    #     if 'picamera2.py' in r.pathname and r.levelno < logging.WARNING:
    #         return False
    #     return True

    logger.debug("starting up drone...")
    from drone import DroneMover
    drone = DroneMover('udp://:14550')
    logger.debug("drone started")

    drone_thread = threading.Thread(
        target = drone_controlling_tread,
        args=(drone,),
        name = "Drone"
    )
    drone_thread.start()

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()

if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    nest_asyncio.apply()
    loop.run_until_complete(main())
