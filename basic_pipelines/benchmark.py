from pathlib import Path
from collections import defaultdict
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.detections_info = defaultdict(list)

    def new_detection(self, class_id, confidence : float):
        self.detections_info[class_id].append(confidence)

    def print_stats(self):
        total_detections = sum(len(confidences) for confidences in self.detections_info.values())
        print(f"Detection statistics:\n\ttotal detections: {total_detections}")
        if total_detections == 0:
            return

        all_confidences = np.asarray(
            [confidence for confidences in self.detections_info.values() for confidence in confidences],
            dtype=np.float32,
        )
        print(
            "\toverall confidence min/median/average/max="
            f"{all_confidences.min():.4f} {np.median(all_confidences):.4f} "
            f"{all_confidences.mean():.4f} {all_confidences.max():.4f}"
        )

        for class_id, confidences in sorted(self.detections_info.items()):
            confidences_np = np.asarray(confidences, dtype=np.float32)
            count = confidences_np.size
            percentage = (count * 100.0) / total_detections
            print(
                f"\tclass {class_id}: count={count} ({percentage:.2f}%), "
                f"\tconfidence min/median/average/max={confidences_np.min():.4f} "
                f"{np.median(confidences_np):.4f} {confidences_np.mean():.4f} {confidences_np.max():.4f}"
            )



# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
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

    frame_id = user_data.get_count()
    # Parse the detections
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        class_id = detection.get_class_id()
        # if label == "person":
            # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        string_to_print += (f"frame#{frame_id:04} detection: #{track_id} class: {class_id} confidence: {confidence:.2f}")
        string_to_print += (f" (x: {bbox.xmin():.3}, y: {bbox.ymin():.3}, w: {bbox.width():.3}, h: {bbox.height():.3})\n")
        detection_count += 1
        user_data.new_detection(class_id, confidence)

    if user_data.use_frame:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        # cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK


class BenchmarkApp(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data):
        super().__init__(app_callback, user_data)

    def on_eos(self):
        self.user_data.print_stats()
        # do not rewind, just stop
        self.loop.quit()
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = BenchmarkApp(app_callback, user_data)
    app.run()
