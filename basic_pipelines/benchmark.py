from pathlib import Path
from collections import defaultdict
from typing import Optional
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.detections_info = defaultdict(list)
        self.save_detection_frames_dir: Optional[Path] = None
        self.save_detection_frames_max = 1
        self.save_detection_frames_every = 1
        self.saved_detection_frames = 0
        self.run_tag = "stream"

    def new_detection(self, class_id, confidence : float):
        self.detections_info[class_id].append(confidence)

    def configure_frame_saving(self, save_dir: Optional[str], max_frames: int, every_n: int, run_tag: str) -> None:
        if not save_dir:
            self.save_detection_frames_dir = None
            return
        self.save_detection_frames_dir = Path(save_dir)
        self.save_detection_frames_dir.mkdir(parents=True, exist_ok=True)
        self.save_detection_frames_max = max_frames
        self.save_detection_frames_every = every_n
        self.run_tag = run_tag

    def should_save_frame(self, frame_id: int, detection_count: int) -> bool:
        if detection_count <= 0:
            return False
        if self.save_detection_frames_dir is None:
            return False
        if self.save_detection_frames_max > 0 and self.saved_detection_frames >= self.save_detection_frames_max:
            return False
        if self.save_detection_frames_every > 1 and frame_id % self.save_detection_frames_every != 0:
            return False
        return True

    def save_detection_frame(self, frame_id: int, frame_bgr) -> None:
        if self.save_detection_frames_dir is None:
            return
        out_path = self.save_detection_frames_dir / f"{self.run_tag}_frame_{frame_id:06d}.jpg"
        if cv2.imwrite(str(out_path), frame_bgr):
            self.saved_detection_frames += 1
            print(f"Saved detection frame: {out_path}")
        else:
            print(f"WARNING: failed to save detection frame: {out_path}")

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
    detections = list(roi.get_objects_typed(hailo.HAILO_DETECTION))

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

    if user_data.use_frame and frame is not None:
        # Note: using imshow will not work here, as the callback function is not running in the main thread
        # Let's print the detection count to the frame
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_h, frame_w = frame.shape[:2]
        for detection in detections:
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            class_id = detection.get_class_id()
            x1 = int(max(0, min(frame_w - 1, bbox.xmin() * frame_w)))
            y1 = int(max(0, min(frame_h - 1, bbox.ymin() * frame_h)))
            x2 = int(max(0, min(frame_w - 1, (bbox.xmin() + bbox.width()) * frame_w)))
            y2 = int(max(0, min(frame_h - 1, (bbox.ymin() + bbox.height()) * frame_h)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_id}:{confidence:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        # Example of how to use the new_variable and new_function from the user_data
        # Let's print the new_variable and the result of the new_function to the frame
        # cv2.putText(frame, f"{user_data.new_function()} {user_data.new_variable}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Convert the frame to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if user_data.should_save_frame(frame_id=frame_id, detection_count=detection_count):
            user_data.save_detection_frame(frame_id=frame_id, frame_bgr=frame_bgr)
        user_data.set_frame(frame_bgr)

    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK


class BenchmarkApp(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        for flag, dest, typ, default in [
            ('--nms-score-threshold',         'nms_score_threshold',         float, 0.3),
            ('--nms-iou-threshold',           'nms_iou_threshold',           float, 0.45),
            ('--tracker-iou-thr',             'tracker_iou_thr',             float, 0.6),
            ('--tracker-kalman-dist-thr',     'tracker_kalman_dist_thr',     float, 0.6),
            ('--tracker-keep-new-frames',     'tracker_keep_new_frames',     int,   1),
            ('--tracker-keep-tracked-frames', 'tracker_keep_tracked_frames', int,   0),
            ('--tracker-keep-lost-frames',    'tracker_keep_lost_frames',    int,   0),
            ('--save-detection-frames-max',   'save_detection_frames_max',   int,   1),
            ('--save-detection-frames-every', 'save_detection_frames_every', int,   1),
        ]:
            try:
                parser.add_argument(flag, dest=dest, type=typ, default=default)
            except Exception:
                pass  # already in default parser
        try:
            parser.add_argument(
                '--save-detection-frames-dir',
                dest='save_detection_frames_dir',
                type=str,
                default=None,
                help='Directory to save callback frames with drawn detections',
            )
        except Exception:
            pass
        super().__init__(app_callback, user_data, parser=parser)
        video_tag = Path(str(self.video_source)).stem if self.video_source else "stream"
        if getattr(self.options_menu, 'save_detection_frames_dir', None):
            # Saving callback frames requires pulling image data from GstBuffer.
            self.user_data.use_frame = True
        self.user_data.configure_frame_saving(
            save_dir=getattr(self.options_menu, 'save_detection_frames_dir', None),
            max_frames=max(0, int(getattr(self.options_menu, 'save_detection_frames_max', 1))),
            every_n=max(1, int(getattr(self.options_menu, 'save_detection_frames_every', 1))),
            run_tag=video_tag,
        )

    def get_pipeline_string(self):
        nms_score = getattr(self.options_menu, 'nms_score_threshold', 0.3)
        nms_iou   = getattr(self.options_menu, 'nms_iou_threshold',   0.45)
        thresholds_str = (
            f"nms-score-threshold={nms_score} "
            f"nms-iou-threshold={nms_iou} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        source_pipeline = SOURCE_PIPELINE(
            video_source=self.video_source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
        )
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=1,
            config_json=self.labels_json,
            additional_params=thresholds_str,
        )
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(
            class_id=1,
            keep_past_metadata='false',
            qos='false',
            iou_thr=getattr(self.options_menu, 'tracker_iou_thr', 0.6),
            kalman_dist_thr=getattr(self.options_menu, 'tracker_kalman_dist_thr', 0.6),
            keep_new_frames=getattr(self.options_menu, 'tracker_keep_new_frames', 1),
            keep_tracked_frames=getattr(self.options_menu, 'tracker_keep_tracked_frames', 0),
            keep_lost_frames=getattr(self.options_menu, 'tracker_keep_lost_frames', 0),
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(
            video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
        )
        return (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )

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
