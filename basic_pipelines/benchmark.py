from pathlib import Path
from collections import defaultdict
import sys
import os

# Add BDD dir to path so app_base can import its sibling modules (pipelines, helpers, etc.)
_BDD_DIR = Path(__file__).resolve().parent.parent / 'BDD'
sys.path.insert(0, str(_BDD_DIR))

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.core.common.core import get_default_parser

from app_base import GStreamerDetectionApp
from pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
)

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
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
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f""

    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    frame_id = user_data.get_count()
    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        class_id = detection.get_class_id()
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        string_to_print += (f"frame#{frame_id:04} detection: #{track_id} class: {class_id} confidence: {confidence:.2f}")
        string_to_print += (f" (x: {bbox.xmin():.3}, y: {bbox.ymin():.3}, w: {bbox.width():.3}, h: {bbox.height():.3})\n")
        detection_count += 1
        user_data.new_detection(class_id, confidence)

    if user_data.use_frame:
        cv2.putText(frame, f"Detections: {detection_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    if string_to_print:
        print(string_to_print)
    return Gst.PadProbeReturn.OK


class BenchmarkApp(GStreamerDetectionApp):

    @staticmethod
    def _make_sweep_parser():
        parser = get_default_parser()
        parser.add_argument('--nms-score-threshold', type=float, default=0.3,
                            help='NMS score threshold for detection (default: 0.3)')
        parser.add_argument('--nms-iou-threshold', type=float, default=0.45,
                            help='NMS IoU threshold for detection (default: 0.45)')
        parser.add_argument('--tracker-iou-thr', type=float, default=0.6,
                            help='Tracker IoU threshold (default: 0.6)')
        parser.add_argument('--tracker-kalman-dist-thr', type=float, default=0.6,
                            help='Tracker Kalman distance threshold (default: 0.6)')
        parser.add_argument('--tracker-keep-new-frames', type=int, default=1,
                            help='Tracker keep_new_frames (default: 1)')
        parser.add_argument('--tracker-keep-tracked-frames', type=int, default=0,
                            help='Tracker keep_tracked_frames (default: 0)')
        parser.add_argument('--tracker-keep-lost-frames', type=int, default=0,
                            help='Tracker keep_lost_frames (default: 0)')
        return parser

    def __init__(self, app_callback, user_data):
        super().__init__(app_callback, user_data, parser=self._make_sweep_parser())
        # Override thresholds with values from CLI args, then rebuild pipeline
        opt = self.options_menu
        self.thresholds_str = (
            f"nms-score-threshold={opt.nms_score_threshold} "
            f"nms-iou-threshold={opt.nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )
        self.create_pipeline()

    def get_pipeline_string(self):
        opt = self.options_menu
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
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str,
        )
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        tracker_pipeline = TRACKER_PIPELINE(
            class_id=1,
            keep_past_metadata='false',
            qos='false',
            iou_thr=opt.tracker_iou_thr,
            kalman_dist_thr=opt.tracker_kalman_dist_thr,
            keep_new_frames=opt.tracker_keep_new_frames,
            keep_tracked_frames=opt.tracker_keep_tracked_frames,
            keep_lost_frames=opt.tracker_keep_lost_frames,
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        if self.source_type == 'rpi':
            display_pipeline = self.get_output_pipeline_string(
                video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
            )
        else:
            display_pipeline = DISPLAY_PIPELINE(
                video_sink=self.video_sink, sync=self.sync, show_fps=self.show_fps
            )
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        return pipeline_string

    def on_eos(self):
        self.user_data.print_stats()
        self.loop.quit()
        self.pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    user_data = user_app_callback_class()
    app = BenchmarkApp(app_callback, user_data)
    app.run()
