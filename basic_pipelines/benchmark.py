from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional
import time
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


def _buffer_correlation_key(buffer: Gst.Buffer):
    pts = buffer.pts
    if pts != Gst.CLOCK_TIME_NONE:
        return ("pts", int(pts))
    return ("id", id(buffer))


class StageLatencyAggregator:
    """
    Wall-clock delays between bench markers on buffer src pads (monotonic_ns).
    The probe on identity_callback.src is registered before app_callback, so stage
    "queues_before_callback" ends right before user callback work; callback_cpu is app_callback only.
    """

    DELTA_LABELS = (
        "inference_wrapper",
        "tracker",
        "queues_before_callback",
    )

    def __init__(self):
        self._n_markers = 4
        self.delta_lists: list[list[int]] = [[] for _ in range(3)]
        self.callback_durations_ns: list[int] = []
        self.e2e_ns: list[int] = []
        self._pending: dict = {}
        self._frame_t0: dict = {}
        self._callback_start: dict = {}

    def record_marker(self, stage_idx: int, buffer: Gst.Buffer) -> None:
        key = _buffer_correlation_key(buffer)
        t = time.monotonic_ns()
        row = self._pending.setdefault(key, [None] * self._n_markers)
        if row[stage_idx] is not None:
            return
        row[stage_idx] = t
        if stage_idx == 0:
            self._frame_t0[key] = t
        if stage_idx > 0 and row[stage_idx - 1] is not None:
            self.delta_lists[stage_idx - 1].append(t - row[stage_idx - 1])
        if stage_idx == self._n_markers - 1:
            self._callback_start[key] = t

    def record_callback_done(self, buffer: Gst.Buffer) -> None:
        key = _buffer_correlation_key(buffer)
        te = time.monotonic_ns()
        t0 = self._frame_t0.pop(key, None)
        if t0 is not None:
            self.e2e_ns.append(te - t0)
        ts = self._callback_start.pop(key, None)
        if ts is not None:
            self.callback_durations_ns.append(te - ts)
        self._pending.pop(key, None)

    def flush(self) -> None:
        self._pending.clear()
        self._frame_t0.clear()
        self._callback_start.clear()

    @staticmethod
    def _fmt_ns(samples: list[int]) -> str:
        if not samples:
            return "(no samples)"
        ms = np.asarray(samples, dtype=np.float64) / 1e6
        return (
            f"n={ms.size} min/med/mean/max ms = "
            f"{ms.min():.3f} {np.median(ms):.3f} {ms.mean():.3f} {ms.max():.3f}"
        )

    def print_report(self) -> None:
        print("Pipeline stage delay (wall clock between identity markers, per frame):")
        for label, samples in zip(self.DELTA_LABELS, self.delta_lists):
            print(f"  {label}: {self._fmt_ns(samples)}")
        print(f"  app_callback (probe work): {self._fmt_ns(self.callback_durations_ns)}")
        print(f"  end-to-end after source marker: {self._fmt_ns(self.e2e_ns)}")


def _make_latency_probe(stage_idx: int, agg: StageLatencyAggregator) -> Callable:
    def _probe(_pad, info, _user):
        buf = info.get_buffer()
        if buf is not None:
            agg.record_marker(stage_idx, buf)
        return Gst.PadProbeReturn.OK

    return _probe


# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.stage_latency: Optional[StageLatencyAggregator] = StageLatencyAggregator()
        self.detections_info = defaultdict(list)
        self.save_detection_frames_dir: Optional[Path] = None
        self.save_detection_frames_max = 0
        self.save_detection_frames_every = 1
        self.saved_detection_frames = 0
        self.run_tag = "stream"
        # Pushes BGR frames to multiprocessing.Queue for cv2.imshow; only safe when the app
        # started the display consumer (--use-frame). Saving JPEGs sets use_frame for numpy
        # extraction but must not fill the queue without a reader (deadlock after maxsize=3).
        self.forward_frames_to_display_queue = True

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

    def should_save_frame(self, frame_id: int) -> bool:
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
        if self.stage_latency is not None:
            self.stage_latency.print_report()
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
        if user_data.should_save_frame(frame_id):
            user_data.save_detection_frame(frame_id=frame_id, frame_bgr=frame_bgr)
        if getattr(user_data, 'forward_frames_to_display_queue', True):
            user_data.set_frame(frame_bgr)

    if string_to_print:
        print(string_to_print)
    if user_data.stage_latency is not None and buffer is not None:
        user_data.stage_latency.record_callback_done(buffer)
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
            ('--save-detection-frames-every', 'save_detection_frames_every', int,   1),
        ]:
            try:
                parser.add_argument(flag, dest=dest, type=typ, default=default)
            except Exception:
                pass  # already in default parser
        try:
            parser.add_argument(
                '--save-detection-frames-max',
                dest='save_detection_frames_max',
                type=int,
                default=0,
                help='How many JPEGs to write at most (0 = no limit). Ignored if --save-detection-frames-dir is not set.',
            )
        except Exception:
            pass
        try:
            parser.add_argument(
                '--save-detection-frames-dir',
                dest='save_detection_frames_dir',
                type=str,
                default=None,
                help='Directory to save callback frames (overlay + count); all frames matching --save-detection-frames-every / -max, including with zero detections',
            )
        except Exception:
            pass
        try:
            parser.add_argument(
                '--no-stage-latency',
                dest='no_stage_latency',
                action='store_true',
                default=False,
                help='Disable identity markers and stage delay measurement',
            )
        except Exception:
            pass
        try:
            parser.add_argument(
                '--video-sink',
                dest='video_sink',
                type=str,
                default='fakesink',
                help=(
                    'Element passed to fpsdisplaysink as video-sink. Default fakesink avoids crashes '
                    'when autovideosink picks DirectFB on headless/SSH sessions. Use autovideosink or '
                    'waylandsink for on-screen preview.'
                ),
            )
        except Exception:
            pass
        super().__init__(app_callback, user_data, parser=parser)
        if getattr(self.options_menu, 'no_stage_latency', False):
            self.user_data.stage_latency = None
        video_tag = Path(str(self.video_source)).stem if self.video_source else "stream"
        if getattr(self.options_menu, 'save_detection_frames_dir', None):
            # Saving callback frames requires pulling image data from GstBuffer.
            self.user_data.use_frame = True
        self.user_data.forward_frames_to_display_queue = bool(
            getattr(self.options_menu, 'use_frame', False)
        )
        self.user_data.configure_frame_saving(
            save_dir=getattr(self.options_menu, 'save_detection_frames_dir', None),
            max_frames=max(0, int(getattr(self.options_menu, 'save_detection_frames_max', 0))),
            every_n=max(1, int(getattr(self.options_menu, 'save_detection_frames_every', 1))),
            run_tag=video_tag,
        )

    def create_pipeline(self):
        vs = getattr(self.options_menu, 'video_sink', None)
        if isinstance(vs, str) and vs.strip():
            self.video_sink = vs.strip()
            print(f"Using GStreamer video sink: {self.video_sink}")
        super().create_pipeline()

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
            class_id=-1,
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
        if getattr(self.options_menu, 'no_stage_latency', False):
            return (
                f'{source_pipeline} ! '
                f'{detection_pipeline_wrapper} ! '
                f'{tracker_pipeline} ! '
                f'{user_callback_pipeline} ! '
                f'{display_pipeline}'
            )
        return (
            f'{source_pipeline} ! identity name=bench_after_source ! '
            f'{detection_pipeline_wrapper} ! identity name=bench_after_infer ! '
            f'{tracker_pipeline} ! identity name=bench_after_tracker ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )

    def _install_stage_latency_probes(self) -> None:
        agg = self.user_data.stage_latency
        if agg is None:
            return
        markers = (
            ("bench_after_source", 0),
            ("bench_after_infer", 1),
            # ("bench_after_tracker", 2),
            ("identity_callback", 3),
        )
        pads_and_idx = []
        for name, idx in markers:
            el = self.pipeline.get_by_name(name)
            if el is None:
                print(f"WARNING: benchmark latency: element '{name}' not found, stage timing disabled.")
                self.user_data.stage_latency = None
                return
            pad = el.get_static_pad("src")
            if pad is None:
                print(f"WARNING: benchmark latency: no src pad on '{name}', stage timing disabled.")
                self.user_data.stage_latency = None
                return
            pads_and_idx.append((pad, idx))
        for pad, idx in pads_and_idx:
            pad.add_probe(Gst.PadProbeType.BUFFER, _make_latency_probe(idx, agg), None)

    def run(self, *args, **kwargs):
        self._benchmark_wall_t0 = time.monotonic()
        self._install_stage_latency_probes()
        super().run(*args, **kwargs)

    def on_eos(self):
        wall_s = time.monotonic() - getattr(self, "_benchmark_wall_t0", time.monotonic())
        print(f"benchmark_file_wall_time_s: {wall_s:.6f}")
        if self.user_data.stage_latency is not None:
            self.user_data.stage_latency.flush()
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
