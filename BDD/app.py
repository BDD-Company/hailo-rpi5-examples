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
import math

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

from helpers import FrameMetadata, Rect, XY,  Detection, Detections, MoveCommand, STOP
from helpers import CameraConfig, CameraSwitcher, DEFAULT_CAMERA_ID
from flight_mode import FlightModeController, remap_tile_bbox
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
USE_TRACKER = False

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
camera_id_caps = Gst.Caps.from_string("camera-id/x-picamera2")

# Detection-mode tile metadata (set by picamera_thread for tiled buffers).
tile_group_caps    = Gst.Caps.from_string("tile-group/x-bdd")
tile_count_caps    = Gst.Caps.from_string("tile-count/x-bdd")
tile_index_caps    = Gst.Caps.from_string("tile-index/x-bdd")
tile_origin_x_caps = Gst.Caps.from_string("tile-origin-x/x-bdd")
tile_origin_y_caps = Gst.Caps.from_string("tile-origin-y/x-bdd")
tile_extent_w_caps = Gst.Caps.from_string("tile-extent-w/x-bdd")
tile_extent_h_caps = Gst.Caps.from_string("tile-extent-h/x-bdd")

# Per-capture tile reassembly buffer, keyed by (camera_id, group_id). In detection
# mode the producer emits one buffer per 640x640 tile; the callback remaps each tile's
# detections to full-frame coords and accumulates them here until all tiles of the
# capture have arrived, then emits ONE merged Detections to the control queue.
_tile_groups: dict = {}

# Engaged by --simulate-detections: overlay a synthetic moving target on top of the
# (real, possibly empty) inference output so the full capture->infer->control path can be
# exercised on a bench with no real object. Set in main().
SIMULATE_DETECTIONS = False


def _read_meta_ts(buffer: Gst.Buffer, caps: Gst.Caps, default=None):
    """Return the uint64 `.timestamp` of a reference-timestamp meta, or `default` if absent."""
    m = buffer.get_reference_timestamp_meta(caps)
    return int(m.timestamp) if m is not None else default


def _synthetic_detection(phase: float) -> Detection:
    """Deterministic, slowly-moving synthetic target in normalized coords (for --simulate)."""
    cx = 0.5 + 0.18 * math.sin(phase)
    cy = 0.5 + 0.18 * math.cos(phase * 0.7)
    half = 0.03
    return Detection(bbox=Rect.from_xyxy(cx - half, cy - half, cx + half, cy + half),
                     confidence=0.8, track_id=None)


def _iou(a: Rect, b: Rect) -> float:
    inter = a.intersection(b).area()
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def _nms_merge(dets: list, iou_thresh: float = 0.5) -> list:
    """Greedy NMS to drop cross-tile duplicates (the y-overlap band is large, so a target
    near the seam is detected in two tiles). Keeps the highest-confidence box per cluster."""
    kept: list = []
    for d in sorted(dets, key=lambda d: d.confidence, reverse=True):
        if all(_iou(d.bbox, k.bbox) <= iou_thresh for k in kept):
            kept.append(d)
    return kept


def _emit_tile_group(user_data, camera_id: int, group_id: int, group: dict):
    merged = _nms_merge(group['dets'])
    user_data.detections_queue.put(
        Detections(group_id, None, merged, meta=group['meta'], camera_id=camera_id)
    )


def _accumulate_tile_group(user_data, camera_id, group_id, tile_count, remapped, meta):
    """Collect a tile's remapped detections; emit the merged capture once all tiles arrive.
    Older incomplete groups for the same camera are flushed when a newer group appears so a
    dropped tile can never stall reassembly."""
    for k in list(_tile_groups.keys()):
        if k[0] == camera_id and k[1] < group_id:
            _emit_tile_group(user_data, k[0], k[1], _tile_groups.pop(k))
    key = (camera_id, group_id)
    group = _tile_groups.get(key)
    if group is None:
        group = {'seen': 0, 'dets': [], 'meta': meta}
        _tile_groups[key] = group
    group['seen'] += 1
    group['dets'].extend(remapped)
    group['meta'] = meta
    if group['seen'] >= tile_count:
        _tile_groups.pop(key, None)
        _emit_tile_group(user_data, camera_id, group_id, group)


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


# Per-camera dedup buffer: frame_id only needs to be unique per producing camera,
# and two cameras can legitimately emit the same offset/PTS. Indexing by camera_id
# keeps a wide-camera buffer from masking a tele-camera frame as "already seen".
seen_frames_by_camera: dict[int, deque] = {}


def _read_camera_id(buffer: Gst.Buffer) -> int:
    meta = buffer.get_reference_timestamp_meta(camera_id_caps)
    if meta is None:
        return DEFAULT_CAMERA_ID
    return int(meta.timestamp)

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
    camera_id = _read_camera_id(buffer)

    # Picamera2 metadata absent when using libcamerasrc; fall back to wall-clock time
    if detection_start_timestamp_ns == 0:
        detection_start_timestamp_ns = detection_end_timestamp_ns
    if sensor_timestamp_ns == 0:
        sensor_timestamp_ns = detection_start_timestamp_ns

    seen_frames = seen_frames_by_camera.get(camera_id)
    if seen_frames is None:
        seen_frames = deque(maxlen=10)
        seen_frames_by_camera[camera_id] = seen_frames
    if frame_id in seen_frames:
        logger.warning("!!!!!!!!!!!! Skipped duplicated frame %s (camera %s)", frame_id, camera_id)
        return Gst.PadProbeReturn.OK

    seen_frames.append(frame_id)

    if False:
        pipeline_clock = pad.get_parent_element().get_clock()
        base_time      = pad.get_parent_element().get_base_time()
        now_running    = pipeline_clock.get_time() - base_time
        end_to_end_ns  = now_running - buffer.pts

        sensor_timestamp_ns = buffer.pts
        detection_start_timestamp_ns = buffer.pts
        detection_end_timestamp_ns = now_running

        logger.info("!!!!!!!!!!!!!!!!!!!!!!! e2e delay: %ums", int(end_to_end_ns / 1_000_000))


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

    # logger.debug(
    #     "frame=#%04d ByteTracker input: %d detections %s",
    #     frame_id,
    #     len(raw_dets),
    #     [(round(r.left_edge, 1), round(r.top_edge, 1), round(r.right_edge, 1), round(r.bottom_edge, 1), round(c, 3)) for r, c in raw_dets],
    # )

    track_id_map: dict[int, int] = {}
    if USE_TRACKER:
        # BYTETracker is not thread-safe; safe here because GStreamer uses a single streaming thread.
        active_tracks = user_data.tracker.update(dets_array, frame_id)

        logger.debug(
            "frame=#%04d ByteTracker output: %d active tracks %s",
            frame_id,
            len(active_tracks),
            [(t.track_id, t.state, round(t.score, 3)) for t in active_tracks],
        )

        # Build index → track_id map before constructing Detection objects
        temp_rects = [r for r, _ in raw_dets]
        for track in active_tracks:
            idx = _match_track_to_detection(track.det_bbox, temp_rects)
            logger.debug(
                "frame=#%04d ByteTracker match: track_id=%d det_bbox=%s → det_idx=%s",
                frame_id,
                track.track_id,
                [round(v, 1) for v in track.det_bbox],
                idx,
            )
            if idx is not None:
                track_id_map[idx] = track.track_id

        logger.debug(
            "frame=#%04d ByteTracker track_id_map: %s (unmatched det indices: %s)",
            frame_id,
            track_id_map,
            [i for i in range(len(raw_dets)) if i not in track_id_map],
        )

    # Construct immutable Detection objects with track_id set at creation time. For a tiled
    # (detection-mode) buffer these coords are TILE-LOCAL [0..1]; for a whole frame they are
    # already full-frame [0..1].
    detections_list = [
        Detection(
            bbox=rect,
            confidence=conf,
            track_id=track_id_map.get(i),
        )
        for i, (rect, conf) in enumerate(raw_dets)
    ]

    meta = FrameMetadata(
        capture_timestamp_ns=sensor_timestamp_ns,
        detection_start_timestamp_ns=detection_start_timestamp_ns,
        detection_end_timestamp_ns=detection_end_timestamp_ns,
    )

    # Detection-mode tiled buffer: remap to full-frame coords and reassemble the 2x2 capture.
    group_id = _read_meta_ts(buffer, tile_group_caps)
    tile_count = _read_meta_ts(buffer, tile_count_caps, 1)
    if group_id is not None and tile_count and tile_count > 1:
        tile_index = _read_meta_ts(buffer, tile_index_caps, 0)
        origin = XY(_read_meta_ts(buffer, tile_origin_x_caps, 0) / 1_000_000,
                    _read_meta_ts(buffer, tile_origin_y_caps, 0) / 1_000_000)
        extent = XY(_read_meta_ts(buffer, tile_extent_w_caps, 1_000_000) / 1_000_000,
                    _read_meta_ts(buffer, tile_extent_h_caps, 1_000_000) / 1_000_000)

        # Inject the synthetic target into tile 0 (local coords) BEFORE remap so the full
        # remap + reassemble path is exercised with a non-empty box.
        if SIMULATE_DETECTIONS and tile_index == 0:
            detections_list.append(_synthetic_detection(group_id / 15.0))

        remapped = [
            Detection(bbox=remap_tile_bbox(d.bbox, origin, extent),
                      confidence=d.confidence, track_id=d.track_id, class_id=d.class_id)
            for d in detections_list
        ]
        _accumulate_tile_group(user_data, camera_id, group_id, tile_count, remapped, meta)
        return Gst.PadProbeReturn.OK

    # Whole-frame buffer (pursuit / detection-mode off): emit directly.
    if SIMULATE_DETECTIONS:
        detections_list.append(_synthetic_detection(frame_id / 15.0))

    user_data.detections_queue.put(
        Detections(
            frame_id,
            frame,
            detections_list,
            meta=meta,
            camera_id=camera_id,
        )
    )

    return Gst.PadProbeReturn.OK


class App(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data, parser=None, video_output_path = None, video_output_chunk_length_s = 30, video_filename_base=None, record_videos=True, flight_mode=None):
        self.video_output_directory = video_output_path or '.'
        self.video_output_chunk_length_s = video_output_chunk_length_s or 30
        self.video_filename_base = video_filename_base
        self.record_videos = record_videos
        # Set BEFORE super().__init__ because that builds the pipeline: get_pipeline_string reads
        # flight_mode to choose the dual-branch cropper detection pipeline, and
        # _detection_flexible_source so the source capsfilter doesn't pin width/height (producer
        # engine flips appsrc caps between tile and full frame; cropper engine uses a constant
        # whole-frame size). run()/picamera_thread pick up the same flight_mode.
        self.flight_mode = flight_mode
        self._detection_flexible_source = bool(flight_mode is not None and flight_mode.enabled)
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
        # Normalize to CONSTANT format+size+PAR before x264enc (matroskamux refuses any caps change
        # after the first, and x264enc re-emits codec_data when its input caps change). In producer
        # detection mode the source flips 640x640 tiles -> 1280x720 full frame at the switch;
        # videoscale absorbs the size and the pinned caps keep x264enc's input byte-identical.
        # Framerate is deliberately NOT pinned: the detection-mode recording rate-limiter
        # (_install_recording_rate_limiter) DROPS buffers to recording_fps via a pad probe — which
        # leaves caps untouched (a videorate would tie the output-caps framerate to the rate and
        # break the muxer when the rate reverts to full in pursuit). Light recording while detecting
        # frees CPU for the detection pipeline (the dominant cost there).
        return f'''
            videoscale \
            ! videoconvert \
            ! video/x-raw, format=I420, width={self.video_width}, height={self.video_height}, pixel-aspect-ratio=1/1 \
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

    start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    configure_logging(level = logging.DEBUG, log_file_name=start_time_str)
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

    # maxsize=1 so the control loop always acts on the FRESHEST detection.
    # The producer (camera->pipeline->callback) runs ~14 fps but the drone loop only
    # ~8.5 fps; with a deep queue + oldest-first get() the loop chewed through a ~20-frame
    # backlog, making every decision act on a frame ~1.3 s stale (the dominant latency).
    # With size 1 the callback overwrites, so get() returns the latest frame and stale
    # frames are dropped instead of queued (same drop count, but keeps new not old).
    detections_queue = OverwriteQueue(maxsize=1)
    output_queue = OverwriteQueue(maxsize=200)


    event = threading.Event()

    arg_parser = get_default_parser()
    arg_parser.add_argument('--action', type=str, choices=["platform", "drone"])
    arg_parser.add_argument('--connection_string', type=str, default=None)
    # Verification harness for dual-camera switching. When set and 2+ cameras
    # are configured, a daemon thread calls CameraSwitcher.toggle() every N
    # seconds so we can confirm in the log that the switch propagates: each
    # toggle yields a `[cam*]` prefix flip in picamera_thread alive lines,
    # a `CAMERA SWITCH` warning in drone_controller, and a refreshed FOV in
    # the angle math. Production policy (e.g. switch by target distance) will
    # call the same set_active() / toggle() API.
    arg_parser.add_argument('--test-camera-switch-s', type=float, default=None,
                            help='Toggle active camera every N seconds (dual-camera verification only)')
    # Live annotated preview. Adds an OpenCVShowImageSink to the debug MultiSink
    # so the recorded frames are also shown in a window. Requires a display
    # (bdd.sh exports DISPLAY=:0); harmless to omit on headless runs.
    arg_parser.add_argument('--preview', action='store_true',
                            help='Show a live annotated preview window (OpenCVShowImageSink); requires a display')
    # Pre-flight detection mode. When on, the drone arms but stays grounded and scans a 2x2
    # grid of native HEF-input tiles (wider search) until N consecutive detections establish a
    # trajectory, then switches to the in-flight pursuit path. Off by default (legacy behavior).
    arg_parser.add_argument('--detection-mode', action='store_true',
                            help='Enable pre-flight detection mode (tiled search before pursuit)')
    arg_parser.add_argument('--detection-camera-id', type=int, default=None,
                            help='Camera id used to scan in detection mode (default: config / 0)')
    arg_parser.add_argument('--detection-engine', choices=['producer', 'cropper'], default=None,
                            help="Detection tiling engine: 'producer' (square tiles, batch-1) or "
                                 "'cropper' (hailotilecropper, batched, unthrottled)")
    arg_parser.add_argument('--detection-recording-fps', type=int, default=None,
                            help='Throttle the debug recorder to this fps while detecting (0 = full rate)')
    arg_parser.add_argument('--simulate-detections', action='store_true',
                            help='Overlay a synthetic moving target on real inference output (bench testing)')

    control_config = {
        'confidence_min': 0.4,
        'confidence_move': 0.3,

        'thrust_takeoff' : 1.0,
        'thrust_cruise' : 0.53,
        'thrust_hover' : 0.4,

        'thrust_min': 0.4,
        'thrust_max': 0.9,

        'thrust_dynamic': False,
        'thrust_proportional_to_distance' : True,
        'thrust_proportional_to_distance_far_coeff' : 1,
        'thrust_proportional_to_distance_medium_distance_m' : 20,
        'thrust_proportional_to_distance_medium_coeff'      : 0.9,
        'thrust_proportional_to_distance_near_distance_m'   : 10,
        'thrust_proportional_to_distance_near_coeff'        : 1.1,

        'target_lost_fade_per_frame': 0.99,
        'target_estimator_clear_history_after_target_lost_frames' : 3,

        'estimation_3d': True,
        'estimation_3d_method': 'numpy',
        'estimation_3d_use_initial_velocity' : False,

        'estimation_lookahead_frames': 1,
        'estimation_lookahead_dynamic': True,
        'estimation_lookahead_dynamic_frames_max' : 5,
        'estimation_lookahead_dynamic_sqrt': False,
        'estimation_lookahead_dynamic_factor': 0.1,
        'estimation_lookahead_dynamic_frames_near':   0,
        'estimation_lookahead_dynamic_frames_medium': 0,
        'estimation_lookahead_dynamic_frames_far':    0, # can't be too big -- estimation will be too FAAR away.

        'optical_methods_to_refine_target_size_and_center': True,
        'adjust_aim_point_at_edge_of_frame': True,
        'adjust_aim_point_at_edge_of_frame_threshold': 0.01,
        'adjust_aim_point_at_edge_of_frame_max_size': 0.25, # w*h, e.g. w=0.5, h=0.5, size=0.25

        'pd_coeff_p': XY(8, 2), # per-axis P gain (x, y)
        'pd_coeff_d': 0, #-1, # -1
        # stating from platform with guides, make sure that initial takeoff is almost perpendicular
        'pd_coeff_p_safe_min': XY(0.1, 0.1),
        'pd_coeff_p_min' : XY(0.5, 0.5),
        'pd_coeff_p_max' : XY(4, 2),

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

        # Multi-camera support. The 'camera' dict is the single source of
        # truth: top-level keys (width/height/fps/video_format/active_id/
        # switch_to_wide_size/switch_to_zoom_size) are SHARED across all
        # cameras (a single appsrc downstream demands identical caps). The
        # nested 'cameras' list holds per-camera CameraConfig dicts; the
        # producer spawns one picamera thread per entry but only the camera
        # with camera_id == active_id feeds inference at any moment. The
        # drone/platform controller looks up frame_angular_size_deg from
        # the active config so the wide FOV is never applied to a tele
        # frame. zoom_factor is auto-derived by CameraSwitcher from the FOVs.
        'camera': {
            'width': 1280,
            'height': 720,
            'fps': 30,
            'video_format': 'RGB',
            'active_id': 0,
            # Relative size of the tracked object (or bigger) that triggers switching to wide-angle camera.
            # With wide 107° / zoom 14° FOVs (zoom_factor ~11x), 0.25 on zoom -> ~0.023 on wide after the
            # switch, leaving a ~1.5x margin above switch_to_wide_size to prevent immediate flip-back.
            'switch_to_wide_size': 0.25,
            # Relative size of the tracked object (or smaller) that triggers switching to narrow-angle (zoomed) camera.
            # 0.015 on wide -> ~0.165 on zoom after the switch, ~1.5x below switch_to_zoom_size.
            'switch_to_zoom_size': 0.015,

            'cameras': [
                # NOTE: must be at least 1 camera
                dict(
                    camera_id=0,
                    name='wide',
                    sensor_index=0,
                    frame_angular_size_deg=XY(107, 85),
                ),
                # dict(
                #     camera_id=1,
                #     name='zoom',
                #     sensor_index=1,
                #     frame_angular_size_deg=XY(14, 8),
                # ),
            ],
        },


        # 'target_size_m' : XY(0.2, 0.2),             # baloon
        'target_size_m' : XY(2, 2),             # large baloon
        # 'target_size_m': XY(1.2, 1.2),            # shahed small
        # 'target_size_m' : XY(3.5, 2.5),             # shahed large
        # 'target_size_m' : XY(1_000_000, 1_000_000), # SUN

        # Inertia correction is in another branch and not used since telemetry data is weird sometimes,
        # leading to completely invalid corrections.
        # 'inertia_correction_gain' : 0, #-0.02, # 0.01 #, 1.0, etc
        # 'inertia_correction_limits': XY(1, 1),
        # 'inertia_correction_min_speed_ms': 5,

        'safe_takeoff_period_ns': 1_000_000_000,
        'delay_takeof_until_n_detection_frames' : 10,

        # Pre-flight detection mode (values-only; the live FlightModeController is built from
        # this and passed explicitly, never embedded in a config dict). When enabled, the drone
        # scans `tiles_x`x`tiles_y` native `tile_size` (= HEF input) crops per frame; overlap is
        # auto-computed from the frame/tile sizes. After `switch_after_consecutive_detections`
        # consecutive detections it switches to pursuit and takes off.
        'detection_mode': {
            'enabled': False,
            'camera_id': 0,
            # 'producer' = picamera crops square HEF tiles, batch-1, callback reassembly (throttled);
            # 'cropper'  = picamera pushes whole frame_size frames, pipeline hailotilecropper +
            #              BATCHED hailonet + hailotileaggregator, unthrottled, switch via valve flip.
            'engine': 'producer',
            'tiles_x': 2,
            'tiles_y': 2,
            'overlap': 0.1,                 # cropper engine: tile overlap fraction
            'tile_size': (640, 640),        # producer engine: square native tile (= HEF input)
            'capture_fps': 10,              # producer engine: throttle so all tiles/capture clear
            'frame_size': (1332, 990),      # cropper engine: whole-frame size before tiling
            'batch_size': None,             # cropper engine: hailonet batch (default = tiles_x*tiles_y)
            'recording_fps': 3,             # throttle the debug recorder to this fps WHILE detecting
            'switch_after_consecutive_detections': 10,
        },
        # Pursuit: how long the target may stay out of sight (s) before reverting to a level
        # loiter (hover); within this window the controller keeps flying the estimated trajectory.
        'pursuit_target_lost_tolerance_s': 1.0,

        'aim_point': XY(0.5, 0.5),
        'aim_point_max_offset': XY(0.5, 0.6),

        'follow_target_position_ned' : False,

        'drone' : {
            # 'connection_string' : 'udpout://192.168.0.3:14540', # direct comm with drone, slow to start, fails to connect on client restart
            #'connection_string' : 'udp://:14550', # mavlink-rounterd
            # 'connection_string' : 'serial:///dev/serial/by-id/usb-Auterion_PX4_FMU_v6X.x_0-if00',
            'connection_string' : 'usb',
            'config' : {
                'upside_down_angle_deg': 130,
                'upside_down_hold_s': 0.2,
                'use_set_attitude': False,
                'min_lift_fraction': 0.1,
                'lift_velocity_headroom_ms': 3.0, # upward velocity when tilt angle restirctions are relaxed significantly
                'lift_accel_headroom_mss': 5.0, # upward acceleration when tilt angle restirctions are relaxed significantly

                # Belly-down yaw assist: yaw the nose toward the ground (forward
                # axis points down) using the accelerometer-estimated gravity dir.
                'belly_down_yaw': True,                  # set False to disable
                'belly_down_yaw_kp': 1.5,                # deg/s per deg of heading error
                'belly_down_yaw_max_rate_deg_s': 90.0,   # clamp on commanded yaw rate
                'belly_down_min_horizontal_g_mss': 2.0,  # min in-plane gravity to engage
            }
        },

        'DEBUG': DEBUG,

        'bytetrack_track_thresh':   0.3,
        'bytetrack_det_thresh':     0.35,
        'bytetrack_match_thresh':   0.3,
        'bytetrack_track_buffer':   30,
        'bytetrack_frame_rate':     30,
        'bytetrack_match_max_dist':    0.2,
        'bytetrack_recovery_max_dist': None,
        'bytetrack_nms_thresh':        0.3,
        'bytetrack_nms_dist_thresh':   0.06,
    }
    global USE_TRACKER
    USE_TRACKER = False

    byte_track_config_params = {k.removeprefix('bytetrack_') : v for k, v in control_config.items() if k.startswith('bytetrack_')}
    for k in byte_track_config_params:
        control_config.pop('bytetrack_' + k)

    bytetracker = BYTETracker(
        **byte_track_config_params
        # track_thresh=control_config['bytetrack_track_thresh'],
        # det_thresh=control_config['bytetrack_det_thresh'],
        # match_thresh=control_config['bytetrack_match_thresh'],
        # track_buffer=control_config['bytetrack_track_buffer'],
        # frame_rate=control_config['bytetrack_frame_rate'],
        # match_max_dist=control_config.get('bytetrack_match_max_dist'),
        # recovery_max_dist=control_config.get('bytetrack_recovery_max_dist'),
        # nms_thresh=control_config.get('bytetrack_nms_thresh'),
        # nms_dist_thresh=control_config.get('bytetrack_nms_dist_thresh'),
    )

    user_data = user_app_callback_class(detections_queue, bytetracker)
    user_data.use_frame = True

    # Detection-mode config (values-only). Popped here so it is not forwarded to the
    # controller as a dict; the live FlightModeController is built below and passed explicitly.
    detection_mode_cfg = control_config.pop('detection_mode', None) or {}

    # Build CameraSwitcher from the 'camera' dict in control_config.
    # The dict is the kwargs for CameraSwitcher except for 'cameras' which
    # is the list of per-camera dicts (each one a CameraConfig kwargs).
    camera_block = control_config.pop('camera', None) or {}
    camera_dicts = camera_block.pop('cameras', None) or []
    camera_switcher = None
    if camera_dicts:
        camera_configs = [CameraConfig(**d) for d in camera_dicts]
        camera_switcher = CameraSwitcher(camera_configs, **camera_block)
        logger.info("!!! Cameras configured: %s, active=%d, shared caps: %dx%d@%dfps %s, thresholds: wide>=%.3f zoom<=%.3f",
                    [(c.camera_id, c.name) for c in camera_configs],
                    camera_switcher.active_id(),
                    camera_switcher.width, camera_switcher.height,
                    camera_switcher.fps, camera_switcher.video_format,
                    camera_switcher.switch_to_wide_size, camera_switcher.switch_to_zoom_size)

    # Build the FlightModeController BEFORE App() — App.__init__ builds the pipeline, and
    # get_pipeline_string needs flight_mode to choose the dual-branch cropper detection pipeline.
    # CLI flags override the values-only config block; parse_known_args because App adds more args.
    early_opts, _ = arg_parser.parse_known_args()
    fm_enabled = bool(detection_mode_cfg.get('enabled', False) or early_opts.detection_mode)
    fm_camera_id = early_opts.detection_camera_id
    if fm_camera_id is None:
        fm_camera_id = detection_mode_cfg.get('camera_id', DEFAULT_CAMERA_ID)
    fm_engine = early_opts.detection_engine or detection_mode_cfg.get('engine', 'producer')
    flight_mode = FlightModeController(
        enabled=fm_enabled,
        detection_camera_id=fm_camera_id,
        engine=fm_engine,
        tiles_x=detection_mode_cfg.get('tiles_x', 2),
        tiles_y=detection_mode_cfg.get('tiles_y', 2),
        overlap=detection_mode_cfg.get('overlap', 0.1),
        tile_size=detection_mode_cfg.get('tile_size', (640, 640)),
        capture_fps=detection_mode_cfg.get('capture_fps', 10),
        frame_size=detection_mode_cfg.get('frame_size', (1332, 990)),
        batch_size=detection_mode_cfg.get('batch_size', None),
        recording_fps=(early_opts.detection_recording_fps if early_opts.detection_recording_fps is not None
                       else detection_mode_cfg.get('recording_fps', 3)),
        switch_after_consecutive_detections=detection_mode_cfg.get('switch_after_consecutive_detections', 10),
    )

    app = App(
        app_callback,
        user_data,
        parser=arg_parser,
        video_output_chunk_length_s=10,
        video_output_path='./_DEBUG',
        video_filename_base=f"RAW_{start_time_str}",
        record_videos=True,
        flight_mode=flight_mode)
    if camera_switcher is not None:
        # Picked up by GStreamerApp.run() to spawn one thread per CameraConfig.
        app.camera_switcher = camera_switcher

    global SIMULATE_DETECTIONS
    SIMULATE_DETECTIONS = bool(app.options_menu.simulate_detections)
    if flight_mode.enabled:
        if camera_switcher is not None:
            if camera_switcher.get_config(flight_mode.detection_camera_id) is not None:
                camera_switcher.set_active(flight_mode.detection_camera_id)
            else:
                logger.error("!!! detection_camera_id=%d not in configured cameras %s; detection mode may get no frames",
                             flight_mode.detection_camera_id, [c.camera_id for c in camera_switcher.configs()])
        if flight_mode.engine == 'cropper':
            logger.warning("!!! DETECTION MODE ENABLED [cropper]: camera_id=%d, %dx%d tiling of %s frame, batch=%d, "
                           "switch after %d consecutive detections; simulate=%s",
                           flight_mode.detection_camera_id, flight_mode.tiles_x, flight_mode.tiles_y,
                           flight_mode.frame_size, flight_mode.batch_size,
                           flight_mode.switch_after_consecutive_detections, SIMULATE_DETECTIONS)
        else:
            logger.warning("!!! DETECTION MODE ENABLED [producer]: camera_id=%d, %dx%d tiles of %s, "
                           "switch after %d consecutive detections; simulate=%s",
                           flight_mode.detection_camera_id, flight_mode.tiles_x, flight_mode.tiles_y,
                           flight_mode.tile_size, flight_mode.switch_after_consecutive_detections, SIMULATE_DETECTIONS)
    else:
        logger.info("!!! Detection mode OFF (pursuit-only behavior); simulate=%s", SIMULATE_DETECTIONS)

    # Optional dual-camera switching verification harness. The thread drives
    # the same CameraSwitcher.toggle() that production policy will use, so a
    # successful run here is direct evidence that the production switch path
    # works end-to-end (producer gate, callback dedup, controller cache purge,
    # FOV refresh). Disabled unless --camera-switch-test-s is passed.
    test_switch_interval_s = app.options_menu.test_camera_switch_s
    if test_switch_interval_s and camera_switcher is not None and len(camera_switcher.configs()) >= 2:
        def _switch_test_thread():
            while True:
                time.sleep(test_switch_interval_s)
                new_id = camera_switcher.toggle()
                cfg = camera_switcher.active_config()
                logger.warning(
                    "!!! CAMERA SWITCH TEST: toggled to camera_id=%s (%s, FOV=%s, zoom=%.2fx)",
                    new_id, cfg.name, cfg.frame_angular_size_deg, cfg.zoom_factor,
                )
        threading.Thread(target=_switch_test_thread, name="camera-switch-test", daemon=True).start()
        logger.warning("!!! Dual-camera switching test harness enabled: toggle every %.2fs", test_switch_interval_s)

    logger.info("!!! Config: %s", control_config)
    if DEBUG:
        import math
        nan = math.nan
        control_config['debug_telemetry_dict'] = {'attitude_euler': {'pitch_deg': 1.1012928485870361, 'roll_deg': -2.5803990364074707, 'timestamp_us': 4597491000, 'yaw_deg': -139.22280883789062}, 'odometry': {'angular_velocity_body': {'pitch_rad_s': 0.005480111576616764, 'roll_rad_s': -0.004354139324277639, 'yaw_rad_s': 0.00451350212097168}, 'child_frame_id': '1 (BODY_NED)', 'frame_id': '1 (BODY_NED)', 'pose_covariance': {'covariance_matrix': (0.0006513984990306199, nan, nan, nan, nan, nan, 0.0006680359947495162, nan, nan, nan, nan, 0.0795387253165245, nan, nan, nan, 0.00014700590691063553, nan, nan, 0.0001532444730401039, nan, 0.0037478541489690542)}, 'position_body': {'x_m': 10155.28515625, 'y_m': 1922.0908203125, 'z_m': 0.21114209294319153}, 'q': {'timestamp_us': 0, 'w': -0.3483775854110718, 'x': -0.0011684682685881853, 'y': -0.024444160982966423, 'z': 0.9370348453521729}, 'time_usec': 4597481426, 'velocity_body': {'x_m_s': 0.014301709830760956, 'y_m_s': -0.007258167490363121, 'z_m_s': 0.04777885600924492}, 'velocity_covariance': {'covariance_matrix': (0.002916615456342697, nan, nan, nan, nan, nan, 0.0030234858859330416, nan, nan, nan, nan, 0.005942340008914471, nan, nan, nan, nan, nan, nan, nan, nan, nan)}}, 'landed_state': None, 'imu': {'acceleration_frd': {'down_m_s2': -10.367236137390137, 'forward_m_s2': -0.11148512363433838, 'right_m_s2': 0.5069471597671509}, 'angular_velocity_frd': {'down_rad_s': -0.0006937360158190131, 'forward_rad_s': 0.004322248511016369, 'right_rad_s': 0.0015081189339980483}, 'magnetic_field_frd': {'down_gauss': 0.2559056580066681, 'forward_gauss': -0.3339233696460724, 'right_gauss': 0.3074171245098114}, 'temperature_degc': 15.0, 'timestamp_us': 4597496423}}

    if app.options_menu.connection_string:
        control_config['drone']['connection_string'] = app.options_menu.connection_string

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
                camera_switcher= camera_switcher,
                flight_mode= flight_mode,
            )
        )
    else:
        drone_params = control_config.pop('drone')
        action_thread = threading.Thread(
            target = drone_controlling_thread,
            args = (
                drone_params.pop('connection_string'), #'udp://10.41.10.2:14540',
                drone_params.pop('config'),
                detections_queue
            ),
            kwargs= dict(
                control_config= control_config,
                output_queue= output_queue,
                signal_event_when_ready= event,
                camera_switcher= camera_switcher,
                flight_mode= flight_mode,
            ),
            name = "Drone"
        )

    # Uncaught crash in the control thread (e.g. drone can't connect to the
    # FMU) used to leave the pipeline running with nothing behind it. Hook
    # the failure: unblock the main thread waiting on `event`, then ask the
    # GStreamer main loop to shut down so app.run() returns and main() can
    # re-raise.
    fatal_action_thread_error: dict = {}
    _default_excepthook = threading.excepthook
    def _action_thread_excepthook(args):
        if args.thread is not action_thread:
            _default_excepthook(args)
            return
        logger.error(
            "Control thread '%s' crashed (%s: %s); shutting down application",
            args.thread.name, args.exc_type.__name__, args.exc_value,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        fatal_action_thread_error['error'] = args.exc_value
        event.set()
        GLib.idle_add(app.shutdown)
        sys.exit(-1)

    threading.excepthook = _action_thread_excepthook

    action_thread.start()
    app.add_shutdown_callback(lambda: detections_queue.put(STOP))

    sinks = [
        # RtspStreamerSink(30, 8554),
        RecorderSink(30,
            "./_DEBUG",
            segment_seconds=10,
            filename_base=f"debug_{start_time_str}",
        ),
    ]
    if app.options_menu.preview:
        logger.info("!!! Preview enabled: adding OpenCVShowImageSink to debug sink")
        sinks.append(OpenCVShowImageSink(window_title='DEBUG IMAGE'))
    sink = MultiSink(sinks)

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

    # Re-raise a control-thread crash captured by the excepthook so the
    # process exits non-zero (caller scripts / systemd / CI must be able
    # to tell a clean stop from a drone-connect failure).
    if 'error' in fatal_action_thread_error:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
