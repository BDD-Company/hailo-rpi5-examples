#!/usr/bin/env python

from math import nan
from pathlib import Path

from dataclasses import dataclass, field
import threading

import os
import sys
import datetime
import time

import hailo
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from app_base import GStreamerDetectionApp
from tiling_policy import (TilingLadderPolicy, TilingSwitchCoordinator,
                           BranchStallWatchdog, build_ladder)
from drone_controller import drone_controlling_thread
from platform_controller import platform_controlling_thread

# from mavsdk.telemetry import EulerAngle

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from bytetrack import BYTETracker

from helpers import FrameMetadata, Rect, XY,  Detection, Detections, MoveCommand, STOP, kalman_or_raw_bbox
from helpers import CameraConfig, CameraSwitcher, DEFAULT_CAMERA_ID
from frame_utils import Frame
from frame_order import FrameOrderGuard, resolve_frame_id
from config import Config
from dataclasses import asdict, replace
from OverwriteQueue import OverwriteQueue
from debug_output import debug_output_thread, DEBUG_OUTPUT_STOP
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
# Phase-2 noise reduction: build matched detections from the tracker's smoothed
# (Kalman) bbox instead of the raw detector rect. Set from config in main().
USE_KALMAN_BBOX = False

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self, detections_queue, tracker: BYTETracker):
        super().__init__()
        self.detections_queue = detections_queue
        self.tracker = tracker

        # ns-per-frame used to turn the branch-independent buffer.pts into a ~1-per-frame
        # index (see normalized_frame_id / frame_order.resolve_frame_id). Set by main()
        # from config.camera.fps; None => the id falls back to the per-branch buffer.offset.
        self.source_frame_duration_ns = None

        # Runtime tiling-ladder switching. The decision lives in the pure, unit-tested
        # TilingLadderPolicy; the serialization in TilingSwitchCoordinator. This class
        # only decides WHEN to hand the (blocking) switch to a worker thread, because
        # the GStreamer streaming thread must never wait on a handover. Wired by main()
        # via attach_tiling_ladder(); None until then.
        self._coordinator = None
        self.auto_switch = False

    def attach_tiling_ladder(self, switch_fn, policy: TilingLadderPolicy,
                             auto_switch: bool = False):
        """Route every tier switch through one serialized coordinator.

        `switch_fn(tier_i) -> bool` is the blocking GStreamer handover
        (app.switch_to_tier). `auto_switch` enables the per-frame size/loss policy; even
        with it off the coordinator is installed, so the --test-switch-s harness goes
        through the same serialized path and keeps the policy's tier truthful."""
        self._coordinator = TilingSwitchCoordinator(policy, switch_fn)
        self.auto_switch = auto_switch

    @property
    def coordinator(self):
        """The serialized switch entry point; None until attach_tiling_ladder()."""
        return self._coordinator

    def request_tier(self, tier_i: int) -> bool:
        """Blocking, thread-safe tier switch. The ONLY entry point — used by the policy
        worker and by the --test-switch-s harness alike."""
        if self._coordinator is None:
            logger.warning("request_tier: the tiling ladder is not configured; ignoring")
            return False
        return self._coordinator.request(tier_i)

    def note_tiling(self, side):
        """Per-frame ladder policy, on the GStreamer streaming thread. `side` is
        max(bbox_w, bbox_h) of the primary tracked target (normalized 0..1), or None
        when no target is matched — which starts the target-lost escalation.

        The coordinator returns a target only when a switch should START (never while
        one is in flight), so at most one worker thread exists at a time. No-op unless
        auto_switch is on."""
        if not self.auto_switch or self._coordinator is None:
            return
        target = self._coordinator.note(side, time.monotonic())
        if target is not None:
            # Never block the streaming thread: the handover waits up to ~1s for the
            # incoming branch's first buffer.
            threading.Thread(target=self._coordinator.request, args=(target,),
                             name="tiling-policy-switch", daemon=True).start()


# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

sensor_timestamp_caps = Gst.Caps.from_string("timestamp/x-picamera2-sensor")
unix_timestamp_caps = Gst.Caps.from_string("timestamp/x-unix")
frame_id_caps = Gst.Caps.from_string("frame-id/x-picamera2")
camera_id_caps = Gst.Caps.from_string("camera-id/x-picamera2")


def normalized_timestamp(ts):
    if ts is not None:
        if isinstance(ts, Gst.ReferenceTimestampMeta):
            return ts.timestamp
        else:
            return int(ts)
    else:
        return 0

def normalized_frame_id(buffer: Gst.Buffer, frame_meta, frame_duration_ns=None) -> int:
    """Return a stable, branch-independent, ~1-per-frame identifier for this buffer.

    Thin GStreamer adapter around frame_order.resolve_frame_id: map the buffer's GST
    NONE sentinels to plain None, delegate the priority logic (producer meta ->
    pts-as-frame-index -> offset), and substitute a wallclock last resort when nothing
    identifies the frame. `frame_duration_ns` (from config.camera.fps, via user_data)
    is what turns the branch-independent pts into a frame index; see resolve_frame_id
    for why pts -- not the per-branch buffer.offset -- is the id the tiled path needs.
    """
    meta_ts = frame_meta.timestamp if frame_meta is not None else None
    pts = None if buffer.pts == Gst.CLOCK_TIME_NONE else int(buffer.pts)
    offset = None if buffer.offset == Gst.BUFFER_OFFSET_NONE else int(buffer.offset)
    fid = resolve_frame_id(meta_ts, pts, offset, frame_duration_ns)
    return fid if fid is not None else time.monotonic_ns()


# Strict per-camera frame-order invariant for the inference/tracker path. Frames
# MUST reach ByteTracker (and, via the queue, the control loop) in monotonically
# increasing per-camera order. This guard drops any stale/reordered/duplicate
# frame (id <= last accepted for that camera) while letting forward skips through
# — so a smoothly-moving object can never step backward in time. It supersedes
# the old last-10 dedup: a duplicate is just id == last, which it also rejects.
frame_order_guard = FrameOrderGuard()


def _read_camera_id(buffer: Gst.Buffer) -> int:
    meta = buffer.get_reference_timestamp_meta(camera_id_caps)
    if meta is None:
        return DEFAULT_CAMERA_ID
    return int(meta.timestamp)


# Stage-A/B latency observability (part of the capture→command latency campaign).
# Accumulate per-frame deltas and emit p50/p95/p99 every N frames so on-device
# benches can read Stage A (sensor→appsrc) and Stage B (appsrc→callback) straight
# from the log, even in --vision-only mode (no control thread). Cheap: two list
# appends per frame plus a sort every N. NOTE on clocks: SensorTimestamp is
# libcamera's CLOCK_BOOTTIME and the appsrc-push stamp is time.monotonic_ns();
# on a Pi that never suspends these share an origin, so the Stage-A delta is
# accurate in practice (and the offset is constant, so cross-run deltas are valid).
_LATENCY_LOG_EVERY_N = int(os.environ.get("BDD_LATENCY_LOG_EVERY_N", "100"))
_stage_a_samples_ms: list[float] = []
_stage_b_samples_ms: list[float] = []
_latency_frame_counter = 0


def _pct(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float('nan')
    i = min(len(sorted_xs) - 1, int(q / 100.0 * len(sorted_xs)))
    return sorted_xs[i]


def _record_stage_latency(stage_a_ms, stage_b_ms):
    global _latency_frame_counter
    if stage_a_ms is not None:
        _stage_a_samples_ms.append(stage_a_ms)
    if stage_b_ms is not None:
        _stage_b_samples_ms.append(stage_b_ms)
    _latency_frame_counter += 1
    if _latency_frame_counter % _LATENCY_LOG_EVERY_N != 0:
        return
    a = sorted(_stage_a_samples_ms)
    b = sorted(_stage_b_samples_ms)
    logger.info(
        "!!! STAGE LATENCY (last %d frames): "
        "StageA sensor->appsrc p50=%.1f p95=%.1f p99=%.1f max=%.1f ms (n=%d) | "
        "StageB appsrc->callback p50=%.1f p95=%.1f p99=%.1f max=%.1f ms (n=%d)",
        _LATENCY_LOG_EVERY_N,
        _pct(a, 50), _pct(a, 95), _pct(a, 99), (a[-1] if a else float('nan')), len(a),
        _pct(b, 50), _pct(b, 95), _pct(b, 99), (b[-1] if b else float('nan')), len(b),
    )
    _stage_a_samples_ms.clear()
    _stage_b_samples_ms.clear()


# Detection throughput (callbacks/s) — time-based so it's meaningful even when
# tiling drops delivered fps well below capture. Logged every ~2 s.
_det_fps_last_t = None
_det_fps_count = 0

def _log_det_fps():
    global _det_fps_last_t, _det_fps_count
    _det_fps_count += 1
    now = time.monotonic()
    if _det_fps_last_t is None:
        _det_fps_last_t = now
        return
    dt = now - _det_fps_last_t
    if dt >= 2.0:
        logger.info("!!! DET FPS: %.1f (%d callbacks / %.2fs)", _det_fps_count / dt, _det_fps_count, dt)
        _det_fps_last_t = now
        _det_fps_count = 0

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
    _log_det_fps()

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    sensor_timestamp_ns  = normalized_timestamp(buffer.get_reference_timestamp_meta(sensor_timestamp_caps))
    detection_start_timestamp_ns  = normalized_timestamp(buffer.get_reference_timestamp_meta(unix_timestamp_caps))
    detection_end_timestamp_ns  = time.monotonic_ns()

    # File input (filesrc -> qtdemux -> identity sync=true) carries none of the
    # producer's timestamp metas, so without this StageB records nothing and the
    # controller's "e2e" silently collapses to stageC. Reconstruct the instant the
    # pace identity released this buffer — base_time + running_time(PTS) — which is
    # the file-path analogue of the capture timestamp. GstSystemClock runs on
    # CLOCK_MONOTONIC, the same domain as time.monotonic_ns(), so the deltas are
    # directly comparable.
    if detection_start_timestamp_ns == 0 and buffer.pts != Gst.CLOCK_TIME_NONE:
        try:
            _seg_event = pad.get_sticky_event(Gst.EventType.SEGMENT, 0)
            _element = pad.get_parent_element()
            if _seg_event is not None and _element is not None:
                _running_time = _seg_event.parse_segment().to_running_time(
                    Gst.Format.TIME, buffer.pts)
                _base_time = _element.get_base_time()
                if (_running_time != Gst.CLOCK_TIME_NONE
                        and _base_time != Gst.CLOCK_TIME_NONE):
                    detection_start_timestamp_ns = _base_time + _running_time
        except Exception:
            pass  # measurement aid only — never let it take down the frame path

    # Stage-A/B latency: record before the fallbacks below zero the deltas out.
    _sensor_present = sensor_timestamp_ns != 0
    _push_present = detection_start_timestamp_ns != 0
    _record_stage_latency(
        (detection_start_timestamp_ns - sensor_timestamp_ns) / 1e6 if (_sensor_present and _push_present) else None,
        (detection_end_timestamp_ns - detection_start_timestamp_ns) / 1e6 if _push_present else None,
    )

    frame_id = normalized_frame_id(buffer, buffer.get_reference_timestamp_meta(frame_id_caps),
                                   getattr(user_data, "source_frame_duration_ns", None))
    camera_id = _read_camera_id(buffer)

    # Picamera2 metadata absent when using libcamerasrc; fall back to wall-clock time
    if detection_start_timestamp_ns == 0:
        detection_start_timestamp_ns = detection_end_timestamp_ns
    if sensor_timestamp_ns == 0:
        sensor_timestamp_ns = detection_start_timestamp_ns

    # Strict-order invariant: drop any frame that is not strictly newer than the
    # last one we handed to the tracker for this camera (a reorder or duplicate).
    # Forward skips pass through. This should never fire in normal operation — the
    # pipeline delivers frames in capture order — so a hit means an upstream regression.
    if not frame_order_guard.accept(camera_id, frame_id):
        logger.warning("!!!!!!!!!!!! Dropping out-of-order/duplicate frame %s (camera %s, last=%s)",
                       frame_id, camera_id, frame_order_guard.last(camera_id))
        return Gst.PadProbeReturn.OK

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    # Wrap in a Frame so downstream consumers handle NV12 (a planar (Y, UV) tuple from
    # get_numpy_from_buffer) and RGB uniformly — control path uses to_gray() (free for
    # NV12), rendering paths use to_rgb(). Frame.coerce(None) stays None.
    frame = Frame.coerce(get_numpy_from_buffer(buffer, format, width, height))

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

    # feat/bytetrack-locking-eval: log the RAW (pre-tracker) detections every frame
    # so a --vision-only run (no control thread => no GOT DETECTIONS line) still
    # captures the full detection stream. Faithful for offline switch analysis AND
    # for re-running BYTETracker with swept params. xyxy + confidence per detection.
    logger.debug(
        "frame=#%04d !!! RAWDETS n=%d %s",
        frame_id, len(raw_dets),
        [(round(r.left_edge, 4), round(r.top_edge, 4), round(r.right_edge, 4),
          round(r.bottom_edge, 4), round(c, 4)) for r, c in raw_dets],
    )

    # Run ByteTracker to assign stable track IDs
    if raw_dets:
        dets_array = np.array([
            [r.left_edge, r.top_edge, r.right_edge, r.bottom_edge, c]
            for r, c in raw_dets
        ])
    else:
        dets_array = np.empty((0, 5))

    _best_conf = max((c for _, c in raw_dets), default=0.0)

    # Throttled detection summary — confirms the model is producing valid detections
    # from the current pixel format (e.g. NV12 fed correctly). Throttle on the callback
    # counter, NOT frame_id: normalized_frame_id falls back to time.monotonic_ns() when
    # no frame-id meta is present, which makes `frame_id % 30` fire pseudo-randomly.
    if user_data.get_count() % 30 == 0:
        logger.info("!!! DETS frame=#%d n=%d maxconf=%.3f", frame_id, len(raw_dets), _best_conf)

    # logger.debug(
    #     "frame=#%04d ByteTracker input: %d detections %s",
    #     frame_id,
    #     len(raw_dets),
    #     [(round(r.left_edge, 1), round(r.top_edge, 1), round(r.right_edge, 1), round(r.bottom_edge, 1), round(c, 3)) for r, c in raw_dets],
    # )

    track_id_map: dict[int, int] = {}
    track_by_idx: dict[int, object] = {}   # det index -> matched STrack (for Kalman bbox)
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
                track_by_idx[idx] = track

        logger.debug(
            "frame=#%04d ByteTracker track_id_map: %s (unmatched det indices: %s)",
            frame_id,
            track_id_map,
            [i for i in range(len(raw_dets)) if i not in track_id_map],
        )

    # Size-driven tiling ladder: side = max(w,h) of the primary tracked target (the
    # matched track with the largest max(w,h)), or None if no track matched — which
    # starts the target-lost escalation. Fed AFTER tracking so it uses stable track
    # identity rather than raw detections: an unmatched blob is clutter, and letting it
    # drive the ladder would hold the rig on a low-latency rung while the real target
    # is unseen. No-op unless the auto-switch ladder is configured (main()).
    primary_side = None
    if USE_TRACKER and track_id_map:
        # track_id_map keys are indices into raw_dets
        primary_side = max(max(raw_dets[i][0].width, raw_dets[i][0].height)
                           for i in track_id_map)
    if user_data.get_count() % 30 == 0:
        logger.info("!!! TILING-SIZE frame=#%d use_tracker=%s tracks=%d primary_side=%s",
                    frame_id, USE_TRACKER, len(track_id_map), primary_side)
    user_data.note_tiling(primary_side)

    # Construct immutable Detection objects with track_id set at creation time.
    # Phase-2: matched detections may use the tracker's smoothed Kalman bbox
    # (default-off; kalman_or_raw_bbox returns the raw rect unchanged when disabled).
    detections_list = [
        Detection(
            bbox=kalman_or_raw_bbox(rect, track_by_idx.get(i), USE_KALMAN_BBOX),
            confidence=conf,
            track_id=track_id_map.get(i),
        )
        for i, (rect, conf) in enumerate(raw_dets)
    ]

    # if len(detections) != 0:
    user_data.detections_queue.put(
        Detections(
            frame_id,
            frame,
            detections_list,
            meta = FrameMetadata(
                capture_timestamp_ns=sensor_timestamp_ns,
                detection_start_timestamp_ns = detection_start_timestamp_ns,
                detection_end_timestamp_ns=detection_end_timestamp_ns),
            camera_id=camera_id,
        )
    )

    return Gst.PadProbeReturn.OK


class App(GStreamerDetectionApp):
    def __init__(self, app_callback, user_data, config : Config, parser=None, video_output_path = None, video_output_chunk_length_s = 30, video_filename_base=None, record_videos=True, camera_switcher = None, video_format=None, ladder_grids=None, tiling_overlap=None, tile_iou_threshold=None, preview=False):
        self.video_output_directory = video_output_path or '.'
        self.video_output_chunk_length_s = video_output_chunk_length_s or 30
        self.video_filename_base = video_filename_base
        self.record_videos = record_videos
        # --preview: fan the display branch out to a live window (autovideosink) so
        # the incoming camera video can be watched during a run. Needs a display
        # (DISPLAY set), so it is opt-in and off on headless benches. Read below in
        # get_output_pipeline_string, which runs during super().__init__() ->
        # create_pipeline(), so it MUST be set before the super() call.
        self.preview = bool(preview)
        self.camera_switcher = camera_switcher
        super().__init__(app_callback, user_data, inference=config.inference, camera_settings=config.camera, parser=parser,
                         video_format=video_format, ladder_grids=ladder_grids,
                         tiling_overlap=tiling_overlap,
                         tile_iou_threshold=tile_iou_threshold)

        #NOTE: unfortunatelly that has to be string, rest of the HAILO python code depends on it
        self.sync = 'false'


    def on_stream_rewound(self):
        # LOAD-BEARING on the file path. File input derives its frame id from buffer.pts
        # (normalized_frame_id turns pts into a frame index), which restarts at 0 on the
        # loop's flush seek. Without this reset the guard's high-water mark would stay at the
        # end of the previous pass and reject every frame of every later pass, leaving the run
        # silently blind. The flush seek drops in-flight buffers, so resetting here cannot
        # admit a stale pre-rewind frame. See FrameOrderGuard.reset.
        logger.info("Video looped; resetting frame-order guard for the new pass.")
        frame_order_guard.reset()


    def get_output_pipeline_string(self, video_sink: str, sync: str = 'true', show_fps: str = 'true'):
        # Live preview branch (--preview): its own leaky, tiny queue gives it a
        # separate streaming thread; the leaky=downstream queue means a slow/stalled
        # window sink can never back-pressure the recorder or add capture latency
        # (it sheds the oldest frame instead). sync=true presents each frame on its
        # (now even, sensor-derived) PTS, so the small capture-thread bursts absorbed
        # by the 2-frame queue play back at the true ~fps cadence instead of flashing
        # by on arrival — that jitter, not any reordering, is what made the preview
        # look jagged. Frame ORDER is identical either way.
        #
        # Sink choice matters on this deployment: the Pi runs a Wayland compositor
        # (labwc) mirrored over the network by wayvnc. autovideosink there falls back
        # to kmssink (direct DRM scanout), which paints a hardware plane the compositor
        # doesn't own — wayvnc never captures it, so over VNC the preview looks like a
        # slideshow. waylandsink renders into a compositor surface that labwc composites
        # and wayvnc streams, giving a smooth remote preview. Fall back to autovideosink
        # off Wayland (X11 / local monitor).
        preview_sink = preview_sink_name()
        preview_branch = (
            'queue name=preview_queue leaky=downstream max-size-buffers=2 '
            '    max-size-bytes=0 max-size-time=0 '
            '! videoconvert '
            f'! {preview_sink} name=preview_sink sync=true'
        )

        if not self.record_videos:
            # Preview only (no recording) — or a plain fakesink terminator.
            if self.preview:
                return preview_branch
            if self.source_type == 'file':
                # A bare fakesink prerolls, holding the pipeline in PAUSED until a first
                # buffer arrives — which the file path's identity sync=true cannot hand
                # over until the clock runs in PLAYING. Deadlock: the run sat in PAUSED
                # forever. async=false skips the preroll handshake; sync=false keeps this
                # terminator from pacing anything (identity already does the pacing).
                # Scoped to file on purpose: live capture keeps the exact element it has
                # always had, so this cannot perturb the flight path.
                return "fakesink sync=false async=false"
            return "fakesink"

        record_start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        video_file_name = Path(self.video_filename_base if self.video_filename_base else f"RAW_{record_start_time_str}.mkv")

        # add "_%05d" so we get multiple files w/o overwriting anything
        video_file_name = video_file_name.stem + "_%05d" + (video_file_name.suffix if video_file_name.suffix else '.mkv')

        video_output_chunk_length_ns = self.video_output_chunk_length_s * 1000_000_000
        record_branch = f'''
            videoconvert \
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
                sink-properties="properties,buffer-mode=2,o-sync=true" \
                max-size-time={video_output_chunk_length_ns} async-finalize=true \
                location="{self.video_output_directory}/{video_file_name}"
        '''

        if not self.preview:
            return record_branch

        # Record + preview: split the display branch with a tee so the preview
        # window and the recorder each get their own leg (recorder unchanged).
        return f'''
            tee name=display_tee
            display_tee. ! {preview_branch}
            display_tee. ! {record_branch}
        '''

    def run(self, wait_event_before_starting=None):
        if wait_event_before_starting:
            wait_event_before_starting.wait()
        logger.info("!!! Starting the application (and generating frames with detections)")

        super().run()


def _stop_debug_output(output_queue):
    """Stop the debug-overlay recorder thread promptly.

    The sentinel is FIFO, so signalling alone is not enough: with a backlogged
    200-deep OverwriteQueue the thread first annotates and x264-encodes every
    pending frame before it even sees the stop. Measured on-device: 10.2 s from
    SIGINT to the thread reacting, writing ~10 s of video nobody will watch and
    blowing past the join timeout below. Those overlay frames are a debug aid, not
    control data — drop them and finalize the file instead.
    """
    output_queue.clear()
    output_queue.put(DEBUG_OUTPUT_STOP)


def preview_sink_name() -> str:
    """GStreamer sink element for the --preview window. See get_output_pipeline_string
    for why Wayland must not get autovideosink. Shared with main() so the startup log
    names the sink that will actually be built."""
    return 'waylandsink' if os.environ.get('WAYLAND_DISPLAY') else 'autovideosink'


def detect_hef_video_format(hef_path, default='RGB'):
    """Read the model's input format from the HEF and map it to a capture video
    format: 'NV12' when the input vstream order is NV12, else 'RGB'. The capture
    format MUST match the model input — the whole-buffer hailocropper requires its
    input format to equal the hailonet input format, so deriving it from the model
    prevents the "Cropper Input and output caps have different formats" crash and
    the silent format mismatches.

    Only a MISSING hailo_platform (dev host, no Hailo SDK) falls back to `default`.
    On a box that HAS the SDK, an unreadable HEF is fatal: guessing the format would
    crash the cropper mid-run. This runs in main() long before the control thread
    starts and before anything is armed, so exiting here is safe."""
    try:
        from hailo_platform import HEF
    except ImportError:
        logger.warning("hailo_platform unavailable (dev host?); assuming %s input for %s",
                       default, hef_path)
        return default

    try:
        info = HEF(str(hef_path)).get_input_vstream_infos()[0]
    except Exception as e:
        raise SystemExit(
            f"FATAL: cannot read the input format of HEF {str(hef_path)!r}: {e}\n"
            f"The capture format must match the model input; guessing it would crash the "
            f"hailocropper once frames flow. Fix --hef-path / config.inference.hef_model_path."
        ) from e

    order = getattr(info.format.order, 'name', str(info.format.order)).upper()
    return 'NV12' if 'NV12' in order else 'RGB'


def _pop_flag(argv: list[str], flag: str) -> bool:
    """Remove a boolean CLI flag from ``argv`` in place; return whether it was
    present. These flags are parsed off sys.argv directly (before the argparser)
    because they gate App() construction."""
    if flag in argv:
        argv.remove(flag)
        return True
    return False


def _pop_value(argv: list[str], flag: str, example: str = 'VALUE') -> str | None:
    """Remove ``<flag> <value>`` or ``<flag>=<value>`` from ``argv`` in place and
    return the value, or None if the flag is absent. Exits with a clear message (not
    an IndexError) if the flag is given with no value following it.

    Both spellings are accepted because argparse accepts both, and a flag popped here
    never reaches argparse — so ``--test-switch-s=10`` would otherwise sail past this
    function and die downstream as an opaque "unrecognized arguments"."""
    for i, tok in enumerate(argv):
        if tok == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value (e.g. {flag} {example})")
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if tok.startswith(flag + "="):
            value = tok[len(flag) + 1:]
            if not value:
                raise SystemExit(f"{flag} requires a value (e.g. {flag}={example})")
            del argv[i]
            return value
    return None


def _pop_deprecated_value(argv: list[str], old_flag: str, new_flag: str,
                          removal_date: str, example: str = 'VALUE') -> str | None:
    """Pop a renamed flag by its OLD name, warning loudly. Kept so scripts, runbooks
    and skills pinned to an older checkout keep working through the transition."""
    value = _pop_value(argv, old_flag, example)
    if value is not None:
        logger.warning("!!! DEPRECATED: %s is now %s; the old name stops working after %s",
                       old_flag, new_flag, removal_date)
    return value


def _parse_float(spec: str, flag: str) -> float:
    """Parse a float CLI value, exiting with a clear message (not a raw ValueError
    traceback) — matching how _parse_grid reports malformed input."""
    try:
        return float(spec)
    except ValueError:
        raise SystemExit(f"{flag} expects a number of seconds, got {spec!r}") from None


def _peek_value(argv: list[str], flag: str) -> str | None:
    """Read the value of ``<flag> <value>`` or ``<flag>=<value>`` from ``argv``
    WITHOUT removing it (unlike ``_pop_value``): argparse still consumes ``--hef-path``
    downstream, so this only needs to peek early to derive the capture format. Returns
    None if the flag is absent; exits with a clear message if it has no value."""
    for i, tok in enumerate(argv):
        if tok == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires a value (e.g. {flag} /path/to/model.hef)")
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok[len(flag) + 1:]
    return None


def _parse_grid(spec: str) -> tuple[int, int]:
    """Parse an ``NxM`` tile-grid spec (case-insensitive, e.g. ``2x2``) into
    ``(nx, ny)``. Exits with a clear message (not an opaque ValueError/IndexError)
    on anything that isn't two positive integers."""
    nx_s, sep, ny_s = spec.lower().partition("x")
    if not sep or not nx_s or not ny_s:
        raise SystemExit(f"tile grid must look like NxM (e.g. 2x2), got {spec!r}")
    try:
        nx, ny = int(nx_s), int(ny_s)
    except ValueError:
        raise SystemExit(f"tile grid must be two integers NxM, got {spec!r}")
    if nx < 1 or ny < 1:
        raise SystemExit(f"tile grid values must be >= 1, got {nx}x{ny}")
    return nx, ny


def main():
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str

    # One timestamp for the whole run. bdd.sh exports BDD_START_TIME so the single
    # durable log (written by scripts/durable_tee.py) and the MKV segments share a
    # run id; fall back to now() when app.py is launched directly.
    start_time_str = os.environ.get("BDD_START_TIME") or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    configure_logging(level = logging.DEBUG, log_file_name=start_time_str)
    # shushing verbose loggers
    logging.getLogger("picamera2").setLevel(logging.WARNING)
    logging.getLogger("mavsdk_server").setLevel(logging.ERROR)

    global DEBUG
    if "--DEBUG" in sys.argv:
        DEBUG=True
        sys.argv.remove('--DEBUG')

    # --config: optional path to an alternate config YAML (defaults to config.yaml
    # next to this file). Parsed from argv directly — like --DEBUG above — because
    # the config is loaded before the main argparser runs. Lets a test/bench rig
    # point at a single-camera (or otherwise tweaked) config without touching the
    # production config.yaml.
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if "--config" in sys.argv:
        i = sys.argv.index("--config")
        config_path = Path(sys.argv[i + 1])
        del sys.argv[i:i + 2]

    # These flags are parsed off sys.argv directly (before the argparser) because
    # they gate App() construction below. NxM grids go through _parse_grid, which
    # validates them; a missing value or malformed grid exits with a clear message.
    # Test/verification-harness flags are prefixed --test-* (see --test-camera-switch-s).

    # --no-record: force RAW recording off regardless of config (frees ~1 CPU core
    # for inference/tiling).
    no_record_flag = _pop_flag(sys.argv, "--no-record")

    # --preview: also show the incoming camera video in a live window. Opt-in because
    # it needs a display; on a headless bench run without it (see the
    # "app.py --preview (DISPLAY=:0)" launch config).
    preview_flag = _pop_flag(sys.argv, "--preview")

    # The tile grid is NOT a CLI knob any more: config.tiling.ladder is the sole tiling
    # control (--tiles / --switch-tiles are gone with the binary switch they configured).
    # For a whole-frame-only run, empty the ladder in the config.

    # --test-switch-s N: manual handover validation — step through the ladder rungs every
    # N seconds. Harness only; needs a >= 2 rung ladder. Old name --switch-test-s still
    # works (deprecated) so runbooks/skills on older checkouts keep running.
    _TEST_SWITCH_S_REMOVAL = "2026-09-30"
    switch_test_s_spec = _pop_value(sys.argv, "--test-switch-s", example="10")
    if switch_test_s_spec is None:
        switch_test_s_spec = _pop_deprecated_value(
            sys.argv, "--switch-test-s", "--test-switch-s", _TEST_SWITCH_S_REMOVAL, example="10")
    switch_test_s = _parse_float(switch_test_s_spec, "--test-switch-s") if switch_test_s_spec else None
    if switch_test_s is not None and switch_test_s <= 0:
        raise SystemExit("--test-switch-s expects a positive number of seconds")

    # --test-kill-tile-after-s N: fault injection. Once a TILED rung is active, shut its
    # valve so the branch dies AFTER warming up — the failure switch_to_tier's warmup
    # timeout cannot catch. Proves BranchStallWatchdog on real hardware.
    kill_tile_after_s_spec = _pop_value(sys.argv, "--test-kill-tile-after-s", example="12")
    kill_tile_after_s = (_parse_float(kill_tile_after_s_spec, "--test-kill-tile-after-s")
                         if kill_tile_after_s_spec else None)
    if kill_tile_after_s is not None and kill_tile_after_s <= 0:
        raise SystemExit("--test-kill-tile-after-s expects a positive number of seconds")

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
    # Run the vision pipeline ALONE: no drone/platform control thread, and the
    # pipeline starts immediately instead of waiting for the controller to signal
    # "ready". For camera/latency benching on a rig with no armable FMU. The Stage
    # A/B latency lines (callback) still report; the controller-side GOT DETECTIONS
    # / e2e LATENCY lines do not (there is no controller).
    arg_parser.add_argument('--vision-only', action='store_true',
                            help='Run the camera+inference pipeline without any control thread (bench/diagnostic).')
    # Verification harness for dual-camera switching. When set and 2+ cameras
    # are configured, a daemon thread calls CameraSwitcher.toggle() every N
    # seconds so we can confirm in the log that the switch propagates: each
    # toggle yields a `[cam*]` prefix flip in picamera_thread alive lines,
    # a `CAMERA SWITCH` warning in drone_controller, and a refreshed FOV in
    # the angle math. Production policy (e.g. switch by target distance) will
    # call the same set_active() / toggle() API.
    arg_parser.add_argument('--test-camera-switch-s', type=float, default=None,
                            help='Toggle active camera every N seconds (dual-camera verification only)')

    config = Config.load(config_path)
    config = replace(config, DEBUG=DEBUG)
    logger.info("!!! Loaded config from %s", config_path)

    # bytetrack is an Optional section: present (non-None) enables tracking,
    # null / enabled=false disables it. Check the object, not a flag.
    global USE_TRACKER
    USE_TRACKER = config.bytetrack is not None

    # Phase-2 noise reduction (default-off): use the tracker's smoothed Kalman bbox
    # for matched detections. Requires the tracker; ignored when tracking is off.
    global USE_KALMAN_BBOX
    USE_KALMAN_BBOX = USE_TRACKER and config.bytetrack.use_kalman_bbox
    if USE_KALMAN_BBOX:
        logger.info("!!! Phase-2 use_kalman_bbox ENABLED: matched detections use the "
                    "tracker's smoothed bbox")

    # RAW recording: on by config unless --no-record forces it off.
    record_videos = config.record_videos and not no_record_flag
    logger.info("!!! RAW video recording: %s", "ENABLED" if record_videos else "DISABLED")

    # The size-driven ladder is the ONLY tiling model. build_ladder validates it (last
    # rung 1x1, strictly-decreasing tile counts) and returns [] for an absent/empty
    # ladder, which means a plain non-switchable whole-frame run. "Switchable" is
    # derived: >= 2 rungs to switch between.
    tiers = build_ladder(config.tiling)
    ladder_grids = [(t.tiles_x, t.tiles_y) for t in tiers]
    switchable = len(tiers) >= 2

    _no_ladder = ("Give tiling.ladder at least two rungs in the config (e.g. 2x1 then 1x1); "
                  "an empty ladder is a plain whole-frame run with nothing to switch.")
    if switch_test_s and not switchable:
        raise SystemExit("--test-switch-s steps through the ladder handover, which only exists "
                         "with a >= 2 rung ladder. " + _no_ladder)
    if kill_tile_after_s and not switchable:
        raise SystemExit("--test-kill-tile-after-s kills a tiled rung, which only exists with a "
                         ">= 2 rung ladder. " + _no_ladder)
    if switch_test_s and config.tiling.auto_switch:
        logger.warning("!!! --test-switch-s together with tiling.auto_switch: the harness and the "
                       "size policy will both drive the tier and fight each other. "
                       "Both switches are serialized, but the run is not representative.")

    if switchable:
        logger.info("!!! TILING LADDER (%d rungs, whole-frame active at startup): %s",
                    len(ladder_grids),
                    " -> ".join(f"{tx}x{ty}" for tx, ty in ladder_grids))
        if not config.tiling.auto_switch:
            logger.warning("!!! tiling.auto_switch is OFF: the ladder is built but nothing drives "
                           "it; the run stays on whole-frame unless --test-switch-s is passed.")
        worst = max(tx * ty for tx, ty in ladder_grids)
        if worst > 2:
            logger.warning(
                "!!! The ladder's most-tiled rung runs %d inferences/frame: while it is active, "
                "capture->command LATENCY will be SUB-PAR. On this rig only ~2 tiles stay under "
                "the 200ms budget (2x2 ~230ms e2e, 3x2 ~300ms). That rung is for reacquiring a "
                "small/lost target, not for low-latency control.", worst)
    else:
        logger.info("!!! Inference tiling: whole-frame (no ladder configured)")

    # Capture format follows the MODEL input: NV12-input hef -> capture NV12,
    # RGB-input hef -> capture RGB. The hailocropper requires capture format ==
    # hailonet input format, so deriving it from the model is the only correct
    # choice (a mismatch crashes the cropper). CLI --hef-path wins over config;
    # accept both `--hef-path X` and `--hef-path=X` (argparse consumes it later too).
    # config.inference.hef_model_path is an ExistingFile (validated on load); a CLI
    # override bypasses that check, so validate it here rather than let it surface as
    # an opaque HEF-read failure.
    cli_hef = _peek_value(sys.argv, '--hef-path')
    if cli_hef is not None and not Path(cli_hef).is_file():
        raise SystemExit(f"--hef-path {cli_hef!r} does not exist (or is not a file)")
    effective_hef = cli_hef if cli_hef is not None else str(config.inference.hef_model_path)
    video_format = detect_hef_video_format(effective_hef)
    if video_format != config.camera.video_format:
        logger.warning("!!! capture video_format: %s (from model input) overrides "
                       "config.camera.video_format=%s", video_format, config.camera.video_format)
    else:
        logger.info("!!! capture video_format: %s (matches model input)", video_format)
    if switchable and video_format == 'NV12':
        logger.warning("!!! NV12 capture + tiling together: extra format handling on the "
                       "tile path may further raise latency/CPU — measure before relying on it.")

    # Fold the model-derived capture format back into config.camera so it is the
    # single source of truth: the appsrc caps (camera_settings.video_format), the
    # producer (camera_switcher.video_format), and the pipeline (self.video_format)
    # then all agree — the base app reads camera settings straight off Config.Camera.
    config = replace(config, camera=replace(config.camera, video_format=video_format))

    bytetracker = BYTETracker(**config.bytetrack.tracker_kwargs()) if config.bytetrack is not None else None

    user_data = user_app_callback_class(detections_queue, bytetracker)
    user_data.use_frame = True

    # Frame duration (ns) that turns the branch-independent buffer.pts into a ~1-per-frame
    # id in normalized_frame_id — the fix for tiling branch switches blinding the callback
    # on file input (per-branch buffer.offset restarts low on every switch). Sourced from
    # config so it is identical on every rung; absent/zero fps leaves it None (falls back
    # to buffer.offset).
    _fps = getattr(config.camera, "fps", None)
    user_data.source_frame_duration_ns = round(Gst.SECOND / _fps) if _fps else None

    # Build CameraSwitcher from the validated 'camera' section. Each CameraEntry
    # maps onto a CameraConfig; the shared caps come from the section itself.

    camera_switcher = None
    if config.camera.cameras:
        camera_configs = [
            CameraConfig(
                camera_id=c.camera_id,
                name=c.name,
                sensor_index=c.sensor_index,
                frame_angular_size_deg=c.frame_angular_size_deg,
            )
            for c in config.camera.cameras
        ]
        camera_switcher = CameraSwitcher(
            camera_configs,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
            video_format=config.camera.video_format,   # model-derived (folded into config.camera above)
            active_id=config.camera.active_id,
            switch_to_wide_size=config.camera.switch_to_wide_size,
            switch_to_zoom_size=config.camera.switch_to_zoom_size,
        )
        logger.info("!!! Cameras configured: %s, active=%d, shared caps: %dx%d@%dfps %s, thresholds: wide>=%.3f zoom<=%.3f",
                    [(c.camera_id, c.name) for c in camera_configs],
                    camera_switcher.active_id(),
                    camera_switcher.width, camera_switcher.height,
                    camera_switcher.fps, camera_switcher.video_format,
                    camera_switcher.switch_to_wide_size, camera_switcher.switch_to_zoom_size)

    app = App(
        app_callback,
        user_data,
        config = config,
        parser=arg_parser,
        video_output_chunk_length_s=10,
        video_output_path='./_DEBUG',
        video_filename_base=f"RAW_{start_time_str}",
        record_videos=record_videos,
        video_format=video_format,
        ladder_grids=ladder_grids,
        tiling_overlap=config.tiling.overlap,
        tile_iou_threshold=config.tiling.tile_iou_threshold,
        preview=preview_flag)
    if preview_flag:
        logger.info("!!! PREVIEW window enabled (%s); needs a display (DISPLAY/WAYLAND_DISPLAY set)",
                    preview_sink_name())
    if camera_switcher is not None:
        # Picked up by GStreamerApp.run() to spawn one thread per CameraConfig.
        app.camera_switcher = camera_switcher

    # Install the switch coordinator whenever the pipeline HAS a ladder, even if the
    # automatic policy is off: the --test-switch-s harness must go through the same
    # serialized path, or it desyncs the policy and races the handover.
    # current_i MUST match the tier the pipeline actually booted on, or the policy's
    # very first decision is made about the wrong rung.
    if switchable:
        whole_i = len(tiers) - 1
        assert app.active_tier == whole_i, (
            f"pipeline booted on tier {app.active_tier} but the ladder's whole-frame "
            f"rung is {whole_i}")
        user_data.attach_tiling_ladder(
            switch_fn=app.switch_to_tier,
            policy=TilingLadderPolicy(
                tiers,
                lost_to_2x1_s=config.tiling.lost_to_2x1_s,
                lost_to_3x2_s=config.tiling.lost_to_3x2_s,
                current_i=whole_i,          # whole-frame is live at startup
            ),
            auto_switch=config.tiling.auto_switch,
        )
        if config.tiling.auto_switch:
            logger.info("!!! auto-switch ladder policy: size = max(bbox_w, bbox_h) of the primary "
                        "tracked target; per-rung thresholds %s; target lost -> rung %d after "
                        "%.1fs, rung 0 after %.1fs",
                        [(f"{t.tiles_x}x{t.tiles_y}", t.up_side, t.down_side) for t in tiers],
                        max(len(tiers) - 2, 0),
                        config.tiling.lost_to_2x1_s, config.tiling.lost_to_3x2_s)
            if not USE_TRACKER:
                logger.error("!!! tiling.auto_switch needs bytetrack (size comes from MATCHED "
                             "tracks): with the tracker off no size is ever fed and the ladder "
                             "will only ever escalate on target-lost timers.")

        # Post-switch watchdog: the handover refuses a rung that never warms up, but
        # nothing else notices a rung that warms up and LATER dies — app_callback simply
        # stops firing and the control loop starves. Poll the callback counter; if it
        # freezes while a tiled rung is active, revert to whole-frame. Daemon thread: it
        # must never hold up shutdown, and losing it at exit costs nothing.
        stall_watchdog = BranchStallWatchdog(
            coordinator=user_data.coordinator,
            frame_count_fn=user_data.get_count,
            stall_timeout_s=config.tiling.stall_timeout_s,
            cooldown_s=config.tiling.stall_cooldown_s,
        )
        # Poll well inside the stall window so detection latency is a fraction of it.
        _poll_s = max(0.05, min(0.25, config.tiling.stall_timeout_s / 4.0))
        def _stall_watchdog_thread():
            while True:
                time.sleep(_poll_s)
                try:
                    stall_watchdog.poll()
                except Exception:
                    logger.exception("branch stall watchdog poll failed")
        threading.Thread(target=_stall_watchdog_thread, name="branch-stall-watchdog",
                         daemon=True).start()
        logger.info("!!! branch stall watchdog: revert to whole-frame after %.1fs without a "
                    "callback on a tiled rung; then refuse to leave whole-frame for %.1fs",
                    config.tiling.stall_timeout_s, config.tiling.stall_cooldown_s)

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

    # Manual hot-switch validation: step through the ladder rungs every N seconds so the
    # handover (valve + input-selector) can be confirmed glitch-free on the log (BRANCH
    # SWITCH lines, DETS continuity, StageB stepping). Only meaningful with a >= 2 rung
    # ladder; the automatic policy calls the same switch_to_tier().
    if switch_test_s:
        # Routed through user_data.request_tier (the coordinator), NOT app.switch_to_tier
        # directly: that is what keeps the policy's belief in sync with the pipeline and
        # stops this thread from racing the policy worker over the valves/selector.
        def _tiling_switch_test():
            # Walk the whole ladder rung by rung and wrap around, so a manual run
            # exercises every branch rather than just one pair.
            tier_i = 0
            while True:
                time.sleep(switch_test_s)
                ok = user_data.request_tier(tier_i)
                grid = ladder_grids[tier_i]
                logger.info("!!! --test-switch-s: requested tier %d (%dx%d) -> %s",
                            tier_i, grid[0], grid[1],
                            "ok" if ok else "ABORTED (rung kept)")
                tier_i = (tier_i + 1) % len(ladder_grids)
        threading.Thread(target=_tiling_switch_test, name="tiling-switch-test", daemon=True).start()
        logger.info("!!! --test-switch-s: stepping through the %d ladder rungs every %.1fs",
                    len(ladder_grids), switch_test_s)

    # Fault injection for the post-switch watchdog: wait for a tiled rung to become
    # active, then shut its valve. That branch has already warmed up and been accepted,
    # so the handover's warmup timeout cannot catch this; only the watchdog can.
    if kill_tile_after_s:
        def _kill_tile_branch():
            time.sleep(kill_tile_after_s)
            if not switch_test_s and not config.tiling.auto_switch:
                # Nothing else will ever leave whole-frame, so select a tiled rung here.
                logger.warning("!!! --test-kill-tile-after-s: no switch driver configured; "
                               "switching to a tiled rung first so there is something to kill")
                user_data.request_tier(0)
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                if app.kill_tile_branch_for_test():
                    return
                time.sleep(0.2)     # still on whole-frame; wait for a driver to leave it
            logger.error("!!! --test-kill-tile-after-s: no tiled rung ever became active; "
                         "nothing killed")
        threading.Thread(target=_kill_tile_branch, name="tile-branch-killer", daemon=True).start()
        logger.warning("!!! --test-kill-tile-after-s: will kill the active tiled rung %.1fs "
                       "from now", kill_tile_after_s)

    logger.info("!!! Config: %s", config)
    if DEBUG:
        import math
        nan = math.nan
        config = replace(config, debug_telemetry_dict={'attitude_euler': {'pitch_deg': 1.1012928485870361, 'roll_deg': -2.5803990364074707, 'timestamp_us': 4597491000, 'yaw_deg': -139.22280883789062}, 'odometry': {'angular_velocity_body': {'pitch_rad_s': 0.005480111576616764, 'roll_rad_s': -0.004354139324277639, 'yaw_rad_s': 0.00451350212097168}, 'child_frame_id': '1 (BODY_NED)', 'frame_id': '1 (BODY_NED)', 'pose_covariance': {'covariance_matrix': (0.0006513984990306199, nan, nan, nan, nan, nan, 0.0006680359947495162, nan, nan, nan, nan, 0.0795387253165245, nan, nan, nan, 0.00014700590691063553, nan, nan, 0.0001532444730401039, nan, 0.0037478541489690542)}, 'position_body': {'x_m': 10155.28515625, 'y_m': 1922.0908203125, 'z_m': 0.21114209294319153}, 'q': {'timestamp_us': 0, 'w': -0.3483775854110718, 'x': -0.0011684682685881853, 'y': -0.024444160982966423, 'z': 0.9370348453521729}, 'time_usec': 4597481426, 'velocity_body': {'x_m_s': 0.014301709830760956, 'y_m_s': -0.007258167490363121, 'z_m_s': 0.04777885600924492}, 'velocity_covariance': {'covariance_matrix': (0.002916615456342697, nan, nan, nan, nan, nan, 0.0030234858859330416, nan, nan, nan, nan, 0.005942340008914471, nan, nan, nan, nan, nan, nan, nan, nan, nan)}}, 'landed_state': None, 'imu': {'acceleration_frd': {'down_m_s2': -10.367236137390137, 'forward_m_s2': -0.11148512363433838, 'right_m_s2': 0.5069471597671509}, 'angular_velocity_frd': {'down_rad_s': -0.0006937360158190131, 'forward_rad_s': 0.004322248511016369, 'right_rad_s': 0.0015081189339980483}, 'magnetic_field_frd': {'down_gauss': 0.2559056580066681, 'forward_gauss': -0.3339233696460724, 'right_gauss': 0.3074171245098114}, 'temperature_degc': 15.0, 'timestamp_us': 4597496423}})

    if app.options_menu.connection_string:
        config = replace(config, drone=replace(config.drone,
                                               connection_string=app.options_menu.connection_string))

    vision_only = getattr(app.options_menu, 'vision_only', False)
    action_thread = None
    if vision_only:
        logger.warning("!!! VISION-ONLY mode: no control thread; pipeline starts immediately (bench/diagnostic).")
        # The pipeline normally waits for the controller to signal readiness on
        # `event`; with no controller, set it now so app.run() starts capture.
        event.set()
    elif app.options_menu.action == 'platform':
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
                control_config= config,
                output_queue= output_queue,
                signal_event_when_ready= event,
                camera_switcher= camera_switcher,
            )
        )
    else:
        action_thread = threading.Thread(
            target = drone_controlling_thread,
            args = (
                config.drone.connection_string, #'udp://10.41.10.2:14540',
                asdict(config.drone.config),
                detections_queue
            ),
            kwargs= dict(
                control_config= config,
                output_queue= output_queue,
                signal_event_when_ready= event,
                camera_switcher= camera_switcher,
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

    if action_thread is not None:
        action_thread.start()
    app.add_shutdown_callback(lambda: detections_queue.put(STOP))

    # Debug overlay recorder (debug_*.mkv): draws detection annotations
    # (annotate_frame_with_detection_info) AND runs a SECOND software x264enc — a full
    # Cortex-A76 core on Pi5 (no HW H264 encoder), on EVERY control-loop frame. Gate it
    # on record_videos so --no-record / record_videos=False frees BOTH encoders. It was
    # previously always-on, which defeated --no-record's "free a core" purpose and added
    # e2e latency by pinning the core the GIL-bound control loop needs. output_queue is a
    # bounded OverwriteQueue, so with the drain thread off the control thread's puts just
    # overwrite (no leak); the queue put itself is a cheap dict of references.
    output_thread = None
    if record_videos:
        sink = MultiSink([
            # RtspStreamerSink(30, 8554),
            RecorderSink(30,
                "./_DEBUG",
                segment_seconds=10,
                filename_base=f"debug_{start_time_str}",
            ),
            # OpenCVShowImageSink(window_title='DEBUG IMAGE')
        ])
        output_thread = threading.Thread(
            target = debug_output_thread,
            args = (output_queue, sink),
            name="DEBUG"
        )
        output_thread.start()
        # The thread is non-daemon and blocks in output_queue.get(); without a sentinel
        # the interpreter hangs joining it at exit and RecorderSink's splitmuxsink never
        # finalizes -> truncated debug_*.mkv.
        app.add_shutdown_callback(lambda: _stop_debug_output(output_queue))

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
    if action_thread is not None:
        action_thread.join()

    # Stop the overlay recorder last. Bounded join — a stuck encoder must not wedge
    # the exit, but say so instead of hanging silently.
    if output_thread is not None:
        _stop_debug_output(output_queue)
        output_thread.join(timeout=5)
        if output_thread.is_alive():
            logger.warning("debug output thread still running after 5s; "
                           "the last debug_*.mkv segment may be truncated")

    # Re-raise a control-thread crash captured by the excepthook so the
    # process exits non-zero (caller scripts / systemd / CI must be able
    # to tell a clean stop from a drone-connect failure).
    if 'error' in fatal_action_thread_error:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
