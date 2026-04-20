#!/usr/bin/env python3
"""Replay a flight log through app_callback and visualize BYTETracker trajectories.

Usage:
    cd BDD
    python debug_tracker.py path/to/_DEBUG_dir/
    python debug_tracker.py path/to/logfile.log --video path/to/video.mp4 \\
        --output video --style boxes --out trajectories.mp4
"""

import argparse
import re
import sys
import types
import time as real_time_module
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

# --- Stubs (must run before BDD imports) ---
import debug_app_callback  # noqa: F401 — side-effect: installs all Gst/Hailo mocks

from debug_app_callback import (
    install_mock_patches,
    set_frame_context,
    MockGstPad,
    MockGstBuffer,
    MockGstPadProbeInfo,
    build_frame_data_list,
)
import app as app_module
from app import app_callback, user_app_callback_class, seen_frames
from debug_drone_controller import parse_log, find_files_in_dir, MockMonotonicNs
from bytetrack import BYTETracker, STrack
from flight_debugger import VideoReader
from OverwriteQueue import OverwriteQueue

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(name)s <%(levelname)s> : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# 20 visually distinct BGR colors
PALETTE = [
    (  56,  56, 255), ( 51, 157, 255), ( 51, 255, 255), ( 51, 255,  51),
    (255, 255,  51), (255, 157,  51), (255,  56,  56), (153,  51, 255),
    (255,  51, 153), ( 51, 255, 153), (153, 255,  51), (255, 204,  51),
    ( 51, 204, 255), (204,  51, 255), (255, 102,   0), (  0, 102, 255),
    (128, 255, 128), (128, 128, 255), (255, 128, 128), (200, 200, 200),
]


def _draw_trajectories(
    canvas: np.ndarray,
    track_history: dict[int, list[tuple[int, np.ndarray]]],
    up_to_frame: int,
    width: int,
    height: int,
    style: str,
) -> None:
    """Draw all track trajectories onto canvas in-place.

    Args:
        canvas:        BGR image to draw on (modified in-place).
        track_history: {track_id: [(frame_id, bbox_xyxy_norm), ...]}
        up_to_frame:   only include points with frame_id <= this value.
        width, height: canvas dimensions in pixels.
        style:         "lines" or "boxes".
    """
    for tid, points in track_history.items():
        color   = PALETTE[tid % len(PALETTE)]
        visible = [(fid, bbox) for fid, bbox in points if fid <= up_to_frame]
        if not visible:
            continue

        centers: list[tuple[int, int]] = []
        for fid, bbox in visible:
            cx = int((bbox[0] + bbox[2]) / 2 * width)
            cy = int((bbox[1] + bbox[3]) / 2 * height)
            centers.append((cx, cy))

            if style == "boxes":
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        if len(centers) > 1:
            pts = np.array(centers, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

        last_cx, last_cy = centers[-1]
        cv2.circle(canvas, (last_cx, last_cy), 4, color, -1)
        cv2.putText(
            canvas, f"ID:{tid}",
            (last_cx + 5, last_cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )


def render(
    output_mode: str,
    style: str,
    track_history: dict[int, list[tuple[int, np.ndarray]]],
    frame_images: dict[int, np.ndarray],
    frame_ids: list[int],
    width: int,
    height: int,
    out_path: Path,
) -> None:
    """Dispatch to the correct rendering mode."""
    if output_mode == "single":
        _render_single(style, track_history, frame_images, frame_ids, width, height, out_path)
    elif output_mode == "frames":
        _render_frames(style, track_history, frame_images, frame_ids, width, height, out_path)
    elif output_mode == "video":
        _render_video(style, track_history, frame_images, frame_ids, width, height, out_path)
    else:
        raise ValueError(f"Unknown output_mode: {output_mode!r}")


def _blank(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _render_single(
    style: str,
    track_history: dict[int, list[tuple[int, np.ndarray]]],
    frame_images: dict[int, np.ndarray],
    frame_ids: list[int],
    width: int,
    height: int,
    out_path: Path,
) -> None:
    """All trajectories overlaid on the last video frame → one PNG."""
    last_fid = frame_ids[-1]
    canvas = frame_images.get(last_fid, _blank(width, height)).copy()
    _draw_trajectories(canvas, track_history, up_to_frame=last_fid,
                       width=width, height=height, style=style)
    if out_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    logger.info("Saved single image → %s", out_path)


def _render_frames(
    style: str,
    track_history: dict[int, list[tuple[int, np.ndarray]]],
    frame_images: dict[int, np.ndarray],
    frame_ids: list[int],
    width: int,
    height: int,
    out_path: Path,
) -> None:
    """Per-frame PNG sequence saved to out_path directory."""
    out_path.mkdir(parents=True, exist_ok=True)
    for fid in frame_ids:
        canvas = frame_images.get(fid, _blank(width, height)).copy()
        if canvas.shape[1] != width or canvas.shape[0] != height:
            canvas = cv2.resize(canvas, (width, height))
        _draw_trajectories(canvas, track_history, up_to_frame=fid,
                           width=width, height=height, style=style)
        cv2.imwrite(str(out_path / f"frame_{fid:04d}.png"), canvas)
    logger.info("Saved %d frames → %s/", len(frame_ids), out_path)


def _render_video(
    style: str,
    track_history: dict[int, list[tuple[int, np.ndarray]]],
    frame_images: dict[int, np.ndarray],
    frame_ids: list[int],
    width: int,
    height: int,
    out_path: Path,
) -> None:
    """Animated MP4 via cv2.VideoWriter."""
    if out_path.suffix.lower() not in (".mp4", ".avi", ".mkv"):
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # avc1 (H.264) works on macOS; fall back to mp4v if unavailable
    for fourcc_str in ("avc1", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (width, height))
        if writer.isOpened():
            break
    if not writer.isOpened():
        logger.error("Failed to open VideoWriter for %s", out_path)
        return
    for fid in frame_ids:
        canvas = frame_images.get(fid, _blank(width, height)).copy()
        # Normalise to canonical resolution in case of stray black frames
        if canvas.shape[1] != width or canvas.shape[0] != height:
            canvas = cv2.resize(canvas, (width, height))
        _draw_trajectories(canvas, track_history, up_to_frame=fid,
                           width=width, height=height, style=style)
        writer.write(canvas)
    writer.release()
    logger.info("Saved video → %s", out_path)


def collect_track_history(
    frame_data_list: list[dict],
    user_data: user_app_callback_class,
    mock_pad: MockGstPad,
    mock_monotonic: MockMonotonicNs,
) -> tuple[dict, dict, int, int]:
    """Run app_callback for every frame, snapshot tracker state after each call.

    Returns:
        track_history: {track_id: [(frame_id, bbox_xyxy_normalized), ...]}
        frame_images:  {frame_id: np.ndarray BGR}
        width, height: frame dimensions in pixels
    """
    track_history: dict[int, list[tuple[int, np.ndarray]]] = {}
    frame_images:  dict[int, np.ndarray] = {}
    width, height = 640, 480

    # USE_TRACKER=False in app.py, so app_callback never calls tracker.update().
    # We drive the tracker here using a sequential index so track_buffer works correctly.
    tracker_frame_idx = 0

    for fdata in frame_data_list:
        fid   = fdata["frame_id"]
        ts_ns = fdata["timestamp_ns"]
        mock_monotonic.set_frame(ts_ns)
        set_frame_context(fdata["frame"], fdata["detections"])
        mock_buffer = MockGstBuffer(fid, ts_ns)
        mock_info   = MockGstPadProbeInfo(mock_buffer)
        app_callback(mock_pad, mock_info, user_data)

        dets = fdata["detections"]
        dets_array = (
            np.array([
                [d.bbox.left_edge, d.bbox.top_edge,
                 d.bbox.right_edge, d.bbox.bottom_edge, d.confidence]
                for d in dets
            ], dtype=float)
            if dets else np.empty((0, 5))
        )
        user_data.tracker.update(dets_array, tracker_frame_idx)
        tracker_frame_idx += 1

        # Snapshot active tracks
        for track in user_data.tracker.tracked_stracks:
            tid  = track.track_id
            bbox = track.bbox.copy()  # [x1, y1, x2, y2] normalized 0-1
            track_history.setdefault(tid, []).append((fid, bbox))

        img = fdata["frame"]
        frame_images[fid] = img.copy()
        # Use the FIRST actual frame to set canonical dimensions,
        # so later black fallback frames (480×640) don't override them.
        if width == 640 and height == 480 and img is not None and img.size > 0:
            h, w = img.shape[:2]
            if h > 0 and w > 0:
                height, width = h, w

    logger.info(
        "Collected %d frames, %d unique tracks (resolution %dx%d)",
        len(frame_images), len(track_history), width, height,
    )
    return track_history, frame_images, width, height


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize BYTETracker trajectories from a recorded flight log.",
    )
    p.add_argument("path", type=Path, help="Log file or _DEBUG_dir/")
    p.add_argument("--video", type=Path, default=None, help="Video file or directory")
    p.add_argument(
        "--output", choices=["single", "frames", "video"], default="single",
        help="Output mode: single PNG / PNG sequence / MP4 video (default: single)",
    )
    p.add_argument(
        "--style", choices=["lines", "boxes"], default="lines",
        help="Track style: lines=center polyline only / boxes=polyline+bbox rects (default: lines)",
    )
    p.add_argument(
        "--recovery-max-dist", type=float, default=None,
        help="Enable Stage 1.5 recovery matching using last observed position. "
             "Value is the max centre distance (0–1 normalised) to recover a "
             "track after a sudden direction change (e.g. 0.3). "
             "Overrides bytetrack_recovery_max_dist from the log config.",
    )
    p.add_argument(
        "--nms-thresh", type=float, default=0.3,
        help="IoU threshold for pre-tracker NMS (default: 0.3). "
             "Suppresses duplicate detections with IoU above this value. "
             "Set to 0 or omit to disable IoU-based suppression.",
    )
    p.add_argument(
        "--nms-dist-thresh", type=float, default=0.06,
        help="Centre-distance threshold for pre-tracker NMS in normalised [0–1] coords "
             "(default: 0.06). Suppresses duplicate detections whose centres are closer "
             "than this value even when their IoU is low. Set to 0 to disable.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output file (single/video) or directory (frames). "
             "Default: tracker_output/ next to log file.",
    )
    return p.parse_args()


_TS_RE = re.compile(r"(\d{8})-(\d{6})")


def _parse_file_ts(path: Path) -> datetime | None:
    m = _TS_RE.search(path.stem)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def _find_videos_for_log(log_file: Path, all_videos: list[Path], window_s: int = 120) -> list[Path]:
    """Return video files whose filename timestamp is within window_s of log_file's timestamp."""
    log_ts = _parse_file_ts(log_file)
    if log_ts is None:
        return all_videos  # no timestamp in name — give all videos
    window = timedelta(seconds=window_s)
    matched = [v for v in all_videos if (vts := _parse_file_ts(v)) and abs(vts - log_ts) <= window]
    return matched


def _process_one(log_file: Path, video_path, output_mode: str, style: str, out_path: Path,
                 recovery_max_dist: float | None = None,
                 nms_thresh: float | None = 0.3,
                 nms_dist_thresh: float | None = 0.06) -> None:
    logger.info("=== Processing %s → %s ===", log_file.name, out_path)

    config_dict, frames, base_ns = parse_log(log_file)
    if not frames:
        logger.warning("No frame data in %s — skipping", log_file.name)
        return

    if not config_dict:
        config_dict = {}

    bytetrack_config = {
        "track_thresh":   config_dict.get("bytetrack_track_thresh",   0.3),
        "det_thresh":     config_dict.get("bytetrack_det_thresh",     0.35),
        "match_thresh":   config_dict.get("bytetrack_match_thresh",   0.3),
        "track_buffer":   config_dict.get("bytetrack_track_buffer",   30),
        "frame_rate":     config_dict.get("bytetrack_frame_rate",     30),
        "match_max_dist":    config_dict.get("bytetrack_match_max_dist",    0.2),
        "recovery_max_dist": (
            recovery_max_dist
            if recovery_max_dist is not None
            else config_dict.get("bytetrack_recovery_max_dist", None)
        ),
        "nms_thresh":      config_dict.get("bytetrack_nms_thresh",      nms_thresh),
        "nms_dist_thresh": config_dict.get("bytetrack_nms_dist_thresh", nms_dist_thresh),
    }

    video_reader = VideoReader(video_path)
    if not video_reader.available:
        logger.warning("No video for %s — black frames will be used", log_file.name)

    frame_data_list = build_frame_data_list(frames, video_reader)

    mock_monotonic = MockMonotonicNs(base_ns)
    mock_time = types.ModuleType("mock_time")
    for attr in dir(real_time_module):
        if not attr.startswith("_"):
            setattr(mock_time, attr, getattr(real_time_module, attr))
    mock_time.monotonic_ns = mock_monotonic
    app_module.time = mock_time

    install_mock_patches()
    seen_frames.clear()

    STrack.reset_counter()
    bytetracker = BYTETracker(**bytetrack_config)
    user_data   = user_app_callback_class(OverwriteQueue(maxsize=1), bytetracker)
    user_data.use_frame = True

    mock_pad = MockGstPad()
    track_history, frame_images, width, height = collect_track_history(
        frame_data_list, user_data, mock_pad, mock_monotonic,
    )

    frame_ids = sorted(frame_images.keys())
    render(output_mode, style, track_history, frame_images, frame_ids, width, height, out_path)
    logger.info("Done → %s", out_path)


def main():
    args = parse_args()

    if args.path.is_dir():
        log_files = sorted(args.path.glob("*.log"))
        if not log_files:
            print(f"No .log files found in {args.path}", file=sys.stderr)
            sys.exit(1)

        all_videos = (
            sorted(args.path.glob("RAW_*.mkv")) + sorted(args.path.glob("RAW_*.mp4"))
            or sorted(args.path.glob("*.mkv")) + sorted(args.path.glob("*.mp4"))
        )

        base_out = args.out or (args.path / "tracker_output")

        for log_file in log_files:
            matched_videos = _find_videos_for_log(log_file, all_videos)
            video_path = matched_videos if matched_videos else None

            stem = log_file.stem
            if args.output == "single":
                out_path = base_out / f"{stem}.png"
            elif args.output == "video":
                out_path = base_out / f"{stem}.mp4"
            else:
                out_path = base_out / stem

            _process_one(log_file, video_path, args.output, args.style, out_path,
                         recovery_max_dist=args.recovery_max_dist,
                         nms_thresh=args.nms_thresh or None,
                         nms_dist_thresh=args.nms_dist_thresh or None)

    else:
        log_file   = args.path
        video_path = args.video
        out_path   = args.out or (log_file.parent / "tracker_output")
        _process_one(log_file, video_path, args.output, args.style, out_path,
                     recovery_max_dist=args.recovery_max_dist,
                     nms_thresh=args.nms_thresh or None,
                     nms_dist_thresh=args.nms_dist_thresh or None)


if __name__ == "__main__":
    main()
