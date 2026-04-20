#!/usr/bin/env python3
"""Replay a flight log through app_callback and visualize BYTETracker trajectories.

Usage:
    cd BDD
    python debug_tracker.py path/to/_DEBUG_dir/
    python debug_tracker.py path/to/logfile.log --video path/to/video.mp4 \\
        --output video --style boxes --out trajectories.mp4
"""

import argparse
import sys
import types
import time as real_time_module
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (width, height))
    for fid in frame_ids:
        canvas = frame_images.get(fid, _blank(width, height)).copy()
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
        if img is not None:
            height, width = img.shape[:2]

    logger.info(
        "Collected %d frames, %d unique tracks",
        len(frame_images), len(track_history),
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
        "--out", type=Path, default=None,
        help="Output file (single/video) or directory (frames). "
             "Default: tracker_output/ next to log file.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    if args.path.is_dir():
        log_file, video_files = find_files_in_dir(args.path)
        if log_file is None:
            print(f"No .log file found in {args.path}", file=sys.stderr)
            sys.exit(1)
        video_path = video_files or None
    else:
        log_file  = args.path
        video_path = args.video

    # Default output path
    out_path = args.out or (log_file.parent / "tracker_output")

    logger.info("Log: %s  Video: %s  Output: %s  Mode: %s  Style: %s",
                log_file, video_path, out_path, args.output, args.style)

    # Parse log
    config_dict, frames, base_ns = parse_log(log_file)
    if not frames:
        print("No frame data in log", file=sys.stderr)
        sys.exit(1)

    if not config_dict:
        config_dict = {}

    bytetrack_config = {
        "track_thresh":   config_dict.get("bytetrack_track_thresh",   0.3),
        "det_thresh":     config_dict.get("bytetrack_det_thresh",     0.35),
        "match_thresh":   config_dict.get("bytetrack_match_thresh",   0.3),
        "track_buffer":   config_dict.get("bytetrack_track_buffer",   30),
        "frame_rate":     config_dict.get("bytetrack_frame_rate",     30),
        "match_max_dist": config_dict.get("bytetrack_match_max_dist", 0.2),
    }

    # Open video
    video_reader = VideoReader(video_path)
    if not video_reader.available:
        logger.warning("No video loaded — black frames will be used")

    frame_data_list = build_frame_data_list(frames, video_reader)

    # Mock time
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
    logger.info("Rendering mode=%s style=%s ...", args.output, args.style)
    render(args.output, args.style, track_history, frame_images, frame_ids,
           width, height, out_path)
    logger.info("Done → %s", out_path)


if __name__ == "__main__":
    main()
