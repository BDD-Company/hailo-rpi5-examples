#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class Prediction:
    frame: int
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class AlignedPrediction:
    frame: int
    original_frame: int
    confidence: float
    bbox: BBox


NUMBER_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
LOG_LINE_RE = re.compile(
    rf"frame#(?P<frame>\d+)\s+detection:\s+#(?P<track>-?\d+)\s+class:\s+(?P<class_id>-?\d+)\s+"
    rf"confidence:\s+(?P<confidence>{NUMBER_RE})\s+\(x:\s*(?P<x>{NUMBER_RE}),\s*"
    rf"y:\s*(?P<y>{NUMBER_RE}),\s*w:\s*(?P<w>{NUMBER_RE}),\s*h:\s*(?P<h>{NUMBER_RE})\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate per-video detection metrics (Precision/Recall/mAP50/mAP50-95, TP/FP/FN) "
            "from benchmark.py logs and GT .txt files."
        )
    )
    parser.add_argument("--videos-dir", type=Path, required=True, help="Directory with source videos.")
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=None,
        help="Directory with GT .txt files. Defaults to --videos-dir.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help=(
            "Directory with benchmark logs. Can be a single HEF report dir (contains *.log) "
            "or benchmark output root with HEF subdirs."
        ),
    )
    parser.add_argument(
        "--video-ext",
        type=str,
        default=".mkv,.mp4,.avi,.mov",
        help="Comma-separated video extensions to process.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Optional class_id filter for predictions from log.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used for TP/FP/FN and Precision/Recall (default: 0.5).",
    )
    parser.add_argument(
        "--frame-align",
        choices=["none", "linear"],
        default="none",
        help=(
            "How to align prediction frame numbers to GT timeline before matching. "
            "'none' keeps original frame ids, 'linear' rescales prediction frames "
            "to GT frame range."
        ),
    )
    parser.add_argument(
        "--frame-tolerance",
        type=int,
        default=0,
        help=(
            "Allow matching GT in neighboring frames within +/-N after frame alignment "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path (file path or existing directory).",
    )
    args = parser.parse_args()
    if args.frame_tolerance < 0:
        parser.error("--frame-tolerance must be >= 0")
    return args


def xywh_center_to_xyxy(x_center: float, y_center: float, width: float, height: float) -> BBox:
    half_w = width / 2.0
    half_h = height / 2.0
    return (x_center - half_w, y_center - half_h, x_center + half_w, y_center + half_h)


def xywh_topleft_to_xyxy(x: float, y: float, width: float, height: float) -> BBox:
    return (x, y, x + width, y + height)


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def parse_gt_file(path: Path) -> Dict[int, List[BBox]]:
    gt_by_frame: Dict[int, List[BBox]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            try:
                frame_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: invalid frame id in line: {line}") from exc

            coords = [c.rstrip(';') for c in parts[1:] if c.rstrip(';')]
            if len(coords) % 4 != 0:
                raise ValueError(
                    f"{path}:{line_no}: expected 4*N bbox values after frame number, got {len(coords)}"
                )

            boxes: List[BBox] = gt_by_frame.setdefault(frame_id, [])
            for idx in range(0, len(coords), 4):
                xc, yc, w, h = (float(coords[idx]), float(coords[idx + 1]), float(coords[idx + 2]), float(coords[idx + 3]))
                boxes.append(xywh_center_to_xyxy(xc, yc, w, h))
    return gt_by_frame


def parse_predictions_file(path: Path, class_id_filter: int | None = None) -> Dict[int, List[Prediction]]:
    pred_by_frame: Dict[int, List[Prediction]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            match = LOG_LINE_RE.search(line)
            if not match:
                continue

            class_id = int(match.group("class_id"))
            if class_id_filter is not None and class_id != class_id_filter:
                continue

            frame = int(match.group("frame"))
            confidence = float(match.group("confidence"))
            x = float(match.group("x"))
            y = float(match.group("y"))
            w = float(match.group("w"))
            h = float(match.group("h"))
            pred = Prediction(frame=frame, confidence=confidence, bbox=xywh_topleft_to_xyxy(x, y, w, h))
            pred_by_frame.setdefault(frame, []).append(pred)
    return pred_by_frame


def _last_gt_frame(gt_by_frame: Dict[int, List[BBox]]) -> int:
    non_empty_frames = [frame for frame, boxes in gt_by_frame.items() if boxes]
    if non_empty_frames:
        return max(non_empty_frames)
    return max(gt_by_frame.keys(), default=0)


def _last_pred_frame(pred_by_frame: Dict[int, List[Prediction]]) -> int:
    non_empty_frames = [frame for frame, preds in pred_by_frame.items() if preds]
    if non_empty_frames:
        return max(non_empty_frames)
    return max(pred_by_frame.keys(), default=0)


def frame_alignment_scale(
    gt_by_frame: Dict[int, List[BBox]],
    pred_by_frame: Dict[int, List[Prediction]],
    frame_align: str,
) -> float:
    if frame_align != "linear":
        return 1.0

    gt_last = _last_gt_frame(gt_by_frame)
    pred_last = _last_pred_frame(pred_by_frame)
    if gt_last <= 0 or pred_last <= 0:
        return 1.0
    return gt_last / float(pred_last)


def _apply_frame_alignment(frame: int, scale: float) -> int:
    if frame <= 0:
        return 0
    return max(1, int(round(frame * scale)))


def align_predictions(
    gt_by_frame: Dict[int, List[BBox]],
    pred_by_frame: Dict[int, List[Prediction]],
    frame_align: str,
) -> Tuple[List[AlignedPrediction], float]:
    scale = frame_alignment_scale(gt_by_frame, pred_by_frame, frame_align)
    aligned_preds: List[AlignedPrediction] = []
    for frame, preds in pred_by_frame.items():
        aligned_frame = _apply_frame_alignment(frame, scale)
        for pred in preds:
            aligned_preds.append(
                AlignedPrediction(
                    frame=aligned_frame,
                    original_frame=pred.frame,
                    confidence=pred.confidence,
                    bbox=pred.bbox,
                )
            )
    aligned_preds.sort(key=lambda p: p.confidence, reverse=True)
    return aligned_preds, scale


def _find_best_unmatched_gt(
    pred_bbox: BBox,
    pred_frame: int,
    gt_by_frame: Dict[int, List[BBox]],
    matched_by_frame: Dict[int, List[bool]],
    frame_tolerance: int,
) -> Tuple[int, int, float] | None:
    start_frame = pred_frame - frame_tolerance
    end_frame = pred_frame + frame_tolerance

    best_iou = 0.0
    best_frame = -1
    best_idx = -1
    best_dist = None

    for frame in range(start_frame, end_frame + 1):
        gt_boxes = gt_by_frame.get(frame, [])
        if not gt_boxes:
            continue

        frame_matches = matched_by_frame.setdefault(frame, [False] * len(gt_boxes))
        for idx, gt in enumerate(gt_boxes):
            if frame_matches[idx]:
                continue

            score = iou(pred_bbox, gt)
            dist = abs(frame - pred_frame)
            if score > best_iou + 1e-12:
                best_iou = score
                best_frame = frame
                best_idx = idx
                best_dist = dist
            elif abs(score - best_iou) <= 1e-12 and best_dist is not None and dist < best_dist:
                best_frame = frame
                best_idx = idx
                best_dist = dist

    if best_idx < 0:
        return None
    return (best_frame, best_idx, best_iou)


def match_predictions(
    gt_by_frame: Dict[int, List[BBox]],
    aligned_preds: List[AlignedPrediction],
    iou_threshold: float,
    frame_tolerance: int,
) -> Tuple[int, int, int, Dict[int, int], Dict[int, int]]:
    matched_by_frame = {frame: [False] * len(boxes) for frame, boxes in gt_by_frame.items()}

    tp = 0
    fp = 0
    fp_frame_counts: Dict[int, int] = {}

    for pred in aligned_preds:
        match = _find_best_unmatched_gt(
            pred_bbox=pred.bbox,
            pred_frame=pred.frame,
            gt_by_frame=gt_by_frame,
            matched_by_frame=matched_by_frame,
            frame_tolerance=frame_tolerance,
        )
        if match is not None and match[2] >= iou_threshold:
            matched_frame, matched_idx, _ = match
            matched_by_frame[matched_frame][matched_idx] = True
            tp += 1
        else:
            fp += 1
            fp_frame_counts[pred.frame] = fp_frame_counts.get(pred.frame, 0) + 1

    fn = 0
    fn_frame_counts: Dict[int, int] = {}
    for frame, matches in matched_by_frame.items():
        unmatched_count = sum(1 for is_matched in matches if not is_matched)
        if unmatched_count > 0:
            fn += unmatched_count
            fn_frame_counts[frame] = unmatched_count

    return (tp, fp, fn, fp_frame_counts, fn_frame_counts)


def calculate_ap(
    gt_by_frame: Dict[int, List[BBox]],
    aligned_preds: List[AlignedPrediction],
    iou_threshold: float,
    frame_tolerance: int,
) -> float:
    total_gt = sum(len(boxes) for boxes in gt_by_frame.values())
    if total_gt == 0:
        return 0.0

    matched_by_frame = {frame: [False] * len(boxes) for frame, boxes in gt_by_frame.items()}
    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for pred in aligned_preds:
        match = _find_best_unmatched_gt(
            pred_bbox=pred.bbox,
            pred_frame=pred.frame,
            gt_by_frame=gt_by_frame,
            matched_by_frame=matched_by_frame,
            frame_tolerance=frame_tolerance,
        )
        if match is not None and match[2] >= iou_threshold:
            matched_frame, matched_idx, _ = match
            matched_by_frame[matched_frame][matched_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    if not tp_flags:
        return 0.0

    tp_cumsum: List[float] = []
    fp_cumsum: List[float] = []
    tp_running = 0.0
    fp_running = 0.0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        tp_running += tp_flag
        fp_running += fp_flag
        tp_cumsum.append(tp_running)
        fp_cumsum.append(fp_running)

    recalls = [tp / float(total_gt) for tp in tp_cumsum]
    precisions = [tp / max(tp + fp, 1e-12) for tp, fp in zip(tp_cumsum, fp_cumsum)]

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

    ap = 0.0
    for idx in range(len(mrec) - 1):
        if mrec[idx + 1] != mrec[idx]:
            ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]
    return ap


def calculate_video_metrics(
    video_path: Path,
    gt_file: Path,
    pred_file: Path,
    class_id_filter: int | None,
    iou_threshold: float,
    frame_align: str,
    frame_tolerance: int,
) -> Dict[str, object]:
    gt_by_frame = parse_gt_file(gt_file)
    pred_by_frame = parse_predictions_file(pred_file, class_id_filter=class_id_filter)
    return calculate_video_metrics_from_data(
        video_name=video_path.name,
        gt_file=gt_file,
        pred_file=pred_file,
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame,
        iou_threshold=iou_threshold,
        frame_align=frame_align,
        frame_tolerance=frame_tolerance,
    )


def calculate_video_metrics_from_data(
    video_name: str,
    gt_file: Path,
    pred_file: Path,
    gt_by_frame: Dict[int, List[BBox]],
    pred_by_frame: Dict[int, List[Prediction]],
    iou_threshold: float,
    frame_align: str,
    frame_tolerance: int,
) -> Dict[str, object]:
    aligned_preds, align_scale = align_predictions(
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame,
        frame_align=frame_align,
    )
    tp_total, fp_total, fn_total, fp_frame_counts, fn_frame_counts = match_predictions(
        gt_by_frame=gt_by_frame,
        aligned_preds=aligned_preds,
        iou_threshold=iou_threshold,
        frame_tolerance=frame_tolerance,
    )

    precision = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    recall = (tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0

    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap_per_threshold = [
        calculate_ap(
            gt_by_frame=gt_by_frame,
            aligned_preds=aligned_preds,
            iou_threshold=thr,
            frame_tolerance=frame_tolerance,
        )
        for thr in iou_thresholds
    ]
    map50 = ap_per_threshold[0]
    map50_95 = float(mean(ap_per_threshold)) if ap_per_threshold else 0.0

    return {
        "video": video_name,
        "gt_file": str(gt_file),
        "prediction_file": str(pred_file),
        "num_gt_frames": len(gt_by_frame),
        "num_gt_boxes": int(sum(len(b) for b in gt_by_frame.values())),
        "num_predictions": int(len(aligned_preds)),
        "frame_align": frame_align,
        "frame_tolerance": int(frame_tolerance),
        "frame_align_scale": float(align_scale),
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "true_positive": int(tp_total),
        "false_positive": int(fp_total),
        "false_negative": int(fn_total),
        "false_positive_frames": sorted(fp_frame_counts.keys()),
        "false_negative_frames": sorted(fn_frame_counts.keys()),
        "false_positive_by_frame": [
            {"frame": frame, "count": count} for frame, count in sorted(fp_frame_counts.items())
        ],
        "false_negative_by_frame": [
            {"frame": frame, "count": count} for frame, count in sorted(fn_frame_counts.items())
        ],
    }


def _calculate_dataset_ap(
    gt_by_video: Dict[str, Dict[int, List[BBox]]],
    pred_tuples: List[Tuple[float, str, int, BBox]],
    iou_threshold: float,
    frame_tolerance: int,
) -> float:
    total_gt = 0
    for gt_by_frame in gt_by_video.values():
        total_gt += sum(len(boxes) for boxes in gt_by_frame.values())
    if total_gt == 0:
        return 0.0

    matched_by_video: Dict[str, Dict[int, List[bool]]] = {
        video_name: {frame: [False] * len(boxes) for frame, boxes in gt_by_frame.items()}
        for video_name, gt_by_frame in gt_by_video.items()
    }
    sorted_preds = sorted(pred_tuples, key=lambda x: x[0], reverse=True)

    tp_flags: List[int] = []
    fp_flags: List[int] = []
    for confidence, video_name, pred_frame, pred_bbox in sorted_preds:
        del confidence  # confidence only used for sorting
        gt_by_frame = gt_by_video.get(video_name, {})
        matched_by_frame = matched_by_video.setdefault(video_name, {})
        match = _find_best_unmatched_gt(
            pred_bbox=pred_bbox,
            pred_frame=pred_frame,
            gt_by_frame=gt_by_frame,
            matched_by_frame=matched_by_frame,
            frame_tolerance=frame_tolerance,
        )

        if match is not None and match[2] >= iou_threshold:
            matched_frame, matched_idx, _ = match
            matched_by_frame.setdefault(matched_frame, [False] * len(gt_by_frame.get(matched_frame, [])))
            matched_by_frame[matched_frame][matched_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    if not tp_flags:
        return 0.0

    tp_cumsum: List[float] = []
    fp_cumsum: List[float] = []
    tp_running = 0.0
    fp_running = 0.0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        tp_running += tp_flag
        fp_running += fp_flag
        tp_cumsum.append(tp_running)
        fp_cumsum.append(fp_running)

    recalls = [tp / float(total_gt) for tp in tp_cumsum]
    precisions = [tp / max(tp + fp, 1e-12) for tp, fp in zip(tp_cumsum, fp_cumsum)]

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

    ap = 0.0
    for idx in range(len(mrec) - 1):
        if mrec[idx + 1] != mrec[idx]:
            ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]
    return ap


def calculate_summary_metrics(
    iou_threshold: float,
    datasets: List[Tuple[str, Dict[int, List[BBox]], Dict[int, List[Prediction]]]],
    frame_align: str,
    frame_tolerance: int,
) -> Dict[str, object]:
    tp_total = 0
    fp_total = 0
    fn_total = 0
    gt_boxes_total = 0
    pred_total = 0

    gt_by_video: Dict[str, Dict[int, List[BBox]]] = {}
    pred_tuples: List[Tuple[float, str, int, BBox]] = []

    for video_name, gt_by_frame, pred_by_frame in datasets:
        gt_boxes_total += sum(len(boxes) for boxes in gt_by_frame.values())
        aligned_preds, _ = align_predictions(
            gt_by_frame=gt_by_frame,
            pred_by_frame=pred_by_frame,
            frame_align=frame_align,
        )
        pred_total += len(aligned_preds)
        gt_by_video[video_name] = gt_by_frame

        for pred in aligned_preds:
            pred_tuples.append((pred.confidence, video_name, pred.frame, pred.bbox))

        tp, fp, fn, _, _ = match_predictions(
            gt_by_frame=gt_by_frame,
            aligned_preds=aligned_preds,
            iou_threshold=iou_threshold,
            frame_tolerance=frame_tolerance,
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    recall = (tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0

    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap_per_threshold = [
        _calculate_dataset_ap(
            gt_by_video=gt_by_video,
            pred_tuples=pred_tuples,
            iou_threshold=thr,
            frame_tolerance=frame_tolerance,
        )
        for thr in iou_thresholds
    ]
    map50 = ap_per_threshold[0] if ap_per_threshold else 0.0
    map50_95 = float(mean(ap_per_threshold)) if ap_per_threshold else 0.0

    return {
        "num_videos": len(datasets),
        "num_gt_boxes": int(gt_boxes_total),
        "num_predictions": int(pred_total),
        "frame_align": frame_align,
        "frame_tolerance": int(frame_tolerance),
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "true_positive": int(tp_total),
        "false_positive": int(fp_total),
        "false_negative": int(fn_total),
    }


def discover_report_sets(reports_dir: Path) -> List[Tuple[str, Path]]:
    direct_logs = sorted(reports_dir.glob("*.log"))
    if direct_logs:
        return [(reports_dir.name, reports_dir)]

    report_sets = []
    for subdir in sorted(reports_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if any(subdir.glob("*.log")):
            report_sets.append((subdir.name, subdir))
    return report_sets


def discover_videos(videos_dir: Path, extensions: Iterable[str]) -> List[Path]:
    ext_set = {ext.strip().lower() for ext in extensions if ext.strip()}
    if not ext_set:
        ext_set = {".mkv"}
    videos = [path for path in videos_dir.iterdir() if path.is_file() and path.suffix.lower() in ext_set]
    return sorted(videos)


def resolve_output_json_path(output_json: Path) -> Path:
    if output_json.exists() and output_json.is_dir():
        return output_json / "benchmark_metrics.json"
    return output_json


def print_report(report_set_name: str, metrics: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    print(f"\nReport set: {report_set_name}")
    print("video | precision | recall | mAP50 | mAP50-95 | TP | FP | FN")
    print("-" * 82)
    for item in metrics:
        print(
            f"{item['video']} | {item['precision']:.4f} | {item['recall']:.4f} | "
            f"{item['mAP50']:.4f} | {item['mAP50_95']:.4f} | "
            f"{item['true_positive']} | {item['false_positive']} | {item['false_negative']}"
        )
        print(f"  FP frames: {item['false_positive_frames']}")
        print(f"  FN frames: {item['false_negative_frames']}")
        print(f"  FP by frame: {item['false_positive_by_frame']}")
        print(f"  FN by frame: {item['false_negative_by_frame']}")
    print("-" * 82)
    print(
        f"TOTAL (all videos) | {summary['precision']:.4f} | {summary['recall']:.4f} | "
        f"{summary['mAP50']:.4f} | {summary['mAP50_95']:.4f} | "
        f"{summary['true_positive']} | {summary['false_positive']} | {summary['false_negative']}"
    )


def main() -> int:
    args = parse_args()
    videos_dir = args.videos_dir
    annotations_dir = args.annotations_dir if args.annotations_dir is not None else args.videos_dir
    reports_dir = args.reports_dir

    if not videos_dir.is_dir():
        print(f"--videos-dir is not a directory: {videos_dir}", file=sys.stderr)
        return 1
    if not annotations_dir.is_dir():
        print(f"--annotations-dir is not a directory: {annotations_dir}", file=sys.stderr)
        return 1
    if not reports_dir.is_dir():
        print(f"--reports-dir is not a directory: {reports_dir}", file=sys.stderr)
        return 1

    videos = discover_videos(videos_dir, args.video_ext.split(","))
    if not videos:
        print(f"No videos found in {videos_dir} with extensions: {args.video_ext}", file=sys.stderr)
        return 1

    report_sets = discover_report_sets(reports_dir)
    if not report_sets:
        print(f"No report sets found in {reports_dir} (*.log files missing)", file=sys.stderr)
        return 1

    all_results: Dict[str, object] = {
        "videos_dir": str(videos_dir),
        "annotations_dir": str(annotations_dir),
        "reports_dir": str(reports_dir),
        "iou_threshold_for_pr": float(args.iou_threshold),
        "frame_align": args.frame_align,
        "frame_tolerance": int(args.frame_tolerance),
        "class_id_filter": args.class_id,
        "report_sets": [],
    }

    has_any_result = False
    for report_set_name, report_set_dir in report_sets:
        per_video_metrics: List[Dict[str, object]] = []
        summary_datasets: List[Tuple[str, Dict[int, List[BBox]], Dict[int, List[Prediction]]]] = []
        for video in videos:
            base_name = video.stem
            gt_file = annotations_dir / f"{base_name}.txt"
            pred_file = report_set_dir / f"{base_name}.log"

            if not gt_file.exists():
                print(f"[WARN] missing GT file: {gt_file} (skipping {video.name})", file=sys.stderr)
                continue
            if not pred_file.exists():
                print(f"[WARN] missing prediction log: {pred_file} (skipping {video.name})", file=sys.stderr)
                continue

            gt_by_frame = parse_gt_file(gt_file)
            pred_by_frame = parse_predictions_file(pred_file, class_id_filter=args.class_id)

            result = calculate_video_metrics_from_data(
                video_name=video.name,
                gt_file=gt_file,
                pred_file=pred_file,
                gt_by_frame=gt_by_frame,
                pred_by_frame=pred_by_frame,
                iou_threshold=args.iou_threshold,
                frame_align=args.frame_align,
                frame_tolerance=args.frame_tolerance,
            )
            per_video_metrics.append(result)
            summary_datasets.append((video.name, gt_by_frame, pred_by_frame))

        if not per_video_metrics:
            continue

        has_any_result = True
        summary = calculate_summary_metrics(
            iou_threshold=args.iou_threshold,
            datasets=summary_datasets,
            frame_align=args.frame_align,
            frame_tolerance=args.frame_tolerance,
        )
        print_report(report_set_name, per_video_metrics, summary)
        all_results["report_sets"].append(
            {
                "name": report_set_name,
                "path": str(report_set_dir),
                "videos": per_video_metrics,
                "summary": summary,
            }
        )

    if not has_any_result:
        print("No videos were evaluated (missing GT or prediction files for all videos).", file=sys.stderr)
        return 1

    if args.output_json:
        output_json = resolve_output_json_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nJSON report saved to: {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
