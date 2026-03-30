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
FrameKey = str


@dataclass(frozen=True)
class Prediction:
    frame: int
    confidence: float
    bbox: BBox


LOG_LINE_RE = re.compile(
    r"frame#(?P<frame>\d+)\s+detection:\s+#(?P<track>-?\d+)\s+class:\s+(?P<class_id>-?\d+)\s+"
    r"confidence:\s+(?P<confidence>\d*\.?\d+)\s+\(x:\s*(?P<x>-?\d*\.?\d+),\s*"
    r"y:\s*(?P<y>-?\d*\.?\d+),\s*w:\s*(?P<w>-?\d*\.?\d+),\s*h:\s*(?P<h>-?\d*\.?\d+)\)"
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
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path (file path or existing directory).",
    )
    return parser.parse_args()


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


def flatten_predictions(pred_by_frame: Dict[int, List[Prediction]]) -> List[Prediction]:
    preds = [pred for preds in pred_by_frame.values() for pred in preds]
    preds.sort(key=lambda p: p.confidence, reverse=True)
    return preds


def match_frame_predictions(preds: List[Prediction], gt_boxes: List[BBox], iou_threshold: float) -> Tuple[int, int, int]:
    if not preds and not gt_boxes:
        return (0, 0, 0)

    sorted_preds = sorted(preds, key=lambda p: p.confidence, reverse=True)
    matched_gt = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if matched_gt[idx]:
                continue
            score = iou(pred.bbox, gt)
            if score > best_iou:
                best_iou = score
                best_gt_idx = idx
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_gt[best_gt_idx] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for m in matched_gt if not m)
    return (tp, fp, fn)


def calculate_ap(gt_by_frame: Dict[int, List[BBox]], sorted_preds: List[Prediction], iou_threshold: float) -> float:
    total_gt = sum(len(boxes) for boxes in gt_by_frame.values())
    if total_gt == 0:
        return 0.0

    matched_by_frame = {frame: [False] * len(boxes) for frame, boxes in gt_by_frame.items()}
    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for pred in sorted_preds:
        gt_boxes = gt_by_frame.get(pred.frame, [])
        gt_matches = matched_by_frame.setdefault(pred.frame, [False] * len(gt_boxes))

        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if gt_matches[idx]:
                continue
            score = iou(pred.bbox, gt)
            if score > best_iou:
                best_iou = score
                best_gt_idx = idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_matches[best_gt_idx] = True
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
    )


def calculate_video_metrics_from_data(
    video_name: str,
    gt_file: Path,
    pred_file: Path,
    gt_by_frame: Dict[int, List[BBox]],
    pred_by_frame: Dict[int, List[Prediction]],
    iou_threshold: float,
) -> Dict[str, object]:
    frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    tp_total = 0
    fp_total = 0
    fn_total = 0
    fp_frame_counts: Dict[int, int] = {}
    fn_frame_counts: Dict[int, int] = {}

    for frame in frames:
        preds = pred_by_frame.get(frame, [])
        gt_boxes = gt_by_frame.get(frame, [])
        tp, fp, fn = match_frame_predictions(preds, gt_boxes, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        if fp > 0:
            fp_frame_counts[frame] = fp_frame_counts.get(frame, 0) + fp
        if fn > 0:
            fn_frame_counts[frame] = fn_frame_counts.get(frame, 0) + fn

    precision = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    recall = (tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0

    sorted_preds = flatten_predictions(pred_by_frame)
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap_per_threshold = [calculate_ap(gt_by_frame, sorted_preds, thr) for thr in iou_thresholds]
    map50 = ap_per_threshold[0]
    map50_95 = float(mean(ap_per_threshold)) if ap_per_threshold else 0.0

    return {
        "video": video_name,
        "gt_file": str(gt_file),
        "prediction_file": str(pred_file),
        "num_gt_frames": len(gt_by_frame),
        "num_gt_boxes": int(sum(len(b) for b in gt_by_frame.values())),
        "num_predictions": int(sum(len(p) for p in pred_by_frame.values())),
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
    gt_by_key: Dict[FrameKey, List[BBox]],
    pred_tuples: List[Tuple[float, FrameKey, BBox]],
    iou_threshold: float,
) -> float:
    total_gt = sum(len(boxes) for boxes in gt_by_key.values())
    if total_gt == 0:
        return 0.0

    matched_by_key = {key: [False] * len(boxes) for key, boxes in gt_by_key.items()}
    sorted_preds = sorted(pred_tuples, key=lambda x: x[0], reverse=True)

    tp_flags: List[int] = []
    fp_flags: List[int] = []
    for confidence, frame_key, pred_bbox in sorted_preds:
        del confidence  # confidence only used for sorting
        gt_boxes = gt_by_key.get(frame_key, [])
        gt_matches = matched_by_key.setdefault(frame_key, [False] * len(gt_boxes))

        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if gt_matches[idx]:
                continue
            score = iou(pred_bbox, gt)
            if score > best_iou:
                best_iou = score
                best_gt_idx = idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_matches[best_gt_idx] = True
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
) -> Dict[str, object]:
    tp_total = 0
    fp_total = 0
    fn_total = 0
    gt_boxes_total = 0
    pred_total = 0

    gt_by_key: Dict[FrameKey, List[BBox]] = {}
    pred_tuples: List[Tuple[float, FrameKey, BBox]] = []

    for video_name, gt_by_frame, pred_by_frame in datasets:
        gt_boxes_total += sum(len(boxes) for boxes in gt_by_frame.values())
        pred_total += sum(len(preds) for preds in pred_by_frame.values())

        for frame, gt_boxes in gt_by_frame.items():
            frame_key = f"{video_name}::{frame}"
            gt_by_key[frame_key] = gt_boxes

        for frame, preds in pred_by_frame.items():
            frame_key = f"{video_name}::{frame}"
            for pred in preds:
                pred_tuples.append((pred.confidence, frame_key, pred.bbox))

        frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
        for frame in frames:
            preds = pred_by_frame.get(frame, [])
            gt_boxes = gt_by_frame.get(frame, [])
            tp, fp, fn = match_frame_predictions(preds, gt_boxes, iou_threshold)
            tp_total += tp
            fp_total += fp
            fn_total += fn

    precision = (tp_total / (tp_total + fp_total)) if (tp_total + fp_total) > 0 else 0.0
    recall = (tp_total / (tp_total + fn_total)) if (tp_total + fn_total) > 0 else 0.0

    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    ap_per_threshold = [_calculate_dataset_ap(gt_by_key, pred_tuples, thr) for thr in iou_thresholds]
    map50 = ap_per_threshold[0] if ap_per_threshold else 0.0
    map50_95 = float(mean(ap_per_threshold)) if ap_per_threshold else 0.0

    return {
        "num_videos": len(datasets),
        "num_gt_boxes": int(gt_boxes_total),
        "num_predictions": int(pred_total),
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
            )
            per_video_metrics.append(result)
            summary_datasets.append((video.name, gt_by_frame, pred_by_frame))

        if not per_video_metrics:
            continue

        has_any_result = True
        summary = calculate_summary_metrics(args.iou_threshold, summary_datasets)
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
