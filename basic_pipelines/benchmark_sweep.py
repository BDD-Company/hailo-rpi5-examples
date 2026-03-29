#!/usr/bin/env python3
"""
Parameter sweep for GStreamerDetectionApp detection and tracking parameters.

Runs benchmark.py with different parameter combinations on all test videos,
then calculates metrics for each combination and ranks them.

Usage examples:
  # Sweep only detection params (12 combos):
  python basic_pipelines/benchmark_sweep.py \\
    --videos-dir test_videos/ --annotations-dir annotations/ \\
    --hef-path model.hef --out-dir sweep_out/ --mode detection

  # Preview all tracking combos without running:
  python basic_pipelines/benchmark_sweep.py \\
    --videos-dir test_videos/ --hef-path model.hef --out-dir sweep_out/ \\
    --mode tracking --dry-run

  # Custom parameter grid from JSON:
  python basic_pipelines/benchmark_sweep.py \\
    --videos-dir test_videos/ --hef-path model.hef --out-dir sweep_out/ \\
    --sweep-config my_grid.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default parameter grids
# ---------------------------------------------------------------------------

DETECTION_SWEEP: dict[str, list] = {
    'nms_score_threshold': [0.2, 0.3, 0.4, 0.5],
    'nms_iou_threshold':   [0.35, 0.45, 0.55],
}

TRACKING_SWEEP: dict[str, list] = {
    'iou_thr':              [0.4, 0.6, 0.8],
    'kalman_dist_thr':      [0.4, 0.6, 0.8],
    'keep_new_frames':      [1, 2, 3],
    'keep_tracked_frames':  [0, 5, 15],
    'keep_lost_frames':     [0, 2],
}

DEFAULTS: dict[str, Any] = {
    'nms_score_threshold':    0.3,
    'nms_iou_threshold':      0.45,
    'iou_thr':                0.6,
    'kalman_dist_thr':        0.6,
    'keep_new_frames':        1,
    'keep_tracked_frames':    0,
    'keep_lost_frames':       0,
}

VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.mov'}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_combos(mode: str, sweep_config: dict | None) -> list[dict[str, Any]]:
    """Return list of parameter dicts for all combinations in the given mode."""
    det_grid  = (sweep_config or {}).get('detection', DETECTION_SWEEP)
    trk_grid  = (sweep_config or {}).get('tracking',  TRACKING_SWEEP)

    def product_of(grid: dict[str, list]) -> list[dict]:
        keys = list(grid.keys())
        return [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]

    if mode == 'detection':
        det_combos = product_of(det_grid)
        trk_fixed  = {k: DEFAULTS[k] for k in TRACKING_SWEEP}
        return [{**trk_fixed, **d} for d in det_combos]

    if mode == 'tracking':
        trk_combos = product_of(trk_grid)
        det_fixed  = {k: DEFAULTS[k] for k in DETECTION_SWEEP}
        return [{**det_fixed, **t} for t in trk_combos]

    # combined
    det_combos = product_of(det_grid)
    trk_combos = product_of(trk_grid)
    return [{**d, **t} for d, t in itertools.product(det_combos, trk_combos)]


def find_videos(videos_dir: Path) -> list[Path]:
    return sorted(p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def run_benchmark_for_video(
    benchmark_py: Path,
    video: Path,
    hef_path: Path,
    log_path: Path,
    params: dict[str, Any],
    timeout_s: int = 300,
) -> int:
    cmd = [
        sys.executable, str(benchmark_py),
        '--i', str(video),
        '--hef-path', str(hef_path),
        '--nms-score-threshold',       str(params['nms_score_threshold']),
        '--nms-iou-threshold',         str(params['nms_iou_threshold']),
        '--tracker-iou-thr',           str(params['iou_thr']),
        '--tracker-kalman-dist-thr',   str(params['kalman_dist_thr']),
        '--tracker-keep-new-frames',   str(params['keep_new_frames']),
        '--tracker-keep-tracked-frames', str(params['keep_tracked_frames']),
        '--tracker-keep-lost-frames',  str(params['keep_lost_frames']),
    ]
    err_path = log_path.with_suffix('.err')
    try:
        with log_path.open('w') as log_fh, err_path.open('w') as err_fh:
            result = subprocess.run(cmd, stdout=log_fh, stderr=err_fh,
                                    timeout=timeout_s,
                                    env={**os.environ, 'DISPLAY': os.environ.get('DISPLAY', ':0')})
        rc = result.returncode
    except subprocess.TimeoutExpired:
        print(f"  WARNING: benchmark timed out after {timeout_s}s for {video.name}", file=sys.stderr)
        rc = -1
    if rc != 0:
        err_text = err_path.read_text().strip() if err_path.exists() else ''
        if err_text:
            print(f"  stderr: {err_text[-500:]}", file=sys.stderr)
    else:
        err_path.unlink(missing_ok=True)
    # Remove blank lines (matches what benchmark.sh does with sed)
    if log_path.exists():
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        log_path.write_text('\n'.join(lines) + ('\n' if lines else ''))
    return rc


def run_metrics(
    metrics_py: Path,
    videos_dir: Path,
    annotations_dir: Path,
    reports_dir: Path,
    output_json: Path,
) -> int:
    cmd = [
        sys.executable, str(metrics_py),
        '--videos-dir',      str(videos_dir),
        '--annotations-dir', str(annotations_dir),
        '--reports-dir',     str(reports_dir),
        '--output-json',     str(output_json),
    ]
    result = subprocess.run(cmd)
    return result.returncode


def extract_summary_metric(report_set: dict, metric: str) -> float:
    """Pull the requested metric from a report_set summary dict."""
    summary = report_set.get('summary', {})
    if metric == 'f1':
        p = summary.get('precision', 0.0) or 0.0
        r = summary.get('recall', 0.0) or 0.0
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    # mAP50_95 stored with underscore in JSON
    key = 'mAP50_95' if metric == 'mAP50_95' else metric
    return summary.get(key, 0.0) or 0.0


def params_short(params: dict[str, Any]) -> str:
    return (
        f"nms_score={params['nms_score_threshold']} nms_iou={params['nms_iou_threshold']} "
        f"iou_thr={params['iou_thr']} kd_thr={params['kalman_dist_thr']} "
        f"kn={params['keep_new_frames']} kt={params['keep_tracked_frames']} kl={params['keep_lost_frames']}"
    )


def print_dry_run_table(combos: list[dict]) -> None:
    header = (
        f"{'#':>4}  {'nms_score':>9}  {'nms_iou':>7}  "
        f"{'iou_thr':>7}  {'kd_thr':>6}  {'kn':>2}  {'kt':>2}  {'kl':>2}"
    )
    print(header)
    print('-' * len(header))
    for i, p in enumerate(combos):
        print(
            f"{i:>4}  {p['nms_score_threshold']:>9.2f}  {p['nms_iou_threshold']:>7.2f}  "
            f"{p['iou_thr']:>7.2f}  {p['kalman_dist_thr']:>6.2f}  "
            f"{p['keep_new_frames']:>2}  {p['keep_tracked_frames']:>2}  {p['keep_lost_frames']:>2}"
        )
    print(f"\nTotal: {len(combos)} combinations")


def print_top_results(ranked: list[dict], top_n: int, metric: str) -> None:
    print(f"\n{'='*70}")
    print(f"TOP {min(top_n, len(ranked))} results by {metric}")
    print('='*70)
    for entry in ranked[:top_n]:
        p = entry['params']
        m = entry['metrics']
        print(
            f"  #{entry['rank']:>2}  {entry['combo_id']}  "
            f"mAP50={m.get('mAP50', 0):.3f}  F1={m.get('f1', 0):.3f}  "
            f"P={m.get('precision', 0):.3f}  R={m.get('recall', 0):.3f}"
        )
        print(f"       {params_short(p)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Parameter sweep for GStreamerDetectionApp detection and tracking parameters.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--videos-dir', type=Path, required=True,
                        help='Directory with test video files')
    parser.add_argument('--annotations-dir', type=Path, default=None,
                        help='Directory with GT .txt annotation files (defaults to --videos-dir)')
    parser.add_argument('--hef-path', type=Path, default=None,
                        help='Path to the Hailo .hef model file (not needed with --metrics-only)')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='Output directory for sweep results')
    parser.add_argument('--mode', choices=['detection', 'tracking', 'combined'],
                        default='combined',
                        help='Which parameters to sweep (default: combined)')
    parser.add_argument('--sweep-config', type=Path, default=None,
                        help='JSON file with custom parameter grids '
                             '(keys: "detection" and/or "tracking", each a dict of param->list)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top results to display (default: 5)')
    parser.add_argument('--metric', choices=['mAP50', 'mAP50_95', 'precision', 'recall', 'f1'],
                        default='mAP50',
                        help='Metric to rank results by (default: mAP50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print parameter combinations and exit without running')
    parser.add_argument('--metrics-only', action='store_true',
                        help='Skip benchmark runs, compute metrics and ranking from existing logs in --out-dir')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    scripts_dir   = Path(__file__).resolve().parent
    benchmark_py  = scripts_dir / 'benchmark.py'
    metrics_py    = scripts_dir / 'benchmark_metrics.py'

    if not metrics_py.exists():
        print(f"ERROR: benchmark_metrics.py not found at {metrics_py}", file=sys.stderr)
        return 1

    if not args.videos_dir.is_dir():
        print(f"ERROR: --videos-dir is not a directory: {args.videos_dir}", file=sys.stderr)
        return 1

    annotations_dir = args.annotations_dir or args.videos_dir

    # ------------------------------------------------------------------
    # metrics-only: skip benchmark runs, read params from existing dirs
    # ------------------------------------------------------------------
    if args.metrics_only:
        if not args.out_dir.is_dir():
            print(f"ERROR: --out-dir does not exist: {args.out_dir}", file=sys.stderr)
            return 1

        combo_params: dict[str, Any] = {}
        for combo_dir in sorted(args.out_dir.iterdir()):
            params_file = combo_dir / 'params.json'
            if combo_dir.is_dir() and params_file.exists():
                combo_params[combo_dir.name] = json.loads(params_file.read_text())

        if not combo_params:
            print(f"ERROR: no combo_XXXX directories with params.json found in {args.out_dir}", file=sys.stderr)
            return 1

        n_combos = len(combo_params)
        print(f"metrics-only: found {n_combos} combos in {args.out_dir}")
        print(f"Metric: {args.metric}  |  Output: {args.out_dir}\n")

        all_metrics_json = args.out_dir / 'all_metrics.json'
        print(f"Calculating metrics → {all_metrics_json} ...")
        rc = run_metrics(metrics_py, args.videos_dir, annotations_dir, args.out_dir, all_metrics_json)
        if rc != 0:
            print(f"WARNING: benchmark_metrics.py exited with code {rc}", file=sys.stderr)

        if not all_metrics_json.exists():
            print("ERROR: metrics file was not created; cannot produce summary.", file=sys.stderr)
            return 1

        all_metrics = json.loads(all_metrics_json.read_text())

        ranked = []
        for report_set in all_metrics.get('report_sets', []):
            name     = report_set.get('name', '')
            combo_id = Path(name).name
            params   = combo_params.get(combo_id) or combo_params.get(Path(report_set.get('path', '')).name, {})
            summary  = report_set.get('summary', {})
            p_val    = summary.get('precision', 0.0) or 0.0
            r_val    = summary.get('recall', 0.0) or 0.0
            f1       = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
            score    = extract_summary_metric(report_set, args.metric)
            ranked.append({
                'combo_id': combo_id,
                'params':   params,
                'score':    score,
                'metrics': {
                    'mAP50':     summary.get('mAP50', 0.0),
                    'mAP50_95':  summary.get('mAP50_95', 0.0),
                    'precision': p_val,
                    'recall':    r_val,
                    'f1':        f1,
                },
            })

        ranked.sort(key=lambda x: x['score'], reverse=True)
        for rank, entry in enumerate(ranked, 1):
            entry['rank'] = rank

        summary_json = args.out_dir / 'sweep_summary.json'
        summary_json.write_text(json.dumps({
            'metric':         args.metric,
            'total_combos':   n_combos,
            'ranked_results': ranked,
        }, indent=2))
        print(f"Sweep summary saved → {summary_json}")
        print_top_results(ranked, args.top_n, args.metric)
        return 0

    # ------------------------------------------------------------------
    # Normal sweep: validate hef-path and run benchmarks
    # ------------------------------------------------------------------
    if not benchmark_py.exists():
        print(f"ERROR: benchmark.py not found at {benchmark_py}", file=sys.stderr)
        return 1

    if args.hef_path is None:
        print("ERROR: --hef-path is required unless --metrics-only is set", file=sys.stderr)
        return 1

    sweep_config = None
    if args.sweep_config:
        sweep_config = json.loads(args.sweep_config.read_text())

    combos = generate_combos(args.mode, sweep_config)

    if args.dry_run:
        print_dry_run_table(combos)
        return 0

    n_combos = len(combos)
    if args.mode == 'combined' and n_combos > 100:
        print(
            f"WARNING: {n_combos} combinations in combined mode. "
            f"This will take a long time. Use --dry-run to preview or "
            f"--sweep-config to reduce the grid.",
            file=sys.stderr,
        )

    videos = find_videos(args.videos_dir)
    if not videos:
        print(f"ERROR: no video files found in {args.videos_dir}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: {n_combos} combos × {len(videos)} videos = {n_combos * len(videos)} benchmark runs")
    print(f"Mode: {args.mode}  |  Metric: {args.metric}  |  Output: {args.out_dir}\n")

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    for ci, params in enumerate(combos):
        combo_id  = f"combo_{ci:04d}"
        combo_dir = args.out_dir / combo_id
        combo_dir.mkdir(exist_ok=True)
        (combo_dir / 'params.json').write_text(json.dumps(params, indent=2))

        for vi, video in enumerate(videos):
            log_path = combo_dir / f"{video.stem}.log"
            prefix = (
                f"[{ci+1:>{len(str(n_combos))}}/{n_combos}  video {vi+1}/{len(videos)}]"
                f"  {combo_id}  {params_short(params)}"
            )
            if log_path.exists() and log_path.stat().st_size > 0:
                print(f"{prefix}  [skip, log exists]", flush=True)
                continue
            print(prefix, flush=True)
            rc = run_benchmark_for_video(benchmark_py, video, args.hef_path, log_path, params)
            if rc != 0:
                print(f"  WARNING: benchmark exited with code {rc} for {video.name}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------------
    all_metrics_json = args.out_dir / 'all_metrics.json'
    print(f"\nCalculating metrics → {all_metrics_json} ...")
    rc = run_metrics(metrics_py, args.videos_dir, annotations_dir, args.out_dir, all_metrics_json)
    if rc != 0:
        print(f"WARNING: benchmark_metrics.py exited with code {rc}", file=sys.stderr)

    if not all_metrics_json.exists():
        print("ERROR: metrics file was not created; cannot produce summary.", file=sys.stderr)
        return 1

    all_metrics = json.loads(all_metrics_json.read_text())

    # ------------------------------------------------------------------
    # Rank results
    # ------------------------------------------------------------------
    # Build combo_id → params mapping
    combo_params = {f"combo_{i:04d}": p for i, p in enumerate(combos)}

    ranked = []
    for report_set in all_metrics.get('report_sets', []):
        name     = report_set.get('name', '')
        combo_id = Path(name).name  # report set name is the subdir name
        params   = combo_params.get(combo_id, {})
        if not params:
            # Try matching by directory basename
            params = combo_params.get(Path(report_set.get('path', '')).name, {})

        summary  = report_set.get('summary', {})
        p_val    = summary.get('precision', 0.0) or 0.0
        r_val    = summary.get('recall', 0.0) or 0.0
        f1       = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

        score = extract_summary_metric(report_set, args.metric)
        ranked.append({
            'combo_id': combo_id,
            'params':   params,
            'score':    score,
            'metrics': {
                'mAP50':      summary.get('mAP50', 0.0),
                'mAP50_95':   summary.get('mAP50_95', 0.0),
                'precision':  p_val,
                'recall':     r_val,
                'f1':         f1,
            },
        })

    ranked.sort(key=lambda x: x['score'], reverse=True)
    for rank, entry in enumerate(ranked, 1):
        entry['rank'] = rank

    summary_json = args.out_dir / 'sweep_summary.json'
    summary_json.write_text(json.dumps({
        'metric':       args.metric,
        'mode':         args.mode,
        'total_combos': n_combos,
        'ranked_results': ranked,
    }, indent=2))
    print(f"Sweep summary saved → {summary_json}")

    print_top_results(ranked, args.top_n, args.metric)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
