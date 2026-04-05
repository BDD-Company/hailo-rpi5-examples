#!/usr/bin/env python3
"""
Собирает *.log от benchmark.py, вытаскивает строки stage delay и (если есть)
benchmark_file_wall_time_s, считает средние по этапам.

Структура каталогов как у benchmark.sh / benchmark_sweep: корень с подпапками
или сразу *.log. Дополнительно можно указать --recursive для обхода всех .log.

Примеры:
  python basic_pipelines/benchmark_latency_aggregate.py --reports-dir ./bench_out
  python basic_pipelines/benchmark_latency_aggregate.py --reports-dir . --recursive
  python basic_pipelines/benchmark_latency_aggregate.py --reports-dir ./bench_out --json summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Строка из StageLatencyAggregator.print_report(), например:
#   inference_wrapper: n=120 min/med/mean/max ms = 1.234 1.345 1.400 2.100
_STAGE_LINE_RE = re.compile(
    r"^\s{2}(?P<name>.+?):\s+n=(?P<n>\d+)\s+min/med/mean/max ms\s*=\s*"
    r"(?P<min>[\d.]+)\s+(?P<med>[\d.]+)\s+(?P<mean>[\d.]+)\s+(?P<max>[\d.]+)\s*$"
)
_WALL_RE = re.compile(r"^benchmark_file_wall_time_s:\s*(?P<s>[\d.]+)\s*$")

_SKIP_LOG_NAMES = frozenset({"hailo_info.log"})


@dataclass
class ParsedLog:
    path: str
    wall_time_s: float | None
    stages: dict[str, dict[str, float | int]] = field(default_factory=dict)
    parse_ok: bool = False


def parse_benchmark_log(text: str) -> tuple[dict[str, dict[str, float | int]], float | None, bool]:
    stages: dict[str, dict[str, float | int]] = {}
    wall: float | None = None
    in_latency_block = False
    for line in text.splitlines():
        if "Pipeline stage delay" in line:
            in_latency_block = True
            continue
        wm = _WALL_RE.match(line.strip())
        if wm:
            wall = float(wm.group("s"))
            continue
        sm = _STAGE_LINE_RE.match(line)
        if sm:
            name = sm.group("name").strip()
            stages[name] = {
                "n": int(sm.group("n")),
                "min_ms": float(sm.group("min")),
                "median_ms": float(sm.group("med")),
                "mean_ms": float(sm.group("mean")),
                "max_ms": float(sm.group("max")),
            }
            in_latency_block = True
            continue
        if in_latency_block and line.strip().startswith("Detection statistics"):
            break
    ok = bool(stages)
    return stages, wall, ok


def discover_logs(reports_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        found = sorted(reports_dir.rglob("*.log"))
    else:
        direct = sorted(reports_dir.glob("*.log"))
        if direct:
            found = direct
        else:
            found = []
            for sub in sorted(reports_dir.iterdir()):
                if sub.is_dir():
                    found.extend(sorted(sub.glob("*.log")))
    return [p for p in found if p.is_file() and p.name not in _SKIP_LOG_NAMES]


def weighted_mean(pairs: list[tuple[float, int]]) -> float | None:
    num = sum(m * n for m, n in pairs)
    den = sum(n for _, n in pairs)
    if den <= 0:
        return None
    return num / den


def aggregate_stage_stats(rows: list[ParsedLog]) -> dict[str, Any]:
    """По каждому имени этапа: среднее mean с весом n, и то же для median/min/max как взвешенное приближение."""
    by_stage: dict[str, list[tuple[float, int]]] = {}
    by_stage_med: dict[str, list[tuple[float, int]]] = {}
    for row in rows:
        if not row.parse_ok:
            continue
        for name, s in row.stages.items():
            n = int(s["n"])
            by_stage.setdefault(name, []).append((float(s["mean_ms"]), n))
            by_stage_med.setdefault(name, []).append((float(s["median_ms"]), n))
    out: dict[str, Any] = {}
    for name in sorted(by_stage.keys()):
        out[name] = {
            "mean_ms_weighted_by_frames": weighted_mean(by_stage[name]),
            "median_ms_weighted_by_frames": weighted_mean(by_stage_med[name]),
            "total_frames": sum(n for _, n in by_stage[name]),
            "logs_with_stage": len(by_stage[name]),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Корень с отчётами (*.log от benchmark.py)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Рекурсивно найти все *.log под каталогом",
    )
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Записать полный результат в JSON",
    )
    args = ap.parse_args()
    root = args.reports_dir.resolve()
    if not root.is_dir():
        print(f"Не каталог: {root}", file=sys.stderr)
        return 1

    log_paths = discover_logs(root, args.recursive)
    if not log_paths:
        print(f"Не найдено *.log в {root}", file=sys.stderr)
        return 1

    rows: list[ParsedLog] = []
    for lp in log_paths:
        text = lp.read_text(encoding="utf-8", errors="replace")
        stages, wall, ok = parse_benchmark_log(text)
        try:
            rel = str(lp.relative_to(root))
        except ValueError:
            rel = str(lp)
        rows.append(
            ParsedLog(
                path=rel,
                wall_time_s=wall,
                stages=stages,
                parse_ok=ok,
            )
        )

    parsed = [r for r in rows if r.parse_ok]
    skipped = [r for r in rows if not r.parse_ok]

    print(f"Логов всего: {len(rows)}, с блоком stage delay: {len(parsed)}, без (пропуск): {len(skipped)}")
    if skipped:
        for r in skipped[:20]:
            print(f"  (нет latency) {r.path}")
        if len(skipped) > 20:
            print(f"  ... ещё {len(skipped) - 20}")

    agg = aggregate_stage_stats(parsed)

    print("\n=== Среднее по этапам (вес = число кадров n в каждом логе, поле mean_ms) ===")
    for name, info in agg.items():
        m = info["mean_ms_weighted_by_frames"]
        med = info["median_ms_weighted_by_frames"]
        print(
            f"  {name}: mean_ms≈{m:.4f}  median_ms≈{med:.4f}  "
            f"кадров={info['total_frames']}  логов={info['logs_with_stage']}"
        )

    walls = [r.wall_time_s for r in parsed if r.wall_time_s is not None]
    print("\n=== Общее время обработки файла (wall clock одного прогона benchmark.py) ===")
    if walls:
        print(f"  Логов с benchmark_file_wall_time_s: {len(walls)} / {len(parsed)}")
        print(f"  Сумма wall time по файлам: {sum(walls):.3f} s")
        print(f"  Среднее wall time на файл: {sum(walls) / len(walls):.3f} s")
    else:
        print(
            "  Нет строк benchmark_file_wall_time_s (нужен свежий benchmark.py, печатает при EOS)."
        )

    print("\n=== По каждому логу: wall time и mean_ms этапов ===")
    for r in sorted(parsed, key=lambda x: x.path):
        wt = f"{r.wall_time_s:.3f}s" if r.wall_time_s is not None else "—"
        parts = [f"{k}={v['mean_ms']:.3f}ms" for k, v in sorted(r.stages.items())]
        print(f"  {wt:>10}  {r.path}")
        if parts:
            print(f"             {' | '.join(parts)}")

    if args.json:
        payload: dict[str, Any] = {
            "reports_dir": str(root),
            "aggregate_by_stage": agg,
            "wall_time_s": {
                "count": len(walls),
                "sum_s": sum(walls) if walls else None,
                "mean_per_file_s": (sum(walls) / len(walls)) if walls else None,
            },
            "per_file": [
                {
                    "path": r.path,
                    "wall_time_s": r.wall_time_s,
                    "stages": r.stages,
                    "parse_ok": r.parse_ok,
                }
                for r in rows
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nJSON: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
