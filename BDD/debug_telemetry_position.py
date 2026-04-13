"""Parse telemetry from a BDD debug log and reconstruct per-frame poses."""

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from math import nan
from pathlib import Path

from telemetry_position import Pose, get_pose

import logging

logger = logging.getLogger(__name__)

TELEMETRY_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s.*?"
    r"frame=#(\d+)\s+(?:!+\s+)?telemetry:\s+(.+)$"
)


@dataclass(frozen=True)
class FramePose:
    timestamp: datetime
    frame_id: int
    pose: Pose


def parse_telemetry_log(path: Path) -> list[FramePose]:
    results: list[FramePose] = []
    with open(path, "r") as f:
        for line in f:
            m = TELEMETRY_RE.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
            frame_id = int(m.group(2))
            try:
                telemetry_dict = eval(m.group(3))  # noqa: S307 — trusted log data
                pose = get_pose(telemetry_dict)
                results.append(FramePose(timestamp=ts, frame_id=frame_id, pose=pose))
            except SyntaxError as e:
                logger.error('Partially broken log?', exc_info=True)
                break

    return results


def main() -> None:
    LOG_PATH = Path(__file__).resolve().parent.parent.parent / "_BACKUPS/2026-04-06/_DEBUG_09/BDD_2026_04_06_13_56_14_02_00_.log"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_PATH
    frames = parse_telemetry_log(path)
    print(f"Parsed {len(frames)} frames from {path.name}\n")

    print(f"{'frame':>6}  {'timestamp':>26}  {'N(m)':>9}  {'E(m)':>9}  {'D(m)':>9}  {'alt(m)':>8}  {'yaw°':>8}  {'pitch°':>8}  {'roll°':>7}")
    print("-" * 120)
    for fp in frames:
        p = fp.pose
        print(
            f"#{fp.frame_id:04d}   "
            f"{fp.timestamp.isoformat(sep=' ')}  "
            f"{p.position.north_m:9.3f}  "
            f"{p.position.east_m:9.3f}  "
            f"{p.position.down_m:9.3f}  "
            f"{p.position.altitude_m:8.3f}  "
            f"{p.orientation.yaw_deg:8.2f}  "
            f"{p.orientation.pitch_deg:8.2f}  "
            f"{p.orientation.roll_deg:7.2f}"
        )


if __name__ == "__main__":
    main()
