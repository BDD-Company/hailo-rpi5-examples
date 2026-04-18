#!/usr/bin/env python3
"""Офлайн-прогон видео: Hailo → ByteTrack → drone_controller → запись debug_* (как в app.py).

Видео идёт через тот же GStreamer-пайплайн детекции, что и в app.py; кадры с треками
попадают в drone_controller (с MockDroneMover). Очередь детекций — blocking queue.Queue
(по умолчанию до 2048 кадров), без вытеснения: OverwriteQueue терял кадры и давал скачки
frame#. Аннотированный вывод пишется в каталог
через RecorderSink в сегменты MKV с префиксом debug_<timestamp>_.

Запуск (из каталога BDD или с указанием пути к скрипту):
    python debug_detections.py /path/to/video.mkv [--out-dir ./_DEBUG] [--segment-seconds 10] [--params '{...}'] [аргументы Hailo]

Первый позиционный аргумент — входное видео; подставляется как --input для парсера Hailo.
"""

from __future__ import annotations

import ast
import datetime
import logging
import os
import queue
import sys
import threading
import types
from pathlib import Path

# --- mavsdk-заглушка до любых импортов, тянущих drone / mavsdk ---
try:
    import mavsdk  # noqa: F401
except ImportError:
    import types as _types

    _mavsdk = _types.ModuleType("mavsdk")
    _mavsdk.System = type("System", (), {"__init__": lambda self: None})
    _offboard = _types.ModuleType("mavsdk.offboard")
    for _name in (
        "PositionNedYaw",
        "VelocityBodyYawspeed",
        "Attitude",
        "VelocityNedYaw",
        "AttitudeRate",
    ):
        setattr(_offboard, _name, type(_name, (), {}))
    _offboard.OffboardError = type("OffboardError", (Exception,), {})
    _telemetry = _types.ModuleType("mavsdk.telemetry")
    _telemetry.Telemetry = type("Telemetry", (), {})
    _telemetry.EulerAngle = type("EulerAngle", (), {})
    _telemetry.LandedState = type("LandedState", (), {"IN_AIR": 1})
    sys.modules["mavsdk"] = _mavsdk
    sys.modules["mavsdk.offboard"] = _offboard
    sys.modules["mavsdk.telemetry"] = _telemetry

from app import App, app_callback, user_app_callback_class
from bytetrack import BYTETracker
from debug_drone_controller import MockDroneMover
from debug_output import debug_output_thread
import drone_controller as drone_controller_module
from drone_controller import drone_controlling_thread
from hailo_apps.hailo_app_python.core.common.core import get_default_parser
from helpers import STOP, XY, configure_logging
from OverwriteQueue import OverwriteQueue
from video_sink_gstreamer import RecorderSink
from video_sink_multi import MultiSink

import time as real_time_module

logger = logging.getLogger(__name__)


class FileEosExitApp(App):
    """Для файлового входа: по EOS завершить пайплайн, без перемотки в начало (как в GStreamerApp для не-file)."""

    def on_eos(self):
        logger.info("Конец видео (EOS) — выход без rewind.")
        self.shutdown()


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    os.environ["HAILO_ENV_FILE"] = str(project_root / ".env")

    configure_logging(level=logging.DEBUG)
    logging.getLogger("picamera2").setLevel(logging.WARNING)
    logging.getLogger("mavsdk_server").setLevel(logging.ERROR)

    if len(sys.argv) < 2:
        print(
            "Использование: python debug_detections.py <видео> "
            "[--out-dir DIR] [--segment-seconds N] [--params '{dict}'] [опции Hailo]",
            file=sys.stderr,
        )
        sys.exit(2)

    video_path = Path(sys.argv[1]).resolve()
    if not video_path.is_file():
        print(f"Файл не найден: {video_path}", file=sys.stderr)
        sys.exit(1)

    sys.argv = [sys.argv[0], "--input", str(video_path)] + sys.argv[2:]

    parser = get_default_parser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("_DEBUG"),
        help="Каталог для debug_*.mkv (и служебного RAW при record_videos)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=10,
        help="Длина сегмента splitmux для debug-записи",
    )
    parser.add_argument(
        "--detections-queue-depth",
        type=int,
        default=2048,
        help="Макс. глубина очереди кадров к drone_controller (0 = без лимита; при полной очереди GStreamer ждёт)",
    )
    parser.add_argument(
        "--params",
        type=lambda x: dict(ast.literal_eval(x)),
        default=None,
        help="Доп. поля control_config (dict через ast.literal_eval)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    dq_max = args.detections_queue_depth
    if dq_max <= 0:
        dq_max = 0
    detections_queue: queue.Queue = queue.Queue(maxsize=dq_max)
    output_queue: OverwriteQueue = OverwriteQueue(maxsize=200)
    ready = threading.Event()

    control_config = {
        "confidence_min": 0.4,
        "confidence_move": 0.3,
        "thrust_takeoff": 0.5,
        "thrust_min": 0.7,
        "thrust_max": 0.9,
        "thrust_dynamic": False,
        "thrust_proportional_to_target_size": False,
        "target_lost_fade_per_frame": 0.99,
        "target_estimator_clear_history_after_target_lost_frames": 3,
        "estimation_3d": True,
        "estimation_3d_mode": "cluster",
        "estimation_3d_use_initial_velocity": True,
        "estimation_lookahead_frames": 2,
        "estimation_lookahead_dynamic": True,
        "estimation_lookahead_dynamic_sqrt": True,
        "estimation_lookahead_dynamic_frames_near": 1,
        "estimation_lookahead_dynamic_frames_medium": 1,
        "estimation_lookahead_dynamic_frames_far": 1,
        "pd_coeff_p": 3,
        "pd_coeff_d": 0,
        "pd_coeff_p_safe_min": 0.6,
        "pd_coeff_p_min": 0.5,
        "pd_coeff_p_max": 10,
        "pd_coeff_p_dynamic": False,
        "pd_coeff_p_dynamic_use_piecewise": False,
        "pd_coeff_p_dynamic_min_target_size": 0.0005,
        "pd_coeff_p_dynamic_min": 0.6,
        "pd_coeff_p_dynamic_max_target_size": 0.0120,
        "pd_coeff_p_dynamic_max": 6,
        "pd_coeff_p_dynamic_stage_1_threshold": 0.01,
        "pd_coeff_p_dynamic_stage_2_threshold": 0.05,
        "pd_coeff_p_dynamic_stage_1_ratio": 1,
        "pd_coeff_p_dynamic_stage_2_ratio": 1,
        "pd_coeff_p_dynamic_stage_3_ratio": 1,
        "frame_angular_size_deg": XY(107, 85),
        "target_size_m": XY(1.8, 1.8),
        "inertia_correction_gain": 0,
        "inertia_correction_limits": XY(1, 1),
        "inertia_correction_min_speed_ms": 5,
        "safe_takeoff_period_ns": 300_000_000,
        "delay_takeof_until_n_detection_frames": 3,
        "aim_point": XY(0.5, 0.5),
        "aim_point_max_offset": XY(0.5, 0.6),
        "follow_target_position_ned": False,
        "drone_use_set_attitude": False,
        "drone_min_lift_fraction": 0.1,
        "drone_lift_velocity_headroom_ms": 3.0,
        "drone_lift_accel_headroom_mss": 5.0,
        "DEBUG": True,
        "bytetrack_track_thresh": 0.3,
        "bytetrack_det_thresh": 0.7,
        "bytetrack_match_thresh": 0.5,
        "bytetrack_track_buffer": 30,
        "bytetrack_frame_rate": 30,
    }
    if args.params:
        control_config.update(args.params)

    bytetracker = BYTETracker(
        track_thresh=control_config["bytetrack_track_thresh"],
        det_thresh=control_config["bytetrack_det_thresh"],
        match_thresh=control_config["bytetrack_match_thresh"],
        track_buffer=control_config["bytetrack_track_buffer"],
        frame_rate=control_config["bytetrack_frame_rate"],
    )
    for _k in (
        "bytetrack_track_thresh",
        "bytetrack_det_thresh",
        "bytetrack_match_thresh",
        "bytetrack_track_buffer",
        "bytetrack_frame_rate",
    ):
        control_config.pop(_k, None)

    user_data = user_app_callback_class(detections_queue, bytetracker)
    user_data.use_frame = False

    drone_controller_module.DroneMover = MockDroneMover
    mock_time = types.ModuleType("mock_time")
    for attr in dir(real_time_module):
        if not attr.startswith("_"):
            setattr(mock_time, attr, getattr(real_time_module, attr))
    mock_time.monotonic_ns = lambda: int(real_time_module.time() * 1_000_000_000)
    drone_controller_module.time = mock_time

    def _drone_worker() -> None:
        try:
            drone_controlling_thread(
                "replay://mock",
                {"upside_down_angle_deg": 130, "upside_down_hold_s": 0.2},
                detections_queue,
                control_config=dict(control_config),
                output_queue=output_queue,
                signal_event_when_ready=ready,
            )
        except Exception:
            logger.exception("drone_controlling_thread")

    action_thread = threading.Thread(target=_drone_worker, name="Drone", daemon=False)

    sink = MultiSink(
        [
            RecorderSink(
                30,
                str(out_dir),
                segment_seconds=args.segment_seconds,
                filename_base=f"debug_{start_time_str}",
            ),
        ]
    )
    output_thread = threading.Thread(
        target=debug_output_thread,
        args=(output_queue, sink),
        name="DEBUG_RECORD",
        daemon=True,
    )

    action_thread.start()
    output_thread.start()

    # RAW с хвоста пайплайна не пишем — только debug_* из debug_output_thread
    app = FileEosExitApp(
        app_callback,
        user_data,
        parser=parser,
        video_output_path=str(out_dir),
        video_output_chunk_length_s=args.segment_seconds,
        video_filename_base=f"RAW_{start_time_str}",
        record_videos=False,
    )
    app.user_data.use_frame = False

    try:
        logger.info(
            "Старт: видео=%s, выход=%s, debug-файлы: %s/debug_%s_*.mkv",
            video_path,
            out_dir,
            out_dir,
            start_time_str,
        )
        app.run(ready)
    except KeyboardInterrupt:
        logger.info("Прервано пользователем")
    finally:
        try:
            detections_queue.put(STOP)
        except Exception:
            logger.debug("Очередь детекций при STOP", exc_info=True)
        action_thread.join(timeout=120)

    logger.info("Готово. Записан отладочный ролик (сегменты) в %s", out_dir)


if __name__ == "__main__":
    main()
