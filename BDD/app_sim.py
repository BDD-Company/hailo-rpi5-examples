#!/usr/bin/env python3
"""
app_sim.py — simulation/PC variant of app.py.

Receives video over UDP RTP H264 (e.g. from MAVLink companion or GStreamer sender)
instead of the Hailo camera pipeline, and detects the target by HSV red-colour
segmentation instead of the Hailo neural-network.

Drone control reuses drone_controller.py unchanged.

Usage:
    python app_sim.py [--DEBUG]
"""

import datetime
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp  # noqa: E402

Gst.init(None)

from drone_controller import drone_controlling_thread
from debug_output import debug_output_thread
from helpers import (
    Detection, Detections, FrameMetadata,
    Rect, XY, STOP,
    configure_logging,
)
from OverwriteQueue import OverwriteQueue
from video_sink_gstreamer import RecorderSink
from video_sink_multi import MultiSink

logger = logging.getLogger(__name__)

DEBUG = False

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

_detect_times: list[float] = []


def detect_red_object(frame):
    """Detect the largest red object in a BGR frame.

    Returns (cx, cy, x, y, w, h) in pixel coordinates, or None if nothing found.
    Appends detection latency (ms) to _detect_times for profiling.
    """
    start = time.perf_counter()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])

    mask = (
        cv2.inRange(hsv, lower1, upper1) |
        cv2.inRange(hsv, lower2, upper2)
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        _detect_times.append((time.perf_counter() - start) * 1000)
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    cx = x + w // 2
    cy = y + h // 2

    _detect_times.append((time.perf_counter() - start) * 1000)

    return cx, cy, x, y, w, h


# ---------------------------------------------------------------------------
# Capture + detect thread  (GStreamer via gi — no OpenCV+GStreamer required)
# ---------------------------------------------------------------------------

UDP_PIPELINE = (
    "udpsrc port=5600 ! "
    "application/x-rtp,encoding-name=H264,payload=96 ! "
    "rtph264depay ! decodebin ! videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink name=sink sync=false drop=true max-buffers=1 emit-signals=false"
)


_PULL_TIMEOUT_NS = 100_000_000  # 100 ms — lets stop_event be checked regularly


def _pull_frame(appsink) -> np.ndarray | None:
    """Pull one BGR frame from a GstApp.AppSink. Returns None on timeout/EOS/error."""
    sample = appsink.emit("try-pull-sample", _PULL_TIMEOUT_NS)
    if sample is None:
        return None

    caps = sample.get_caps()
    if caps is None:
        return None
    struct = caps.get_structure(0)
    width = struct.get_value("width")
    height = struct.get_value("height")

    buf = sample.get_buffer()
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(height, width, 3).copy()
    finally:
        buf.unmap(map_info)

    return frame


def capture_thread_fn(
    detections_queue: OverwriteQueue,
    stop_event: threading.Event,
    ready_event: threading.Event | None = None,
    show: bool = False,
):
    """Read frames from UDP RTP H264 via GStreamer, run detect_red_object, push Detections."""

    pipeline = Gst.parse_launch(UDP_PIPELINE)
    appsink = pipeline.get_by_name("sink")

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        logger.error("Failed to start GStreamer pipeline: %s", UDP_PIPELINE)
        return

    logger.info("GStreamer pipeline started. Waiting for drone ready signal…")
    if ready_event:
        ready_event.wait()

    logger.info("Starting capture loop")

    frame_id = 0
    frames_with_detection = 0
    last_log_time = time.monotonic()

    while not stop_event.is_set():
        capture_ts_ns = time.monotonic_ns()

        frame = _pull_frame(appsink)
        if frame is None:
            msg = pipeline.get_bus().timed_pop_filtered(
                0, Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
            if msg is not None:
                logger.warning("GStreamer pipeline ended: %s", msg.type)
                break
            time.sleep(0.005)
            continue

        h_frame, w_frame = frame.shape[:2]

        detect_start_ns = time.monotonic_ns()
        result = detect_red_object(frame)
        detect_end_ns = time.monotonic_ns()

        detections_list = []
        if result is not None:
            frames_with_detection += 1
            cx, cy, x, y, w, h = result
            detections_list.append(Detection(
                bbox=Rect.from_xywh(
                    x / w_frame, y / h_frame,
                    w / w_frame, h / h_frame,
                ),
                confidence=1.0,
                track_id=None,
            ))

            if show:
                vis = frame.copy()
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(vis, f"cx={cx} cy={cy}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("app_sim detection", vis)
                cv2.waitKey(1)
        elif show:
            cv2.imshow("app_sim detection", frame)
            cv2.waitKey(1)

        # Log stats every 5 seconds
        now = time.monotonic()
        if now - last_log_time >= 5.0:
            elapsed = now - last_log_time
            fps = frame_id / max(elapsed, 1e-6) if frame_id > 0 else 0
            det_avg = (
                sum(_detect_times[-100:]) / len(_detect_times[-100:])
                if _detect_times else 0
            )
            logger.info(
                "frames=%d  detections=%d  fps≈%.1f  detect_avg=%.1fms  frame=%dx%d",
                frame_id, frames_with_detection, fps, det_avg, w_frame, h_frame,
            )
            last_log_time = now

        detections_queue.put(Detections(
            frame_id=frame_id,
            frame=frame,
            detections=detections_list,
            meta=FrameMetadata(
                capture_timestamp_ns=capture_ts_ns,
                detection_start_timestamp_ns=detect_start_ns,
                detection_end_timestamp_ns=detect_end_ns,
            ),
        ))

        frame_id += 1

    pipeline.set_state(Gst.State.NULL)
    if show:
        cv2.destroyAllWindows()
    logger.info("Capture thread stopped (processed %d frames, %d with detection)",
                frame_id, frames_with_detection)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _add_file_logging(log_path: str) -> None:
    """Add a FileHandler that writes the same format as the console to *log_path*."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(threadName)s]"
        " @ { %(filename)s:%(lineno)s : %(funcName)20s() }"
        " <%(levelname)s> :\t%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)


def main():
    configure_logging(level=logging.DEBUG)
    logging.getLogger("mavsdk_server").setLevel(logging.ERROR)

    global DEBUG
    if "--DEBUG" in sys.argv:
        DEBUG = True
        sys.argv.remove("--DEBUG")

    show = "--show" in sys.argv
    if show:
        sys.argv.remove("--show")

    if DEBUG:
        logger.error("")
        logger.error("!!! ============================================================== !!!")
        logger.error("!!! Will run in DEBUG mode, behaviour might differ from production !!!")
        logger.error("!!! ============================================================== !!!")
        logger.error("")

    start_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_path = f"./_DEBUG/BDD_SIM_{start_time_str}_.log"
    _add_file_logging(log_path)
    logger.info("Logging to %s", log_path)

    detections_queue = OverwriteQueue(maxsize=20)
    output_queue = OverwriteQueue(maxsize=200)

    # drone_controlling_thread sets this when armed and ready;
    # capture thread waits on it before pushing frames.
    ready_event = threading.Event()
    stop_event = threading.Event()

    control_config = {
        'confidence_min': 0.4,
        'confidence_move': 0.3,

        'thrust_takeoff': 0.5,
        'thrust_min': 0.5,
        'thrust_max': 0.5,
        'thrust_dynamic': False,
        'thrust_proportional_to_target_size': False,

        'target_lost_fade_per_frame': 0.99,
        'target_estimator_clear_history_after_target_lost_frames': 3,

        'estimation_3d': True,
        'estimation_3d_mode': 'wls',
        'estimation_lookahead_frames': 2,
        'estimation_lookahead_dynamic': False,
        'estimation_lookahead_dynamic_frames_near':   1,
        'estimation_lookahead_dynamic_frames_medium': 2,
        'estimation_lookahead_dynamic_frames_far':    4,

        'pd_coeff_p': 3,
        'pd_coeff_d': 0,
        'pd_coeff_p_safe_min': 0.6,
        'pd_coeff_p_min': 0.5,
        'pd_coeff_p_max': 10,

        'pd_coeff_p_dynamic': False,
        'pd_coeff_p_dynamic_use_piecewise': False,
        'pd_coeff_p_dynamic_min_target_size': 0.0005,
        'pd_coeff_p_dynamic_min': 0.6,
        'pd_coeff_p_dynamic_max_target_size': 0.0120,
        'pd_coeff_p_dynamic_max': 6,

        'pd_coeff_p_dynamic_stage_1_threshold': 0.01,
        'pd_coeff_p_dynamic_stage_2_threshold': 0.05,
        'pd_coeff_p_dynamic_stage_1_ratio': 1,
        'pd_coeff_p_dynamic_stage_2_ratio': 1,
        'pd_coeff_p_dynamic_stage_3_ratio': 1,

        'frame_angular_size_deg': XY(107, 85),

        # 'target_size_m': XY(0.2, 0.2),           # balloon
        'target_size_m': XY(1.8, 1.8),             # shahed small
        # 'target_size_m': XY(3.5, 2.5),           # shahed large

        'inertia_correction_gain': 0,
        'inertia_correction_limits': XY(1, 1),
        'inertia_correction_min_speed_ms': 5,

        'safe_takeoff_period_ns': 300_000_000,
        'delay_takeof_until_n_detection_frames': 20,

        'aim_point': XY(0.5, 0.5),
        'aim_point_max_offset': XY(0.5, 0.6),

        'DEBUG': DEBUG,
    }

    logger.info("Config: %s", control_config)

    drone_thread = threading.Thread(
        target=drone_controlling_thread,
        args=(
            'udp://:14540',
            {
                'upside_down_angle_deg': 130,
                'upside_down_hold_s': 0.2,
            },
            detections_queue,
        ),
        kwargs=dict(
            control_config=control_config,
            output_queue=output_queue,
            signal_event_when_ready=ready_event,
        ),
        name="Drone",
    )
    drone_thread.start()

    sink = MultiSink([
        RecorderSink(
            30,
            "./_DEBUG",
            segment_seconds=10,
            filename_base=f"debug_{start_time_str}",
        ),
    ])

    output_thread = threading.Thread(
        target=debug_output_thread,
        args=(output_queue, sink),
        name="DEBUG",
    )
    output_thread.start()

    cap_thread = threading.Thread(
        target=capture_thread_fn,
        args=(detections_queue, stop_event, ready_event),
        kwargs={"show": show},
        name="Capture",
    )
    cap_thread.start()

    try:
        while drone_thread.is_alive():
            drone_thread.join(timeout=1.0)
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down…")
    finally:
        stop_event.set()
        detections_queue.put(STOP)

    cap_thread.join()
    drone_thread.join()
    logger.info("Done")


if __name__ == "__main__":
    main()
