"""Stereo camera calibration utilities.

Provides:
- FramePairer: timestamp-matches raw numpy frames from two cameras.
- calibrate_stereo / save_calibration / load_calibration: OpenCV-backed math.
- run_calibration_mode / run_verify_mode: interactive entry points used by app.py.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import Optional

import cv2
import numpy as np

from stereo_distance import StereoCalibration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FramePairer — matches raw numpy frames from two cameras by timestamp
# ---------------------------------------------------------------------------

class FramePairer:
    """Buffers (frame, timestamp_ns) from 'left' and 'right'.

    `put()` returns the matched (left_frame, right_frame) pair on success, else None.
    Matched pairs are also pushed to an internal queue so that consumers running
    on a different thread can call `wait_for_pair(timeout_s)` to block on them.
    """

    def __init__(self, max_gap_ns: int = 33_000_000, buffer_maxlen: int = 3):
        self._max_gap_ns = max_gap_ns
        self._buffers: dict[str, deque] = {
            'left':  deque(maxlen=buffer_maxlen),
            'right': deque(maxlen=buffer_maxlen),
        }
        self._lock = threading.Lock()
        self._ready: queue.Queue = queue.Queue()

    def put(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp_ns: int,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        assert camera_id in ('left', 'right')
        other = 'right' if camera_id == 'left' else 'left'

        with self._lock:
            other_buf = self._buffers[other]
            best_idx, best_diff = None, self._max_gap_ns

            for i, (_, other_ts) in enumerate(other_buf):
                diff = abs(timestamp_ns - other_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is None:
                self._buffers[camera_id].append((frame, timestamp_ns))
                return None

            other_frame, _ = other_buf[best_idx]
            remaining = [item for j, item in enumerate(other_buf) if j != best_idx]
            self._buffers[other] = deque(remaining, maxlen=other_buf.maxlen)

            pair = (frame, other_frame) if camera_id == 'left' else (other_frame, frame)

        self._ready.put(pair)
        return pair

    def wait_for_pair(self, timeout_s: float = 1.0) -> Optional[tuple[np.ndarray, np.ndarray]]:
        try:
            return self._ready.get(timeout=timeout_s)
        except queue.Empty:
            return None


# ---------------------------------------------------------------------------
# OpenCV calibration math
# ---------------------------------------------------------------------------

def calibrate_stereo(
    obj_pts: list[np.ndarray],
    img_pts_left: list[np.ndarray],
    img_pts_right: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict:
    """Run cv2.stereoCalibrate + stereoRectify + initUndistortRectifyMap.

    Returns dict with: rms, map_left_x, map_left_y, map_right_x, map_right_y,
    focal_px, K1, K2.
    """
    flags = cv2.CALIB_RATIONAL_MODEL

    rms, K1, D1, K2, D2, R, T, _E, _F = cv2.stereoCalibrate(
        obj_pts, img_pts_left, img_pts_right,
        None, None, None, None,
        image_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    )

    R1, R2, P1, P2, _Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    return {
        'rms':         rms,
        'map_left_x':  map_l_x,
        'map_left_y':  map_l_y,
        'map_right_x': map_r_x,
        'map_right_y': map_r_y,
        'focal_px':    float(P1[0, 0]),
        'K1': K1, 'K2': K2,
    }


def save_calibration(
    path: str,
    map_left_x: np.ndarray,
    map_left_y: np.ndarray,
    map_right_x: np.ndarray,
    map_right_y: np.ndarray,
    focal_px: float,
    baseline_m: float,
    image_size: tuple[int, int],
    rms: float,
) -> None:
    np.savez(
        path,
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        focal_px=np.float64(focal_px),
        baseline_m=np.float64(baseline_m),
        image_size=np.array(image_size, dtype=np.int32),
        rms=np.float64(rms),
    )


def load_calibration(path: str) -> StereoCalibration:
    return StereoCalibration.load(path)


# ---------------------------------------------------------------------------
# Interactive calibration / verification entry points
# ---------------------------------------------------------------------------

def run_calibration_mode(
    pairer: FramePairer,
    baseline_m: float,
    output_path: str,
    board_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
    min_pairs: int = 20,
    poll_timeout_s: float = 1.0,
) -> None:
    """Capture chessboard frame pairs, run stereoCalibrate, save to output_path."""
    obj_pts:       list[np.ndarray] = []
    img_pts_left:  list[np.ndarray] = []
    img_pts_right: list[np.ndarray] = []

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size_m
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_size: Optional[tuple[int, int]] = None

    logger.info("Calibration: show %dx%d chessboard. Need %d pairs.", *board_size, min_pairs)

    while len(obj_pts) < min_pairs:
        pair = pairer.wait_for_pair(timeout_s=poll_timeout_s)
        if pair is None:
            continue
        left_frame, right_frame = pair
        if image_size is None:
            h, w = left_frame.shape[:2]
            image_size = (w, h)

        gray_l = cv2.cvtColor(left_frame,  cv2.COLOR_RGB2GRAY)
        gray_r = cv2.cvtColor(right_frame, cv2.COLOR_RGB2GRAY)
        found_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, None)

        if found_l and found_r:
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts_left.append(corners_l)
            img_pts_right.append(corners_r)
            logger.info("Captured pair %d/%d", len(obj_pts), min_pairs)
        else:
            logger.debug("Chessboard not found (l=%s r=%s)", found_l, found_r)

    logger.info("Running stereoCalibrate...")
    result = calibrate_stereo(obj_pts, img_pts_left, img_pts_right, image_size)
    logger.info("Calibration RMS: %.4f", result['rms'])

    save_calibration(
        path=output_path,
        map_left_x=result['map_left_x'],
        map_left_y=result['map_left_y'],
        map_right_x=result['map_right_x'],
        map_right_y=result['map_right_y'],
        focal_px=result['focal_px'],
        baseline_m=baseline_m,
        image_size=image_size,
        rms=result['rms'],
    )
    logger.info("Calibration saved to %s", output_path)


def run_verify_mode(
    calib: StereoCalibration,
    left_frame: np.ndarray,
    right_frame: np.ndarray,
) -> None:
    """Display rectified frames side-by-side with horizontal epipolar lines."""
    h, w = left_frame.shape[:2]
    rect_l = cv2.remap(left_frame,  calib.map_left_x,  calib.map_left_y,  cv2.INTER_LINEAR)
    rect_r = cv2.remap(right_frame, calib.map_right_x, calib.map_right_y, cv2.INTER_LINEAR)

    combined = np.hstack([rect_l, rect_r])
    for y in range(0, h, max(1, h // 10)):
        cv2.line(combined, (0, y), (w * 2, y), (0, 255, 0), 1)

    cv2.imshow("Stereo rectification verify (press any key to close)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
