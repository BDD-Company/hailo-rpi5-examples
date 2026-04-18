from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np

from helpers import Detection, Rect


@dataclass
class StereoCalibration:
    map_left_x:  np.ndarray
    map_left_y:  np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    focal_px:    float
    baseline_m:  float
    image_size:  tuple[int, int]  # (width, height)

    def save(self, path: str) -> None:
        np.savez(
            path,
            map_left_x=self.map_left_x,
            map_left_y=self.map_left_y,
            map_right_x=self.map_right_x,
            map_right_y=self.map_right_y,
            focal_px=np.float64(self.focal_px),
            baseline_m=np.float64(self.baseline_m),
            image_size=np.array(self.image_size, dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str) -> 'StereoCalibration':
        data = np.load(path)
        return cls(
            map_left_x=data['map_left_x'],
            map_left_y=data['map_left_y'],
            map_right_x=data['map_right_x'],
            map_right_y=data['map_right_y'],
            focal_px=float(data['focal_px']),
            baseline_m=float(data['baseline_m']),
            image_size=tuple(int(x) for x in data['image_size']),
        )


def _compute_iou(a: Rect, b: Rect) -> float:
    ix1 = max(a.left_edge,  b.left_edge)
    iy1 = max(a.top_edge,   b.top_edge)
    ix2 = min(a.right_edge, b.right_edge)
    iy2 = min(a.bottom_edge, b.bottom_edge)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def match_stereo_detections(
    left_dets: list[Detection],
    right_dets: list[Detection],
    iou_threshold: float = 0.1,
) -> list[tuple[Detection, Detection]]:
    """Return matched (left, right) Detection pairs by IoU. Each detection used at most once."""
    pairs: list[tuple[Detection, Detection]] = []
    used_right: set[int] = set()
    for ld in left_dets:
        best_idx, best_iou = None, iou_threshold
        for i, rd in enumerate(right_dets):
            if i in used_right:
                continue
            iou = _compute_iou(ld.bbox, rd.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx is not None:
            pairs.append((ld, right_dets[best_idx]))
            used_right.add(best_idx)
    return pairs


def stereo_distance(
    left_det: Detection,
    right_det: Detection,
    calib: StereoCalibration,
    frame_width: int,
    frame_height: int,
) -> Optional[float]:
    """
    Return distance in metres from stereo disparity, or None if disparity is invalid.

    Looks up the rectified x-coordinate for each detection's centre pixel using
    the preloaded remap arrays (one point lookup — no full-frame remap at runtime).
    """
    w, h = calib.image_size
    assert frame_width == w and frame_height == h, (
        f"Frame size {frame_width}x{frame_height} != calibration size {w}x{h}"
    )
    cx_l = int(np.clip(left_det.bbox.center.x  * frame_width,  0, w - 1))
    cy_l = int(np.clip(left_det.bbox.center.y  * frame_height, 0, h - 1))
    cx_r = int(np.clip(right_det.bbox.center.x * frame_width,  0, w - 1))
    cy_r = int(np.clip(right_det.bbox.center.y * frame_height, 0, h - 1))

    rect_x_l = float(calib.map_left_x[cy_l, cx_l])
    rect_x_r = float(calib.map_right_x[cy_r, cx_r])

    disparity = rect_x_l - rect_x_r
    if disparity <= 0:
        return None

    return calib.baseline_m * calib.focal_px / disparity
