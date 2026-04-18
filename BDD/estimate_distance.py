#/usr/bin/env python

import math
from typing import Optional, Tuple
from enum import Enum

import cv2
import numpy as np

from helpers import XY


def measure_object_size(frame: np.ndarray, bbox: 'Rect') -> 'XY | None':
    """Return the normalized (0-1) size of the object inside bbox.

    Uses inverted Otsu threshold on grayscale — reliable for dark objects
    on bright backgrounds (drone against sky). Returns None on failure;
    caller should fall back to bbox.size.
    """
    if frame is None:
        return None

    fh, fw = frame.shape[:2]

    x1 = int(np.clip(bbox.p1.x * fw, 0, fw - 1))
    y1 = int(np.clip(bbox.p1.y * fh, 0, fh - 1))
    x2 = int(np.clip(bbox.p2.x * fw, 0, fw))
    y2 = int(np.clip(bbox.p2.y * fh, 0, fh))

    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None

    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    bbox_px_area = (x2 - x1) * (y2 - y1)
    if cv2.contourArea(largest) < 0.05 * bbox_px_area:
        return None

    _, _, w, h = cv2.boundingRect(largest)
    return XY(w / fw, h / fh)


def estimate_distance(
    target_size_m: XY,
    frame_angular_size_deg: XY,
    target_frame_size : float
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate range to a target from known true size and image occupancy.

    Returns:
        (distance_from_x, distance_from_y, combined_distance)

    combined_distance:
        - average of valid X/Y estimates if both are available
        - otherwise whichever one is available
        - None if neither is available
    """
    def one_axis(size_m: float, fov_deg: float, frac: float) -> Optional[float]:
        if size_m <= 0 or fov_deg <= 0 or not (0.0 < frac <= 1.0):
            return None

        angular_size_rad = math.radians(fov_deg * frac)

        # Prevent tan(0) or near-zero instability
        if angular_size_rad <= 0:
            return None

        return size_m / (2.0 * math.tan(angular_size_rad / 2.0))

    dx = one_axis(target_size_m.x, frame_angular_size_deg.x, target_frame_size.x)
    dy = one_axis(target_size_m.y, frame_angular_size_deg.y, target_frame_size.y)

    if dx is not None and dy is not None:
        d = 0.5 * (dx + dy)
    else:
        d = dx if dx is not None else dy

    return dx, dy, d

class DistanceClass(Enum):
    NEAR = 1
    MEDIUM = 2
    FAR = 3

def estimate_distance_class(
    target_size_m: XY,
    frame_angular_size_deg: XY,
    target_frame_size : XY
    # , distance_classes : list[float] = None
) -> tuple[DistanceClass, float] | tuple[None, None]:
    # distance_classes = distance_classes if distance_classes and len(distance_classes) == 2 else [5, 10]
    max_size = max(target_size_m.x, target_size_m.y)
    dx, dy, d = estimate_distance(target_size_m, frame_angular_size_deg, target_frame_size)
    if d is None:
        return None, None

    if d < max_size * 5:
        return DistanceClass.NEAR, d
    if d < max_size * 20:
        return DistanceClass.MEDIUM, d
    else:
        return DistanceClass.FAR, d
