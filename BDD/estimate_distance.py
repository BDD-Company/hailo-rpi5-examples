#/usr/bin/env python

import math
from typing import Optional, Tuple
from enum import Enum

import cv2
import numpy as np

from helpers import XY, Rect


def _extract_largest_object_contour(frame: np.ndarray, bbox: Rect) -> tuple[np.ndarray, int, int, int, int] | None:
    """Extract largest segmented contour inside bbox using inverted Otsu threshold."""
    if frame is None:
        return None

    fh, fw = frame.shape[:2]
    if fw == 0 or fh == 0:
        return None

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
    area = cv2.contourArea(largest)
    if area < 0.05 * bbox_px_area or area > 0.90 * bbox_px_area:
        return None

    return largest, x1, y1, fw, fh

def measure_object_size_from_contour(contour) -> 'XY | None':
    if contour is None:
        return None

    largest, _x1, _y1, fw, fh = contour
    _, _, w, h = cv2.boundingRect(largest)
    return XY(w / fw, h / fh)


def measure_object_size(frame: np.ndarray, bbox: Rect) -> 'XY | None':
    """Return the normalized (0-1) size of the object inside bbox.

    Uses inverted Otsu threshold on grayscale — reliable for dark objects
    on bright backgrounds (drone against sky). Returns None on failure;
    caller should fall back to bbox.size.
    """
    return measure_object_size_from_contour(_extract_largest_object_contour(frame, bbox))


def measure_object_circle_center_from_contour(contour) -> 'XY | None':
    if contour is None:
        return None

    largest, x1, y1, fw, fh = contour
    x, y, w, h = cv2.boundingRect(largest)
    if w < 2 or h < 2:
        return None

    # Build filled-object mask in contour-local ROI and find the farthest
    # interior point from boundary; this is the inscribed-circle center.
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = largest - np.array([[[x, y]]], dtype=largest.dtype)
    cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(dist)
    if max_val <= 0:
        return None

    cx = x + float(max_loc[0])
    cy = y + float(max_loc[1])
    return XY((x1 + cx) / fw, (y1 + cy) / fh)


def measure_object_circle_center(frame: np.ndarray, bbox: Rect, contour = None) -> 'XY | None':
    """Return normalized center of the inscribed circle for segmented object in bbox."""
    return measure_object_circle_center_from_contour(_extract_largest_object_contour(frame, bbox))


class OpticalObjectInfo:
    """
    Detailed object info based on pixes from the frame
    """

    def __init__(self, frame: np.ndarray, bbox: Rect):
        self._countour = _extract_largest_object_contour(frame, bbox)

    def object_circle_center(self):
        return measure_object_circle_center_from_contour(self._countour)

    def object_size(self):
        return measure_object_size_from_contour(self._countour)



def estimate_distance(
    target_size_m: XY,
    frame_angular_size_deg: XY,
    target_frame_size : XY
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

        half_fov_rad = math.radians(fov_deg / 2.0)
        if half_fov_rad <= 0:
            return None

        # Pinhole (rectilinear) projection: pixel position ∝ tan(angle)
        return size_m / (2.0 * frac * math.tan(half_fov_rad))

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

    if d < max_size * 2:
        return DistanceClass.NEAR, d
    if d < max_size * 10:
        return DistanceClass.MEDIUM, d
    else:
        return DistanceClass.FAR, d
