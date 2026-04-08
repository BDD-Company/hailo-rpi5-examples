#/usr/bin/env python

import math
from typing import Optional, Tuple
from enum import Enum

from helpers import XY

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
    target_frame_size : XY,
    distance_classes : list[float] = None
) -> tuple[DistanceClass, float] | None:
    distance_classes = distance_classes if distance_classes and len(distance_classes) == 2 else [5, 10]

    dx, dy, d = estimate_distance(target_size_m, frame_angular_size_deg, target_frame_size)
    if d is None:
        return None

    if d < distance_classes[0]:
        return DistanceClass.NEAR, d
    if d < distance_classes[1]:
        return DistanceClass.MEDIUM, d
    else:
        return DistanceClass.FAR, d
