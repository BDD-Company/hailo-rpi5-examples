#!/usr/bin/env python3
"""Minimal ByteTrack — pure numpy, no external dependencies."""

from __future__ import annotations
from enum import IntEnum
import numpy as np

# -----------------------------------------------------------------------
# Kalman Filter
# -----------------------------------------------------------------------

_STD_WEIGHT_POS = 1.0 / 20
_STD_WEIGHT_VEL = 1.0 / 160


class KalmanFilter:
    """Constant-velocity Kalman filter.

    State:       [cx, cy, w, h, vcx, vcy, vw, vh]
    Observation: [cx, cy, w, h]
    """

    def __init__(self, frame_rate: float = 30.0):
        # State transition: pos += vel (dt = 1 frame)
        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        # Observation: extract first 4 components
        self._H = np.eye(4, 8)

    def initiate(self, bbox_cxcywh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Initialise state from [cx, cy, w, h] bounding box."""
        h = max(float(bbox_cxcywh[3]), 1e-6)
        std_p = _STD_WEIGHT_POS * h
        std_v = _STD_WEIGHT_VEL * h
        mean = np.concatenate([np.array(bbox_cxcywh, dtype=float), np.zeros(4)])
        cov = np.diag(np.array([
            2 * std_p, 2 * std_p,       std_p, 2 * std_p,
            10 * std_v, 10 * std_v, 0.1 * std_v, 10 * std_v,
        ]) ** 2)
        return mean, cov

    def predict(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1e-6)
        std_p = _STD_WEIGHT_POS * h
        std_v = _STD_WEIGHT_VEL * h
        Q = np.diag(np.array([
            std_p, std_p, std_p, std_p,
            std_v, std_v, std_v, std_v,
        ]) ** 2)
        return self._F @ mean, self._F @ cov @ self._F.T + Q

    def update(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        bbox_cxcywh: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = max(float(mean[3]), 1e-6)
        std = _STD_WEIGHT_POS * h
        R = np.diag(np.array([std, std, std / 10, std]) ** 2)
        S = self._H @ cov @ self._H.T + R
        K = cov @ self._H.T @ np.linalg.inv(S)
        innovation = np.array(bbox_cxcywh, dtype=float) - self._H @ mean
        return mean + K @ innovation, (np.eye(8) - K @ self._H) @ cov
