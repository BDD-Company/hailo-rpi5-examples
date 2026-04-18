#!/usr/bin/env python3
"""Minimal ByteTrack — pure numpy, no external dependencies."""

from __future__ import annotations
from enum import IntEnum  # used by TrackState added in Task 2
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
        # frame_rate accepted for API compatibility with BYTETracker;
        # noise is scaled by box height, not frame rate.
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
        K = np.linalg.solve(S.T, (cov @ self._H.T).T).T
        innovation = np.array(bbox_cxcywh, dtype=float) - self._H @ mean
        return mean + K @ innovation, (np.eye(8) - K @ self._H) @ cov


# -----------------------------------------------------------------------
# Track state machine
# -----------------------------------------------------------------------

class TrackState(IntEnum):
    New     = 0
    Tracked = 1
    Lost    = 2
    Removed = 3


class STrack:
    """Single tracked object with Kalman filter state."""

    # Shared across all BYTETracker instances; IDs are globally unique within a process.
    # Intentional for single-tracker use; use reset_counter() in tests.
    _id_counter: int = 0

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    @classmethod
    def reset_counter(cls) -> None:
        """Reset ID counter — use in tests only."""
        cls._id_counter = 0

    def __init__(self, det_bbox: np.ndarray, score: float, kf: KalmanFilter):
        """
        Args:
            det_bbox: [x1, y1, x2, y2] in normalised 0-1 coords.
            score: detection confidence.
            kf: shared KalmanFilter instance.
        """
        self._kf = kf
        self._det_bbox = np.array(det_bbox, dtype=float)
        self.score = float(score)
        self.state = TrackState.New
        self.track_id: int | None = None
        self.frame_id: int = 0
        self.start_frame: int = 0
        self.tracklet_len: int = 0
        self.mean: np.ndarray | None = None
        self.cov:  np.ndarray | None = None

    # -- coordinate helpers ------------------------------------------------

    @staticmethod
    def _xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
        return np.array([(b[0]+b[2])/2, (b[1]+b[3])/2, b[2]-b[0], b[3]-b[1]])

    @staticmethod
    def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
        return np.array([b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2])

    @property
    def bbox(self) -> np.ndarray:
        """[x1,y1,x2,y2] Kalman estimate when activated, raw detection otherwise."""
        if self.mean is not None:
            return self._cxcywh_to_xyxy(self.mean[:4])
        return self._det_bbox.copy()

    @property
    def det_bbox(self) -> np.ndarray:
        """Last raw detection [x1,y1,x2,y2] — used for back-mapping in app.py."""
        return self._det_bbox.copy()

    # -- lifecycle ---------------------------------------------------------

    def activate(self, frame_id: int) -> None:
        self.track_id = self._next_id()
        self.mean, self.cov = self._kf.initiate(self._xyxy_to_cxcywh(self._det_bbox))
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 1

    def predict(self) -> None:
        if self.mean is None:
            return
        mean = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean[6] = 0.0  # zero size velocities when lost
            mean[7] = 0.0
        self.mean, self.cov = self._kf.predict(mean, self.cov)

    def update(self, det_bbox: np.ndarray, score: float, frame_id: int) -> None:
        if self.mean is None:
            raise RuntimeError("STrack.update() called before activate()")
        self._det_bbox = np.array(det_bbox, dtype=float)
        self.score = float(score)
        self.mean, self.cov = self._kf.update(
            self.mean, self.cov, self._xyxy_to_cxcywh(self._det_bbox)
        )
        self.state = TrackState.Tracked
        self.tracklet_len += 1
        self.frame_id = frame_id

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed


# -----------------------------------------------------------------------
# IoU helpers
# -----------------------------------------------------------------------

def _iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. Inputs: (N,4) and (M,4) xyxy. Returns (N,M)."""
    N, M = len(bboxes_a), len(bboxes_b)
    if N == 0 or M == 0:
        return np.zeros((N, M))
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0])
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1])
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2])
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    iou = np.zeros_like(inter)
    np.divide(inter, union, out=iou, where=union > 0)
    return iou


def _greedy_match(
    iou_matrix: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy IoU matching, highest IoU first.

    Uses greedy (not optimal Hungarian) assignment. Adequate for single-target
    use; with many similar-IoU detections the optimal assignment may differ.

    Returns (matched_pairs, unmatched_row_indices, unmatched_col_indices).
    """
    matched: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    for flat_idx in np.argsort(-iou_matrix, axis=None):
        r, c = divmod(int(flat_idx), iou_matrix.shape[1])
        if r in used_r or c in used_c:
            continue
        if iou_matrix[r, c] < thresh:
            break
        matched.append((r, c))
        used_r.add(r)
        used_c.add(c)
    unm_r = [r for r in range(iou_matrix.shape[0]) if r not in used_r]
    unm_c = [c for c in range(iou_matrix.shape[1]) if c not in used_c]
    return matched, unm_r, unm_c


def _associate(
    stracks: list[STrack], dets: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match stracks to dets (N,5) by IoU."""
    if not stracks or len(dets) == 0:
        return [], list(range(len(stracks))), list(range(len(dets)))
    track_bboxes = np.array([t.bbox for t in stracks])
    return _greedy_match(_iou_batch(track_bboxes, dets[:, :4]), thresh)


# -----------------------------------------------------------------------
# Tracker
# -----------------------------------------------------------------------

class BYTETracker:
    """ByteTrack with two-stage IoU association and Kalman prediction.

    Stage 1: high-confidence detections matched against all active + lost tracks.
    Stage 2: low-confidence detections matched against unmatched active tracks.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        det_thresh:   float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int   = 30,
        frame_rate:   float = 30.0,
    ):
        self.track_thresh = track_thresh
        self.det_thresh   = det_thresh
        self.match_thresh = match_thresh
        self.max_lost_age = int(frame_rate / 30.0 * track_buffer)
        self._kf = KalmanFilter(frame_rate=frame_rate)
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks:    list[STrack] = []

    def update(self, dets: np.ndarray, frame_id: int) -> list[STrack]:
        """
        Args:
            dets: (N, 5) array [x1, y1, x2, y2, score] in normalised 0-1 coords.
            frame_id: monotonically increasing frame counter (NOT a timestamp).
                      Must increment by ~1 per frame; track_buffer expiry is
                      based on frame_id differences, not wall time.
        Returns:
            Active STrack list (state == Tracked).
        """
        if len(dets) > 0:
            mask      = dets[:, 4] >= self.track_thresh
            high_dets = dets[mask]
            low_dets  = dets[~mask]
        else:
            high_dets = np.empty((0, 5))
            low_dets  = np.empty((0, 5))

        n_tracked   = len(self.tracked_stracks)
        strack_pool = self.tracked_stracks + self.lost_stracks

        for t in strack_pool:
            t.predict()

        # Stage 1: high_dets vs full pool (tracked + lost)
        matches1, unm_pool1, unm_high = _associate(strack_pool, high_dets, self.match_thresh)
        for ti, di in matches1:
            strack_pool[ti].update(high_dets[di, :4], high_dets[di, 4], frame_id)

        unm_tracked = [strack_pool[ti] for ti in unm_pool1 if ti < n_tracked]
        unm_lost    = [strack_pool[ti] for ti in unm_pool1 if ti >= n_tracked]

        # Stage 2: low_dets vs unmatched tracked stracks
        matches2, unm_r2, _ = _associate(unm_tracked, low_dets, 0.5)
        for ti, di in matches2:
            unm_tracked[ti].update(low_dets[di, :4], low_dets[di, 4], frame_id)
        newly_lost = [unm_tracked[i] for i in unm_r2]
        for t in newly_lost:
            t.mark_lost()

        # New tracks from unmatched high detections
        new_tracks: list[STrack] = []
        for di in unm_high:
            t = STrack(high_dets[di, :4], high_dets[di, 4], self._kf)
            if t.score >= self.det_thresh:
                t.activate(frame_id)
                new_tracks.append(t)

        # Prune timed-out lost tracks
        all_lost = newly_lost + unm_lost
        surviving_lost: list[STrack] = []
        for t in all_lost:
            if frame_id - t.frame_id <= self.max_lost_age:
                surviving_lost.append(t)
            else:
                t.mark_removed()

        self.tracked_stracks = (
            [t for t in strack_pool if t.state == TrackState.Tracked] + new_tracks
        )
        self.lost_stracks = surviving_lost

        return list(self.tracked_stracks)
