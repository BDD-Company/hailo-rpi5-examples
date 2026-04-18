# ByteTrack Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal pure-numpy ByteTracker to `BDD/app.py` that replaces the currently-disabled Hailo GStreamer tracker and assigns stable `track_id` values to each `Detection`.

**Architecture:** Three-class implementation in a new `BDD/bytetrack.py`: `KalmanFilter` (8-state constant-velocity), `STrack` (single track with state machine), `BYTETracker` (two-stage IoU association). `BYTETracker` is instantiated in `main()`, stored on `user_app_callback_class`, and called inside `app_callback` after Hailo detections are parsed.

**Tech Stack:** Python 3.10+, numpy (already a project dependency). No additional packages required.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `BDD/bytetrack.py` | Full ByteTrack — KalmanFilter, STrack, BYTETracker, IoU helpers |
| Create | `BDD/test_bytetrack.py` | pytest tests for all three classes |
| Modify | `BDD/app.py` | Wire tracker into `user_app_callback_class`, `app_callback`, `main()` |

---

## Task 1: KalmanFilter

**Files:**
- Create: `BDD/bytetrack.py`
- Create: `BDD/test_bytetrack.py`

- [ ] **Step 1.1: Create `test_bytetrack.py` with failing KalmanFilter tests**

```python
#!/usr/bin/env python3
"""Tests for bytetrack.py"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pytest
from bytetrack import KalmanFilter


def test_initiate_shape():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)


def test_initiate_mean_values():
    kf = KalmanFilter()
    bbox = np.array([0.3, 0.4, 0.08, 0.12])
    mean, _ = kf.initiate(bbox)
    assert np.allclose(mean[:4], bbox)
    assert np.allclose(mean[4:], 0.0)


def test_predict_advances_position_with_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    mean[4] = 0.05  # inject vcx
    mean_pred, _ = kf.predict(mean, cov)
    assert mean_pred[0] > 0.5        # cx moved right
    assert np.isclose(mean_pred[0], 0.55, atol=1e-9)


def test_predict_covariance_grows():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    _, cov2 = kf.predict(mean, cov)
    assert np.trace(cov2) > np.trace(cov)


def test_update_moves_mean_toward_measurement():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    measurement = np.array([0.6, 0.5, 0.1, 0.1])
    mean_upd, _ = kf.update(mean, cov, measurement)
    assert 0.5 < mean_upd[0] < 0.6   # pulled toward measurement


def test_update_reduces_uncertainty():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([0.5, 0.5, 0.1, 0.1]))
    _, cov_pred = kf.predict(mean, cov)
    _, cov_upd = kf.update(mean, cov_pred, np.array([0.5, 0.5, 0.1, 0.1]))
    assert np.trace(cov_upd) < np.trace(cov_pred)


if __name__ == "__main__":
    sys.exit(pytest.main(["-qq", __file__]))
```

- [ ] **Step 1.2: Run tests — verify they fail with ImportError**

```bash
cd /path/to/BDD_drone/BDD && python3 -m pytest test_bytetrack.py -qq 2>&1 | head -5
```
Expected: `ModuleNotFoundError: No module named 'bytetrack'`

- [ ] **Step 1.3: Create `bytetrack.py` with `KalmanFilter`**

```python
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
```

- [ ] **Step 1.4: Run tests — all should pass**

```bash
cd /path/to/BDD_drone/BDD && python3 -m pytest test_bytetrack.py -qq
```
Expected: `6 passed`

- [ ] **Step 1.5: Commit**

```bash
git add BDD/bytetrack.py BDD/test_bytetrack.py
git commit -m "feat: add KalmanFilter to bytetrack.py with tests"
```

---

## Task 2: STrack and IoU helpers

**Files:**
- Modify: `BDD/bytetrack.py` (append)
- Modify: `BDD/test_bytetrack.py` (append)

- [ ] **Step 2.1: Append STrack tests to `test_bytetrack.py`**

Add after the last existing test:

```python
from bytetrack import STrack, TrackState


def _make_strack(x1=0.1, y1=0.1, x2=0.3, y2=0.3, score=0.8):
    kf = KalmanFilter()
    STrack.reset_counter()
    return STrack(np.array([x1, y1, x2, y2]), score, kf)


def test_strack_initial_state():
    t = _make_strack()
    assert t.state == TrackState.New
    assert t.track_id is None


def test_strack_activate_assigns_id_and_tracked():
    t = _make_strack()
    t.activate(frame_id=0)
    assert t.state == TrackState.Tracked
    assert t.track_id == 1


def test_strack_ids_increment():
    STrack.reset_counter()
    kf = KalmanFilter()
    t1 = STrack(np.array([0.1, 0.1, 0.3, 0.3]), 0.9, kf)
    t2 = STrack(np.array([0.5, 0.5, 0.7, 0.7]), 0.9, kf)
    t1.activate(0)
    t2.activate(0)
    assert t2.track_id == t1.track_id + 1


def test_strack_bbox_before_activate_returns_raw():
    t = _make_strack(0.1, 0.1, 0.3, 0.3)
    assert np.allclose(t.bbox, [0.1, 0.1, 0.3, 0.3])


def test_strack_bbox_after_activate_is_kalman():
    t = _make_strack(0.1, 0.1, 0.3, 0.3)
    t.activate(0)
    # Kalman bbox should be very close to initial detection
    assert np.allclose(t.bbox, [0.1, 0.1, 0.3, 0.3], atol=1e-6)


def test_strack_predict_changes_bbox():
    t = _make_strack()
    t.activate(0)
    bbox_before = t.bbox.copy()
    t.mean[4] = 0.05  # inject vcx
    t.predict()
    assert t.bbox[0] > bbox_before[0]


def test_strack_update_sets_det_bbox_and_tracked():
    t = _make_strack()
    t.activate(0)
    new_bbox = np.array([0.2, 0.2, 0.4, 0.4])
    t.update(new_bbox, 0.9, frame_id=1)
    assert t.state == TrackState.Tracked
    assert np.allclose(t.det_bbox, new_bbox)
    assert t.frame_id == 1


def test_strack_mark_lost_and_removed():
    t = _make_strack()
    t.activate(0)
    t.mark_lost()
    assert t.state == TrackState.Lost
    t.mark_removed()
    assert t.state == TrackState.Removed
```

- [ ] **Step 2.2: Run — verify new tests fail**

```bash
cd /path/to/BDD_drone/BDD && python3 -m pytest test_bytetrack.py -qq 2>&1 | tail -5
```
Expected: `ImportError` or `cannot import name 'STrack'`

- [ ] **Step 2.3: Append `TrackState`, `STrack`, and IoU helpers to `bytetrack.py`**

Add after the `KalmanFilter` class:

```python
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
    return np.where(union > 0, inter / union, 0.0)


def _greedy_match(
    iou_matrix: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy IoU matching, highest IoU first.

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
```

- [ ] **Step 2.4: Run — all tests pass**

```bash
cd /path/to/BDD_drone/BDD && python3 -m pytest test_bytetrack.py -qq
```
Expected: `15 passed`

- [ ] **Step 2.5: Commit**

```bash
git add BDD/bytetrack.py BDD/test_bytetrack.py
git commit -m "feat: add STrack, TrackState, and IoU helpers to bytetrack.py"
```

---

## Task 3: BYTETracker

**Files:**
- Modify: `BDD/bytetrack.py` (append)
- Modify: `BDD/test_bytetrack.py` (append)

- [ ] **Step 3.1: Append BYTETracker tests to `test_bytetrack.py`**

Add after the last STrack test:

```python
from bytetrack import BYTETracker


def _det(x1, y1, x2, y2, score):
    return [x1, y1, x2, y2, score]


def _tracker():
    STrack.reset_counter()
    return BYTETracker(track_thresh=0.5, det_thresh=0.6, match_thresh=0.8,
                       track_buffer=5, frame_rate=30)


def test_tracker_no_dets_returns_empty():
    t = _tracker()
    result = t.update(np.empty((0, 5)), frame_id=0)
    assert result == []


def test_tracker_single_detection_creates_track():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    tracks = t.update(dets, frame_id=0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].state == TrackState.Tracked


def test_tracker_same_detection_keeps_track_id():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    tracks = t.update(dets, frame_id=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_tracker_lost_track_remembered():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    # Frame with no detections
    tracks = t.update(np.empty((0, 5)), frame_id=1)
    assert tracks == []
    assert len(t.lost_stracks) == 1


def test_tracker_lost_track_recovered():
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    t.update(np.empty((0, 5)), frame_id=1)  # goes lost
    # Reappears
    tracks = t.update(dets, frame_id=2)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1          # same ID recovered


def test_tracker_lost_track_removed_after_buffer():
    # track_buffer=5 → max_lost_age = 5 frames
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(dets, frame_id=0)
    # 6 empty frames
    for fid in range(1, 7):
        t.update(np.empty((0, 5)), frame_id=fid)
    assert len(t.lost_stracks) == 0


def test_tracker_low_confidence_below_det_thresh_not_tracked():
    # score=0.4 < det_thresh=0.6 → should not create a new track
    t = _tracker()
    dets = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.4)])
    tracks = t.update(dets, frame_id=0)
    assert tracks == []


def test_tracker_low_conf_rescues_lost_track():
    t = _tracker()
    # Establish a track
    high_det = np.array([_det(0.1, 0.1, 0.3, 0.3, 0.8)])
    t.update(high_det, frame_id=0)
    # Next frame: only low-confidence detection at same position
    low_det = np.array([_det(0.11, 0.11, 0.31, 0.31, 0.35)])
    tracks = t.update(low_det, frame_id=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1          # rescued via Stage 2


def test_tracker_two_detections_two_tracks():
    t = _tracker()
    dets = np.array([
        _det(0.1, 0.1, 0.3, 0.3, 0.9),
        _det(0.6, 0.6, 0.8, 0.8, 0.9),
    ])
    tracks = t.update(dets, frame_id=0)
    assert len(tracks) == 2
    ids = {tr.track_id for tr in tracks}
    assert len(ids) == 2                    # distinct IDs
```

- [ ] **Step 3.2: Run — verify new tests fail**

```bash
cd /path/to/BDD_drone/BDD && python3 -m pytest test_bytetrack.py -qq 2>&1 | tail -5
```
Expected: `cannot import name 'BYTETracker'`

- [ ] **Step 3.3: Append `BYTETracker` to `bytetrack.py`**

Add after the `_associate` function:

```python
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
            frame_id: monotonically increasing frame counter.
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

        matched2_set = {ti for ti, _ in matches2}
        newly_lost = [unm_tracked[i] for i in range(len(unm_tracked))
                      if i not in matched2_set]
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
```

- [ ] **Step 3.4: Run — all tests pass**

```bash
cd /path/to/BDD_drone/BDD && python3 -m pytest test_bytetrack.py -qq
```
Expected: `25 passed`

- [ ] **Step 3.5: Commit**

```bash
git add BDD/bytetrack.py BDD/test_bytetrack.py
git commit -m "feat: add BYTETracker with two-stage association"
```

---

## Task 4: Integrate BYTETracker into app.py

**Files:**
- Modify: `BDD/app.py`

- [ ] **Step 4.1: Add import and helper at top of `app.py`**

After the existing imports (after `from helpers import ...`), add:

```python
import numpy as np
from bytetrack import BYTETracker, STrack
```

Then, after the `seen_frames = deque(maxlen=10)` line (line 97), add the helper function:

```python
def _match_track_to_detection(
    track_det_bbox: np.ndarray, detections: list
) -> int | None:
    """Return index of detection in `detections` with highest IoU against track_det_bbox."""
    best_idx, best_iou = None, 0.0
    x1, y1, x2, y2 = track_det_bbox
    for i, det in enumerate(detections):
        b = det.bbox
        ix1 = max(x1, b.left_edge);  iy1 = max(y1, b.top_edge)
        ix2 = min(x2, b.right_edge); iy2 = min(y2, b.bottom_edge)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (x2 - x1) * (y2 - y1)
        area_b = b.width * b.height
        union = area_a + area_b - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    return best_idx
```

- [ ] **Step 4.2: Update `user_app_callback_class` to hold the tracker**

Replace (lines 50–53 in `app.py`):
```python
class user_app_callback_class(app_callback_class):
    def __init__(self, detections_queue):
        super().__init__()
        self.detections_queue = detections_queue
```

With:
```python
class user_app_callback_class(app_callback_class):
    def __init__(self, detections_queue, tracker: BYTETracker):
        super().__init__()
        self.detections_queue = detections_queue
        self.tracker = tracker
        self._bt_frame_id = 0
```

- [ ] **Step 4.3: Replace Hailo track-ID read with ByteTracker call in `app_callback`**

In `app_callback`, find this block (around lines 149–175):

```python
    # Parse the detections
    detection_count = 0
    detections_list = []
    for detection in detections:
        detection : hailo.HailoDetection = detection

        # DEBUG_dump('detecion: ', detection)

        # label = detection.get_label()
        bbox = detection.get_bbox()

        confidence = detection.get_confidence()
        # if label == "person":
        # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()
        else:
            track_id = None

        detection_count += 1
        detections_list.append(Detection(
            bbox = Rect.from_xyxy(bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()),
            confidence = confidence,
            track_id = track_id,
            # class_id = detection.class_id,
        ))
```

Replace it entirely with:

```python
    # Parse the detections
    detection_count = 0
    detections_list = []
    for detection in detections:
        detection: hailo.HailoDetection = detection
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        detection_count += 1
        detections_list.append(Detection(
            bbox=Rect.from_xyxy(bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()),
            confidence=confidence,
            track_id=None,
        ))

    # Run ByteTracker to assign stable track IDs
    if detections_list:
        dets_array = np.array([
            [d.bbox.left_edge, d.bbox.top_edge, d.bbox.right_edge, d.bbox.bottom_edge, d.confidence]
            for d in detections_list
        ])
    else:
        dets_array = np.empty((0, 5))

    active_tracks = user_data.tracker.update(dets_array, user_data._bt_frame_id)
    user_data._bt_frame_id += 1

    for track in active_tracks:
        idx = _match_track_to_detection(track.det_bbox, detections_list)
        if idx is not None:
            detections_list[idx].track_id = track.track_id
```

- [ ] **Step 4.4: Update `main()` — add bytetrack params to `control_config` and wire tracker**

In `control_config` dict (around line 287 in `main()`), add these entries anywhere in the dict:

```python
        'bytetrack_track_thresh': 0.5,
        'bytetrack_det_thresh':   0.6,
        'bytetrack_match_thresh': 0.8,
        'bytetrack_track_buffer': 30,
        'bytetrack_frame_rate':   30,
```

Then find the line `user_data = user_app_callback_class(detections_queue)` (around line 273) and replace it with:

```python
    bytetracker = BYTETracker(
        track_thresh=control_config['bytetrack_track_thresh'],
        det_thresh=control_config['bytetrack_det_thresh'],
        match_thresh=control_config['bytetrack_match_thresh'],
        track_buffer=control_config['bytetrack_track_buffer'],
        frame_rate=control_config['bytetrack_frame_rate'],
    )
    user_data = user_app_callback_class(detections_queue, bytetracker)
```

Note: `user_data = user_app_callback_class(detections_queue)` appears **before** `control_config` in the original `main()`. Move or reorder so `bytetracker` is created after `control_config` is defined. The correct final order in `main()` is:

1. `control_config = { ... }` (with bytetrack keys added)
2. `bytetracker = BYTETracker(...)` (using control_config values)
3. `user_data = user_app_callback_class(detections_queue, bytetracker)`

The original line `user_data = user_app_callback_class(detections_queue)` at line 273 must be removed; the replacement above goes after `control_config` is fully defined (around line 361 in the original).

- [ ] **Step 4.5: Verify syntax**

```bash
cd /path/to/BDD_drone/BDD && python3 -c "import app" 2>&1
```
Expected: no output (clean import)

- [ ] **Step 4.6: Run existing tests to confirm nothing is broken**

```bash
cd /path/to/BDD_drone/BDD && python3 -m pytest test_bytetrack.py test_OverwriteQueue.py -qq
```
Expected: all pass

- [ ] **Step 4.7: Commit**

```bash
git add BDD/app.py
git commit -m "feat: integrate BYTETracker into app_callback, replace Hailo track ID"
```

---

## Self-Review Checklist

- [x] **KalmanFilter** — initiate / predict / update all covered in Task 1
- [x] **STrack state machine** — New → Tracked → Lost → Removed covered in Task 2
- [x] **IoU batch + greedy match** — exercised indirectly through BYTETracker tests
- [x] **BYTETracker two-stage association** — Task 3 covers: no dets, single track, ID persistence, lost/recovered, buffer expiry, low-conf rescue, two-track scenario
- [x] **Hailo UNIQUE_ID removal** — Step 4.3 replaces the block entirely
- [x] **control_config params** — Step 4.4 adds all five keys
- [x] **user_app_callback_class signature change** — Step 4.2 adds `tracker` and `_bt_frame_id`
- [x] **`_match_track_to_detection` helper** — Step 4.1, uses `det_bbox` (raw detection, not Kalman)
- [x] **Move `user_data` init after `control_config`** — noted explicitly in Step 4.4
