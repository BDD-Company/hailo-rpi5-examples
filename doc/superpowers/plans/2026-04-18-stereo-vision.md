# Stereo Vision Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add binocular stereo vision to the platform controller using two independent libcamera Hailo inference branches in a single GStreamer pipeline, with OpenCV-based calibration mode and disparity-based distance estimation.

**Architecture:** Two `libcamerasrc` branches each run through their own `INFERENCE_PIPELINE_WRAPPER` then a named `identity` element. Two pad probes call `app_callback` with a `camera_id` tag. A `StereoPairer` buffers and timestamp-matches detections from both cameras into `StereoDetections`, which `platform_controller` consumes for disparity-based distance. A separate `--mode calibrate` builds rectification maps via OpenCV and saves `calibration.npz`.

**Tech Stack:** Python 3.11, GStreamer/Hailo (existing), OpenCV (`cv2`), NumPy, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `BDD/helpers.py` | Modify | Add `StereoDetections` dataclass |
| `BDD/stereo_pairer.py` | Create | `StereoPairer`: buffer + timestamp-match `Detections` from left/right; emit `StereoDetections` or mono fallback |
| `BDD/stereo_distance.py` | Create | `StereoCalibration` dataclass with `.load()`; `match_stereo_detections()`; `stereo_distance()` |
| `BDD/stereo_calibration.py` | Create | `FramePairer`; `calibrate_stereo()`; `save_calibration()`; `load_calibration()`; `run_calibration_mode()`; `run_verify_mode()` |
| `BDD/app.py` | Modify | `user_app_callback_class`: add `camera_id`, replace `detections_queue` with `sink`; move `seen_frames` to instance; update `app_callback`; add stereo pipeline + probes; add `--mode`, `--input-right`, `--calibration-file` args; wire calibrate/verify modes |
| `BDD/platform_controller.py` | Modify | Accept `StereoDetections \| Detections`; call `stereo_distance` when available; fallback to monocular |
| `BDD/test_stereo_pairer.py` | Create | Unit tests for `StereoPairer` |
| `BDD/test_stereo_distance.py` | Create | Unit tests for `stereo_distance`, `match_stereo_detections`, `StereoCalibration.load()` |
| `BDD/test_stereo_calibration.py` | Create | Unit tests for `FramePairer`, `calibrate_stereo`, `save_calibration`/`load_calibration` |

---

## Task 1: `StereoDetections` in `helpers.py`

**Files:**
- Modify: `BDD/helpers.py` (after the `Detections` dataclass, ~line 422)

- [ ] **Step 1: Write the failing test**

Create `BDD/test_stereo_pairer.py` (will grow in Task 2; start with just the import test):

```python
# BDD/test_stereo_pairer.py
import pytest
from helpers import Detections, StereoDetections, FrameMetadata


def _make_detections(frame_id=1, ts=1_000_000_000):
    return Detections(
        frame_id=frame_id,
        frame=None,
        detections=[],
        meta=FrameMetadata(capture_timestamp_ns=ts),
    )


def test_stereo_detections_fields():
    left = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=1_000_500_000)
    sd = StereoDetections(left=left, right=right, pair_timestamp_ns=1_000_250_000)
    assert sd.left is left
    assert sd.right is right
    assert sd.pair_timestamp_ns == 1_000_250_000
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd BDD && python -m pytest test_stereo_pairer.py::test_stereo_detections_fields -v
```

Expected: `ImportError: cannot import name 'StereoDetections' from 'helpers'`

- [ ] **Step 3: Add `StereoDetections` to `helpers.py`**

Add after the `Detections` class (around line 422):

```python
@dataclass(slots=True, frozen=True)
class StereoDetections:
    left: 'Detections'
    right: 'Detections'
    pair_timestamp_ns: int
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd BDD && python -m pytest test_stereo_pairer.py::test_stereo_detections_fields -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add BDD/helpers.py BDD/test_stereo_pairer.py
git commit -m "feat: add StereoDetections dataclass to helpers"
```

---

## Task 2: `StereoPairer` in `stereo_pairer.py`

**Files:**
- Create: `BDD/stereo_pairer.py`
- Modify: `BDD/test_stereo_pairer.py`

- [ ] **Step 1: Write failing tests**

Append to `BDD/test_stereo_pairer.py`:

```python
import time
import threading
from collections import deque
from stereo_pairer import StereoPairer


class _CollectQueue:
    """Simple list-backed queue stub for tests."""
    def __init__(self):
        self.items = []
    def put(self, item):
        self.items.append(item)
    def get(self, timeout=None):
        return self.items.pop(0)


def test_pair_emitted_when_timestamps_close():
    q = _CollectQueue()
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000)
    left  = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=1_010_000_000)  # 10 ms apart
    pairer.put('left', left)
    pairer.put('right', right)
    assert len(q.items) == 1
    pair = q.items[0]
    assert isinstance(pair, StereoDetections)
    assert pair.left is left
    assert pair.right is right


def test_no_pair_when_timestamps_too_far():
    q = _CollectQueue()
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000)
    left  = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=2_000_000_000)  # 1 second apart
    pairer.put('left', left)
    pairer.put('right', right)
    assert len(q.items) == 0


def test_mono_fallback_after_timeout():
    q = _CollectQueue()
    # 50 ms timeout
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000, pair_timeout_ns=50_000_000)
    # Use a very old timestamp so age calculation fires immediately
    old_ts = time.monotonic_ns() - 100_000_000  # 100 ms ago
    left = _make_detections(1, ts=old_ts)
    pairer.put('left', left)
    # Trigger flush by putting a right frame with unmatched timestamp
    right = _make_detections(2, ts=old_ts + 500_000_000)
    pairer.put('right', right)
    # left should have been forwarded as mono fallback
    mono_items = [i for i in q.items if isinstance(i, Detections) and not isinstance(i, StereoDetections)]
    assert any(i is left for i in q.items)


def test_pair_timestamp_is_average():
    q = _CollectQueue()
    pairer = StereoPairer(output_queue=q, max_pair_gap_ns=33_000_000)
    left  = _make_detections(1, ts=1_000_000_000)
    right = _make_detections(2, ts=1_020_000_000)
    pairer.put('left', left)
    pairer.put('right', right)
    assert q.items[0].pair_timestamp_ns == 1_010_000_000
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd BDD && python -m pytest test_stereo_pairer.py -v
```

Expected: `ModuleNotFoundError: No module named 'stereo_pairer'`

- [ ] **Step 3: Implement `stereo_pairer.py`**

```python
# BDD/stereo_pairer.py
import time
import threading
from collections import deque

from helpers import Detections, StereoDetections


class StereoPairer:
    """
    Buffers Detections from 'left' and 'right' cameras and emits StereoDetections
    when a matching pair is found (by capture timestamp proximity).

    If a frame waits longer than pair_timeout_ns without a match it is forwarded
    as a plain Detections (mono fallback).
    """

    def __init__(self, output_queue, max_pair_gap_ns=33_000_000, pair_timeout_ns=100_000_000, buffer_maxlen=5):
        self._output_queue = output_queue
        self._max_pair_gap_ns = max_pair_gap_ns
        self._pair_timeout_ns = pair_timeout_ns
        self._buffer_maxlen = buffer_maxlen
        # Each entry: (detections, wall_time_received_ns)
        self._buffers: dict[str, deque] = {
            'left':  deque(maxlen=buffer_maxlen),
            'right': deque(maxlen=buffer_maxlen),
        }
        self._lock = threading.Lock()

    def put(self, camera_id: str, detections: Detections) -> None:
        assert camera_id in ('left', 'right'), f"Unknown camera_id: {camera_id}"
        received_ns = time.monotonic_ns()
        capture_ns = detections.meta.capture_timestamp_ns or received_ns
        other = 'right' if camera_id == 'left' else 'left'

        with self._lock:
            other_buf = self._buffers[other]
            best_idx, best_diff = None, self._max_pair_gap_ns

            for i, (other_det, _) in enumerate(other_buf):
                other_cap = other_det.meta.capture_timestamp_ns or 0
                diff = abs(capture_ns - other_cap)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is not None:
                other_det, _ = other_buf[best_idx]
                # Rebuild deque without the matched item
                remaining = [item for j, item in enumerate(other_buf) if j != best_idx]
                self._buffers[other] = deque(remaining, maxlen=self._buffer_maxlen)

                left  = detections if camera_id == 'left' else other_det
                right = other_det  if camera_id == 'left' else detections
                l_ts  = left.meta.capture_timestamp_ns or 0
                r_ts  = right.meta.capture_timestamp_ns or 0
                avg   = (l_ts + r_ts) // 2

                self._output_queue.put(StereoDetections(left=left, right=right, pair_timestamp_ns=avg))
            else:
                self._buffers[camera_id].append((detections, received_ns))
                self._flush_timed_out()

    def _flush_timed_out(self) -> None:
        now = time.monotonic_ns()
        for side in ('left', 'right'):
            buf = self._buffers[side]
            while buf:
                oldest_det, oldest_received = buf[0]
                if now - oldest_received > self._pair_timeout_ns:
                    buf.popleft()
                    self._output_queue.put(oldest_det)
                else:
                    break
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd BDD && python -m pytest test_stereo_pairer.py -v
```

Expected: all 5 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add BDD/stereo_pairer.py BDD/test_stereo_pairer.py
git commit -m "feat: add StereoPairer for timestamp-based stereo frame matching"
```

---

## Task 3: `stereo_distance.py`

**Files:**
- Create: `BDD/stereo_distance.py`
- Create: `BDD/test_stereo_distance.py`

- [ ] **Step 1: Write failing tests**

```python
# BDD/test_stereo_distance.py
import numpy as np
import pytest
import tempfile, os
from helpers import Detection, Rect, XY
from stereo_distance import (
    StereoCalibration,
    stereo_distance,
    match_stereo_detections,
)


def _make_identity_calib(w=640, h=480, focal_px=500.0, baseline_m=0.12):
    """Identity rectification maps (no distortion) for testing."""
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return StereoCalibration(
        map_left_x=xs, map_left_y=ys,
        map_right_x=xs, map_right_y=ys,
        focal_px=focal_px,
        baseline_m=baseline_m,
        image_size=(w, h),
    )


def _det(cx, cy, w=0.1, h=0.1):
    """Make a Detection with center at (cx, cy) in normalized coords."""
    return Detection(bbox=Rect.from_xywh(cx - w/2, cy - h/2, w, h))


def test_stereo_distance_basic():
    calib = _make_identity_calib(w=640, h=480, focal_px=500.0, baseline_m=0.12)
    # Left center at pixel 320, right center at pixel 260 → disparity = 60px
    # Expected: 0.12 * 500 / 60 = 1.0 m
    left_det  = _det(320/640, 240/480)
    right_det = _det(260/640, 240/480)
    d = stereo_distance(left_det, right_det, calib, frame_width=640, frame_height=480)
    assert d is not None
    assert abs(d - 1.0) < 0.01


def test_stereo_distance_zero_disparity_returns_none():
    calib = _make_identity_calib()
    det = _det(0.5, 0.5)
    result = stereo_distance(det, det, calib, frame_width=640, frame_height=480)
    assert result is None


def test_stereo_distance_negative_disparity_returns_none():
    calib = _make_identity_calib(w=640)
    # right center is to the LEFT of left center → negative disparity
    left_det  = _det(200/640, 0.5)
    right_det = _det(300/640, 0.5)
    result = stereo_distance(left_det, right_det, calib, frame_width=640, frame_height=480)
    assert result is None


def test_match_stereo_detections_matches_by_iou():
    left  = [_det(0.3, 0.3), _det(0.7, 0.7)]
    right = [_det(0.71, 0.71), _det(0.31, 0.31)]
    pairs = match_stereo_detections(left, right, iou_threshold=0.1)
    assert len(pairs) == 2
    centers_l = {(round(p[0].bbox.center.x, 1), round(p[0].bbox.center.y, 1)) for p in pairs}
    centers_r = {(round(p[1].bbox.center.x, 1), round(p[1].bbox.center.y, 1)) for p in pairs}
    assert (0.3, 0.3) in centers_l
    assert (0.7, 0.7) in centers_l


def test_match_stereo_detections_no_match_below_threshold():
    left  = [_det(0.1, 0.1)]
    right = [_det(0.9, 0.9)]
    pairs = match_stereo_detections(left, right, iou_threshold=0.1)
    assert pairs == []


def test_stereo_calibration_save_load_roundtrip():
    calib = _make_identity_calib(w=320, h=240, focal_px=400.0, baseline_m=0.10)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name
    try:
        calib.save(path)
        loaded = StereoCalibration.load(path)
        assert loaded.focal_px == pytest.approx(400.0)
        assert loaded.baseline_m == pytest.approx(0.10)
        assert loaded.image_size == (320, 240)
        assert np.allclose(loaded.map_left_x, calib.map_left_x)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd BDD && python -m pytest test_stereo_distance.py -v
```

Expected: `ModuleNotFoundError: No module named 'stereo_distance'`

- [ ] **Step 3: Implement `stereo_distance.py`**

```python
# BDD/stereo_distance.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd BDD && python -m pytest test_stereo_distance.py -v
```

Expected: all 6 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add BDD/stereo_distance.py BDD/test_stereo_distance.py
git commit -m "feat: add StereoCalibration and stereo_distance computation"
```

---

## Task 4: `stereo_calibration.py` — FramePairer + calibration

**Files:**
- Create: `BDD/stereo_calibration.py`
- Create: `BDD/test_stereo_calibration.py`

- [ ] **Step 1: Write failing tests**

```python
# BDD/test_stereo_calibration.py
import numpy as np
import pytest
import tempfile, os
import time
from stereo_calibration import FramePairer, calibrate_stereo, save_calibration, load_calibration


def test_frame_pairer_returns_pair_when_close():
    pairer = FramePairer(max_gap_ns=33_000_000)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pairer.put('left',  frame, timestamp_ns=1_000_000_000)
    result = pairer.put('right', frame, timestamp_ns=1_010_000_000)
    assert result is not None
    left_frame, right_frame = result
    assert left_frame is frame
    assert right_frame is frame


def test_frame_pairer_returns_none_when_too_far():
    pairer = FramePairer(max_gap_ns=33_000_000)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pairer.put('left',  frame, timestamp_ns=1_000_000_000)
    result = pairer.put('right', frame, timestamp_ns=2_000_000_000)
    assert result is None


def test_calibration_save_load_roundtrip():
    map_x = np.arange(640 * 480, dtype=np.float32).reshape(480, 640)
    map_y = np.zeros((480, 640), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        path = f.name
    try:
        save_calibration(
            path=path,
            map_left_x=map_x, map_left_y=map_y,
            map_right_x=map_x, map_right_y=map_y,
            focal_px=480.0,
            baseline_m=0.12,
            image_size=(640, 480),
            rms=0.42,
        )
        calib = load_calibration(path)
        assert calib.focal_px == pytest.approx(480.0)
        assert calib.baseline_m == pytest.approx(0.12)
        assert calib.image_size == (640, 480)
        assert np.allclose(calib.map_left_x, map_x)
    finally:
        os.unlink(path)


def test_calibrate_stereo_returns_rms_and_maps():
    # Use a known synthetic chessboard (9x6, 25mm squares)
    # Generate synthetic object points and image points
    board_size = (6, 4)  # inner corners
    square_size = 0.025
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

    # Fake image points (identity projection — not physically accurate, just checks the function runs)
    img_pts = objp[:, :2].copy()
    img_pts_left  = [img_pts * 20 + np.array([50, 50]) for _ in range(5)]
    img_pts_right = [img_pts * 20 + np.array([40, 50]) for _ in range(5)]  # slight horizontal shift
    obj_pts = [objp for _ in range(5)]
    image_size = (640, 480)

    result = calibrate_stereo(obj_pts, img_pts_left, img_pts_right, image_size)
    assert 'rms' in result
    assert 'map_left_x' in result
    assert result['map_left_x'].shape == (image_size[1], image_size[0])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd BDD && python -m pytest test_stereo_calibration.py -v
```

Expected: `ModuleNotFoundError: No module named 'stereo_calibration'`

- [ ] **Step 3: Implement `stereo_calibration.py`**

```python
# BDD/stereo_calibration.py
"""
Stereo camera calibration utilities.

For interactive calibration mode, see run_calibration_mode().
"""
from __future__ import annotations
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from stereo_distance import StereoCalibration


# ---------------------------------------------------------------------------
# FramePairer — matches raw numpy frames from two cameras by timestamp
# ---------------------------------------------------------------------------

class FramePairer:
    """
    Buffers (frame, timestamp_ns) from 'left' and 'right'.
    put() returns (left_frame, right_frame) on a match, else None.
    """

    def __init__(self, max_gap_ns: int = 33_000_000, buffer_maxlen: int = 3):
        self._max_gap_ns = max_gap_ns
        self._buffers: dict[str, deque] = {
            'left':  deque(maxlen=buffer_maxlen),
            'right': deque(maxlen=buffer_maxlen),
        }
        self._lock = threading.Lock()

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

            for i, (other_frame, other_ts) in enumerate(other_buf):
                diff = abs(timestamp_ns - other_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i

            if best_idx is not None:
                other_frame, _ = other_buf[best_idx]
                remaining = [item for j, item in enumerate(other_buf) if j != best_idx]
                self._buffers[other] = deque(remaining, maxlen=other_buf.maxlen)

                if camera_id == 'left':
                    return frame, other_frame
                else:
                    return other_frame, frame
            else:
                self._buffers[camera_id].append((frame, timestamp_ns))
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
    """
    Run cv2.stereoCalibrate + stereoRectify + initUndistortRectifyMap.

    Returns dict with keys:
        rms, map_left_x, map_left_y, map_right_x, map_right_y, focal_px, K1, K2
    """
    flags = cv2.CALIB_RATIONAL_MODEL

    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_left, img_pts_right,
        None, None, None, None,
        image_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )

    map_l_x, map_l_y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    focal_px = float(P1[0, 0])

    return {
        'rms':         rms,
        'map_left_x':  map_l_x,
        'map_left_y':  map_l_y,
        'map_right_x': map_r_x,
        'map_right_y': map_r_y,
        'focal_px':    focal_px,
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
# Interactive calibration mode (called from app.py --mode calibrate)
# ---------------------------------------------------------------------------

def run_calibration_mode(
    pipeline,            # running GStreamer pipeline (already PLAYING)
    pairer: FramePairer,
    baseline_m: float,
    output_path: str,
    board_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
    min_pairs: int = 20,
) -> None:
    """
    Capture chessboard frame pairs, run stereoCalibrate, save to output_path.
    Blocks until min_pairs collected or pipeline stops.
    """
    import logging
    logger = logging.getLogger(__name__)

    obj_pts:      list[np.ndarray] = []
    img_pts_left: list[np.ndarray] = []
    img_pts_right: list[np.ndarray] = []

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = (
        np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size_m
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_size = None

    logger.info("Calibration mode: show chessboard %dx%d to both cameras. Need %d pairs.", *board_size, min_pairs)

    while len(obj_pts) < min_pairs:
        pair = pairer._pending_pair()  # blocks up to 1 s, see note below
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
            logger.debug("Chessboard not found in both frames (l=%s r=%s)", found_l, found_r)

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


def run_verify_mode(calib: StereoCalibration, left_frame: np.ndarray, right_frame: np.ndarray) -> None:
    """
    Draw horizontal epipolar lines on both rectified frames and show via OpenCV window.
    """
    h, w = left_frame.shape[:2]
    rect_l = cv2.remap(left_frame,  calib.map_left_x,  calib.map_left_y,  cv2.INTER_LINEAR)
    rect_r = cv2.remap(right_frame, calib.map_right_x, calib.map_right_y, cv2.INTER_LINEAR)

    combined = np.hstack([rect_l, rect_r])
    for y in range(0, h, h // 10):
        cv2.line(combined, (0, y), (w * 2, y), (0, 255, 0), 1)

    cv2.imshow("Stereo rectification verify (press any key to close)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

**Note:** `run_calibration_mode` references `pairer._pending_pair()` which is a blocking helper not yet in `FramePairer`. Add it:

```python
# Add inside FramePairer class:
def _pending_pair(self, timeout_s: float = 1.0) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Poll for a ready pair, blocking up to timeout_s."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with self._lock:
            if self._ready:
                return self._ready.pop(0)
        time.sleep(0.01)
    return None
```

And add `self._ready: list = []` to `__init__`, and in `put()` when a pair is found, instead of returning, append to `self._ready` and return `None` from `put()`. **But** this would break the test in Task 4 Step 1 which checks the return value of `put()`.

To keep both tests passing, `FramePairer.put()` returns the pair directly (for unit tests) AND also appends to `self._ready` (for `run_calibration_mode`). Update the implementation:

```python
# In FramePairer.put(), after computing left_frame/right_frame:
pair = (left_frame, right_frame) if camera_id == 'left' else (other_frame, frame)
self._ready.append(pair)
return pair
```

And in `__init__`: `self._ready: list = []`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd BDD && python -m pytest test_stereo_calibration.py -v
```

Expected: all 4 tests `PASSED` (`test_calibrate_stereo_returns_rms_and_maps` may take a few seconds due to OpenCV)

- [ ] **Step 5: Commit**

```bash
git add BDD/stereo_calibration.py BDD/test_stereo_calibration.py
git commit -m "feat: add stereo calibration — FramePairer, calibrate_stereo, save/load"
```

---

## Task 5: `app.py` — refactor callback + stereo mode

**Files:**
- Modify: `BDD/app.py`

This task has multiple steps. Read `BDD/app.py` in full before starting.

- [ ] **Step 1: Refactor `user_app_callback_class`**

Replace the existing class definition (lines ~52-57):

```python
# OLD:
class user_app_callback_class(app_callback_class):
    def __init__(self, detections_queue, tracker: BYTETracker):
        super().__init__()
        self.detections_queue = detections_queue
        self.tracker = tracker

# NEW:
class user_app_callback_class(app_callback_class):
    def __init__(self, sink, tracker: BYTETracker, camera_id: str = 'mono'):
        super().__init__()
        self.sink = sink           # OverwriteQueue (mono) or StereoPairer (stereo)
        self.tracker = tracker
        self.camera_id = camera_id
        self.seen_frames = deque(maxlen=10)
```

- [ ] **Step 2: Update `app_callback` to use `sink` and per-instance `seen_frames`**

In `app_callback`, replace the module-level `seen_frames` usage and the final `user_data.detections_queue.put(...)` call:

```python
# Replace the two module-level seen_frames lines (~lines 152-155):
# OLD:
if frame_id in seen_frames:
    return Gst.PadProbeReturn.OK
seen_frames.append(frame_id)

# NEW:
if frame_id in user_data.seen_frames:
    return Gst.PadProbeReturn.OK
user_data.seen_frames.append(frame_id)
```

```python
# Replace the final put call (~lines 215-225):
# OLD:
user_data.detections_queue.put(
    Detections(frame_id, frame, detections_list, meta=FrameMetadata(...))
)

# NEW:
detections_obj = Detections(
    frame_id, frame, detections_list,
    meta=FrameMetadata(
        capture_timestamp_ns=sensor_timestamp_ns,
        detection_start_timestamp_ns=detection_start_timestamp_ns,
        detection_end_timestamp_ns=detection_end_timestamp_ns,
    )
)
from stereo_pairer import StereoPairer as _StereoPairer
if isinstance(user_data.sink, _StereoPairer):
    user_data.sink.put(user_data.camera_id, detections_obj)
else:
    user_data.sink.put(detections_obj)
```

Move the `from stereo_pairer import StereoPairer` import to the top of the file (with other imports).

Also remove the now-unused module-level `seen_frames = deque(maxlen=10)` line.

- [ ] **Step 3: Update `main()` — add CLI args and stereo wiring**

In `main()`, add arguments after the existing `arg_parser.add_argument('--action', ...)` line:

```python
arg_parser.add_argument(
    '--mode',
    type=str,
    choices=['mono', 'stereo', 'calibrate', 'verify-calibration'],
    default='mono',
)
arg_parser.add_argument('--input-right', type=str, default=None,
                        help='Right camera source for stereo/calibrate modes')
arg_parser.add_argument('--calibration-file', type=str, default='calibration.npz',
                        help='Path to stereo calibration file')
arg_parser.add_argument('--baseline-m', type=float, default=0.12,
                        help='Stereo baseline in metres (used during calibration)')
arg_parser.add_argument('--board-size', type=str, default='9x6',
                        help='Chessboard inner corners WxH (e.g. 9x6)')
```

- [ ] **Step 4: Update `user_data` construction in `main()` for stereo**

Replace the single `user_data` construction block (~lines 395-403):

```python
# OLD:
bytetracker = BYTETracker(...)
user_data = user_app_callback_class(detections_queue, bytetracker)
user_data.use_frame = True

# NEW:
args_parsed, _ = arg_parser.parse_known_args()
run_mode = args_parsed.mode

bytetracker_left = BYTETracker(
    track_thresh=control_config['bytetrack_track_thresh'],
    det_thresh=control_config['bytetrack_det_thresh'],
    match_thresh=control_config['bytetrack_match_thresh'],
    track_buffer=control_config['bytetrack_track_buffer'],
    frame_rate=control_config['bytetrack_frame_rate'],
)

if run_mode == 'stereo':
    from stereo_pairer import StereoPairer
    stereo_pairer = StereoPairer(output_queue=detections_queue)
    bytetracker_right = BYTETracker(
        track_thresh=control_config['bytetrack_track_thresh'],
        det_thresh=control_config['bytetrack_det_thresh'],
        match_thresh=control_config['bytetrack_match_thresh'],
        track_buffer=control_config['bytetrack_track_buffer'],
        frame_rate=control_config['bytetrack_frame_rate'],
    )
    user_data_left  = user_app_callback_class(stereo_pairer, bytetracker_left,  camera_id='left')
    user_data_right = user_app_callback_class(stereo_pairer, bytetracker_right, camera_id='right')
    user_data_left.use_frame  = True
    user_data_right.use_frame = True
    user_data = user_data_left  # primary user_data passed to App (probe added manually for right)
else:
    user_data = user_app_callback_class(detections_queue, bytetracker_left)
    user_data.use_frame = True
    user_data_right = None
```

- [ ] **Step 5: Add stereo `get_pipeline_string()` override to `App` class**

Add this method to the `App` class (after `get_output_pipeline_string`):

```python
def get_pipeline_string(self):
    if not getattr(self, '_stereo_mode', False):
        return super().get_pipeline_string()

    from pipelines import SOURCE_PIPELINE, INFERENCE_PIPELINE, INFERENCE_PIPELINE_WRAPPER, USER_CALLBACK_PIPELINE

    def _branch(name, source):
        src = SOURCE_PIPELINE(
            video_source=source,
            video_width=self.video_width,
            video_height=self.video_height,
            frame_rate=self.frame_rate,
            sync=self.sync,
            name=name,
        )
        inf = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=self.batch_size,
            config_json=self.labels_json,
            additional_params=self.thresholds_str,
            name=f'{name}_inf',
            vdevice_group_id=1,
        )
        wrapper = INFERENCE_PIPELINE_WRAPPER(inf, name=f'{name}_wrapper')
        cb = USER_CALLBACK_PIPELINE(name=f'{name}_cb')
        return f'{src} ! {wrapper} ! {cb} ! fakesink sync=false'

    left_branch  = _branch('left',  self.video_source)
    right_branch = _branch('right', self._video_source_right)
    return f'{left_branch}\n{right_branch}'
```

- [ ] **Step 6: Override `run()` in `App` to add right-camera probe in stereo mode**

Add this override to the `App` class:

```python
def run(self, wait_event_before_starting=None):
    if getattr(self, '_stereo_mode', False) and self._user_data_right is not None:
        right_elem = self.pipeline.get_by_name('right_cb')
        if right_elem is None:
            raise RuntimeError("Stereo pipeline missing 'right_cb' identity element")
        right_pad = right_elem.get_static_pad('src')
        right_pad.add_probe(Gst.PadProbeType.BUFFER, self.app_callback, self._user_data_right)
    super().run(wait_event_before_starting)
```

- [ ] **Step 7: Wire stereo state onto `App` instance in `main()`**

After creating the `App` instance and before calling `app.run()`, add:

```python
if run_mode == 'stereo':
    app._stereo_mode = True
    app._video_source_right = args_parsed.input_right
    app._user_data_right = user_data_right
    if args_parsed.input_right is None:
        raise SystemExit("--input-right is required for --mode stereo")
else:
    app._stereo_mode = False
    app._video_source_right = None
    app._user_data_right = None
```

Also wire the `calibration_file` path onto the platform controller config:

```python
if run_mode == 'stereo':
    control_config['calibration_file'] = args_parsed.calibration_file
```

- [ ] **Step 8: Handle `--mode calibrate` and `--mode verify-calibration` in `main()`**

Before the `App(...)` construction, add:

```python
if run_mode in ('calibrate', 'verify-calibration'):
    _run_calibration_or_verify(args_parsed, control_config)
    return

def _run_calibration_or_verify(args, config):
    from stereo_calibration import FramePairer, run_calibration_mode, run_verify_mode, load_calibration
    from pipelines import SOURCE_PIPELINE, USER_CALLBACK_PIPELINE
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib

    if args.input_right is None:
        raise SystemExit("--input-right is required for calibration modes")

    board_w, board_h = (int(x) for x in args.board_size.split('x'))
    pairer = FramePairer(max_gap_ns=33_000_000)

    # Build minimal two-source pipeline (no inference)
    left_src  = SOURCE_PIPELINE(video_source=args.input,       name='left_cal')
    right_src = SOURCE_PIPELINE(video_source=args.input_right, name='right_cal')
    left_cb   = USER_CALLBACK_PIPELINE(name='left_cal_cb')
    right_cb  = USER_CALLBACK_PIPELINE(name='right_cal_cb')
    pipeline_str = (
        f'{left_src}  ! {left_cb}  ! fakesink sync=false\n'
        f'{right_src} ! {right_cb} ! fakesink sync=false'
    )

    Gst.init(None)
    pipeline = Gst.parse_launch(pipeline_str)

    def _make_cal_callback(camera_id):
        def _cb(pad, info, _user_data):
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            fmt, w, h = get_caps_from_pad(pad)
            frame = get_numpy_from_buffer(buf, fmt, w, h)
            ts = buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else time.monotonic_ns()
            pairer.put(camera_id, frame, timestamp_ns=ts)
            return Gst.PadProbeReturn.OK
        return _cb

    for side in ('left', 'right'):
        elem = pipeline.get_by_name(f'{side}_cal_cb')
        elem.get_static_pad('src').add_probe(Gst.PadProbeType.BUFFER, _make_cal_callback(side), None)

    pipeline.set_state(Gst.State.PLAYING)

    if args.mode == 'calibrate':
        run_calibration_mode(
            pipeline=pipeline,
            pairer=pairer,
            baseline_m=args.baseline_m,
            output_path=args.calibration_file,
            board_size=(board_w, board_h),
            min_pairs=20,
        )
    else:  # verify-calibration
        calib = load_calibration(args.calibration_file)
        pair = pairer._pending_pair(timeout_s=5.0)
        if pair is None:
            raise SystemExit("No frame pair received within 5 seconds")
        run_verify_mode(calib, pair[0], pair[1])

    pipeline.set_state(Gst.State.NULL)
```

- [ ] **Step 9: Verify `mono` mode still works (regression check)**

```bash
cd BDD && python -c "
from app import user_app_callback_class
from bytetrack import BYTETracker
from OverwriteQueue import OverwriteQueue
q = OverwriteQueue(maxsize=5)
bt = BYTETracker(track_thresh=0.5, det_thresh=0.6, match_thresh=0.8, track_buffer=30, frame_rate=30)
ud = user_app_callback_class(q, bt)
assert ud.camera_id == 'mono'
print('OK')
"
```

Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add BDD/app.py
git commit -m "feat: add stereo pipeline, dual probes, and calibrate/verify modes to app.py"
```

---

## Task 6: `platform_controller.py` — stereo distance integration

**Files:**
- Modify: `BDD/platform_controller.py`

Read `BDD/platform_controller.py` in full before starting.

- [ ] **Step 1: Add imports at top of `platform_controller.py`**

```python
# Add after existing imports:
from helpers import StereoDetections
from stereo_distance import StereoCalibration, stereo_distance, match_stereo_detections
```

- [ ] **Step 2: Load calibration at startup**

In `platform_controlling_thread_async`, after the `control_config.pop(...)` block, add:

```python
CALIBRATION_FILE = control_config.pop('calibration_file', None)
stereo_calib: StereoCalibration | None = None
if CALIBRATION_FILE:
    try:
        stereo_calib = StereoCalibration.load(CALIBRATION_FILE)
        logger.info("Loaded stereo calibration from %s (focal=%.1f baseline=%.3fm)",
                    CALIBRATION_FILE, stereo_calib.focal_px, stereo_calib.baseline_m)
    except FileNotFoundError:
        raise SystemExit(
            f"Stereo mode requires calibration file. Run with --mode calibrate first. "
            f"(looked for: {CALIBRATION_FILE})"
        )
```

- [ ] **Step 3: Unpack `StereoDetections | Detections` from the queue**

In the main loop, replace the existing queue read and `detections_obj = r` assignment:

```python
# OLD:
r : Detections = detections_queue.get(0.01)
if r is STOP:
    logger.info("stopping")
    break
detections_obj = r

# NEW:
r = detections_queue.get(0.01)
if r is STOP:
    logger.info("stopping")
    break
if isinstance(r, StereoDetections):
    detections_obj = r.left
    stereo_right   = r.right
else:
    detections_obj = r
    stereo_right   = None
```

- [ ] **Step 4: Replace monocular distance with stereo distance (with fallback)**

Find the existing `estimated_distance = estimate_distance_class(...)` call in the detection block and replace:

```python
# OLD:
estimated_distance = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, detection.bbox.size)

# NEW:
estimated_distance = None
if stereo_calib is not None and stereo_right is not None:
    frame_w, frame_h = stereo_calib.image_size
    pairs = match_stereo_detections(
        [detection],
        [d for d in stereo_right.detections if d.confidence >= MIN_CONFIDENCE],
    )
    if pairs:
        _, right_det = pairs[0]
        d_m = stereo_distance(detection, right_det, stereo_calib, frame_w, frame_h)
        if d_m is not None:
            from estimate_distance import DistanceClass
            max_size = max(TARGET_SIZE_M.x, TARGET_SIZE_M.y)
            if d_m < max_size * 5:
                dc = DistanceClass.NEAR
            elif d_m < max_size * 20:
                dc = DistanceClass.MEDIUM
            else:
                dc = DistanceClass.FAR
            estimated_distance = (dc, d_m)

if estimated_distance is None:
    estimated_distance = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, detection.bbox.size)
```

- [ ] **Step 5: Verify no existing tests break**

```bash
cd BDD && python -m pytest test_stereo_pairer.py test_stereo_distance.py test_stereo_calibration.py test_estimate_distance.py test_helpers.py test_bytetrack.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add BDD/platform_controller.py
git commit -m "feat: integrate stereo distance into platform_controller with monocular fallback"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task |
|---|---|
| Single GStreamer app, two branches | Task 5 Steps 5-7 |
| `--mode` CLI flag (mono/stereo/calibrate/verify-calibration) | Task 5 Steps 3, 8 |
| `StereoPairer` timestamp matching + timeout fallback | Task 2 |
| `StereoDetections` dataclass | Task 1 |
| `stereo_distance` + `match_stereo_detections` | Task 3 |
| `StereoCalibration` save/load (calibration.npz) | Task 3 |
| `FramePairer` for calibration mode | Task 4 |
| `calibrate_stereo` via OpenCV | Task 4 |
| `run_calibration_mode` interactive loop | Task 4, Task 5 Step 8 |
| `run_verify_mode` epipolar line display | Task 4, Task 5 Step 8 |
| Hard error if calibration.npz missing in stereo mode | Task 6 Step 2 |
| Hard error if one camera unavailable | GStreamer pipeline parse will fail — inherent |
| Mono fallback when disparity invalid | Task 6 Step 4 |
| Mono fallback when no IoU match | Task 6 Step 4 |
| Mono fallback from StereoPairer timeout | Task 2 |
| `platform_controller` uses left frame for control | Task 6 Step 3 |
| Test: `test_stereo_pairer.py` | Task 2 |
| Test: `test_stereo_distance.py` | Task 3 |
| Test: `test_stereo_calibration.py` | Task 4 |

All spec requirements covered. ✓

### Type consistency check

- `StereoDetections.left` / `.right` → `Detections` ✓ (defined Task 1, used Task 2, consumed Task 6)
- `StereoCalibration` fields: `map_left_x`, `map_left_y`, `map_right_x`, `map_right_y`, `focal_px`, `baseline_m`, `image_size` — consistent across Task 3 (`stereo_distance.py`), Task 4 (`stereo_calibration.py` save/load), Task 6 (`platform_controller.py`)  ✓
- `stereo_distance(left_det, right_det, calib, frame_width, frame_height)` — defined Task 3, called Task 6 ✓
- `match_stereo_detections(left_dets, right_dets, iou_threshold)` → `list[tuple[Detection, Detection]]` — defined Task 3, called Task 6 ✓
- `FramePairer.put(camera_id, frame, timestamp_ns)` → `Optional[tuple[np.ndarray, np.ndarray]]` — defined Task 4, called Task 5 Step 8 ✓
- `StereoPairer.put(camera_id, detections)` — defined Task 2, called Task 5 Step 2 ✓
- `user_app_callback_class(sink, tracker, camera_id)` — defined Task 5 Step 1, constructed Task 5 Step 4 ✓
