# Object Size Refinement via Binary Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw `detection.bbox.size` in distance estimation with the actual object size computed by Otsu binary thresholding inside the bbox crop.

**Architecture:** A new `measure_object_size(frame, bbox) -> XY | None` function is added to `estimate_distance.py`. It crops the frame to the bbox, applies inverted Otsu threshold (dark object on bright sky), finds the largest contour, and returns its normalized bounding rect size. Both call sites in `drone_controller.py` and `platform_controller.py` use it with a `or detection.bbox.size` fallback.

**Tech Stack:** Python, OpenCV (`cv2` — already used in the project), NumPy, pytest

---

### Task 1: Add `measure_object_size` to `estimate_distance.py` with tests

**Files:**
- Modify: `BDD/estimate_distance.py` (add function + `import cv2`, `import numpy as np`)
- Create: `BDD/test_estimate_distance.py`

- [ ] **Step 1: Write the failing tests**

Create `BDD/test_estimate_distance.py`:

```python
#!/usr/bin/env python3
import numpy as np
import pytest
from helpers import Rect, XY
from estimate_distance import measure_object_size


def _make_frame(h=100, w=100):
    """White frame (bright sky)."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _draw_dark_rect(frame, x1, y1, x2, y2):
    """Draw a dark rectangle (drone) on the frame."""
    frame[y1:y2, x1:x2] = 20
    return frame


def test_returns_none_when_frame_is_none():
    bbox = Rect.from_xyxy(0.1, 0.1, 0.5, 0.5)
    assert measure_object_size(None, bbox) is None


def test_returns_none_when_roi_too_small():
    frame = _make_frame(100, 100)
    # bbox maps to 3x3 px — below 4x4 guard
    bbox = Rect.from_xyxy(0.0, 0.0, 0.03, 0.03)
    assert measure_object_size(frame, bbox) is None


def test_returns_none_when_no_dark_object():
    # Uniform white frame — Otsu will find nothing meaningful
    frame = _make_frame(100, 100)
    bbox = Rect.from_xyxy(0.0, 0.0, 1.0, 1.0)
    result = measure_object_size(frame, bbox)
    # Either None (noise guard) or a tiny value — must not crash
    assert result is None or (0.0 < result.x <= 1.0 and 0.0 < result.y <= 1.0)


def test_detects_dark_object_smaller_than_bbox():
    frame = _make_frame(100, 100)
    # Dark rect occupies 20x20 px in the center of a 60x60 bbox
    _draw_dark_rect(frame, 20, 20, 40, 40)
    bbox = Rect.from_xyxy(0.1, 0.1, 0.7, 0.7)  # 60x60 px bbox
    result = measure_object_size(frame, bbox)
    assert result is not None
    # Object is 20px wide / 100px frame → 0.20; allow ±5%
    assert abs(result.x - 0.20) < 0.05
    assert abs(result.y - 0.20) < 0.05


def test_result_smaller_than_bbox():
    frame = _make_frame(100, 100)
    _draw_dark_rect(frame, 30, 30, 50, 50)
    bbox = Rect.from_xyxy(0.2, 0.2, 0.8, 0.8)
    result = measure_object_size(frame, bbox)
    assert result is not None
    assert result.x < bbox.size.x
    assert result.y < bbox.size.y


def test_clips_bbox_to_frame_bounds():
    frame = _make_frame(100, 100)
    _draw_dark_rect(frame, 90, 90, 99, 99)
    # bbox extends beyond frame edges
    bbox = Rect.from_xyxy(0.8, 0.8, 1.2, 1.2)
    # Must not raise, may return None or a result
    result = measure_object_size(frame, bbox)
    assert result is None or isinstance(result, XY)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/yanker/Developer/BDD_drone/BDD
python -m pytest test_estimate_distance.py -v
```

Expected: `ImportError` or `AttributeError` — `measure_object_size` does not exist yet.

- [ ] **Step 3: Implement `measure_object_size` in `estimate_distance.py`**

Add at the top of `BDD/estimate_distance.py` after existing imports:

```python
import cv2
import numpy as np
```

Add this function before `estimate_distance`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/yanker/Developer/BDD_drone/BDD
python -m pytest test_estimate_distance.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add BDD/estimate_distance.py BDD/test_estimate_distance.py
git commit -m "feat: add measure_object_size for object size refinement via Otsu threshold"
```

---

### Task 2: Use `measure_object_size` in `drone_controller.py`

**Files:**
- Modify: `BDD/drone_controller.py:13` (import), `BDD/drone_controller.py:514` (call site)

- [ ] **Step 1: Update import in `drone_controller.py`**

At line 13, change:
```python
from estimate_distance import estimate_distance_class, DistanceClass
```
to:
```python
from estimate_distance import estimate_distance_class, DistanceClass, measure_object_size
```

- [ ] **Step 2: Replace call site at line 514**

Change:
```python
estimated_distance_class, estimated_distance_m = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, detection.bbox.size)
```
to:
```python
object_size = measure_object_size(detections_obj.frame, detection.bbox) or detection.bbox.size
estimated_distance_class, estimated_distance_m = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, object_size)
```

- [ ] **Step 3: Verify no import errors**

```bash
cd /Users/yanker/Developer/BDD_drone/BDD
python -c "import drone_controller; print('OK')"
```

Expected: `OK` (may print warnings about missing hardware — that's fine).

- [ ] **Step 4: Commit**

```bash
git add BDD/drone_controller.py
git commit -m "feat: use measure_object_size for distance estimation in drone_controller"
```

---

### Task 3: Use `measure_object_size` in `platform_controller.py`

**Files:**
- Modify: `BDD/platform_controller.py:13` (import), `BDD/platform_controller.py:286` (call site)

- [ ] **Step 1: Update import in `platform_controller.py`**

At line 13, change:
```python
from estimate_distance import estimate_distance_class, DistanceClass
```
to:
```python
from estimate_distance import estimate_distance_class, DistanceClass, measure_object_size
```

- [ ] **Step 2: Replace call site at line 286**

Change:
```python
estimated_distance = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, detection.bbox.size)
```
to:
```python
object_size = measure_object_size(detections_obj.frame, detection.bbox) or detection.bbox.size
estimated_distance = estimate_distance_class(TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, object_size)
```

- [ ] **Step 3: Verify no import errors**

```bash
cd /Users/yanker/Developer/BDD_drone/BDD
python -c "import platform_controller; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Run all BDD tests**

```bash
cd /Users/yanker/Developer/BDD_drone/BDD
python -m pytest test_estimate_distance.py test_OverwriteQueue.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add BDD/platform_controller.py
git commit -m "feat: use measure_object_size for distance estimation in platform_controller"
```
