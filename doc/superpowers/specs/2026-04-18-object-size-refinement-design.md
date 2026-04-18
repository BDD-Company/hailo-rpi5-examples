# Object Size Refinement via Binary Segmentation

**Date:** 2026-04-18  
**Branch:** simulator  
**Status:** Approved

## Problem

Distance estimation (`estimate_distance_class`) uses `detection.bbox.size` — the full bounding box dimensions — as a proxy for the real object size. Neural-network and HSV-based bounding boxes are consistently larger than the object itself, causing the distance estimate to undershoot (object appears closer than it is).

## Goal

Compute the actual linear size (width, height) of the object *inside* the bbox by segmenting it from the background, and use that refined size instead of the raw bbox dimensions.

## Scope

Both `drone_controller.py` and `platform_controller.py`. Works with frames from both `app.py` (Hailo pipeline) and `app_sim.py` (RTP/HSV pipeline) since both pass the numpy frame through `Detections.frame`.

## Target Appearance

Dark drone against bright sky — high contrast, bimodal intensity histogram. Otsu's automatic threshold is reliable for this scenario.

## Algorithm: `measure_object_size`

**Signature:**
```python
def measure_object_size(frame: np.ndarray, bbox: Rect) -> XY | None
```

**Steps:**

1. Guard: if `frame is None`, return `None`.
2. Compute pixel coords from normalized bbox, clipped to frame bounds.
3. Guard: if ROI is smaller than 4×4 px, return `None`.
4. Crop: `crop = frame[y1:y2, x1:x2]`
5. Grayscale: `cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)` (pipeline produces RGB).
6. Otsu threshold inverted: `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)` — inverted because object is dark.
7. Find contours (`cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE`). If none, return `None`.
8. Take largest contour by area. If area < 5% of bbox pixel area, return `None` (noise guard).
9. `x, y, w, h = cv2.boundingRect(largest_contour)`
10. Normalize: `XY(w / frame.shape[1], h / frame.shape[0])` — same units as `bbox.size`.

**Returns:** refined `XY` size, or `None` on any failure/edge case.

## Call Sites

`drone_controller.py` and `platform_controller.py`, replacing the direct `detection.bbox.size` argument:

```python
object_size = measure_object_size(detections_obj.frame, detection.bbox) or detection.bbox.size
estimated_distance_class, estimated_distance_m = estimate_distance_class(
    TARGET_SIZE_M, FRAME_ANGLUAR_SIZE_DEG, object_size
)
```

## Edge Cases

| Case | Handling |
|------|----------|
| `frame is None` | Return `None` → fallback to bbox |
| bbox out of frame bounds | `np.clip` pixel coords before crop |
| ROI < 4×4 px | Return `None` → fallback to bbox |
| Otsu finds no contours | Return `None` → fallback to bbox |
| Largest contour area < 5% bbox area | Return `None` → fallback to bbox |
| Contour spans > 95% bbox | Valid result, keep it |

## Files Changed

| File | Change |
|------|--------|
| `BDD/estimate_distance.py` | Add `measure_object_size` function, add `import cv2` |
| `BDD/drone_controller.py` | Replace `detection.bbox.size` with `measure_object_size(...)` call |
| `BDD/platform_controller.py` | Same |
