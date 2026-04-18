# ByteTrack Integration Design

**Date:** 2026-04-18  
**Scope:** Add a minimal pure-numpy ByteTracker to replace Hailo's disabled GStreamer tracker

---

## Context

The Hailo GStreamer pipeline includes a `TRACKER_PIPELINE` (hailotracker) that is intentionally disabled (`if False:` in `app_base.py:651`). As a result, `track_id` in every `Detection` is always `None`. The drone controller relies on stable track IDs to follow a target across frames. This design adds a Python-side ByteTracker with no external dependencies beyond numpy.

---

## Files

| File | Change |
|---|---|
| `BDD/bytetrack.py` | New — full ByteTrack implementation |
| `BDD/app.py` | Modified — integrate tracker into `app_callback` and `user_app_callback_class` |

---

## Architecture

### `bytetrack.py` — three classes

#### `KalmanFilter`

8-dimensional state vector: `[x, y, w, h, vx, vy, vw, vh]` where `(x, y)` is the bbox centre, `(w, h)` is width/height, and velocities are their derivatives.

Implemented with numpy only (no scipy). Standard constant-velocity linear Kalman filter:
- State transition matrix `F` encodes `pos += vel * dt` (dt=1 frame)
- Observation matrix `H` extracts `[x, y, w, h]` from state
- Process noise `Q` and measurement noise `R` scaled by `frame_rate`
- Methods: `initiate(bbox) → (mean, cov)`, `predict(mean, cov)`, `update(mean, cov, bbox)`

#### `STrack`

Represents a single tracked object.

State machine: `New → Tracked → Lost → Removed`

- `New`: created this frame, not yet confirmed
- `Tracked`: matched in the current frame
- `Lost`: not matched; kept for `track_buffer` frames then promoted to `Removed`
- `Removed`: pruned at the end of each update cycle

Fields: `track_id` (monotonically incrementing int), `kalman_state (mean, cov)`, `score`, `bbox` (last matched, normalised 0–1 xyxy), `state`, `frame_id` (last matched), `tracklet_len`

Methods: `predict()`, `update(det, frame_id)`, `mark_lost()`, `mark_removed()`

#### `BYTETracker`

Main tracker. Holds lists of `tracked_stracks`, `lost_stracks`.

```
update(dets: np.ndarray, frame_id: int) → list[STrack]
```

`dets` shape: `(N, 5)` — `[x1, y1, x2, y2, score]` in normalised 0–1 coords.  
Returns active `STrack` objects for the current frame.

---

## Two-Stage Association Algorithm

```
Input: dets (N×5 array), current tracked/lost stracks

1. Predict all tracked + lost stracks via Kalman

2. Split dets:
   high_dets = dets[score >= track_thresh]
   low_dets  = dets[det_thresh <= score < track_thresh]

Stage 1 — match high_dets against tracked stracks:
   iou_matrix = iou(tracked_bboxes, high_det_bboxes)   # shape (T × H)
   greedy match by descending IoU, threshold match_thresh
   → matched pairs: update strack; unmatched stracks → candidate_lost
   → unmatched high_dets → unmatched_high

Stage 2 — match low_dets against candidate_lost (not yet fully lost):
   iou_matrix = iou(candidate_lost_bboxes, low_det_bboxes)
   greedy match, same threshold
   → matched pairs: update strack (rescue from lost)
   → remaining candidate_lost → mark_lost()

New tracks:
   unmatched_high dets → new STrack(New)
   (confirmed to Tracked on next frame they are matched)

Lost stracks surviving > track_buffer frames → mark_removed()
Prune removed stracks.

Return: all stracks with state == Tracked
```

Greedy matching is sufficient for single-target drone tracking and avoids a scipy dependency. If multiple targets are ever needed, the same structure supports replacement with a proper Hungarian solver.

---

## Coordinate Convention

All bounding boxes use **normalised (0–1) xyxy** throughout — identical to the existing `Detection.bbox` format. No pixel conversion is needed.

---

## Integration in `app.py`

### `user_app_callback_class`

```python
class user_app_callback_class(app_callback_class):
    def __init__(self, detections_queue, tracker):
        super().__init__()
        self.detections_queue = detections_queue
        self.tracker = tracker
        self._frame_id = 0
```

### `app_callback` — changes

Remove Hailo UNIQUE_ID read (current lines 162–167).

After building `detections_list`, add:

```python
if detections_list:
    dets_array = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence]
                           for d in detections_list])
else:
    dets_array = np.empty((0, 5))

active_tracks = user_data.tracker.update(dets_array, user_data._frame_id)
user_data._frame_id += 1

# map track_id back to closest detection by IoU
for track in active_tracks:
    best_idx, best_iou = _match_track_to_detection(track.bbox, detections_list)
    if best_idx is not None:
        detections_list[best_idx].track_id = track.track_id
```

`_match_track_to_detection` is a small helper (< 10 lines) that finds the detection with highest IoU to the track's bbox.

### `main()` — changes

```python
bytetracker = BYTETracker(
    track_thresh  = control_config['bytetrack_track_thresh'],
    det_thresh    = control_config['bytetrack_det_thresh'],
    match_thresh  = control_config['bytetrack_match_thresh'],
    track_buffer  = control_config['bytetrack_track_buffer'],
    frame_rate    = control_config['bytetrack_frame_rate'],
)
user_data = user_app_callback_class(detections_queue, bytetracker)
```

---

## Parameters in `control_config`

```python
'bytetrack_track_thresh': 0.5,   # high-confidence detection threshold (stage 1)
'bytetrack_det_thresh':   0.6,   # min score to create a new track
'bytetrack_match_thresh': 0.8,   # IoU threshold for greedy matching
'bytetrack_track_buffer': 30,    # frames a Lost track is kept (30fps → 1 s)
'bytetrack_frame_rate':   30,    # used to scale Kalman process noise
```

---

## What Does NOT Change

- `Detection` and `Detections` dataclasses in `helpers.py` — unchanged
- `drone_controller.py` and downstream consumers — unchanged; they already handle `track_id: int | None`
- GStreamer pipeline — unchanged; Hailo hailotracker remains disabled
