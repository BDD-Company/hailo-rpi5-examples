# Stereo Vision Pipeline Design

**Date:** 2026-04-18  
**Status:** Approved

## Overview

Add binocular stereo vision to the platform controller. Two libcamera cameras (standard horizontal stereo pair with known fixed baseline) run Hailo inference independently. Detections from both cameras are paired by timestamp, and distance is computed via stereo disparity. A separate calibration mode captures chessboard pairs and produces rectification maps.

## Architecture

### Pipeline (Approach B — single GStreamer app, two branches)

```
libcamerasrc (left)  → SOURCE_PIPELINE(name='left')  → INFERENCE_PIPELINE(name='left_inf')  → identity(name='left_cb')
                                                                                                        ↓
                                                                                             app_callback(camera_id='left')
                                                                                                        ↓
                                                                                             StereoPairer (in-process)
                                                                                                        ↓
                                                                                             detections_queue → platform_controller
libcamerasrc (right) → SOURCE_PIPELINE(name='right') → INFERENCE_PIPELINE(name='right_inf') → identity(name='right_cb')
                                                                                                        ↓
                                                                                             app_callback(camera_id='right')
```

Both camera branches live in one GStreamer pipeline sharing a single GLib mainloop and clock. Each branch gets its own pad probe on its `identity` element, tagged with `camera_id`.

### Modes (CLI flag `--mode`)

| Mode | Description |
|---|---|
| `mono` | Current single-camera mode (default, backwards-compatible) |
| `stereo` | Both cameras active, stereo distance estimation |
| `calibrate` | Capture chessboard pairs, produce `calibration.npz` |
| `verify-calibration` | Load `calibration.npz`, capture one stereo frame pair, draw horizontal epipolar lines on both rectified frames, display via OpenCV window. Used to visually confirm rectification quality before deployment. |

## New Components

### `stereo_pairer.py` — `StereoPairer`

Buffers `Detections` from left and right cameras in separate deques. On each incoming item, searches the opposite buffer for the closest timestamp. If `|t_left - t_right| < MAX_PAIR_GAP_NS` (default: 33 ms = one frame at 30 fps), emits a `StereoDetections` to the controller queue. Otherwise the frame waits or is evicted when the buffer fills.

If one side waits longer than `PAIR_TIMEOUT_NS`, the unpaired frame is forwarded as a mono fallback `Detections`.

`MAX_PAIR_GAP_NS` and `PAIR_TIMEOUT_NS` are configurable parameters.

### `stereo_distance.py` — `stereo_distance`

1. Match left↔right detections by IoU of raw bboxes (configurable IoU threshold).
2. For each matched pair, remap the center pixel coordinate `(cx, cy)` using the preloaded rectification maps (`map_x[cy, cx]`, `map_y[cy, cx]`). Full-frame remap is not applied at runtime — only the single center point is looked up for efficiency.
3. Compute disparity: `d = rect_center_left.x - rect_center_right.x` (pixels).
4. Compute distance: `Z = baseline_m * focal_px / d`.
5. If no match or disparity ≤ 0: return `None` (caller falls back to monocular `estimate_distance_class`).

### `stereo_calibration.py` — calibration mode

Launched via `--mode calibrate --calibration-output calibration.npz`.

1. Start both `SOURCE_PIPELINE`s (no inference). In calibration mode a simple `FramePairer` (not `StereoPairer`) buffers raw numpy frames by timestamp and emits `(left_frame, right_frame)` pairs.
2. User presents a chessboard (e.g. 9×6) in varied positions.
3. For each synced pair: `cv2.findChessboardCorners` on both frames.
4. Accumulate until `MIN_PAIRS` (default: 20) valid pairs captured. Log progress.
5. Run `cv2.stereoCalibrate` → `K1, D1, K2, D2, R, T`.
6. Run `cv2.stereoRectify` → `R1, R2, P1, P2`.
7. Run `cv2.initUndistortRectifyMap` → left/right remap arrays.
8. Save to `calibration.npz`. Log RMS reprojection error.

**`calibration.npz` structure:**

| Key | Description |
|---|---|
| `map_left_x`, `map_left_y` | Remap arrays for left camera |
| `map_right_x`, `map_right_y` | Remap arrays for right camera |
| `focal_px` | Focal length in pixels (from `P1[0,0]`) |
| `baseline_m` | Fixed baseline (provided via CLI, stored for reference) |
| `image_size` | `(width, height)` |
| `rms` | Reprojection RMS error |

### `helpers.py` — `StereoDetections`

```python
@dataclass
class StereoDetections:
    left: Detections
    right: Detections
    pair_timestamp_ns: int  # average of left/right capture timestamps
```

## Data Flow

```
app_callback (left)  → StereoPairer.put('left', detections)
app_callback (right) → StereoPairer.put('right', detections)
                              ↓
                    StereoPairer matches by timestamp
                              ↓
              StereoDetections → detections_queue
                              ↓
                    platform_controller
                      selects best detection from left frame
                      calls stereo_distance(left_det, right_det, calibration)
                        → distance_m (or None → fallback to monocular)
                      rest of control logic unchanged
```

## Changes to Existing Files

### `app.py`

- Two pad probes instead of one, each tagged with `camera_id='left'` or `camera_id='right'`.
- `user_app_callback_class` holds a `StereoPairer` instance instead of directly writing to `detections_queue`.
- Pipeline string extended with two source+inference branches.
- Mode switch via `--mode` CLI flag.

### `platform_controller.py`

- Accepts `StereoDetections | Detections` from queue (union type).
- If `StereoDetections`: use `stereo_distance` for distance; use `left.detections` for target position as before.
- If `Detections` (mono fallback): existing logic unchanged.

### `pipelines.py`

- No structural changes. Two calls to `SOURCE_PIPELINE` and `INFERENCE_PIPELINE` with distinct `name` prefixes from `app.py`.

## Error Handling

| Situation | Behaviour |
|---|---|
| One camera unavailable at startup | Hard error, app exits with clear message |
| Frames not paired (timestamp gap too large) | Log warning every N misses; fallback to mono after `PAIR_TIMEOUT_NS` |
| Zero or negative disparity | `stereo_distance` returns `None`; controller uses monocular fallback |
| `calibration.npz` missing in stereo mode | Hard error: `"Stereo mode requires calibration file. Run with --mode calibrate first."` |
| No IoU match between left/right detections | Detection forwarded without stereo distance (`distance_m = None`) |

## Testing

| File | Coverage |
|---|---|
| `test_stereo_pairer.py` | Timestamp matching, missed-frame behaviour, timeout fallback |
| `test_stereo_distance.py` | Disparity computation, zero-disparity fallback, IoU bbox matching |
| `test_stereo_calibration.py` | `calibration.npz` save/load, structure validation |

Existing tests (`test_bytetrack.py`, `test_estimate_distance.py`, `test_helpers.py`) are unaffected.
