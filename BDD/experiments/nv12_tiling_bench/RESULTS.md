# NV12 tiling benchmark — yolov11n_nv12.hef on bdd-sd9 (2026-06-23)

## What was tested
- **HEF:** `/home/bdd/models/yolov11n_nv12.hef` — verified NV12 input: `Input UINT8, NV12(320x640x3)`, 640×640 yolov11n, 3 classes, on-chip NMS.
- **Box:** `bdd@192.168.0.7` (bdd-sd9), **Hailo-8** (HEF is compiled for **8L** → runs with HailoRT warning *"lower performance"*).
- **NV12 tiling works** end-to-end through the Hailo GStreamer plugin (`hailotilecropper` → `hailonet` → `hailotileaggregator`), NV12 in cropper/aggregator caps and accepted by `hailonet`.
- **Tool:** `nv12_tiling_bench.py` (+ `run_matrix.sh`). FIFO pad-probe latency (cropper-sink → aggregator-src), per-branch FPS, tiles/s.

### ⚠️ Camera caveat — live capture NOT tested
Both CSI sensors fail to probe right now: `ov5647 11-0036: i2c read error … probe failed -121` (EREMOTEIO = bad cable/connection); `arducam_64mp` is seen on the bus but libcamera enumerates **0 cameras** (`Picamera2.global_camera_info() == []`). So **"whole-frame NV12 capture from camera"** could not be run — needs a cable reseat / sensor fix.
The benchmark uses a **synthetic NV12 source** (`videotestsrc`, 1280×720). This is valid for comparing tiling modes: the camera stage is identical across all modes, so what differs is purely the inference path. Add the measured camera **Stage-A capture latency ~135–170 ms** (p50 ~150 ms, see `latency-budget` memory) to get true sensor→command e2e.

## Key constant
**NPU ceiling ≈ 68 inferences/s** (~14.7 ms/inference) for this 8L-HEF-on-H8, *independent of tiling geometry*. Throughput of any mode = `68 / (crops per frame)`.

## Results (uncapped = ceiling; latency is inference-path only, p50 ms)

### A. Pure tiling, single net-group (no +1) — the efficient baseline
| grid | crops | FPS | inf/s | lat p50 (maxq4) | lat p50 (maxq1) |
|------|-------|-----|-------|-----------------|-----------------|
| 1×1  | 1     | 66.5| 67.3  | 86  | **41** |
| 2×1  | 2     | 33.2| 67.7  | 173 | — |
| 2×2  | 4     | 16.6| 67.7  | 255 | 172 |
| 3×2  | 6     | 11.1| 67.8  | 315 | — |
| 3×3  | 9     | 7.4 | 67.6  | 405 | 393 |

### B. NxM tiles + whole-frame "+1" — TWO hailonets, round-robin (naïve architecture)
| mode  | inf/frame | sys FPS | tile lat p50 | whole-frame FPS | total inf/s |
|-------|-----------|---------|--------------|-----------------|-------------|
| 2×1+1 | 3  | 16.6 | 348 | 33.5 | 68.7 |
| 2×2+1 | 5  | 8.3  | 472 | 33.3 | 68.6 |
| 3×2+1 | 7  | 5.5  | 614 | 33.3 | 68.5 |
| 3×3+1 | 10 | 3.7  | 813 | 33.3 | 68.6 |

➡ The round-robin scheduler is **frame-fair, not cost-fair**: the cheap whole-frame branch runs at ~33 fps and eats **half** the NPU (33 of 68 inf/s) when it needs far less, starving the tiling branch to **half** its solo rate.

### C. Same "+1" load merged into ONE net-group (the fix)
| load | inf/frame | merged FPS | merged lat p50 | vs round-robin |
|------|-----------|-----------|----------------|----------------|
| 2×2+1 | 5  | **13.3** | 285 | **+60% fps, −40% lat** |
| 3×2+1 | 7  | **9.4**  | 344 | **+71% fps, −44% lat** |
| 3×3+1 | 10 | **6.5**  | 435 | **+76% fps, −47% lat** |

### D. Operating point: source capped at 30 fps (camera-like)
Same picture; whole-frame branch capped at 29.5 fps, tiling slightly better than uncapped (e.g. 2×1+1 → 18.5 fps). NPU still pinned at ~68 inf/s.

## End-to-end (add camera Stage-A ~150 ms p50)
| config | pipeline p50 | + camera | e2e p50 | FPS |
|--------|-------------|----------|---------|-----|
| whole-frame only (1×1, maxq1) | 41 ms | 150 | **~190 ms** | 66 |
| 2×2+1 merged (maxq1–2)        | ~210–285 ms | 150 | **~360–435 ms** | 13 |
| 3×3+1 merged                  | ~435 ms | 150 | **~585 ms** | 6.5 |

## Proposals (priority order)
1. **Merge tiles + whole-frame into ONE net-group** (biggest win: +60–76% fps, −40–47% latency). Use a single cropper emitting the N×M grid **plus** one full-frame crop into one `hailonet` + one aggregator. *Not* multi-scale mode — that over-generates a full pyramid (scale-2 ≈ 9 crops, scale-3 ≈ 25, measured), far more than N×M+1. Needs a custom crop `.so` (extend the BDD whole-buffer cropper) or two croppers funneled via `hailoroundrobin` into one `hailonet`.
2. **Recompile the NV12 HEF with `--hw-arch hailo8`.** The box is a Hailo-8 but the HEF is 8L → explicit "lower performance" warning. Free ceiling lift for *every* mode.
3. **Queue depth is a free latency knob.** `maxq=1` on bypass/post queues cut 1×1 latency 86→41 ms with **zero** throughput loss. Use minimal depth in the control path.
4. **If you keep two branches, rate-limit the whole-frame branch** to ~10 fps (it doesn't need 33) — recovers ~20 inf/s for tiling without re-architecting.
5. **Tiling ↔ latency tension.** The 50–120 ms sensor→command target is already exceeded by the camera Stage-A (~150 ms) alone; tiling adds 170–435 ms more. Treat tiling as a small-object-recall tool that costs reaction time. **2×2+1 merged (~285 ms pipe, 13 fps)** is the sweet spot; 3×3+1 (435–813 ms) is likely too slow to close a control loop.
6. **Switch yolo11n → yolov8n** (per `runtime-tiling-branch-switch-arch` memory) for higher inf/s on both chips.

## Repro
```
# on Pi
~/hailo-rpi5-examples/venv_hailo_rpi_examples/bin/python \
  BDD/experiments/nv12_tiling_bench/nv12_tiling_bench.py \
  --nx 2 --ny 2 --duration 5 --uncapped [--single] [--maxq 1]
# full matrix:
bash BDD/experiments/nv12_tiling_bench/run_matrix.sh   # -> /tmp/nv12_matrix.txt
```
Raw data: `results/nv12_matrix_2026-06-23.txt`.
