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

## LIVE CAMERA retest (2026-06-23, after reboot — imx477 back online)
Camera recovered after a Pi reboot. Re-ran on the **real imx477** (NV12 via PiSP ISP, 1280×720@30, exposure pinned 8 ms). `capP50` = **capture→aggregator** latency (buffer PTS vs pipeline clock) = true sensor→detection e2e.

- ✅ **Whole-frame NV12 capture + inference works**: **28 fps, 58 ms e2e**, zero drops (210/210). Comfortably inside the 50–120 ms target.
- ✅ **Tiling works** on the real camera — but needed a fix (see DMABUF gotcha below).

| config | inf/frame | FPS | pipe p50 | **e2e p50 (capture→det)** | e2e p90 | drops |
|--------|-----------|-----|----------|---------------------------|---------|-------|
| whole-frame (1×1) | 1 | **28.0** | 17 | **58** | 53→58 | 0 |
| 2×1+1 round-robin | 3 | 18.3 | 204 | **296** | 310 | ~3% |
| 2×2+1 round-robin | 5 | 9.5 | 413 | **506** | 520 | ~5% |
| 3×2+1 round-robin | 7 | 6.3 | 548 | **644** | 660 | ~7% |
| 3×3+1 round-robin | 10 | 4.1 | 705 | **802** | 815 | 10% |
| *single tiling 2×1 (2 crops)* | 2 | 29.7 | 32 | 75 | 76 | 0 |
| *single tiling 2×2 (4 crops)* | 4 | 16.5 | 234 | 326 | 339 | ~4% |
| *single tiling 3×2 (6 crops)* | 6 | 10.9 | 315 | 406 | 421 | ~5% |
| *single tiling 3×3 (9 crops)* | 9 | 7.2 | 405 | 494 | 509 | ~6% |

Camera Stage-A here ≈ **41 ms** for whole-frame (capP50 58 − pipe 17), rising to ~90 ms under tiling load (CPU contention from the RGB-roundtrip copy + deeper source queue). Much lower than the ~150 ms in the dual-camera picamera2/appsrc path — the single libcamerasrc + pinned-8ms-exposure path is leaner. Merging the "+1" into one net-group still applies and would bring 2×2+1 from 506 → ~360 ms e2e / 9.5 → ~13.5 fps.

### ⚠️ DMABUF gotcha (live camera + tiling) — and why the "conversion" is a red herring
`libcamerasrc` hands `hailotilecropper` **DMABUF** memory it can't map → **SIGSEGV** as soon as frames flow (even 1×1). The root cause is **memory type, not pixel format**: `videotestsrc`'s system-memory NV12 fed the cropper with zero conversion. So no format change is *needed* — only a system-memory copy. A passthrough `videoconvert NV12→NV12` does *not* copy (still crashes); forcing a copy fixes it.

Cost of the forced copy is **memory-bandwidth-bound**, ~90 ms under tiling CPU load. The format of the copy barely matters: `NV12→I420→NV12` (YUV-only) = 317 ms e2e vs `NV12→RGB→NV12` = 326 ms — only ~9 ms apart. The benchmark `--camera` path uses the YUV-only copy, but **this is a benchmark workaround, NOT a production pattern.**

**Do NOT capture NV12 → convert → convert back in production.** Correct paths:
- **Whole-frame:** no cropper, no extra copy. `camera → videoscale(640²) → hailonet` — the resize to the HEF input is necessary work and resolves the dmabuf in one pass. Measured pipeline latency **~24 ms** (scale+inference). `hailonet` maps dmabuf NV12 fine on its own.
- **Tiling:** feed **system-memory NV12** so there's no dmabuf to copy. The app's existing **picamera2 → appsrc** path is already system-memory; switch it to deliver NV12 (instead of today's RGB888) and feed the cropper directly — zero copy, zero conversion.
- **Upstream fix:** teach `hailotilecropper` to import/map dmabuf (it already does for whole-buffer `hailocropper` on the appsrc path).

### Reboot/tmpfs note
The reboot that fixed the camera **wiped tmpfs `~/models/`** — had to re-rsync `yolov11n_nv12.hef` from the laptop (`/media/Pets/BDD/_MODELS/experimental/`). Missing HEF → `HAILO_OPEN_FILE_FAILURE(13)` then segfault once frames flow (per deploy-model memory).

## picamera2 → appsrc zero-copy validation (the production-correct tiling source)
Confirmed the recommended fix: **picamera2 delivers system-memory NV12** (1280×720, stride 1280 tightly-packed — `format="NV12"` configures natively), so feeding `appsrc → hailotilecropper` directly has **no dmabuf, no SIGSEGV, no format conversion**, and **no forced copy hop**. The producer does one system-memory copy (numpy → Gst buffer, identical to the app's existing `new_wrapped_full`).

`--picam` mode, live, vs the libcamerasrc+copy hack:
| mode | libcamerasrc + forced copy | **picamera2 → appsrc** |
|------|----------------------------|------------------------|
| 2×2 single | 16.5 fps, 107/111 (drops) | 16.2 fps, **128/128 (no drops)** |
| 2×1+1 | 18.3 fps / 296 ms / drops | 17.2 fps / **253 ms** / no drops |
| 2×2+1 | 9.5 fps / 506 ms / drops | 9.6 fps / **465 ms** / no drops |
| 3×2+1 | 6.3 fps / 644 ms | 6.1 fps / **604 ms** |
| 3×3+1 | 4.1 fps / 802 ms | 3.9 fps / **757 ms** |

Same throughput, ~40 ms lower e2e, and no drops at low tile counts — **and it removes the NV12→…→NV12 waste entirely.** Caveat: the `--picam` `capP50` uses `appsrc do-timestamp` (PTS stamped at push), so it excludes picamera2's internal capture depth and is not directly comparable to the libcamerasrc `capP50` (which carried the sensor PTS); the inference-path `tP50` IS comparable and matches (~415 ms for 2×2). The real, durable win is correctness: production-consistent, conversion-free, drop-free.

**Recommendation for the app:** when running tiling, switch the picamera2/appsrc producer from RGB888 to `format="NV12"` and feed the cropper directly. Whole-frame stays simplest of all (`hailonet`, no cropper).

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
