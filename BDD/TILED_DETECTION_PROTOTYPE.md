# Tiled detection-only pipeline — prototype & findings

Prototype: [tiled_detection_prototype.py](tiled_detection_prototype.py). A parallel, detection-only
pipeline that splits a large pre-tiling frame into a configurable `NxM` grid with **hailotilecropper**,
**batches** all tiles of a frame through `hailonet` (batch-size = N·M), and reassembles + cross-tile
NMS with **hailotileaggregator** — the native-Hailo alternative to the app's producer-side,
one-buffer-per-tile, batch-1 detection mode.

Run on the Pi (inside the hailo venv, `setup_env.sh` sourced):
```
python BDD/tiled_detection_prototype.py --source test --width 1332 --height 990 --tiles 2x2
python BDD/tiled_detection_prototype.py --source test --width 2028 --height 1520 --tiles 3x3 --lean
```

## Hardware reality first

- **Device is HAILO8L, not Hailo-8** (`hailortcli fw-control identify` → `Device Architecture: HAILO8L`;
  the "Board Name: Hailo-8" string is generic). The HEFs are 8L-compiled. A true **Hailo-8 benchmark is
  not possible on this bench** — it needs Hailo-8 silicon *and* an 8-compiled HEF, neither present.
- HailoRT 4.20.0; model = `yolov11n` @ 640×640, on-chip NMS, 5-context HEF.

## Batch benchmark — `hailortcli benchmark` (HW-only, the NPU ceiling), 8L

| batch | tile-inferences/s (hw) | HW latency |
|------:|-----------------------:|-----------:|
| 1  | **79**  | ~12.6 ms |
| 4  | **137** | 24.5 ms |
| 8  | **155** | 44.3 ms |
| 12 | **163** | 64.6 ms |

Batching helps the NPU (4 → 1.7×, 8 → 1.95×, diminishing after 8); latency grows because the batch must
fill before inference. **Hailo-8 would be ~2× these** (26 vs 13 TOPS) but is unmeasured here.

## Pipeline FPS — prototype, test source (RGB), 8L

| config | full-frame FPS | tile-inf/s | note |
|--------|---------------:|-----------:|------|
| 1×1 1280×720 (whole-frame baseline) | **60** | 60 | 1 scale/convert per frame |
| 2×2 1332×990 batch1 | 19 | 77 | |
| 2×2 1332×990 batch4 | **22** | 88 | best 2×2 |
| 2×2 1332×990 batch8 | 21 | 86 | no gain past batch4 |
| 2×2 1332×990 batch4 `--lean` | 22 | 88 | cropper-scale ≈ videoscale |
| 3×3 1920×1080 batch9 | 11 | 99 | |
| 3×3 2028×1520 batch9 | 10 | 92 | |

## The key finding: this pipeline is **CPU-bound, not Hailo-bound**

`tile-inf/s` plateaus at **~88–105** regardless of batch size (4 vs 8) or `--lean`, while the NPU can do
**137–163**. The bottleneck is the **Pi5 CPU doing the per-tile scale + colour-convert** (the
hailotilecropper resize + videoconvert of N tiles per frame), not the Hailo inference. Consequences:

- **Batching buys little end-to-end** here: batch-4 is only ~14% over batch-1 in the full pipeline
  (vs 1.7× HW-only), because the CPU can't feed the NPU fast enough to exploit the batch.
- **`--lean` (let the cropper scale straight to the model, dropping the explicit `videoscale`) does not
  help** — the resize cost is the same wherever it happens.
- **A Hailo-8 upgrade would NOT raise this FPS** — same Pi5 CPU, same scaling bottleneck. Hailo-8 only
  helps once the pipeline is NPU-bound (e.g. far more/larger tiles, or off-loading the resize).
- This native-tiling + batch-4 path (~22 fps for 2×2) is only marginally faster than the current
  producer-side batch-1 detection mode (~20 fps at 2×2 if un-throttled) — **for the same reason** (both
  are CPU-resize-bound).

## Max frame size vs tile grid

Tile size is fixed at the HEF input (640). For native (no-upscale) tiles, the pre-tiling frame should be
≈ `grid × 640`: **2×2 → ~1280–1332**, **3×3 → ~1920–2028** (imx477 sensor modes: 1332×990@120,
2028×1080@75, 2028×1520@54, 4056×2160@20). Bigger frame ⇒ more tiles ⇒ lower FPS (≈ 2× tiles → ½ FPS),
because cost scales with tile count, not pixels.

## NV12-direct: measured, and it's a DEAD END (it's slower)

Hypothesis was to feed the cropper NV12 (12 bpp vs RGB 24) so the resize moves half the bytes. Measured
back-to-back (same thermal state, `--lean`):

| config | RGB | NV12 |
|--------|----:|-----:|
| 2×2 1332×990 batch4 | **28.2 fps** (112 inf/s) | 18.7 fps (74) |
| 3×3 1920×1080 batch9 | **12.2 fps** (110) | 8.9 fps (80) |
| 1×1 1280×720 (no tiling) | 59.6 | 59.2 (equal) |

**NV12 is ~34% SLOWER for tiling.** The hailotilecropper preserves format, so every tile pays a
YUV→RGB colour convert (the model needs RGB) and that per-tile convert costs more than the cheaper NV12
resize saves. With RGB the per-tile convert is a passthrough (~free), so the resize is the only cost.
At 1×1 (no tiling) they're equal, confirming it's the per-tile convert that hurts. **Keep RGB.**

(FPS varies ±~25% with the uncooled Pi5's thermal throttling, so all comparisons here are back-to-back;
the 28-vs-22 spread between sweeps above is mostly temperature.)

## To actually go faster (future work, in rough order of payoff)

The format lever is exhausted (NV12 loses). The resize itself is the irreducible CPU cost, so:

1. **Fewer, larger tiles** when detection range allows — 2×2 is ~2× the FPS of 3×3.
2. **Smaller overlap** (fewer redundant pixels to scale).
3. **Offload the resize from the CPU** — the only structural win, and the hard one: Pi5 GStreamer has no
   GPU videoscale; the `pisp` ISP can emit downscaled outputs but that isn't exposed through
   hailotilecropper. Without this, ~20–28 fps (2×2) is the ceiling regardless of NPU.
4. **batch-size (4–8)** and a **Hailo-8** only matter once the pipeline is NPU-bound — it is not here.

Bottom line: the native-cropper path tops out near the current producer-side detection mode on this
hardware (both CPU-resize-bound). The cropper buys cleaner cross-tile NMS and native batching, not a
step change in FPS — worth it for code simplicity, not for speed.

## Camera path

`--source camera` uses `libcamerasrc` (with `format=NV12` pinned — without it, libcamerasrc negotiates
raw Bayer at exact sensor-mode sizes and fails). It negotiates correctly but **end-to-end flow is flaky**
(libcamerasrc live-source quirks — the same reason the main app uses the **picamera2/appsrc** producer,
see `.claude/notes/latency-failed-steps.md`). The compute ceiling above (test source) is the binding
constraint regardless of capture, so it characterizes real-world FPS. **For integration, drive
hailotilecropper from the existing picamera2 producer** (push whole big frames into a tiling inference
wrapper) rather than libcamerasrc.

## Integration sketch (not done — prototype only)

Add a `hailotilecropper`-based inference wrapper as an alternative to `INFERENCE_PIPELINE_WRAPPER` in
[pipelines.py](pipelines.py), selected when detection mode is on; the picamera2 producer pushes whole
frames at the larger size (RGB — NV12 loses, see above), and the existing detection→pursuit caps flip
(640-tile vs full-frame) is replaced by a cropper-vs-whole-buffer wrapper choice. Don't expect a FPS
win (both paths are CPU-resize-bound); do it for the native cross-tile NMS + simpler producer, and use
2×2 with minimal overlap to stay nearest the 60 fps whole-frame ceiling.
