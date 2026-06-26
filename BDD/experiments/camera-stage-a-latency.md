# Camera Stage-A latency reduction (sensor → appsrc)

**Status:** IMPLEMENTED (Fixes 2 & 3) + measured on bench. **Date:** 2026-06-17 (design), 2026-06-23 (impl).

---

## Results / implementation (2026-06-23, branch `camera-stage-a-latency`)

**Bench rig:** single imx477 on sensor 0 (the box's only camera; ov5647 absent), RGB ball model
`2026-04-07_11n_ball_v2.hef`, well-lit indoor room. Measured with new `--vision-only` mode (no FMU on the
bench can arm, and the pipeline normally waits for the controller "ready" signal) + a new callback Stage-A/B
latency log (p50/p95/p99 every 100 frames) + the existing picamera alive-fps log. Config via new `--config`
override pointing at `config.test-single-imx477.yaml`.

**Key finding — the low-light premise did not reproduce here.** In good light the *baseline already ran at a
steady 30 fps* (auto-exposure picked a short shutter on its own), so the predicted "~15 fps → 150 ms Stage A"
did not occur. Baseline Stage A was already p50 ≈ 21 ms. The implemented fixes therefore won in the **tail /
determinism**, not the median:

| Config (exposure / buffers)        | fps  | Stage-A p50 | Stage-A p99 | Stage-A max | clean? |
|------------------------------------|------|-------------|-------------|-------------|--------|
| baseline: auto-exposure, 3 buffers | 30.0 | ~21 ms      | 24–41 ms    | ~41 ms      | yes    |
| Fix 3: pin 8000 µs, 3 buffers      | 30.0 | ~21 ms      | 22–24 ms    | ~24 ms      | yes    |
| Fix 3+2: pin 8000 µs, 2 buffers    | 30.0 | ~21 ms      | 21–22.6 ms  | ~22 ms      | yes    |

So pinning exposure removes the AE-settle jitter (p99/max 41 → 24 ms) and `buffer_count` 3→2 shaves the
remaining queue-depth tail (24 → 22 ms); cumulatively the p99/max **~halved**. Stage B (appsrc→callback,
incl. inference) ≈ p50 14 ms throughout. No libcamera "no buffers"/dropped-frame warnings at `buffer_count=2`.
Thermal: ~54 °C, ARM 2.4 GHz, no active throttle at 30 fps.

**Light-gating tradeoff + the gain rescue:** at 8000 µs with AE off the gain defaults to 1.0, so a dim indoor
scene underexposes (recorded mean luma ~27/255). The fix is `analogue_gain` (paired with the exposure pin):
raise sensor gain instead of lengthening the shutter. Pinning `analogue_gain: 8.0` lifted the same scene to
mean luma **~109/255** (well exposed) with **Stage-A latency unchanged** (p50 ~21 ms, p99 ~22–23 ms, 30 fps) —
gain is a pure amplifier, it doesn't touch integration time. So a short, low-latency shutter is now usable
well into dim light by trading gain (more noise) for brightness. 8000 µs is still a *daylight/sky* exposure;
production `config.yaml` defaults `exposure_time_us: 0` / `analogue_gain: 0.0` (auto) and the operator tunes
both per field light. Gain only applies when AE is off (`exposure_time_us > 0`); set otherwise it's warned and
ignored by the AGC.

**Auto-estimate-then-pin (`exposure_auto_pin_s`, recommended over a hand-tuned fixed pin).** Run AE for a
short warmup, then read back its converged ExposureTime/AnalogueGain and pin them — scene-adapted *and*
deterministic/short. The AE estimate is a guide, clamped by `exposure_min_us` / `exposure_max_us` / `gain_max`;
when AE wants a longer exposure than `exposure_max_us`, the clipped light is shifted into gain (up to
`gain_max`) so brightness is preserved. Re-runs on each camera (re)activation so a switched-to camera adapts
to its own scene.

This also corrected a wrong assumption: the bench room is actually **dim**. Given 1 s, AE converged to
**33 ms @ gain 6.74** (i.e. baseline "30 fps" was AE pushing the shutter to the frame-duration limit, not
abundant light). With `exposure_auto_pin_s: 1.0, exposure_max_us: 10000, gain_max: 16`:
`measured 33013 µs / 6.74 → pinned 10000 µs / 16.0` (shutter 3.3× shorter, light moved to gain). Result: mean
luma **~149/255**, **fps steady 30**, Stage-A **p99 ~22 ms** after warmup, clean. So the operator sets a
shutter ceiling (latency/fps guard) + gain ceiling (noise guard) once, and the camera self-exposes within them.

**What shipped (config-driven; exposure/gain knobs live in the optional `Config.Camera.autoexposure`
section — set it to `null` to disable; `buffer_count` stays directly on `Config.Camera`):**
- `exposure_time_us` (0 = auto; >0 = AE off + pinned shutter). Plumbed CameraSwitcher → picamera_thread; AE
  stays off across camera activation when pinned. (Fix 3)
- `analogue_gain` (0 = AGC; >0 = pin sensor gain, only with AE off) — rescues a short pinned shutter in dim
  light by raising gain instead of the shutter; brightness-only, no latency cost. (Fix 3b)
- `exposure_auto_pin_s` + `exposure_min_us` / `exposure_max_us` / `gain_max` — auto-estimate exposure over a
  warmup then pin it clamped to the limits (shutter ceiling preserves fps/latency, gain absorbs the rest).
  Re-pins on camera (re)activation. Supersedes the fixed pin. (Fix 3c)
- `buffer_count` (default 2, `Range(min=2)` enforces the floor — 1 is rejected). (Fix 2)
- Harness: `app.py --vision-only`, `--config PATH`, callback Stage-A/B percentile log, picamera alive log now
  prints actual ExposureTime/AnalogueGain. `config.test-single-imx477.yaml` for the bench box.

**Fix 1 (capture 30 / drop surplus newest-first) — deliberately NOT implemented.** It only pays off when
inference runs *slower* than capture (then 30 conversions/s are wasted on frames the leaky downstream queues
drop). On this rig inference keeps pace at 30 fps (the callback logs at the same cadence as capture), so there
is no surplus and no thermal pressure — implementing a drop path now would be untested speculative complexity.
It becomes relevant when inference drops below capture, i.e. the **2×2 tiling** task (`runtime-tiling-branch-
switch.md`), and should be built + validated there.

---

**Original design (2026-06-17):**

**Status:** design / analysis (not yet implemented). **Date:** 2026-06-17.
**Owner concept:** part of the capture→command latency campaign (see memory `latency-budget-and-fixes`).
**Why this is its own doc:** Stage A is the **dominant** remaining latency and the **prerequisite** for every
other latency-spending feature (e.g. tiling — see `runtime-tiling-branch-switch.md`). Nothing here may be lost.

---

## TL;DR

Stage-A latency ≈ **pipeline depth (in frames) × frame-interval**. Cut BOTH factors together:

1. **Raise capture to 30 fps + drop surplus frames newest-first** (inference rate stays ~15 fps).
2. **`buffer_count` 3 → 2** (the floor — **NOT 1**; 1 breaks double-buffering).
3. **Pin exposure short** (`exposure-time-mode=manual`, ~8000 µs) — this is what *enables* 30 fps and also
   independently cuts latency.

Expected: **Stage A ~135–170 ms → ~60–70 ms** (`3 buf × 1/15 s = 200 ms → 2 buf × 1/30 s = 67 ms`).
Hard gate: enough light to use a short shutter. Hard rule: drop surplus frames *cheaply* (request-level,
skip the numpy conversion) or the doubled ISP/CPU load heats the no-cooling Pi 5 and the win is lost.

---

## Why it matters (budget framing)

- Control loop needs **capture→command e2e ≈ 50–120 ms**.
- Current p50 ≈ **234 ms**, broken down: **Stage A ≈ 135–170 ms (dominant)** + Stage B ≈ 30 ms (GStreamer +
  1 inference) + Stage C ≈ 65 ms. Stage A alone nearly eats the entire target budget.
- e2e = StageA + StageB + StageC; all stages **add into one fixed envelope**. Any latency-spending feature
  (tiling adds +45–60 ms for 2×2) is unaffordable until Stage A is cut. **Sequence: fix Stage A first.**
- Cameras: OV5647 noir (wide) + IMX477 (zoom), **dual-camera**, currently effective **~14–18 fps**. Stage A ≈
  ~2 frame-intervals of sensor/ISP depth → at ~15 fps a frame-interval is ~55–70 ms, so each buffer of depth
  is expensive.

## Root model

`StageA ≈ depth(frames) × frame_interval`. Two independent multipliers:
- **depth** = how many frames are in flight between sensor and app = `buffer_count` (FIFO delivery, see below).
- **frame_interval** = `1 / capture_fps`.

Raising capture rate is the stronger lever because it shrinks *every* frame-interval-based delay in the whole
pipeline (not just the camera pool), and makes the inferred frame younger. Shrinking buffers caps worst-case
queue depth. They multiply: `3×(1/15)=200 ms → 2×(1/30)=67 ms`.

---

## Fix 1 — capture 30 fps, drop surplus newest-first

- picamera2/libcamera delivers completed requests **in capture order (FIFO), not newest-first**. So buffered
  frames age by `depth × interval` before the app reads them.
- Run the sensor at **30 fps** but keep inference at its current ~15 fps; **drop the excess**. Two effects:
  1. every frame-interval delay halves (33 ms vs 67 ms);
  2. the frame inference actually consumes is **≤ 33 ms old instead of ≤ 67 ms** — fresher target position.
- Inference throughput is unchanged; this is purely a *freshness/latency* win, not a throughput change.
- **Dropping MUST be newest-first / keep-latest** (the existing leaky-downstream GStreamer queues already do
  this). Never FIFO-drain a backlog of stale frames.

## Fix 2 — `buffer_count` 3 → 2 (NOT 1)

- `buffer_count` = size of the DMA pool the sensor→ISP (PiSP) fills in rotation; it caps the in-flight queue
  depth (= worst-case frames of staleness). Fewer → lower worst-case Stage-A latency. 3→2 saves ~one
  frame-interval.
- **2 is the floor.** It preserves **double-buffering**: the sensor DMAs buffer #2 while the app holds #1.
- **1 is harmful, not better.** With one buffer there is no second buffer to fill while the app holds the only
  one → the PiSP pipeline either **stalls** (waiting for the buffer back, jittery cadence) or **drops** the
  next frame. Effective fps falls (often ~halves) → delivered frames are **staler** and the **p99 tail gets
  worse** from stall jitter. You give back (and exceed) the latency you "saved." The mechanism guarantees 1
  cannot win — don't bother testing it.
- Dual-camera contention makes a 1-deep pool even more fragile.

## Fix 3 — pin exposure short (the enabler)

- The camera is at ~15 fps almost certainly because **auto-exposure picks a long integration time** in low
  light (~1/15 s ≈ 66 ms shutter). You physically **cannot run 30 fps with a 66 ms exposure** (30 fps needs
  shutter ≤ 33 ms). So **pinning exposure short is the prerequisite that unlocks Fix 1.**
- Set `exposure-time-mode=manual exposure-time≈8000 µs` (already added on the libcamerasrc path; the
  **picamera2/appsrc path does NOT pin exposure yet** — that's the gap to close).
- Independent latency benefit: a 66 ms integration **smears the "moment" across 66 ms** (effective timestamp
  ~33 ms stale); an 8 ms exposure **centres it ~4 ms before readout**. Short shutter also removes AE-settle
  jitter (helps p99) and speeds AE/AWB convergence.
- **Hard constraint:** short exposure needs **enough light**. If the scene is too dark to expose well at
  ≤ 33 ms (ideally ~8 ms), 30 fps is not achievable — this is the gating risk to validate first.

---

## Implementation rules (do not skip)

- **Drop surplus frames CHEAPLY.** Capturing 30 fps but converting all 30 to numpy then dropping half doubles
  the ~13 MB/s memcpy + ISP/CPU load for frames you discard. On the no-cooling Pi 5 (`no-cooling-thermal-
  throttle-by-design`) that extra heat → throttling → latency back. Rule: re-queue the surplus buffer **at the
  request level WITHOUT `make_array()` / numpy copy** (same idea as the existing L1 "skip make_array on inactive
  camera"). Keep conversions at ~15/s, just sourced from a 30 fps stream so the chosen frame is fresher.
- **Sensor mode:** pick a mode that **natively does 30 fps @ 1280×720** (binned/cropped) so PiSP isn't
  throttling or upscaling to hit the rate.
- **Dual-camera:** keep the inactive camera at reduced rate via the existing `INACTIVE_FPS` (currently 15);
  only the active camera needs 30 fps. Re-check CSI/ISP bandwidth with one cam at 30.
- **appsrc caps:** capture rate / caps are set once in `GStreamerApp.run()` before producers spawn (L4); update
  the single source of truth (`CameraSwitcher`: width/height/target_fps/video_format) — don't introduce a race.

## Risks & constraints

- **Light level** (primary): not enough light → can't use short shutter → can't sustain 30 fps. Validate first.
- **`buffer_count=2` may not stream cleanly** on the dual-camera PiSP config — if it drops frames, you're stuck
  at 3. Test before assuming.
- **Thermal:** even with cheap drops, 30 fps capture raises ISP load vs 15; watch throttling.
- Don't try `buffer_count=1` — mechanism says it loses.

## Validation / measurement plan

Set `buffer_count=2`, capture 30 fps, exposure pinned, cheap newest-first drop. Then measure:
1. **Delivered fps vs 30** (is the sensor actually sustaining it?).
2. **`detection delay` p50 AND p99** (the `!!! LATENCY` line / `GOT DETECTIONS`) — confirm Stage A dropped and
   the tail didn't get worse.
3. **libcamera "no buffers" / dropped-frame warnings** — must be clean; any spam means the pool is too shallow.
4. Sweep: 3 buf @15 (baseline) → 2 buf @30 (target). Optionally 3 buf @30 to isolate the rate effect.

## Code locations

- `BDD/app.py`: `picamera_thread` (~line 884), `CameraSwitcher` (~399–441) — capture loop, where surplus frames
  must be dropped cheaply and where target_fps/exposure live.
- `BDD/app_base.py`: `GStreamerApp.run()` sets appsrc caps once; `update_fps_caps()` (~427–459) toggles
  videorate/capsfilter; appsrc `set_property` sites (~537–545, 693–694, 774–779).
- Existing related fixes (in git, branch `dual-camera-latency-opt-2026-05-27`): L1 skip make_array on inactive
  cam, L2 `INACTIVE_FPS` 5→15, L4 single-source-of-truth caps, `new_wrapped_full(bytes)` half-copy, dropped
  no-op videoscale/convert/rate on appsrc source.

## Expected outcome

Stage A **~150 ms → ~60–70 ms**. That recovered headroom is the budget that later makes a 2×2 tiling branch
(+45–60 ms) affordable within the 50–120 ms e2e target. This task **gates** `runtime-tiling-branch-switch.md`.
