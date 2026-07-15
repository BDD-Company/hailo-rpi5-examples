# Camera Stage-A latency: pin/auto-pin exposure, floor buffers, vision-only bench

**Branch:** `camera-stage-a-latency` → `main` · 13 commits · `BDD/{app.py, app_base.py, config.py,
helpers.py, config.yaml}` + bench config + docs.

## Summary

Stage A (sensor → appsrc) is the dominant, gating stage of the capture→command latency campaign. This PR cuts
its **tail and jitter** and makes the capture path **scene-adaptively low-latency** by controlling exposure
and the DMA pool depth — all config-driven, defaulted on, and validated on the Pi.

The headline lever is **auto-estimate-then-pin exposure**: let auto-exposure converge for a short warmup, then
*pin* the measured shutter/gain (clamped to limits). You get AE's scene adaptation **and** a deterministic,
short shutter (no ongoing AE jitter), with a shutter ceiling that protects the frame rate.

## What changed

Exposure/gain knobs live in an optional nested section `camera.autoexposure` (set to `null` / `enabled: false`
→ plain auto-exposure). All durations are **integer milliseconds** (`_ms`).

| Knob (`camera.autoexposure`) | Effect |
|---|---|
| `exposure_auto_pin_ms` | warmup AE this long (startup + each camera activation), then pin the measured exposure/gain — clamped. Supersedes the fixed pin. **The recommended mode.** |
| `exposure_min_ms` / `exposure_max_ms` | clamp the pinned shutter. `exposure_max_ms` is the fps/latency guard; when AE wants longer, the clipped light is shifted into gain. |
| `gain_max` | ceiling on the compensation gain (noise guard). |
| `exposure_time_ms` / `analogue_gain` | fixed manual pin (used when `exposure_auto_pin_ms == 0`). |
| `camera.buffer_count` | picamera2 DMA pool depth; default **3→2** (floor, `Range(min=2)` rejects 1). |

Supporting changes:
- **Producer (`picamera_thread`)**: applies the fixed pin, runs the warmup→pin state machine (timed from
  camera activation, re-pins on re-activation), and clamps with brightness-preserving gain compensation. Reads
  its config straight off the validated **`Config.Camera`** object (`app.camera_settings`) — converting ms →
  picamera2-native µs / seconds at the boundary.
- **Separation of concerns**: `CameraSwitcher` carries only runtime active-camera state + the shared appsrc
  caps; the static exposure/buffer knobs come from `Config.Camera`, not mirrored onto the switcher.
- **Bench harness**: `app.py --vision-only` (run the vision pipeline without the FMU/controller-ready gate),
  `--config PATH` override, a callback **Stage-A/B percentile latency log**, the picamera alive-log now prints
  actual ExposureTime/AnalogueGain, and `config.test-single-imx477.yaml` for the single-camera bench box.

## Results (measured on Pi 5 + imx477 bench, vision-only)

**Key finding:** on this rig the baseline *already ran at a steady 30 fps* (AE picks a short-ish shutter on its
own), so the predicted "15 fps → 150 ms Stage A" did not reproduce — the win is in the **tail/determinism**,
not the median.

| Config | fps | Stage-A p50 | Stage-A p99 | max | clean? |
|---|---|---|---|---|---|
| baseline: auto-exposure, 3 buffers | 30.0 | ~21 ms | 24–41 ms | ~41 ms | yes |
| pin + 2 buffers | 30.0 | ~21 ms | 21–23 ms | ~22 ms | yes |

- Pinning removes AE-settle jitter (p99/max **41 → 22 ms, ~halved**); `buffer_count` 3→2 shaves the remaining
  queue-depth tail. No libcamera "no buffers"/dropped-frame warnings at `buffer_count=2`. ~54 °C, no throttle.
- **Auto-pin in dim light:** AE converged to **33 ms @ gain ~6–10** (the room is genuinely dim — baseline
  "30 fps" was AE pinned to the frame-duration limit). Auto-pin clamped to **10 ms @ gain 16**: shutter 3.3×
  shorter, light moved into gain, recorded mean luma ~149/255 (well exposed), 30 fps, Stage-A p99 ~22 ms.
- **Gain rescue:** a short pinned shutter that would otherwise underexpose (mean luma ~27) lifts to ~109–149
  via `analogue_gain`/`gain_max` — **latency unchanged** (gain is a pure amplifier).

**e2e latency estimate.** Stage-A `SensorTimestamp` is stamped at end-of-exposure (verified: the Stage-A metric
is flat across 8/10/30 ms shutters), so the exposure-centering benefit — `exposure/2` — sits *outside* the
measured Stage-A number. Including it: ~**10–15 ms p50 / ~30 ms p99** win in adequate light; ~**60–90 ms** in
the low-light / dual-camera-contended regime the feature targets (a short shutter holding 30 fps instead of AE
drifting to a long exposure / low fps).

## Defaults

Production `config.yaml` ships **auto-pin enabled**: `exposure_auto_pin_ms: 500`, `exposure_max_ms: 10`,
`gain_max: 16`, `buffer_count: 2`. A commented **dim-light profile** (`gain_max: 22`) is included for a
one-line toggle. In bright field/sky AE picks a short exposure at low gain, so the caps don't bind.

## Not in this PR (scope)

- **Fix 1 (capture 30 / drop surplus newest-first):** deliberately **not** implemented — verified inference
  keeps pace at 30 fps on this rig, so there is no surplus to drop. It becomes relevant only when inference
  drops below capture (the 2×2 tiling task) and is **explicitly deferred to that work** (see memory
  `fix1-drop-surplus-deferred-to-tiling` and the tiling experiment doc's prerequisite note).
- **Field light tuning:** the right `exposure_max_ms`/`gain_max` depend on real scene light (noise vs
  brightness); the defaults are bench-tuned and safe, but want an on-scene check.
- **Multi-camera re-pin-on-activation:** implemented but only the single-camera startup path is bench-tested
  (the box currently has one camera). Worth a check when the ov5647 (wide) is connected — it also has a lower
  max analogue gain (~8×), so `gain_max: 16/22` clamps to that and it may read darker in dim light.

## Testing

- `pytest BDD/test_config.py` — 78 passed (schema, nested-optional parse, int/float enforcement,
  `null`/`enabled:false` → None).
- On-device vision-only bench at every step (exposure pin, gain, auto-pin, buffer_count, ms→int refactor,
  Config.Camera sourcing), confirming delivered fps, Stage-A/B percentiles, exposure/gain readback,
  recorded-frame brightness, and a clean libcamera stream.

## Commits

```
b48f844 feat: config-driven manual exposure pin + vision-only bench harness
8c042b5 feat: config-driven buffer_count, default 3->2 (floor enforced)
fc6012e docs: record implementation + bench results
01abe66 feat: add analogue_gain knob to rescue dim scenes (Fix 3b)
cdd37c1 feat: auto-estimate-then-pin exposure with limits (Fix 3c)
185f709 feat: enable auto-pin by default with scene-tuned limits
e82caca refactor: group exposure/gain knobs into Camera.autoexposure
b2b950e docs: add commented dim-light autoexposure profile
9ae53b3 refactor: unify AutoExposure time fields to milliseconds
0840be7 refactor: make AutoExposure _ms time fields integers
261f0f4 docs: tidy AutoExposure field grouping + comments
dd18839 minor: comments
c6af916 refactor: read exposure/buffer_count from Config.Camera, not CameraSwitcher
```
