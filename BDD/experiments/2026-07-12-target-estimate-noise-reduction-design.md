# Target position/velocity noise reduction — design

**Date:** 2026-07-12
**TODO item:** "implement noise reduction on target position and speed estimation"
**Goal:** stop high-frequency noise in the target position/velocity estimate from
corrupting the aim point, without adding aim lag.

## 1. Problem

The aim point buzzes. The buzz gets worse with range, and it turns into misses.

Two separate defects hide behind that one symptom, and they must not be conflated:

- **Noise** — frame-to-frame shake of the detector bounding box.
- **Bias** — a systematic range underestimate at long range (a true 80 m target
  reports ~50 m; 10–40 m is much closer to correct).

No filter fixes bias, and no calibration fixes noise. Both are in scope.

### 1.1 Why the noise compounds instead of cancelling

`target_center` and `target_size` are both derived from *the same four bbox edges*
(`drone_controller.py:912-913`). When an edge shakes:

- the **center** moves → bearing error,
- the **size** changes → range error,

and the two errors are born together, from one cause. Filtering "position" and
"distance" as if they were two independent signals — the naive reading of the TODO
— cannot exploit that. Filtering the *box* once, at the source, fixes both
**coherently**: the smoothed center and smoothed size stay mutually consistent.

### 1.2 Why range noise reaches the aim point at all

The measurement is strongly **anisotropic**:

- **Bearing** (from `bbox.center`) is good — a pixel of jitter is a fraction of a degree.
- **Range** (from `bbox.size` via `estimate_distance.py:117`) is bad, and its error
  is **multiplicative**: `d ∝ 1/apparent_size`, so ±1 px on an 8 px target at 80 m
  is a ±10 m range swing per frame.

So the NED position error is a long cigar along the line of sight. Critically:

> **A purely radial position error is invisible in the aim point.** Reproject it
> through the pinhole model and it lands on the same pixel.

Radial noise becomes lethal only through **velocity**:

1. `TargetEstimator3D` fits a slope through ~15 samples over a 500 ms window
   (`drone_controller.py:521-522`).
2. Differencing a ±10 m radial jitter over 0.5 s manufactures a **phantom radial
   velocity of tens of m/s**, randomly signed each frame.
3. `estimate()` (`TargetEstimator.py:347-358`) returns `raw_newest_pos + v·dt`,
   where `dt` is the lookahead — currently `0.1 × distance` frames.
4. The extrapolated point is off the line of sight, so it **reprojects to a
   different pixel**. P=8 on X turns that into an attitude command.

Note step 4's perversity: **lookahead grows with distance, and so does the noise.**
The design extrapolates longest exactly where the estimate is least trustworthy.

`FOLLOW_TARGET_POSITION_NED` is `false`, so the live path is zenith/attitude and the
aim point is what matters. (In NED-follow mode radial error would *not* be benign —
it would move the commanded point directly. Out of scope while that mode is off.)

### 1.3 Three smaller chatter sources

- **`estimate()` discards its own fit.** It anchors extrapolation on the *raw newest
  sample* and throws away the regression's intercept. Position smoothing today is
  therefore **exactly zero** — only velocity is filtered.
- **No hysteresis on distance class** (`estimate_distance.py:159-176`). A range
  hovering near 10 m flips NEAR/MEDIUM every frame, stepping `thrust` and
  `pd_coeff_p` with it.
- **Optical refinement flip-flop.** `drone_controller.py:915-918` overwrites size and
  center from an Otsu contour, and silently falls back to the raw bbox when the
  contour check fails — a *discrete* step, not smooth noise.

### 1.4 The bug at the heart of it

`bytetrack.py:16` runs a constant-velocity Kalman filter on every track in
`(cx, cy, w, h + velocities)`, and `STrack.bbox` (`bytetrack.py:129-134`) returns
that filtered estimate.

`app.py:390-397` builds every `Detection` from `raw_dets` — the **raw** detector box
— and borrows only the `track_id`.

**The smoothed bounding box is computed, unit-tested, and discarded every frame.**

## 2. The bias model

Hypothesis: the detector's box carries a roughly **constant pixel margin**:

    s_det ≈ s_true + m

Because `d ∝ 1/s`, this produces a scale-dependent range error that matches the
observed curve:

| true range | true size | effect of `+m` | reported range |
|---|---|---|---|
| 80 m | small | dominates | badly short (~50 m) |
| 10–40 m | large | negligible | close to correct |

A *symmetric* margin inflates the size without moving the center — which also
explains why range is biased but bearing is not.

The detector runs on a fixed-size resized input, so a constant margin in
*model-input pixels* is a **constant in normalized bbox coordinates**: one
camera-independent config number, not a per-camera table.

**Correction:** `s_true = max(s_det − m, ε)`, applied inside `estimate_distance`,
**after** filtering and **before** the range computation.

**The sting:** subtracting `m` at long range divides by a smaller number, so
**correcting the bias amplifies the range noise.** The bias fix makes the noise fix
more necessary, not less — and it forces the filter to sit *before* the margin
subtraction. Ordering is not negotiable: **filter `s_det`, then correct.**

When `s_det` approaches `m`, the corrected size is not merely noisy but
*unreliable*. That is a signal to shorten lookahead, never to fail.

## 3. Approach

### 3.1 Phase 0 — metric and truth (prerequisite, not optional)

Every later phase is an A/B against these numbers. No filter is written until they exist.

**Metrics, always reported as a pair:**

- **Jitter** — RMS of the *second difference* of the final aim point across frames,
  i.e. `rms(p[n] − 2·p[n−1] + p[n−2])`. Chosen because it is blind to both a constant
  aim offset and a constant aim *velocity* — the two things we legitimately want the
  estimator to produce — while being maximally sensitive to frame-to-frame shake.
  Reported per axis. Computed identically at every stage (bbox edges → size → distance
  → NED → velocity → aim point), which locates the amplification and confirms or kills
  the phantom-radial-velocity theory of §1.2.
- **Lag** — aim-point lag against target motion. **Non-negotiable companion to
  jitter.** Every fix here trades quiet for lag, and the project's north star is
  capture→command latency: a change that halves jitter while adding 100 ms of aim lag
  is a *worse* miss on a closing target. Any result quoting jitter alone is invalid.

**Corpus:** the 3 UAE 2026-04-27 logs (near-static / cruise / 39 m/s dive), replayed
via `debug_drone_controller.py <log> --headless` with dotted `--params` for A/B.

**Truth for the bias — fit on pixels, validate on motion:**

- **Fit (pixel truth):** hand-annotate the target's true extent on ~30 frames of the
  paired UAE videos, spanning the size range from a few px to filling the frame.
  Regress `s_det` against annotated `s_true`; the intercept *is* `m`. This **tests**
  the constant-margin model rather than assuming it — if the fit wants a
  multiplicative term, or neither shape fits, the theory dies cheaply and early.
- **Validate (relative-motion truth):** global telemetry NED is not trustworthy
  enough to serve as a range reference, but *short-baseline displacement* is. Over a
  closing segment, `d(t)` must shrink by the integrated closing speed. Fit
  `s_det = k/d + m` using only the *change* in range, never its absolute value, and
  confirm it reproduces the `m` fitted from pixels.

### 3.2 Phase 1 — bias

Apply §2. Config: one normalized margin constant. Graceful degradation: if the
corrected size collapses to `ε`, mark the range unreliable and shorten lookahead
rather than emitting a poisoned distance.

### 3.3 Phase 2 — noise at the source

Use `track.bbox` (the Kalman estimate) for matched detections instead of the raw
rect. Config-gated so it can be A/B'd against raw.

**The switch is not the leverage; the retune is.** ByteTrack scales its measurement
noise by box height, so for an 8-px target it assumes sub-pixel measurement accuracy
and barely filters at all. Add a **noise floor** — assume the detector is never
better than ~1–2 px regardless of box size. This makes the filter distrust precisely
the small, far targets that are in fact untrustworthy, and is expected to be the
single highest-value change in this document.

Expose the filter's process/measurement noise weights and the floor as config.

### 3.4 Phase 3 — optical refinement: measured, not assumed

A/B refinement on/off against the jitter+lag metrics. It sits *downstream* of the
tracker's Kalman box, so nothing currently damps its discrete fallback steps.
Outcomes, in order of preference if the numbers permit:

1. **Delete it** — if it is a net noise source at these target sizes.
2. **Re-feed it** — run it on the smoothed bbox crop, one filter only, minimum lag.
3. **Smooth after it** — a second small filter on the final `(center, log size)`.
   `log` because size noise is multiplicative and `d ∝ 1/s`, so a linear filter on
   `log s` is a linear filter on `log d` — the coordinate where the error is
   well-behaved. Cost: two filters in series, whose lag must be budgeted jointly.

The decision is made by the measurement. It is not pre-made here.

### 3.5 Phase 4 — stop the downstream amplification

- **Use the fit.** `estimate()` returns `intercept + slope·dt` instead of
  `raw_newest + slope·dt`. Free position smoothing; no added lag for
  constant-velocity motion; removes the raw sample from the aim path entirely.
- **Physical velocity clamp.** Targets do not do 200 m/s. Clamp and log — never abort.
- **Outlier gate.** Reject/down-weight NED samples implying an impossible jump given
  the current fit. (The cheap version of an EKF's innovation gate.)
- **Distance-class hysteresis.** A band around the NEAR/MEDIUM/FAR boundaries so
  `thrust` and `pd_coeff_p` stop chattering.
- **Lookahead schedule — A/B, don't guess.** Sweep on the replay corpus:
  (a) today's `0.1 × distance`; (b) fixed, equal to measured capture→command latency;
  (c) confidence-scaled (longer when the estimate is trustworthy, shorter when it is
  not — inverting today's coupling). The numbers pick.

### 3.6 Rejected: a full EKF in NED

An EKF with anisotropic measurement noise (σ_range growing as d², σ_bearing small and
constant) is the textbook-correct answer: it structurally refuses to manufacture
radial velocity and gives an innovation gate for free.

Rejected **for now**: it is a large commitment on a control path governed by
*graceful degradation, never hard-fail*, it can diverge, and it inherits telemetry
quaternion quality as a new failure mode. Revisit if Phases 1–4 leave the aim point
too noisy. The metrics from Phase 0 are what would justify it.

## 4. Constraints and invariants

- **Latency is the north star.** Jitter and lag are reported together, always.
- **Graceful degradation, never hard-fail.** Every filter falls back to the previous
  good value or the raw measurement; nothing on the control path throws.
- **Config is values-only.** No live objects in `control_config`; filters are
  constructed explicitly and passed as parameters.
- **Every new config knob is mirrored into `test_config.py`'s `TestConfig`,** with a
  test.
- **Every phase is default-off** until its A/B on the replay corpus says otherwise.

## 5. Validation

- **Off-rig:** replay all 3 UAE logs per phase; report jitter *and* lag; A/B via
  dotted `--params`.
- **Unit tests:** margin correction (including the `s_det → m` collapse), Kalman-bbox
  passthrough, fit-anchored `estimate()`, velocity clamp, outlier gate, class
  hysteresis.
- **On-rig:** verify per the box launch/debug runbook once the replay numbers hold.
