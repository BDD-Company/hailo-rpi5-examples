# Lower priority for targets below the horizon

Design, 2026-07-12. Task: `BDD/TODO.md`, "targets below the drone must have lower
priority that targets above the drone". Nothing is implemented yet — this document is
the handoff for that work.

## Problem

`_pick_target_detection()` (`BDD/drone_controller.py:154`) chooses what the drone
chases, and its entire policy is *the most confident detection wins*:

```python
pool = [d for d in detections if d is not None and d.confidence >= confidence_min]
...
return max(pool, key=lambda d: d.confidence)
```

Geometry plays no part in it. A confident false positive on the ground — a bush, a
vehicle, a person, a glint — outranks a genuine but less confident airborne target,
and on the launch pad it can be the *only* thing in frame. The drone is an
interceptor: a target below the horizon is either clutter or something we should not
be diving at, and the cost of latching onto one at launch is a flight aimed at the
ground.

The detection's elevation relative to the true horizon is what matters, not its
position in the image. A nose-down drone sees sky at the bottom of the frame. So the
test has to be attitude-corrected, which is what the TODO means by "adjust for drone
nose attitude".

## Approach

Score each candidate detection by its *effective* confidence — raw confidence times a
penalty that depends on whether the detection's ray points above or below the horizon
— and rank on that instead of on raw confidence. Before takeoff, and briefly after it,
a below-horizon detection is not merely demoted but removed from the pool: on the pad
there is no cost to refusing to launch, and every cost to launching at a bush.

### The horizon test reuses existing, validated geometry

`telemetry_position.project_camera_to_ned()` (`BDD/telemetry_position.py:160`) already
converts a bbox centre + FOV + attitude quaternion into an absolute NED position, and
it is the single place that encodes the camera mounting convention:

> Camera is mounted pointing up (-Z in FRD body frame). Frame bottom = drone front,
> frame left = drone left.

Call it with `distance_m = 1.0` and `drone_pos = PositionNED(0, 0, 0)` and what comes
back is the detection's unit ray in the world frame. Its `down_m` component is the
whole answer:

    down_m > 0  =>  the ray points downward  =>  below the horizon

That is the entire test. Roll, pitch and nose attitude are handled because the
quaternion handles them.

Writing fresh trigonometry for this would mean a *second* copy of the mounting
convention, free to drift out of sync with the first. Reusing
`project_camera_to_ned` means the horizon test and the 3D estimator are either both
right or both wrong, and the 3D estimator is exercised on every flight. That is worth
more than the handful of multiplications the unit-distance call wastes. Cost is
negligible regardless: one quaternion rotation per detection per frame, on a pool that
is rarely more than a few detections.

The quaternion comes from `telemetry["odometry"]["q"]`, via
`get_orientation_quaternion()` — **not** from `attitude_euler`, and not from a
telemetry aspect named `attitude_quaternion` (no such key exists in this codebase; do
not go looking for one). The 2026-04-27 UAE replay logs carry it, so the whole feature
is verifiable off-rig.

### Phases

Behaviour is a function of one thing: how long ago the drone took off.
`takeoff_time_ns` (`BDD/drone_controller.py:507`, set at `:863`) already tracks this,
and is `None` until the controller first commands movement.

| Phase | Condition | Below-horizon detection |
|---|---|---|
| Pre-takeoff | `takeoff_time_ns is None` | **Excluded from the pool entirely** |
| Early flight | `flight_time_ns < early_stage_duration_s` | **Excluded from the pool entirely** |
| Rest of flight | after that | `score = confidence * confidence_multiplier` |

Above-horizon detections always score their raw confidence, in every phase.

The early window is simply how long the pre-takeoff hard-ignore survives past liftoff;
it is not a second, softer penalty. This is why the TODO's fourth knob,
`early_stage_multiplier`, **is deliberately not implemented** — with the window
covered by an outright exclusion there is nothing left for a multiplier to scale.
Adding it would be a config field that does nothing, which is worse than no field.

Note that the existing post-takeoff window (`takeoff.duration_ns`, `SAFE_TAKEOFF_PERIOD_NS`)
is a *different* mechanism that shapes thrust and PD gains. The two windows are
independent and must not be conflated: `early_stage_duration_s` belongs to this
feature and touches only target selection.

### The penalty ranks, it never rejects (in flight)

The `confidence >= confidence_min` admission gate keeps testing **raw** confidence.
The multiplier decides who wins among candidates; it can never push a detection under
the gate and out of the pool. A below-horizon target that is the *only* target in
frame is still tracked, at full confidence.

This is deliberate, and it follows the project's standing rule that the control path
degrades rather than hard-fails (memory: `graceful-degradation-never-hard-fail`).
Multiplying before the gate would mean a 0.45-confidence detection with an 0.8
multiplier scores 0.36, falls under `confidence_min = 0.4`, and vanishes — leaving a
flying drone with nothing to chase. Deprioritising a target is safe; blinding the
controller is not.

The pre-takeoff exclusion is not a contradiction of this. Rejection is only dangerous
once airborne. On the ground, "no acceptable target, so do not launch" is the correct
and safe outcome.

### Missing telemetry

The horizon is undefined without an attitude quaternion, and each phase fails to the
safe side:

- **In flight, no fresh quaternion** (MAVSDK dropout): no penalty, no exclusion. Every
  detection scores its raw confidence and the feature silently no-ops for that frame.
  A stale attitude is worthless in flight — the airframe rotates fast — so the code
  declines to guess.
- **Pre-takeoff, no fresh quaternion**: fall back to the **last known** quaternion. The
  drone is sitting on the ground; its attitude has not meaningfully changed. This
  requires caching the most recent good quaternion — the only piece of state the
  feature owns.
- **No quaternion has ever been seen** (no telemetry at all, replay log without
  odometry): the feature disables itself completely. Nothing is excluded, nothing is
  penalised. It must never be possible for a missing telemetry stream to reject every
  detection.

### ByteTrack target lock

`_pick_target_detection` narrows the pool to the locked `track_id` *before* ranking, so
once a lock is held the multiplier has nothing to compare against and does nothing.
That is the intended behaviour: **the horizon rule shapes lock acquisition, not lock
retention.**

- While the lock is free, exclusions and penalties decide which track gets latched — so
  a ground target cannot be locked on the pad.
- Once locked, the lock is honoured even if the target descends below the horizon. A
  real target being dived on legitimately ends up below the drone, and breaking the
  lock at that exact moment is the most expensive possible time to lose it.

## Components

**`telemetry_position.is_below_horizon(...) -> bool`** — new, pure, ~10 lines. Wraps
`project_camera_to_ned` at unit distance from the origin and returns `down_m > 0`.
Signature mirrors `project_camera_to_ned`'s (centre, aim point, FOV, quaternion), minus
distance and position.

**`BDD/target_priority.py`** — new module, the only place that knows about phases. Holds
the last-known-quaternion cache and exposes the scoring function that maps a
`Detection` to an effective score, or to `None` meaning "exclude". Keeping this out of
`drone_controller.py` matters: that file is already ~1300 lines, and this logic is
pure, phase-driven and heavily table-testable, which is exactly the kind of thing that
should not be buried in the control loop.

**`_pick_target_detection()`** — gains an optional `score_fn` parameter, defaulting to
`None` (current behaviour exactly). When supplied: the `confidence_min` gate still
tests raw confidence, detections whose `score_fn` returns `None` are dropped, and
`max()` ranks on the score. The ByteTrack lock filter stays exactly where it is.

**Call site** (`BDD/drone_controller.py:889`) — passes a `score_fn` closed over the
current phase, the frame's `CameraConfig` FOV, `AIM_POINT`, and the quaternion from
this iteration's telemetry. Per the project rule that config dicts hold values only
(memory: `no-active-objects-in-config-dicts`), the live objects are explicit
parameters, not config entries.

**Debug output** — a `below_horizon_active` flag in `debug_info` (was the rule actually
scoring this frame, or had it degraded to a no-op?), plus one INFO line when the rule
leaves no candidate at all. That is enough to read the A/B off a replay log. Deliberately
no per-detection elevation field: this is the hot control thread, and a float per
detection per frame buys nothing the flag and the log line do not already give.

## Config

New optional top-level section, following the `optical_refinement` / `speed_reduction`
pattern: absent, or `enabled: false`, means the parser returns `None` and the feature
is off. Check `config.below_horizon is not None` — never `.enabled`, which is a
file-only toggle and not a dataclass field.

```yaml
# Deprioritize targets below the horizon (attitude-corrected, not image-relative).
below_horizon:
  enabled: true
  # Effective confidence of a below-horizon detection, once the early window has
  # passed: score = confidence * confidence_multiplier. Ranking only -- it can never
  # push a detection under confidence_min.
  confidence_multiplier: 0.8    # [0.5 .. 1.0]
  # How long the pre-takeoff hard-ignore of below-horizon targets persists past
  # takeoff. 0 ends it at liftoff.
  early_stage_duration_s: 1.0   # [0 .. inf]
```

Constraints: `confidence_multiplier` is `Annotated[float, Range(0.5, 1.0)]`,
`early_stage_duration_s` is `Annotated[float, Range(min=0.0)]`. Both defaults come from
the TODO.

This adds no new *parser capability* — optional sections and `Range` bounds already
exist — so `test_config.py`'s synthetic `TestConfig` (which mirrors parser features,
not the real `Config`) needs no change. What is required is a parse test for the real
section, and adding `below_horizon` to `test_config_has_optional_toggle_sections`. The
existing `test_consumers_gate_on_presence_not_enabled` already covers the new section
automatically: it enumerates every `Optional[<dataclass>]` field and fails any consumer
that reads `.below_horizon.enabled`.

## Testing

**Unit — the horizon test.** Hand-built quaternions with known answers: level hover
(camera up, frame centre well above the horizon); nose-down; rolled 90° (the case that
proves the test is not secretly image-relative — the image horizon is vertical, so a
pitch-only implementation gets this wrong); inverted. Detections at frame centre and at
each edge.

**Unit — the scoring table.** Every cell of the phase table above, plus each of the
three missing-telemetry paths, plus: a lone below-horizon target in flight survives at
full confidence (the ranking-only guarantee), and an excluded detection never reaches
`max()`.

**Unit — `_pick_target_detection`.** `score_fn=None` reproduces today's behaviour
byte-for-byte; a `score_fn` reorders the pick; a `None` score excludes; the ByteTrack
lock still short-circuits the pool.

**Replay A/B — the gate.** `debug_drone_controller.py <log> --headless` against all
three 2026-04-27 UAE logs (near-static, cruise, 39 m/s dive — memory:
`uae-replay-log-corpus`), feature off vs on. What to look at: how many detections the
rule excluded pre-takeoff, how many it demoted in flight, and whether the picked target
ever changed. On a real flight the ground must land below the horizon and the target
above it.

**This A/B is a correctness gate, not a formality** — see the risk below.

## Risks

**The sign convention is the whole feature.** If `project_camera_to_ned`'s "camera
points up (−Z in FRD)" docstring is stale — a remount, a sensor swap, an inverted
image — then this feature does the precise opposite of what is wanted: it will
deprioritise real airborne targets and, worse, refuse to launch at them while happily
locking onto the ground. Nothing about the code will look wrong. This is why the
horizon test reuses `project_camera_to_ned` (so any convention error is shared with the
already-flying 3D estimator rather than being a fresh, independent guess), and why the
replay A/B must be read for *sign sanity* before this goes near the rig: on the ground,
before launch, the earth must be the thing below the horizon.

**Telemetry/frame skew.** The quaternion comes from the current iteration's telemetry
while the bbox comes from a frame captured ~100 ms earlier. The existing 3D projection
already has this skew and flies with it, so this feature inherits it rather than
introducing it — but at high angular rates near the horizon it can flip a marginal
detection's classification. The soft multiplier tolerates this (a misclassified target
is demoted, not dropped). The pre-takeoff hard exclusion tolerates it too, because on
the pad angular rates are ~zero. No mitigation planned; noted so the next person does
not rediscover it as a bug.

**Interaction with `takeoff.delay_until_n_detection_frames`.** Excluding below-horizon
detections pre-takeoff also keeps them out of the target estimator's history, so a
launch-pad scene with nothing above the horizon will never accumulate the N detection
frames that gate takeoff, and the drone will simply not launch. That is the intended
behaviour, but it *will* look like a hang to someone bench-testing indoors with a
target on a table. Worth a log line at INFO when the rule excludes every candidate.

---

# Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank candidate detections by an attitude-corrected *effective* confidence, so
targets below the horizon lose to targets above it — and are ignored outright until the
drone is airborne.

**Architecture:** One new pure geometry helper (`is_below_horizon`, a thin wrapper over
the existing `project_camera_to_ned`), one new module holding the phase policy
(`target_priority.HorizonPriority`), an optional `score_fn` parameter on
`_pick_target_detection`, and ~15 lines of wiring in the control loop. No new
dependencies.

**Tech stack:** Python 3.11, `pytest`, existing `parse_config`/`Config` dataclass
schema. Everything here runs on the host — no Hailo, no GStreamer, no MAVSDK.

## Global constraints

- All commands run from the `BDD/` directory. Tests: `python -m pytest <file> -v`.
  Baseline before starting: `python -m pytest -q` → the suite passes except the known
  pre-existing failure `test_OverwriteQueue.py::test_fifo_when_not_overwritten` (see
  `TODO.md`; it is not a regression, do not chase it).
- Imports in `BDD/` are flat: `from config import Config`, never `from BDD.config import ...`.
- The optional-section contract: a feature behind `Optional[<dataclass>]` is gated on
  `config.below_horizon is not None`. **Never** read `.below_horizon.enabled` —
  `test_consumers_gate_on_presence_not_enabled` scans `drone_controller.py`'s source and
  will fail the build if you do.
- Config carries values only. Live objects (`HorizonPriority`, quaternions, FOV) are
  explicit function parameters, never entries in a config dict.
- The control path degrades, it never hard-fails. Any new code on the per-frame path
  that can raise must be caught and fall back to "feature off for this frame".
- Config defaults, verbatim: `confidence_multiplier = 0.8` (range `[0.5, 1.0]`),
  `early_stage_duration_s = 1.0` (range `[0, inf)`).

---

## Task 1: `is_below_horizon()` geometry helper

**Files:**
- Modify: `BDD/telemetry_position.py` (add after `project_camera_to_ned`, which ends at
  `:229`)
- Create: `BDD/test_target_priority.py`

**Interfaces:**
- Consumes: `project_camera_to_ned`, `PositionNED`, `Quaternion` (all already in
  `telemetry_position.py`).
- Produces: `is_below_horizon(detection_center_x, detection_center_y, aim_point_x,
  aim_point_y, fov_h_deg, fov_v_deg, quaternion) -> bool`. Task 3 calls this.

**Why this shape:** the ray math and the camera mounting convention already live in
`project_camera_to_ned`. Calling it at unit distance from the origin turns it into a
direction vector, and the sign of `down_m` is the answer. Do **not** re-derive the
trigonometry — a second copy of the convention is free to drift out of sync with the
first, and that failure mode is silent and inverts the whole feature.

- [ ] **Step 1: Write the failing tests**

Create `BDD/test_target_priority.py`:

```python
#!/usr/bin/env python3
"""Host tests for the below-horizon target-priority rule.

No drone, no hardware — pure geometry and pure scoring policy.
"""

import math

import pytest

from telemetry_position import Quaternion, is_below_horizon

# Camera FOV of the rig's imx477 entry in config.yaml, and the default aim point.
FOV_H, FOV_V = 107.0, 85.0
AIM_X, AIM_Y = 0.5, 0.5


def _q_axis_angle(ax: float, ay: float, az: float, deg: float) -> Quaternion:
    """Body->NED quaternion for a rotation of `deg` about the unit axis (ax, ay, az)."""
    half = math.radians(deg) / 2.0
    s = math.sin(half)
    return Quaternion(w=math.cos(half), x=ax * s, y=ay * s, z=az * s)


# Level hover: the camera looks along body -Z = straight up, so EVERYTHING in frame
# is above the horizon.
Q_LEVEL = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
# Rolled 180 degrees: the camera looks straight down, so everything is below it.
Q_INVERTED = _q_axis_angle(1.0, 0.0, 0.0, 180.0)


def _below(cx, cy, q) -> bool:
    return is_below_horizon(cx, cy, AIM_X, AIM_Y, FOV_H, FOV_V, q)


def test_level_hover_puts_frame_centre_above_horizon():
    assert _below(0.5, 0.5, Q_LEVEL) is False


def test_level_hover_puts_every_frame_corner_above_horizon():
    # The camera points straight up and the FOV is < 180 deg, so no pixel can be
    # below the horizon. This is the guard against an image-relative implementation:
    # a "bbox is in the lower half of the frame" rule would call (0.5, 0.9) below.
    for cx, cy in ((0.5, 0.9), (0.5, 0.1), (0.05, 0.95), (0.95, 0.05)):
        assert _below(cx, cy, Q_LEVEL) is False, f"({cx}, {cy}) should be above"


def test_inverted_puts_frame_centre_below_horizon():
    assert _below(0.5, 0.5, Q_INVERTED) is True


def test_roll_90_splits_the_frame_across_the_horizon():
    # Rolled 90 deg the boresight lies ON the horizon, so the two sides of the frame
    # straddle it: exactly one of these is below. Attitude decides the split, and the
    # split is along the ROLL axis -- an image-y rule cannot produce this.
    q = _q_axis_angle(1.0, 0.0, 0.0, 90.0)
    left = _below(0.05, 0.5, q)
    right = _below(0.95, 0.5, q)
    assert left != right


def test_roll_90_split_is_not_trivially_true():
    # The same two points are BOTH above the horizon when level -- so the test above
    # is really observing attitude, not just any asymmetry in the projection.
    assert _below(0.05, 0.5, Q_LEVEL) is False
    assert _below(0.95, 0.5, Q_LEVEL) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest test_target_priority.py -v
```

Expected: collection error — `ImportError: cannot import name 'is_below_horizon' from 'telemetry_position'`.

- [ ] **Step 3: Implement `is_below_horizon`**

Append to `BDD/telemetry_position.py`, directly after `project_camera_to_ned` (i.e. after
its closing `return PositionNED(...)` at `:229`, before `rotate_ned_to_frd`):

```python
def is_below_horizon(
    detection_center_x: float,
    detection_center_y: float,
    aim_point_x: float,
    aim_point_y: float,
    fov_h_deg: float,
    fov_v_deg: float,
    quaternion: Quaternion,
) -> bool:
    """True when the detection's line of sight points BELOW the horizon.

    Attitude-corrected: this is about where the target is in the world, not where it
    sits in the image. A nose-down drone sees sky at the bottom of the frame.

    Implemented by reusing project_camera_to_ned at unit distance from the origin, so
    the resulting NED point IS the line-of-sight direction and its `down_m` component
    carries the sign (NED: down is positive). Sharing that call keeps ONE definition of
    the camera mounting convention in the codebase — re-deriving the ray here would let
    a second copy drift out of sync, silently inverting this test.
    """
    ray = project_camera_to_ned(
        detection_center_x,
        detection_center_y,
        aim_point_x,
        aim_point_y,
        fov_h_deg,
        fov_v_deg,
        1.0,                                            # unit distance -> direction
        quaternion,
        PositionNED(north_m=0.0, east_m=0.0, down_m=0.0),   # from the origin
    )
    return ray.down_m > 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest test_target_priority.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add BDD/telemetry_position.py BDD/test_target_priority.py
git commit -m "feat(telemetry): is_below_horizon() — attitude-corrected horizon test"
```

---

## Task 2: `Config.BelowHorizon` section

**Files:**
- Modify: `BDD/config.py` (add the nested dataclass after `OpticalRefinement`, which ends
  at `:271`)
- Modify: `BDD/config.yaml` (add the section after `optical_refinement`, which ends at
  `:139`)
- Modify: `BDD/test_config.py` (add parse tests; extend
  `test_config_has_optional_toggle_sections` at `:1069`)

**Interfaces:**
- Produces: `Config.BelowHorizon` with fields `confidence_multiplier: float` and
  `early_stage_duration_s: float`; top-level field `Config.below_horizon:
  Optional[BelowHorizon]`, which is `None` when the section is absent or has
  `enabled: false`. Tasks 3 and 5 consume it.

- [ ] **Step 1: Write the failing tests**

Add to the end of `BDD/test_config.py`:

```python
# ===========================================================================
# below_horizon: de-prioritize targets below the horizon (optional section)
# ===========================================================================

def test_below_horizon_defaults():
    cfg = Config.load(CONFIG_YAML)
    assert cfg.below_horizon is not None
    assert cfg.below_horizon.confidence_multiplier == 0.8
    assert cfg.below_horizon.early_stage_duration_s == 1.0


def test_below_horizon_disabled_is_none():
    text = CONFIG_YAML.read_text().replace(
        "below_horizon:\n  enabled: true",
        "below_horizon:\n  enabled: false",
    )
    cfg = loads_config(Config, text, source="<<below_horizon-off>>")
    assert cfg.below_horizon is None


def test_below_horizon_multiplier_out_of_range_is_rejected():
    text = CONFIG_YAML.read_text().replace(
        "confidence_multiplier: 0.8",
        "confidence_multiplier: 0.4",   # below the 0.5 floor
    )
    with pytest.raises(ConfigError) as excinfo:
        loads_config(Config, text, source="<<below_horizon-bad>>")
    assert "confidence_multiplier" in str(excinfo.value)
```

And extend the existing assertion at `test_config_has_optional_toggle_sections`
(`test_config.py:1069`) to name the new section:

```python
def test_config_has_optional_toggle_sections():
    assert {'optical_refinement', 'bytetrack', 'below_horizon'} <= set(_optional_dataclass_sections(Config))
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest test_config.py -v -k below_horizon
```

Expected: FAIL — `AttributeError: 'Config' object has no attribute 'below_horizon'`
(and the toggle-sections test fails its subset assertion).

- [ ] **Step 3: Add the section to `config.py`**

In `BDD/config.py`, immediately after the `optical_refinement: Optional[OpticalRefinement]`
line (`:271`) and before the `PDCoeff` dataclass:

```python
    @dataclass(slots=True, kw_only=True, frozen=True)
    class BelowHorizon:
        """De-prioritize targets below the horizon (attitude-corrected).

        enabled=False disables the whole feature (parser returns None for the
        section); check `config.below_horizon is not None`, not .enabled.

        Ranking only: `confidence_multiplier` scales the confidence used to CHOOSE
        between candidates, never the confidence tested against `confidence_min`. A
        below-horizon target that is the only target in frame is still tracked.

        Until the drone has been airborne for `early_stage_duration_s`, below-horizon
        detections are not merely demoted but ignored outright — on the pad there is
        no cost to refusing to launch, and every cost to launching at a bush.
        """
        confidence_multiplier:  Annotated[float, Range(0.5, 1.0)] = 0.8
        early_stage_duration_s: Annotated[float, Range(min=0.0)]  = 1.0
    below_horizon: Optional[BelowHorizon] = None
```

- [ ] **Step 4: Add the section to `config.yaml`**

In `BDD/config.yaml`, after the `optical_refinement` block (which ends at `:139`) and
before `pd_coeff:`:

```yaml
# Targets below the horizon are clutter or worse: an interceptor should not dive at
# the ground. Attitude-corrected, so this is about the real horizon, not the image.
below_horizon:
  enabled: true   # set false to disable the feature (section is still validated)
  # Effective confidence of a below-horizon detection once the drone is airborne:
  #   score = confidence * confidence_multiplier
  # Ranking only -- it can never push a detection under confidence_min.
  confidence_multiplier: 0.8    # [0.5 .. 1.0]
  # Below-horizon targets are IGNORED ENTIRELY before takeoff and for this long
  # after it. 0 ends the hard-ignore at liftoff. Unrelated to takeoff.duration_ns,
  # which shapes thrust and PD gains.
  early_stage_duration_s: 1.0   # [0 .. inf]
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
python -m pytest test_config.py -v
```

Expected: all pass, including the three new `below_horizon` tests.

- [ ] **Step 6: Commit**

```bash
git add BDD/config.py BDD/config.yaml BDD/test_config.py
git commit -m "feat(config): below_horizon section (optional, off via enabled:false)"
```

---

## Task 3: `target_priority.HorizonPriority` — the phase policy

**Files:**
- Create: `BDD/target_priority.py`
- Modify: `BDD/test_target_priority.py` (append the policy tests)

**Interfaces:**
- Consumes: `is_below_horizon` (Task 1), `Config.BelowHorizon` (Task 2), `helpers.XY`,
  `helpers.Detection`.
- Produces:
  - `HorizonPriority(cfg: Config.BelowHorizon | None, aim_point: XY)`
  - `HorizonPriority.score_fn(quaternion: Quaternion | None, fov_deg: XY,
    flight_time_s: float | None) -> Callable[[Detection], float | None] | None`
    — call once per frame. Returns `None` when the feature cannot or should not act
    this frame; otherwise returns the per-detection scorer that Task 4's
    `_pick_target_detection` takes as `score_fn`. `flight_time_s is None` means
    "not airborne yet".

  Task 5 calls `score_fn` and passes its result straight into `_pick_target_detection`.

**Why a separate module:** `drone_controller.py` is already ~1300 lines. This logic is
pure, phase-driven and exhaustively table-testable, which is exactly what should not be
buried inside the control loop.

- [ ] **Step 1: Write the failing tests**

Append to `BDD/test_target_priority.py`:

```python
# ===========================================================================
# HorizonPriority — the phase policy
# ===========================================================================

from config import Config
from helpers import XY, Detection, Rect
from target_priority import HorizonPriority

AIM = XY(0.5, 0.5)
FOV = XY(107.0, 85.0)

BelowHorizon = Config.BelowHorizon


def _cfg(confidence_multiplier=0.8, early_stage_duration_s=1.0) -> BelowHorizon:
    return BelowHorizon(
        confidence_multiplier=confidence_multiplier,
        early_stage_duration_s=early_stage_duration_s,
    )


def _det(confidence=0.9) -> Detection:
    # Centred bbox: above the horizon under Q_LEVEL, below it under Q_INVERTED.
    return Detection(bbox=Rect.from_xywh(0.45, 0.45, 0.1, 0.1), confidence=confidence)


# --- pre-takeoff: below-horizon targets are excluded outright ---

def test_pre_takeoff_excludes_below_horizon():
    score = HorizonPriority(_cfg(), AIM).score_fn(Q_INVERTED, FOV, flight_time_s=None)
    assert score(_det(0.9)) is None


def test_pre_takeoff_keeps_above_horizon_at_raw_confidence():
    score = HorizonPriority(_cfg(), AIM).score_fn(Q_LEVEL, FOV, flight_time_s=None)
    assert score(_det(0.9)) == pytest.approx(0.9)


# --- the early window extends the hard-ignore past takeoff ---

def test_early_flight_still_excludes_below_horizon():
    score = HorizonPriority(_cfg(early_stage_duration_s=1.0), AIM).score_fn(
        Q_INVERTED, FOV, flight_time_s=0.5)
    assert score(_det(0.9)) is None


def test_after_early_window_below_horizon_is_only_demoted():
    score = HorizonPriority(_cfg(early_stage_duration_s=1.0), AIM).score_fn(
        Q_INVERTED, FOV, flight_time_s=2.0)
    assert score(_det(0.9)) == pytest.approx(0.9 * 0.8)


def test_after_early_window_above_horizon_is_untouched():
    score = HorizonPriority(_cfg(), AIM).score_fn(Q_LEVEL, FOV, flight_time_s=2.0)
    assert score(_det(0.9)) == pytest.approx(0.9)


def test_zero_duration_ends_the_hard_ignore_at_liftoff():
    score = HorizonPriority(_cfg(early_stage_duration_s=0.0), AIM).score_fn(
        Q_INVERTED, FOV, flight_time_s=0.0)
    assert score(_det(0.9)) == pytest.approx(0.9 * 0.8)


# --- degradation: the feature must never blind the controller ---

def test_disabled_config_produces_no_scorer():
    assert HorizonPriority(None, AIM).score_fn(Q_INVERTED, FOV, flight_time_s=None) is None


def test_in_flight_without_attitude_produces_no_scorer():
    # A stale attitude is worthless in flight -- the airframe rotates fast. Decline to
    # guess: no penalty, no exclusion, every detection keeps its raw confidence.
    p = HorizonPriority(_cfg(), AIM)
    p.score_fn(Q_INVERTED, FOV, flight_time_s=2.0)      # feed it a good attitude first
    assert p.score_fn(None, FOV, flight_time_s=2.0) is None


def test_pre_takeoff_without_attitude_reuses_the_last_known_one():
    # On the ground the attitude has not changed, so the cached one is still valid and
    # the hard-ignore must survive a telemetry dropout on the pad.
    p = HorizonPriority(_cfg(), AIM)
    p.score_fn(Q_INVERTED, FOV, flight_time_s=None)     # cache an inverted attitude
    score = p.score_fn(None, FOV, flight_time_s=None)
    assert score is not None
    assert score(_det(0.9)) is None


def test_never_any_attitude_disables_the_feature():
    # No telemetry has EVER arrived: nothing may be excluded on the strength of a
    # horizon we cannot locate.
    p = HorizonPriority(_cfg(), AIM)
    assert p.score_fn(None, FOV, flight_time_s=None) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest test_target_priority.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'target_priority'`.

- [ ] **Step 3: Implement the module**

Create `BDD/target_priority.py`:

```python
"""Below-horizon target de-prioritization.

An interceptor should not dive at the ground. A confident false positive down there
— a bush, a vehicle, a glint — must not outrank a genuine airborne target, and on the
launch pad it must not be chased at all.

This module owns the *policy*; telemetry_position.is_below_horizon owns the geometry.
The policy is a function of exactly one thing: how long ago the drone took off.

    pre-takeoff, and for early_stage_duration_s after it:
        below-horizon detections are EXCLUDED from the pool entirely
    after that:
        below-horizon detections score confidence * confidence_multiplier

Above-horizon detections always score their raw confidence.

Two rules the tests pin down, and which must not be "simplified" away:

1. RANKING ONLY, once airborne. The multiplier decides who wins among candidates. It
   never feeds the `confidence >= confidence_min` admission gate, so it can never drop
   the last remaining target and leave a flying drone with nothing to chase.
   Deprioritizing a target is safe; blinding the controller is not.

2. The horizon is undefined without an attitude quaternion, and each phase fails to
   the safe side: in flight, no attitude means no penalty (the feature no-ops); on the
   ground it falls back to the last known attitude (the drone is sitting still); and if
   no attitude has ever been seen, the feature disables itself rather than exclude
   detections on the strength of a horizon it cannot locate.
"""

import logging

from helpers import XY, Detection
from telemetry_position import Quaternion, is_below_horizon

logger = logging.getLogger(__name__)


class HorizonPriority:
    """Per-frame scoring policy. Owns exactly one piece of state: the last attitude."""

    __slots__ = ('_cfg', '_aim_point', '_last_quaternion')

    def __init__(self, cfg, aim_point: XY):
        """`cfg` is a Config.BelowHorizon, or None when the feature is off (the config
        section is absent or `enabled: false`)."""
        self._cfg = cfg
        self._aim_point = aim_point
        self._last_quaternion: Quaternion | None = None

    def score_fn(self, quaternion, fov_deg: XY, flight_time_s):
        """Build this frame's scorer, or None when the feature must not act.

        `flight_time_s` is None until the drone is airborne. Returns a callable
        `score(detection) -> float | None`, where None means EXCLUDE the detection.
        A None return from this method means "no scoring at all this frame" — the
        caller then ranks on raw confidence, exactly as before the feature existed.
        """
        if self._cfg is None:
            return None

        airborne = flight_time_s is not None

        if quaternion is not None:
            self._last_quaternion = quaternion
            attitude = quaternion
        elif airborne:
            # In flight the airframe rotates fast; a cached attitude is worse than none.
            attitude = None
        else:
            # On the ground it has not moved, so the last known attitude still holds.
            attitude = self._last_quaternion

        if attitude is None:
            return None

        cfg = self._cfg
        hard_ignore = (not airborne) or (flight_time_s < cfg.early_stage_duration_s)
        aim, multiplier = self._aim_point, cfg.confidence_multiplier

        def score(detection: Detection):
            centre = detection.bbox.center
            try:
                below = is_below_horizon(
                    centre.x, centre.y,
                    aim.x, aim.y,
                    fov_deg.x, fov_deg.y,
                    attitude,
                )
            except (ValueError, ZeroDivisionError):
                # Never let a geometry edge case take down the control loop: an
                # unclassifiable detection is treated as above the horizon (no penalty).
                logger.warning("horizon test failed for %s; treating as above", detection)
                return detection.confidence

            if not below:
                return detection.confidence
            if hard_ignore:
                return None
            return detection.confidence * multiplier

        return score
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest test_target_priority.py -v
```

Expected: 15 passed (5 geometry + 10 policy).

- [ ] **Step 5: Commit**

```bash
git add BDD/target_priority.py BDD/test_target_priority.py
git commit -m "feat(target_priority): HorizonPriority phase policy"
```

---

## Task 4: `_pick_target_detection(score_fn=...)`

**Files:**
- Modify: `BDD/drone_controller.py:154-176`
- Modify: `BDD/test_drone_controller.py` (append)

**Interfaces:**
- Consumes: nothing new (the scorer is injected, so this task has no dependency on
  Tasks 1-3 and can be reviewed on its own).
- Produces: `_pick_target_detection(detections, confidence_min, locked_track_id,
  use_track_lock, score_fn=None)`. Task 5 passes `score_fn`.

**The invariant this task exists to protect:** the `confidence >= confidence_min` gate
keeps testing **raw** confidence. `score_fn` only supplies the ranking key. Moving the
gate onto the scored value would let an 0.8 multiplier drop a 0.45-confidence detection
under a 0.4 threshold and blind the controller — the single most dangerous mistake
available in this feature.

- [ ] **Step 1: Write the failing tests**

Append to `BDD/test_drone_controller.py` (and extend its import line from
`drone_controller` to include `_pick_target_detection`):

```python
from helpers import Detection, Rect
from drone_controller import _pick_target_detection


def _d(confidence, track_id=1) -> Detection:
    return Detection(bbox=Rect.from_xywh(0.45, 0.45, 0.1, 0.1),
                     confidence=confidence, track_id=track_id)


def test_pick_without_score_fn_is_unchanged():
    low, high = _d(0.5, track_id=1), _d(0.9, track_id=2)
    assert _pick_target_detection([low, high], 0.4, None, False) is high


def test_score_fn_can_demote_the_most_confident_detection():
    # The 0.9 detection is below the horizon (0.9 * 0.8 = 0.72) and loses to an
    # honest 0.8 above it.
    below, above = _d(0.9, track_id=1), _d(0.8, track_id=2)
    score = lambda d: d.confidence * 0.8 if d is below else d.confidence
    assert _pick_target_detection([below, above], 0.4, None, False, score_fn=score) is above


def test_score_fn_returning_none_excludes_the_detection():
    excluded, kept = _d(0.9, track_id=1), _d(0.5, track_id=2)
    score = lambda d: None if d is excluded else d.confidence
    assert _pick_target_detection([excluded, kept], 0.4, None, False, score_fn=score) is kept


def test_all_detections_excluded_yields_no_target():
    assert _pick_target_detection([_d(0.9), _d(0.8)], 0.4, None, False,
                                  score_fn=lambda d: None) is None


def test_gate_uses_raw_confidence_not_the_score():
    # THE safety invariant: a lone below-horizon target scores 0.45 * 0.8 = 0.36, under
    # confidence_min=0.4 -- and is STILL picked, because the gate tests raw confidence.
    # A flying drone must never be left with nothing to chase.
    lone = _d(0.45)
    picked = _pick_target_detection([lone], 0.4, None, False,
                                    score_fn=lambda d: d.confidence * 0.8)
    assert picked is lone


def test_track_lock_still_short_circuits_the_pool():
    # With a lock held the pool is narrowed to the locked track before ranking, so the
    # scorer never gets to override the lock.
    locked, other = _d(0.5, track_id=7), _d(0.9, track_id=8)
    picked = _pick_target_detection([locked, other], 0.4, 7, True,
                                    score_fn=lambda d: d.confidence)
    assert picked is locked
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest test_drone_controller.py -v -k "score_fn or pick or gate or track_lock"
```

Expected: FAIL — `TypeError: _pick_target_detection() got an unexpected keyword argument 'score_fn'`.

- [ ] **Step 3: Implement**

Replace `_pick_target_detection` in `BDD/drone_controller.py:154-176` with:

```python
def _pick_target_detection(
    detections: list,
    confidence_min: float,
    locked_track_id: int | None,
    use_track_lock: bool,
    score_fn = None,
) -> Detection | None:
    """
    Выбор одной детекции для сопровождения. При use_track_lock и заданном
    locked_track_id возвращает только детекции с этим ByteTrack track_id,
    чтобы не перескакивать на случайные высоко-уверенные ложные срабатывания.
    Если заблокированный трек в кадре отсутствует — None (ведём себя как
    при потере цели, оценщик/затухание сами продолжают движение).

    `score_fn(detection) -> float | None` (optional) supplies the RANKING key —
    typically the below-horizon penalty from target_priority.HorizonPriority. Returning
    None EXCLUDES a detection from the pool.

    The `confidence >= confidence_min` admission gate always tests the RAW confidence,
    never the score. A penalty may demote a detection, but it must never be able to push
    one under the threshold and leave a flying drone with no target at all.

    The track lock is applied BEFORE scoring: once a lock is held the scorer has nothing
    to compare against, so the horizon rule shapes lock ACQUISITION, not retention. A
    target being dived on legitimately ends up below the drone, and that is the worst
    possible moment to drop the lock.
    """
    pool = [d for d in detections if d is not None and d.confidence >= confidence_min]
    if not pool:
        return None
    effective_lock = locked_track_id if use_track_lock else None
    if effective_lock is not None:
        locked = [d for d in pool if d.track_id == effective_lock]
        if locked:
            return max(locked, key=lambda d: d.confidence)
        return None
    if score_fn is None:
        return max(pool, key=lambda d: d.confidence)

    scored = [(score, d) for score, d in ((score_fn(d), d) for d in pool) if score is not None]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[0])[1]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
python -m pytest test_drone_controller.py -v
```

Expected: 26 passed (the 20 existing + 6 new).

- [ ] **Step 5: Commit**

```bash
git add BDD/drone_controller.py BDD/test_drone_controller.py
git commit -m "feat(control): _pick_target_detection takes an optional score_fn"
```

---

## Task 5: Wire it into the control loop, and verify on real flights

**Files:**
- Modify: `BDD/drone_controller.py` — the import block at `:17-24`; the setup block near
  `AIM_POINT` at `:406`; the pick site at `:889-891`; the debug dict near `:1320`.

**Interfaces:**
- Consumes: `HorizonPriority` (Task 3), `Config.below_horizon` (Task 2),
  `_pick_target_detection(score_fn=...)` (Task 4), and
  `telemetry_position.get_orientation_quaternion` (already exists at
  `telemetry_position.py:109`).
- Produces: the running feature.

**Two facts that will bite you if you skip them:**

1. The quaternion lives at `telemetry["odometry"]["q"]`. There is no
   `attitude_quaternion` telemetry aspect in this codebase — do not go looking for one,
   and do not use `attitude_euler` instead.
2. `drone_pose = get_pose(telemetry_dict)` is computed at `:957`, which is **after** the
   pick at `:889`. You cannot reuse it. Extract the quaternion yourself, before the pick,
   from the `telemetry_dict` already fetched at `:846`.

- [ ] **Step 1: Extend the imports**

In `BDD/drone_controller.py`, add `get_orientation_quaternion` to the existing
`telemetry_position` import (`:17-24`) and import the new policy:

```python
from telemetry_position import (
    # get_position_ned,
    get_orientation_quaternion,
    project_camera_to_ned,
    get_pose,
    project_ned_to_camera
)
from target_priority import HorizonPriority
```

- [ ] **Step 2: Construct the policy at controller startup**

In `drone_controlling_thread_async`, immediately after `AIM_POINT = control_config.aim_point`
(`:406`):

```python
    # Below-horizon de-prioritization. `control_config.below_horizon` is None when the
    # feature is off — gate on presence, never on `.enabled`.
    horizon_priority = HorizonPriority(control_config.below_horizon, AIM_POINT)
    if control_config.below_horizon is None:
        logger.info("below-horizon target de-prioritization: OFF")
```

- [ ] **Step 3: Score the pool at the pick site**

Replace the pick at `BDD/drone_controller.py:889-891`:

```python
            picked = _pick_target_detection(
                detections, CONFIDENCE_MIN, locked_track_id, BYTETRACK_TARGET_LOCK
            )
```

with:

```python
            # The horizon test needs THIS frame's attitude. Note that drone_pose is not
            # built until later (it needs the distance estimate), so read the quaternion
            # straight from the telemetry dict fetched above. A missing/garbled reading
            # yields None, and the policy degrades on its own — it must never raise here.
            try:
                current_quaternion = get_orientation_quaternion(telemetry_dict)
            except (KeyError, TypeError):
                current_quaternion = None

            # None until the drone is airborne — the policy reads this as "pre-takeoff".
            flight_time_s = (flight_time_ns / 1e9) if takeoff_time_ns is not None else None
            horizon_score = horizon_priority.score_fn(
                current_quaternion, FRAME_ANGLUAR_SIZE_DEG, flight_time_s
            )

            picked = _pick_target_detection(
                detections, CONFIDENCE_MIN, locked_track_id, BYTETRACK_TARGET_LOCK,
                score_fn=horizon_score,
            )
            if picked is None and detections and horizon_score is not None:
                # Expected on the pad with nothing but ground clutter in frame — and it
                # will also stall takeoff, because the estimator never accumulates its
                # delay_until_n_detection_frames. That reads as a hang on a bench test,
                # so say so out loud.
                logger.info(
                    "below-horizon rule left no candidate among %d detection(s)%s",
                    len(detections),
                    "" if flight_time_s is not None else " (pre-takeoff)",
                )
```

`FRAME_ANGLUAR_SIZE_DEG` is refreshed from the frame's camera at `:821`, so it is already
correct for this frame's `camera_id` by the time you read it here.

- [ ] **Step 4: Expose it to the flight debugger**

In the `debug_info` dict near `:1320` (which already carries `'frame_angular_size_deg'`),
add:

```python
                    'below_horizon_active' : horizon_score is not None,
```

- [ ] **Step 5: Verify the whole suite still passes**

```bash
python -m pytest -q
```

Expected: everything passes except the known pre-existing
`test_OverwriteQueue.py::test_fifo_when_not_overwritten` (see Global Constraints).

- [ ] **Step 6: Replay against the real flights — this is the correctness gate**

Run all three 2026-04-27 UAE logs (near-static, cruise, and the 39 m/s dive) through the
real control thread, feature on:

```bash
python debug_drone_controller.py /media/BDD/_BACKUPS/UAE/2026-04/2026-04-27/_DEBUG_09/BDD_20260427-150025.log --headless
```

(The other two logs are named in the `uae-replay-log-corpus` memory. Pass the log FILE,
never the directory.)

Then A/B it with the feature off. `enabled` is a file-only toggle, **not** a dataclass
field, so `--params below_horizon.enabled=false` will not work — the dotted `--params`
path is validated against the real dataclass schema. Disable it by pointing the harness
at a copy of the config instead:

```bash
sed 's/^  enabled: true/  enabled: false/' config.yaml > /tmp/config-no-horizon.yaml
# check you flipped the right one -- optical_refinement also has an `enabled: true`:
grep -n -A1 'below_horizon:\|optical_refinement:' /tmp/config-no-horizon.yaml
python debug_drone_controller.py <same log> --headless --config /tmp/config-no-horizon.yaml
```

(That `sed` flips **every** `enabled: true` at that indent, which also disables
`optical_refinement`. For a clean single-variable A/B, edit the copy by hand and change
only the `below_horizon` one.)

**What you are checking, and why it matters more than the unit tests:** if
`project_camera_to_ned`'s "camera points up (−Z in FRD)" docstring is stale, every test
above still passes and the feature does the exact *opposite* of what is wanted — it
deprioritizes real airborne targets and refuses to launch at them, while happily locking
onto the ground. Nothing in the code would look wrong.

So read the sign, don't just read the exit code:

- Grep the run for `below-horizon rule left no candidate`. On a real intercept it should
  be rare-to-absent once airborne. A flood of it means the sign is inverted.
- Compare the picked target across the A/B. On these logs (a genuine airborne target,
  tracked to impact) the two runs should pick the **same** detection almost always. A
  large divergence means the rule is rejecting the real target.
- Both runs must exit 0 with zero exceptions.

If the sign looks inverted, **stop** and flip the comparison in `is_below_horizon`
(Task 1) rather than papering over it in the policy — and fix the stale docstring in
`project_camera_to_ned`, which is then wrong for the 3D estimator too.

- [ ] **Step 7: Commit**

```bash
git add BDD/drone_controller.py
git commit -m "feat(control): deprioritize targets below the horizon"
```

- [ ] **Step 8: Close out the TODO**

Mark the task in `BDD/TODO.md:12` as done (`+` prefix, per the file's own legend), noting
what shipped, that `early_stage_multiplier` was deliberately not implemented, and that
tuning `confidence_multiplier` on the rig is the remaining work.

```bash
git add BDD/TODO.md
git commit -m "docs(todo): below-horizon target priority done"
```

## What is NOT in this plan, on purpose

- **Rig tuning.** `confidence_multiplier = 0.8` and `early_stage_duration_s = 1.0` are the
  TODO's guesses, not measured values. Tuning them is a flight-test activity, not an
  implementation step.
- **A horizon dead-band.** Detections straddling the horizon flicker between the two
  classifications at high angular rates. The soft multiplier tolerates it by design (a
  misclassified target is demoted, not dropped), so hysteresis would be speculative
  complexity. Add it only if a flight log shows it actually hurting.
- **`early_stage_multiplier`** from the TODO — see the Approach section above for why it
  does not exist.
