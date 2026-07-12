# Speed-dependant PD coefficient — design

Date: 2026-07-12
TODO item: `!!! speed-dependant PD coeff` (BDD/TODO.md)

## Problem

`P` is currently chosen from target size (and nudged by distance), with no regard for how
fast the drone is actually moving. The same gain that is well-damped in a slow approach is
twitchy at speed: a given angular error produces the same commanded correction whether the
airframe is doing 5 m/s or 35 m/s, so at high speed the loop over-corrects.

Fix: make `P` fall off as speed rises past a configurable threshold.

## Behaviour

Above a `start_speed_ms` threshold, `P` decays geometrically with speed:

    P *= coeff ** ((speed_ms - start_speed_ms) / speed_step_ms)

At or below the threshold, `P` is untouched. Worked examples from the TODO (P = 10,
start = 10 m/s, speed = 20 m/s, coeff = 0.9):

| speed_step_ms | result                    |
|---------------|---------------------------|
| 1             | 10 * 0.9**10 = 3.4868     |
| 2             | 10 * 0.9**5  = 5.9049     |

The reduction is applied **last**, after every other `P` modification, and the result is
then clamped into the configured `[p_min, p_max]`.

**Speed** is the full 3D magnitude of `odometry.velocity_body` (FRD), i.e.
`sqrt(vx**2 + vy**2 + vz**2)`. The drone dives at its targets, so vertical rate is a real
part of how fast the airframe is moving and how twitchy a given `P` becomes.

## Config

New optional section under `control.pd_coeff`, following the `Optional[...] = None` pattern
already used by `thrust.proportional_to_distance`. **The section being absent means the
feature is off** — there is no separate `enabled` flag.

```python
@dataclass(slots=True, kw_only=True, frozen=True)
class SpeedReduction:
    start_speed_ms: Annotated[float, Range(0.0, 100.0)] = 15.0
    coeff:          Annotated[float, Range(0.0, 1.0)]   = 0.947
    speed_step_ms:  Annotated[float, Range(1.0, 10.0)]  = 1.0
speed_reduction: Optional[SpeedReduction] = None
```

Ranges and defaults are taken from the TODO. `coeff = 1.0` is a legal no-op; `coeff = 0.0`
collapses `P` to the `p_min` floor above the threshold.

## Components

Two pure, module-level functions in `drone_controller.py`, next to `clamp_xy`. Both are
unit-testable on the host with no drone and no hardware (`drone_controller` imports clean).

### `speed_from_telemetry(telemetry_dict) -> float | None`

3D magnitude of `odometry.velocity_body`. Returns `None` — never raises — when odometry is
missing, the velocity sub-dict is missing, or a component is absent/non-numeric. Callers
decide what a `None` means; this keeps the parsing total.

### `speed_reduced_p(p, speed_ms, cfg) -> XY`

`cfg` is the `Config.Control.PDCoeff.SpeedReduction` value object (frozen, values-only —
safe to pass whole; this does not violate the "no active objects in config dicts" rule).

- `cfg is None` (feature off) → return `p` unchanged.
- `speed_ms <= cfg.start_speed_ms` → return `p` unchanged.
- otherwise → `p * cfg.coeff ** ((speed_ms - cfg.start_speed_ms) / cfg.speed_step_ms)`.

Stateless and total. Per-axis: both components of the `XY` are scaled by the same factor,
because speed is a scalar property of the airframe, not of an axis.

## Data flow / call site

In the control loop, immediately after the thrust/distance block (which today ends around
`drone_controller.py:1083`) and before `command_regulator.set_coeffs(...)`:

1. `speed = speed_from_telemetry(telemetry_dict)`.
2. If `speed is None`, reuse `last_known_speed_ms`; otherwise update `last_known_speed_ms`.
3. `pd_coeff_p = speed_reduced_p(pd_coeff_p, speed, PD_COEFF_P_SPEED_REDUCTION)`.
4. `pd_coeff_p = clamp_xy(PD_COEFF_P_MIN, pd_coeff_p, PD_COEFF_P_MAX)` — **unconditionally**.

`last_known_speed_ms` starts at `None`; while it is `None` (we have never had a reading) no
reduction is applied.

### Error handling / graceful degradation

A telemetry dropout must never stall or crash the control loop (a stalled control loop can
wreck the airframe). It degrades by **reusing the last known speed**, which fails toward the
safe side: a dropout mid-dive holds the reduced, calmer `P` rather than snapping the gain
back up to its full twitchy value. Only a drone that has never reported a velocity gets no
reduction at all — and that drone is not moving fast yet.

### Note: the final clamp fixes an existing latent bug

Today `pd_coeff_p_for_target_size()` clamps into `[p_min, p_max]`, but the NEAR / MEDIUM
branches then multiply `pd_coeff_p` by 1.1 *after* that clamp, so the `P` actually handed to
the regulator can already exceed the configured `p_max`. Moving the clamp to the end of the
chain — and running it even when speed reduction is disabled — makes `p_max` mean what it
says. This is a deliberate, small behaviour change in the direction the config already
promises.

## Logging

The computed speed and the applied reduction factor are appended to the existing per-frame
`mode`/`extra` flight-log string, so the reduction is visible in post-flight analysis
alongside the `p:` value already logged there.

## Testing

New host test file `BDD/test_drone_controller.py` (pytest, no hardware), written test-first:

- Both TODO worked examples asserted literally (3.4868 @ step 1, 5.9049 @ step 2).
- At and below `start_speed_ms`: `P` unchanged.
- `cfg is None`: `P` unchanged.
- `coeff = 1.0`: `P` unchanged at any speed.
- Result clamped up to `p_min` (extreme speed) and never above `p_max`.
- Both `XY` axes scaled by the same factor; per-axis `P` ratio preserved.
- `speed_from_telemetry`: correct 3D magnitude; `None` for missing odometry, missing
  `velocity_body`, and a malformed/partial velocity dict — never raises.
- Dropout path: a `None` reading reuses the last known speed; a never-seen speed disables
  the reduction.

## Out of scope

- Logging `P` to the PX4 ulog (separate TODO item).
- Any change to `D`, to the target-size `P` profile, or to `platform_controller.py`
  (already marked stale).
