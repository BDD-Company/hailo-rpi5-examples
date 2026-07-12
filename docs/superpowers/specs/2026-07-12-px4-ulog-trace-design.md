# Write extra logs into the PX4's persistent ulog

Design, 2026-07-12. Branch `feat/px4-ulog-trace`. Task: `BDD/TODO.md`, "write extra
logs into the px4's persistent ulog for post-crash analisys".

## Problem

After a crash the Pi's SD card may be unreadable, taking every `_DEBUG/` log and
video with it. The PX4's own ulog survives far more reliably. So a small trace of
what the decision pipeline was doing needs to live in the PX4 log, not only in ours.

The trace must answer, from the flight controller's log alone: was the target seen,
where was it, how confident were we, what gain were we running, and what did we
command?

## Approach

Send one `DEBUG_FLOAT_ARRAY` MAVLink message per tick, at 5 Hz, over the *existing*
MAVSDK link. PX4's MAVLink receiver republishes it as the `debug_array` uORB topic,
which the logger writes into the ulog.

### Why this message

PX4 accepts four inbound debug messages: `DEBUG` (`debug_value`), `NAMED_VALUE_FLOAT`
(`debug_key_value`), `DEBUG_VECT` (`debug_vect`), and `DEBUG_FLOAT_ARRAY`
(`debug_array`). Only `DEBUG_FLOAT_ARRAY` carries all our fields in a single
message: it is `time_usec` + `name[10]` + `array_id` + `data[58]` (MAVLink msg 350).

58 float32 slots is far more than the ~12 values we need, so the float16 /
bit-packing idea sketched in the TODO is unnecessary. One message per tick costs
~1.3 KB/s on a 1 Mbps serial link. Packing would save bandwidth we do not need and
would make the log unreadable without a bespoke decoder. Every value goes in as a
plain float32 and `pyulog` / FlightReview read it directly.

The alternative, `NAMED_VALUE_FLOAT` per field, is more readable in FlightReview but
needs ~12 messages per tick and — decisively — the fields would no longer be
guaranteed to come from the same control iteration. A trace whose fields are skewed
across ticks is misleading exactly when you most need to trust it.

### Why we can send it at all

MAVSDK's Python bindings have no `mavlink_passthrough` plugin, and the rig's link is
serial-USB (`connection_string: usb` → `serial:///dev/serial/by-id/...:1000000`),
which `mavsdk_server` opens exclusively. A second pymavlink connection to the same
port is therefore impossible.

The installed MAVSDK (3.15.3) ships the `mavlink_direct` plugin, which sends an
arbitrary MAVLink message as JSON fields over the link MAVSDK already holds. No
second connection, no new dependency.

## Trace layout

`DEBUG_FLOAT_ARRAY`, `name = "BDD"`, `array_id = 0`.

| slot | field | notes |
|-----:|-------|-------|
| 0 | format version | `1`. Lets a decoder tell layouts apart. |
| 1 | frames since a usable target | `0` on a tick that selected a target. |
| 2 | iterations with no frame at all | the loop's `skipped_detetions`; `0` when a frame arrived. |
| 3 | detection centre x | normalised `[0, 1]`; `NaN` if no target. |
| 4 | detection centre y | " |
| 5 | detection width | " |
| 6 | detection height | " |
| 7 | detection confidence | `NaN` if no target. |
| 8 | `pd_coeff_p.x` | after the final clamp into `[p_min, p_max]`. |
| 9 | `pd_coeff_p.y` | " |
| 10 | commanded roll, degrees | verbatim, as passed to `DroneMover`. |
| 11 | commanded pitch, degrees | " |
| 12 | commanded thrust | " |
| 13–57 | zero | reserved. |

Slots 1 and 2 are two different starvation modes and neither implies the other.
Slot 1 rising means vision lost the ball while frames kept coming. Slot 2 rising
means the camera/inference pipeline itself starved and the controller had nothing to
act on. Post-crash you cannot distinguish those from either counter alone, and the
distinction points at completely different root causes, so both are recorded.

`NaN` rather than `0.0` for absent detection fields: a detection at the exact frame
centre with zero confidence is a legal (if unlikely) reading, so zeros would be
ambiguous. `NaN` is not.

## Components

### `BDD/ulog_trace.py` (new)

- `TRACE_FORMAT_VERSION`, `TRACE_NAME`, and the slot indices, in one place.
- `UlogTraceSample` — a frozen dataclass of the fields above, with `to_floats()`
  returning the 58-slot list. Pure, trivially testable.
- `UlogTracer(send_fn, rate_hz, enabled, now_fn=time.monotonic)`.
  - `send_fn` is an **explicit parameter**, an async callable
    `(name: str, array_id: int, data: list[float]) -> None`. It is a live object and
    so must never travel inside a config dict (project rule: `control_config` is
    values-only).
  - `submit(sample)` is the hot path, called from the control loop. Synchronous and
    cheap: if disabled, return. If less than `1 / rate_hz` has elapsed since the last
    send, return. Otherwise fire `send_fn` as a detached `asyncio` task and return
    immediately — it never awaits the send.
  - If a previous send is still in flight, drop the sample and count it. A stalled
    link can then never build a backlog nor stall the control loop.
  - Send exceptions are caught and counted, logged once and thereafter periodically.
    A broken trace degrades to no trace; it never propagates into the control loop.
    (Project rule: on the control path, degrade — never hard-fail.)
  - Counters (`sent`, `dropped`, `failed`) are exposed for the shutdown summary.

### `BDD/drone.py` — `DroneMover`

- `send_debug_array(name, array_id, data)` — wraps `mavlink_direct.send_message` with
  a `DEBUG_FLOAT_ARRAY` `MavlinkMessage`. Raises on failure; `UlogTracer` is what
  swallows it.
- Record the commanded attitude **verbatim**, as handed to `DroneMover`, before any
  adjustment:
  - `move_to_target_zenith_async` stores `(roll_degree, pitch_degree, thrust)` on its
    **first line** — before the upside-down veto and before `_clamp_tilt_for_lift`.
    This is deliberate: we are logging the decision the vision pipeline made, not
    what the FC ultimately did with it. What the FC did is already in the ulog's own
    attitude topics, so recording the pre-clamp value adds information rather than
    duplicating it. `idle()` funnels through this method, so it is covered for free.
  - `standstill(thrust)` stores `(0.0, 0.0, thrust)` — the only other attitude path.
  - `move_to_target_ned` is a *position* command with no roll/pitch/thrust; it stores
    `NaN`s. That mode is off on the rig (`follow_target_position_ned: false`), and a
    stale-but-plausible attitude would be worse than an honest "no attitude command".
- `last_attitude_command()` returns the stored triple. (Distinct from the existing
  `last_command()`, which returns a formatted *string* from the `debug_collect_call_info`
  proxy and carries no usable numbers.)
- At the end of `startup_sequence`, read the `SDLOG_PROFILE` param and log a loud
  WARNING naming the exact fix if the Debug bit (32) is clear. The app never *writes*
  flight-controller params — see "Provisioning".

### `BDD/drone_controller.py`

- Build the tracer once, before the loop, from `control_config.ulog_trace` plus the
  live `drone`, and call `tracer.submit(...)` at exactly one site: immediately after
  `command_sent_ns = time.monotonic_ns()`.

  That point is **after** the awaited command has already gone out, so the trace adds
  nothing to capture→command latency — the metric this project optimises above all
  others. Every field is also in scope there regardless of which branch ran.

### `BDD/config.py` + `config.yaml`

New optional section, values only:

```yaml
ulog_trace:
  enabled: true
  rate_hz: 5.0        # [0.1 .. 50]
```

Absent or `enabled: false` → the tracer is a no-op and nothing is sent. Mirrored into
`test_config.py`'s `TestConfig` with a test, per the project's config-mirror rule.

### `BDD/debug_drone_controller.py`

`MockDroneMover` gains `send_debug_array` (recording, not sending) and
`last_attitude_command`, so the replay harness exercises the real trace path off-rig.

## Provisioning

PX4 only writes `debug_array` into the ulog when `SDLOG_PROFILE` has its Debug bit
(32) set, and the logger reads that param at startup — so a change needs an FC reboot
and cannot help the flight during which it is made.

The app therefore **checks and warns, but does not write**. Auto-setting the param
would give the vision app the power to mutate flight-controller configuration, which
is a new and invasive capability for a benefit (a trace that only starts working
after the next reboot anyway) that does not justify it. Setting it is a documented,
one-time rig-provisioning step; the startup warning is what stops a misconfigured FC
from being discovered post-crash, which is the one moment it cannot be fixed.

## Failure modes

| failure | behaviour |
|---|---|
| `mavlink_direct` unsupported by the running `mavsdk_server` | first send raises; counted, warned once, tracer keeps no-oping. Flight unaffected. |
| link saturated / send stalls | next `submit` sees the in-flight task, drops the sample, counts it. Loop never blocks. |
| `SDLOG_PROFILE` Debug bit clear | messages still sent (harmless); loud WARNING at startup naming the fix. |
| `ulog_trace` section absent | tracer disabled, zero overhead. |
| PX4 not connected / MockDroneMover | `send_fn` records or no-ops; replay harness runs clean. |

Nothing in this feature can abort the control loop. A stalled control loop can wreck
the hardware, and a *debugging* feature must never be the thing that stalls it.

## Testing

**Host unit tests** — `BDD/test_ulog_trace.py`, with a fake async `send_fn`:

- `to_floats()` puts each field in its documented slot; the array is exactly 58 long;
  the format version is slot 0; unused slots are zero.
- Absent detection → `NaN` in slots 3–7, not `0.0`.
- Rate limiting: submitting at 20 Hz against `rate_hz=5` sends ~1 in 4 (driven by an
  injected `now_fn`, so the test is deterministic and does not sleep).
- Drop-on-inflight: with a send that never completes, subsequent submits are dropped
  and counted, and `submit` still returns promptly.
- A raising `send_fn` is swallowed, counted, and does not propagate.
- `enabled: false` → `send_fn` is never called.

**Config test** — `ulog_trace` parses, defaults apply when the section is absent, and
an out-of-range `rate_hz` is rejected. Mirrored in `TestConfig`.

**Replay, off-rig** — run `debug_drone_controller.py` against the 39 m/s UAE dive log
(`uae-replay-log-corpus`) with a capturing fake `send_fn`; assert the trace count
matches ~5 Hz over the log's duration, that slot 0 is the format version on every
sample, and that commanded roll/pitch/thrust match what the controller passed to
`MockDroneMover`. This is the check that the wiring is real rather than merely
well-unit-tested.

**On-rig verification** is a separate, later step (needs the FC param set + a reboot):
fly or bench-run, pull the ulog, and confirm a `debug_array` topic named `BDD` is
present with sane values.

## Out of scope

- Setting `SDLOG_PROFILE` from the app.
- Any change to what the Pi-side `_DEBUG/` logs record.
- A decoder/plotting script for the ulog side — `pyulog` already dumps `debug_array`,
  and the slot table above is the whole format.
