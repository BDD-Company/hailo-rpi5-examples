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

| slot | field | absent / idle | notes |
|-----:|-------|---------------|-------|
| 0 | format version | — | `1`. Non-zero, so a message can never truncate to nothing. |
| 1 | frame id | — | the controller's monotonic counter, the one in our log prefix `frame=#0123`. |
| 2 | frames since a usable target | `0` on a tick that selected one | frames arrived, vision found nothing. |
| 3 | iterations with no frame at all | `0` when a frame arrived | the loop's `skipped_detetions`; the pipeline itself starved. |
| 4 | `pd_coeff_p.x` | — | after the final clamp into `[p_min, p_max]`. |
| 5 | `pd_coeff_p.y` | — | " |
| 6 | commanded roll | `0.0` | verbatim, as passed to `DroneMover`, pre-clamp. **Units vary — see below.** |
| 7 | commanded pitch | `0.0` | " |
| 8 | commanded thrust | `0.0` | " |
| 9 | detection centre x | `0.0` | ← detection block. Normalised `[0, 1]`. |
| 10 | detection centre y | `0.0` | " |
| 11 | detection width | `0.0` | " |
| 12 | detection height | `0.0` | " |
| 13 | detection confidence | `0.0` | " |
| 14–57 | reserved | `0.0` | always trimmed on the wire. |

Slots 2 and 3 are two different starvation modes and neither implies the other.
Slot 2 rising means vision lost the ball while frames kept coming. Slot 3 rising
means the camera/inference pipeline itself starved and the controller had nothing to
act on. Post-crash you cannot distinguish those from either counter alone, and the
distinction points at completely different root causes, so both are recorded.

### Why the detection block goes last

MAVLink 2 trims trailing zero bytes from the payload on the wire. Putting the
detection fields at the end means a no-detection tick physically shrinks, and the 44
reserved slots trim away on *every* message (~180 bytes each). Wire cost at 5 Hz:
~84 B/msg with a detection, ~64 B/msg without, against 264 B untrimmed.

Note this saves link bandwidth only. The ulog still records all 58 floats, because
the `debug_array` uORB struct is fixed-size.

### The command slots are NOT always degrees

`DroneMover.move_to_target_zenith_async` forwards its arguments to one of two
different MAVSDK calls, depending on `drone.config.use_set_attitude`:

| `use_set_attitude` | MAVSDK call | slots 6–7 mean |
|---|---|---|
| `true` | `set_attitude` | **degrees** (an angle) |
| `false` ← **rig default** | `set_attitude_rate` | **deg/s** (a rate) |

The replay of the 2026-04-27 UAE dive shows commanded roll ranging to ±235, which is
absurd as a bank angle and entirely normal as a roll *rate*. Anyone reading the ulog
after a crash must know which one they are looking at, so the slots are named
`cmd_roll` / `cmd_pitch` rather than `..._deg`, and this is called out in
`UlogTraceSample`.

### Absent values are `0.0` everywhere — never NaN

`NaN` is the obvious sentinel for "no value", and it is **unusable here**. MAVSDK
serialises message fields as JSON; `json.dumps` emits a bare `NaN` token; MAVSDK's own
C++ parser then rejects the message outright — `Failed to parse JSON`
(`mavlink_direct_impl.cpp`). One `NaN` anywhere in the array silently costs the
*entire* sample. This was found by sending real messages through a real
`mavsdk_server`, not by reading the docs, and it is why `to_floats()` clamps every
non-finite value to `0.0`: better to lose one field than the whole sample.

Zeros are unambiguous in both blocks anyway, because a zeroed reading is not something
the live system can produce:

- **detection**: a width, height, and confidence of exactly zero is a far more
  reliable signature of "nothing was detected" than of a real target sitting
  dead-centre at zero confidence.
- **command**: a real command always carries non-zero thrust — `THRUST_CRUISE`, and
  even `idle()` uses `IDLE_THRUST / 2` — so an all-zero roll/pitch/thrust means no
  command has been issued.

### `frame_id` precision

float32 holds integers exactly to 2²⁴ = 16,777,216; the next integer, 16,777,217,
rounds down, and above that only even integers exist, so `+1` increments stop being
distinguishable. At 3 fps that is ~65 days of continuous running, at 20 fps ~9.7 days,
and the counter resets every process start. Not a real limit — but a property of the
format worth recording rather than rediscovering from a log full of duplicate frame
ids.

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

**On the rig this is already satisfied.** Checked against the real Auterion PX4 on
`bdd-sd9-mandarin` (2026-07-13): `SDLOG_PROFILE = 1057` = 1024 (internal temperature
sensors) + **32 (debug)** + 1 (default). So no param change and no FC reboot are
needed — the trace will be logged as soon as the app sends it.

The app nonetheless **checks and warns, but never writes**. Auto-setting the param
would give the vision app the power to mutate flight-controller configuration, which
is a new and invasive capability for a benefit — a trace that only starts working
after the next reboot anyway — that does not justify it. The startup warning is what
stops a *differently* configured FC (a swapped board, a params reset) from being
discovered post-crash, which is the one moment it cannot be fixed.

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
- Absent detection → `0.0` in slots 9–13, so that everything from slot 9 on is zero
  and the payload trims. Absent command → `NaN` in slots 6–8.
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

**On-rig, against the real Auterion PX4** (`bdd-sd9-mandarin`, 2026-07-13) — DONE:

- The host suite passes on the Pi's Python 3.11 as well as the dev host's 3.12.
- `SDLOG_PROFILE = 1057` — debug bit already set. `SDLOG_MODE = 0`, so PX4 logs only
  while ARMED, which is exactly the crash window we care about.
- Over the real 1 Mbps serial link: 3/3 direct `send_debug_array` calls, 25 tracer
  sends at 5 Hz, zero failures.
- **The decisive check.** Streaming traces while asking PX4's own nsh shell for
  `listener debug_array` echoed our payload straight back out of the flight
  controller — `3.25, 1.75, -2.5, 1.5, 0.625, 0.375, 0.4375, 0.0625, 0.03125, 0.875`,
  every slot matching what we sent, tail zeroed. That proves PX4 *received* the
  DEBUG_FLOAT_ARRAY, *parsed* it, and *published* it on the `debug_array` uORB topic —
  which, with the debug bit set, is precisely what the logger writes.

  This route replaced the original plan of downloading the ulog over MAVLink FTP:
  `log_files.get_entries()` times out on this link, and the FC only logs while armed
  anyway. The listener proves the same thing without arming the aircraft.

Two traps worth recording for anyone repeating this. Killed runs leave `mavsdk_server`
processes alive holding the serial port; a second one then silently contends for it and
every `param`/`shell` call starts timing out. Kill them with `pkill -9 -x mavsdk_server`
— **not** `pkill -f`, which matches the SSH shell's own command line and kills your
session. And `DroneMover.__del__` calls `asyncio.run()` from a running loop, so any
script that connects must `os._exit()` rather than fall off the end.

Still outstanding: a live ARMED flight, where the trace lands in a real ulog on the FC's
SD card. Nothing in the code path is untested at this point — only the arming is.

## Stress test: what happens with no rate limiting (2026-07-13, on the rig)

Run at 5 / 20 / 50 / 100 Hz and unlimited, against the real PX4.

**The control loop is unaffected.** The tracer's send shares the control loop's asyncio
loop, so the risk was that it delays commands. Measured as the lateness of a simulated
20 Hz control tick: **p50 0.54 ms quiet → 0.60 ms at one-message-per-frame → 0.70 ms at
50 Hz**. Noise.

**Telemetry is unaffected.** The control loop reads attitude + odometry at 50 Hz over the
same link, so starving them would make it fly on stale data. They hold **50.0 Hz exactly**
at every trace rate, including a 2200 msg/s flood.

**The link is not the bottleneck, and it is not 1 Mbps.** PX4's own `uorb top` reports its
`debug_array` publish rate tracking ours 1:1 all the way to **2221 msg/s** — which at
~88 B/msg is ~1.95 Mbps, impossible on a 1 Mbps UART. The device is a **USB CDC-ACM**
virtual serial port (`/dev/serial/by-id/usb-Auterion_PX4_FMU_v6X...`), where the
`:1000000` baud in the connection string is *ignored*. The real ceiling is USB.

**The bottleneck is the FC's LOGGER, and it fails silently.** Measured as the growth of
the actual `.ulg` on the FC's SD card (each `debug_array` record is ~256 B):

| trace rate | log growth | trace contribution | records persisted |
|---|---|---|---|
| quiet | 38.7 kB/s | — | — |
| **20 Hz (per frame)** | 43.7 kB/s | **+5.0 kB/s** | **~20/s — all of them** |
| unlimited (2644/s) | 88.5 kB/s | +49.8 kB/s | **~195/s — 7% of them** |

At 20 Hz the +5.0 kB/s is exactly 20 × 256 B, which independently confirms the records
really are being written. But an unlimited flood only grows the log by ~195 records/s: the
logger polls the uORB topic more slowly than we publish and keeps just the latest sample
per poll. It reports **`dropouts: 0` while discarding 93% of them** — these are not buffer
overruns, so nothing warns you.

**Conclusion.** Per-frame logging (20–30 Hz) is safe and fully persisted, costing ~5 kB/s
on top of the FC's own ~39 kB/s of logging. Unlimited is not dangerous, merely pointless:
it burns ~2 Mbps of USB and 50 kB/s of SD to persist a seventh of what it sends, and
bloats the flight log 2.3× with a stream full of invisible holes. The config's 50 Hz
ceiling sits comfortably inside the fully-logged region; **raising it means re-measuring
that ~195/s logger ceiling, not just editing the number.**

## Out of scope

- Setting `SDLOG_PROFILE` from the app.
- Any change to what the Pi-side `_DEBUG/` logs record.
- A decoder/plotting script for the ulog side — `pyulog` already dumps `debug_array`,
  and the slot table above is the whole format.
