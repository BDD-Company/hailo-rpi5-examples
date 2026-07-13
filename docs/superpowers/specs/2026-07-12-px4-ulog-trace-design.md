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

58 float32 slots is far more than the ~16 we need, so the float16 / bit-packing idea
sketched in the TODO is unnecessary for the *values*. Measured cost is ~375 B/s at 5 Hz.
Packing every field would save bandwidth we do not need and make the log unreadable
without a bespoke decoder, so each value goes in as a plain float32 that `pyulog` /
FlightReview read directly. (The one place bit-packing *does* earn its keep is the
per-frame history, where the alternative would be one slot per frame.)

The alternative, `NAMED_VALUE_FLOAT` per field, is more readable in FlightReview but
needs ~12 messages per tick and — decisively — the fields would no longer be
guaranteed to come from the same control iteration. A trace whose fields are skewed
across ticks is misleading exactly when you most need to trust it.

### Why we can send it at all

MAVSDK's Python bindings have no `mavlink_passthrough` plugin, and the rig's link is
serial-USB (`connection_string: usb` → `serial:///dev/serial/by-id/...:1000000`),
which `mavsdk_server` opens exclusively. A second pymavlink connection to the same
port is therefore impossible.

(That `:1000000` baud is a red herring — it is a USB CDC-ACM virtual serial port, so
the baud is ignored entirely and the real ceiling is USB. See the stress test below.)

The installed MAVSDK (3.15.3) ships the `mavlink_direct` plugin, which sends an
arbitrary MAVLink message as JSON fields over the link MAVSDK already holds. No
second connection, no new dependency.

## A record is an INTERVAL SUMMARY, not a snapshot

The control loop runs at ~20–30 fps and we send at 5 Hz, so one record stands for ~5–6
iterations. A snapshot of the sending frame would throw the other five away — and worse,
it would *misreport*: a good detection followed by a single miss would log as "no
detection", erasing the detection from the log entirely. That is the one thing a
post-crash trace must never do.

So every field is aggregated over the interval **since the last record was SENT**:

- **counts** — how many frames arrived carrying no viable target, and how many
  iterations got no frame at all;
- **a per-frame history** — the outcome of up to 21 frames, packed into 2 slots with an
  explicit frame count;
- **the LAST VIABLE DETECTION of the interval**, not the last frame's.

The accumulator resets on **dispatch**, not when a record merely becomes due. If a send
is skipped (previous one still in flight), the interval simply keeps growing, so the
next record that does go out accounts for every frame since the last one that did. A
slow link can never silently eat the data we are trying to preserve.

`pd_coeff_p` and the command stay **momentary** — they are the values *in force* at send
time, which is what you want to know. The history tells you how many frames ago the
logged detection actually occurred, so there is no ambiguity about whether the gain was
computed from it.

## Trace layout

`DEBUG_FLOAT_ARRAY`, `name = "BDD"`, `array_id = 0`. **Format version 3.**

| slot | field | absent / idle | notes |
|-----:|-------|---------------|-------|
| 0 | format version | — | `3`. Non-zero, so a message can never truncate to nothing. |
| 1 | frame id | — | the **most recent** frame of the interval; the one in our log prefix `frame=#0123`. |
| 2 | frames with no viable detection | `0` | **count over the interval**: frames that arrived carrying no usable target. |
| 3 | iterations with no frame at all | `0` | **count over the interval**: the pipeline itself starved. |
| 4 | frame history: count + frames 0–8 | `0` | count in bits 0–5, then 2 bits per frame. |
| 5 | frame history: frames 9–20 | `0` | 2 bits per frame. |
| 6 | `pd_coeff_p.x` | — | in force at send time, after the final clamp into `[p_min, p_max]`. |
| 7 | `pd_coeff_p.y` | — | " |
| 8 | commanded roll | `0.0` | verbatim, as passed to `DroneMover`, pre-clamp. **Units vary — see below.** |
| 9 | commanded pitch | `0.0` | " |
| 10 | commanded thrust | `0.0` | " |
| 11 | detection centre x | `0.0` | ← detection block. The **last viable** detection of the interval. Normalised `[0, 1]`. |
| 12 | detection centre y | `0.0` | " |
| 13 | detection width | `0.0` | " |
| 14 | detection height | `0.0` | " |
| 15 | detection confidence | `0.0` | " |
| 16–57 | reserved | `0.0` | always trimmed on the wire. |

Slots 2 and 3 are two different starvation modes and neither implies the other.
Slot 2 rising means vision lost the ball while frames kept coming. Slot 3 rising
means the camera/inference pipeline itself starved and the controller had nothing to
act on. Post-crash you cannot distinguish those from either counter alone, and the
distinction points at completely different root causes, so both are recorded.

### The frame history (slots 4–5)

Each frame's outcome is one of four, so **2 bits each**:

| value | outcome |
|---:|---|
| 0 | `NO_FRAME` — the loop got no frame at all; the pipeline starved |
| 1 | `NO_DETECTIONS` — a frame arrived, the detector found nothing |
| 2 | `NO_VIABLE` — a frame arrived with detections, none good enough to act on |
| 3 | `VIABLE` — a frame arrived and we picked a target |

**The history carries its own frame count.** Without it, a partially-filled slot's
padding zeros would be indistinguishable from genuine `NO_FRAME` starvation (`NO_FRAME`
*is* zero), and every short interval would read as though the pipeline had died. v2
solved that with a fifth `NO_DATA` value at 4 bits per frame; the explicit count solves it
for 6 bits total and buys back the depth — **21 frames instead of 12**, in the same two
slots.

Bit layout, fixed by the format version:

```
slot 4:  bits  0..5    frame count (0..63; capacity is 21)
         bits  6..23   9 outcomes, 2 bits each   -> interval frames 0..8
slot 5:  bits  0..23   12 outcomes, 2 bits each  -> interval frames 9..20
```

No outcome straddles the slot boundary, so a decoder never stitches two floats.

**24 bits per slot, deliberately.** A float32 represents integers exactly only up to 2²⁴,
so that is the widest field that survives the trip intact. Anything wider would silently
round — and a rounded bitfield decodes to *garbage*, not to an obviously-wrong number.

**Oldest first**: index 0 is the first frame of the interval. On overflow — an interval
longer than 21 frames, which needs `rate_hz` set very low — the **latest** frames that do
not fit are simply not recorded, and the count reports what was. The counts in slots 2 and
3 still cover the whole interval, so this loses detail, never data.

(The tradeoff is worth naming: in the final record before a crash, the newest frames are
the ones nearest impact. It only bites at pathologically low rates — at 0.1 Hz on a 30 fps
loop an interval is 300 frames — and the aggregate stays exact regardless.)

### Why the detection block goes last

MAVLink 2 trims trailing zero bytes from the payload on the wire. Putting the
detection fields at the end means an interval with no viable detection physically
shrinks, and the 42 reserved slots trim away on *every* message. Measured over the
2026-04-27 UAE dive replay: **85 B/msg mean, against 264 B untrimmed** (~375 B/s at
5 Hz).

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

- `TRACE_FORMAT_VERSION`, `TRACE_NAME`, the history widths, and the slot indices, in
  one place — so the packer, the tests and any decoder share one definition.
- `FrameOutcome` — the 5-value enum packed into the history (`NO_DATA`, `NO_FRAME`,
  `NO_DETECTIONS`, `NO_VIABLE`, `VIABLE`).
- `pack_history()` / `unpack_history()` — the bit layout, and its inverse. The decoder
  exists so the tests, and anyone reading a real ulog, never re-derive it by hand.
- `UlogTraceSample` — a frozen dataclass of the fields above, with `to_floats()`
  returning the 58-slot list. Pure, trivially testable.
- `TraceAccumulator` — gathers the interval between two SENT records: the counts, the
  history ring, and the last viable detection. **Kept separate from `UlogTracer` and
  free of asyncio**, because the aggregation rules are the part with the interesting
  behaviour and they should be testable without an event loop.
  - `reset()` happens on **dispatch**, not when a record becomes due. A skipped send
    therefore extends the interval rather than discarding it.
  - `reset()` also clears the detection: if it survived, an interval that saw nothing
    would keep re-reporting an old target as though it were still in view.
- `UlogTracer(send_fn, rate_hz, name, array_id, now_fn=time.monotonic)`.
  - `send_fn` is an **explicit parameter**, an async callable
    `(name: str, array_id: int, data: list[float]) -> None`. It is a live object and
    so must never travel inside a config dict (project rule: `control_config` is
    values-only).
  - `note(frame_id, outcome, pd_p, cmd, detection)` is the hot path, called once per
    control-loop iteration. Synchronous and cheap: if disabled, return. Otherwise fold
    the iteration into the accumulator; then, if `1 / rate_hz` has elapsed, build the
    record, reset, and fire `send_fn` as a detached `asyncio` task. It never awaits the
    send.
  - If a previous send is still in flight, skip *the send* and count it — but keep
    accumulating. A stalled link can then never build a backlog nor stall the control
    loop, and equally never loses the frames it was too slow to report.
  - Send exceptions are caught and counted, logged once and thereafter periodically.
    A broken trace degrades to no trace; it never propagates into the control loop.
    (Project rule: on the control path, degrade — never hard-fail.)
  - Counters (`sent`, `skipped`, `failed`) are exposed for the shutdown summary.

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
    zeros. That mode is off on the rig (`follow_target_position_ned: false`), and a
    stale-but-plausible attitude would be worse than an honest "no attitude command".
- `last_attitude_command()` returns the stored triple. (Distinct from the existing
  `last_command()`, which returns a formatted *string* from the `debug_collect_call_info`
  proxy and carries no usable numbers.)
- At the end of `startup_sequence`, read the `SDLOG_PROFILE` param and log a loud
  WARNING naming the exact fix if the Debug bit (32) is clear. The app never *writes*
  flight-controller params — see "Provisioning".

### `BDD/drone_controller.py`

Build the tracer once, before the loop, from `control_config.ulog_trace` plus the live
`drone`. Then `note()` **every** iteration, from two sites:

1. Immediately after `command_sent_ns = time.monotonic_ns()`, classifying the frame:
   `VIABLE` if a target was picked, `NO_DETECTIONS` if the detector returned nothing at
   all, `NO_VIABLE` if it returned detections but none good enough to act on. (The
   classification is exact: `_pick_target_detection` already filters by
   `confidence_min`, so `picked is not None` *is* viability.)

   That point is **after** the awaited command has gone out, so the trace adds nothing
   to capture→command latency — the metric this project optimises above all others.

2. From the **starved path**, as `NO_FRAME`, before its `continue`. If the vision
   pipeline dies outright the loop never reaches site 1, and the trace would simply
   fall silent — losing the one datum that says *why*. Here it keeps beating with a
   rising starvation count, which names the failure.

Because the tracer aggregates, both sites just record an outcome; the record is built
and sent on whichever `note()` happens to cross the period boundary.

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

## Rate: uncapped, and what a record covers

`ulog_trace.rate_hz` sets records per second. **`0` means uncapped**: one record per
control-loop iteration, each covering exactly one frame — the finest resolution the ulog
can hold. That is *not* the same as "off" (an absent section, or `enabled: false`).

Frames per record must track **both** knobs — the configured rate (explicit) and the
control-loop rate (implicit: the loop slows when the hardware thermally throttles, and
nobody reconfigures anything). Swept against the 39 m/s UAE dive, varying the log rate and
decimating the log's frames to simulate a throttled loop:

| loop fps | rate Hz | records | frames/record | accounted | lost |
|---:|---:|---:|---:|---:|---:|
| 20.0 | **0 (uncapped)** | 1655 | **1** | 1655 / 1655 | 0 |
| 20.0 | 2.0 | 158 | 11 | 1652 / 1655 | 3 |
| 20.0 | 5.0 | 366 | 5 | 1655 / 1655 | 0 |
| 20.0 | 20.0 | 1023 | 2 | 1655 / 1655 | 0 |
| **10.0** (throttled) | 5.0 | 322 | **3** | 827 / 828 | 1 |
| **6.7** (throttled) | 5.0 | 276 | **2** | 551 / 552 | 1 |

Both knobs move it, and **nothing is ever lost**: at every combination the accounted-for
iterations equal the frames the loop ran, minus only the final interval still sitting in
the accumulator when the loop stopped. (In a crash the process dies mid-interval and that
tail is unrecoverable anyway — there is nothing to flush it to.)

A record covers `floor(fps / rate) + 1` frames, not `fps / rate`: the frame that
*triggers* the send is itself part of the interval it reports.

Uncapped is safe on this rig — at 20–30 fps it sits far below the flight controller's
~195 records/s logger ceiling (see the stress test), costs ~5 kB/s of log, and perturbs
neither the control loop nor the telemetry streams.

## Failure modes

| failure | behaviour |
|---|---|
| `mavlink_direct` unsupported by the running `mavsdk_server` | first send raises; counted, warned once, tracer keeps no-oping. Flight unaffected. |
| link saturated / send stalls | `note()` sees the in-flight task and skips *the send*, but keeps accumulating. The loop never blocks, and the frames it was too slow to report are not lost — the next record covers them. |
| `SDLOG_PROFILE` Debug bit clear | messages still sent (harmless); loud WARNING at startup naming the fix. |
| `ulog_trace` section absent | tracer disabled, zero overhead. |
| interval longer than the 12-frame history | the oldest per-frame outcomes are dropped; the counts still cover the whole interval, so nothing aggregate is lost. |
| PX4 not connected / MockDroneMover | `send_fn` records or no-ops; replay harness runs clean. |

Nothing in this feature can abort the control loop. A stalled control loop can wreck
the hardware, and a *debugging* feature must never be the thing that stalls it.

## Testing

**Host unit tests** — `BDD/test_ulog_trace.py`, 22 of them:

*Layout* — `to_floats()` puts each field in its documented slot; the array is exactly 58
long; the format version is slot 0 and **is 2**; reserved slots are zero; a no-detection
interval zeroes everything from slot 11 on so the payload trims; non-finite values are
clamped.

*History packing* — round-trips newest-first; a short interval pads with `NO_DATA` and
**not** `NO_FRAME` (the whole reason the field is 4 bits wide); a full history stays
exactly representable in float32; overflow drops the *oldest*.

*Aggregation* — the rules that actually changed, tested directly against
`TraceAccumulator` with no event loop:

- the worked example from the task: 2 no-viable + 1 no-frame then a detection →
  `frames_no_detection = 2`, `frames_without_frame = 1`, detection present;
- **a detection on the first of six frames survives five later misses** — the case a
  snapshot design gets wrong, and the reason for this whole change;
- the *last* viable detection of an interval wins;
- `reset()` clears the detection, so an interval that saw nothing reports zeros rather
  than re-reporting a target already logged;
- `pd_p` and the command are the values in force at the end of the interval.

*Tracer* — one record per period summarising every frame between; **a stalled send does
not lose the frames it could not report** (the interval keeps growing and the next record
accounts for all of them); the history reaches the wire; a raising `send_fn` is swallowed
and counted; disabled → never sends; `note()` from a thread with no event loop does not
raise.

**Config test** — `ulog_trace` parses, defaults apply when the section is absent, and
an out-of-range `rate_hz` is rejected. Mirrored in `TestConfig`.

**Replay, off-rig** — drive the REAL control thread against the 39 m/s UAE dive log
(`uae-replay-log-corpus`) with a capturing fake `send_fn`. This is the check that the
wiring is real rather than merely well-unit-tested, and it produced the decisive
number: over the 1655-frame dive,

    1012 counted in slots 2+3  +  643 VIABLE frames in the histories  =  1655

**every single control-loop iteration is accounted for**, with none double-counted and
none lost between records. Median 5 frames per record (max 5, inside the 12-frame
history). 20 records carry a detection whose *newest* frame was a miss — under the old
snapshot design all 20 would have logged `det = 0`. Records carrying a detection rose
141 → 161.

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
