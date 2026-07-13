#! /usr/bin/env python3
"""Mirror a slice of the decision pipeline into the PX4's own persistent ulog.

After a crash the Pi's SD card may be unreadable, taking every `_DEBUG/` log and
video with it. The flight controller's log survives far more reliably, so a small
trace of what we saw and what we commanded goes there too, as a MAVLink
DEBUG_FLOAT_ARRAY that PX4 republishes as the `debug_array` uORB topic.

A record is a SUMMARY OF AN INTERVAL, not a snapshot of one frame. The control loop
runs at ~20-30 fps and we send at 5 Hz, so each record stands for ~5-6 frames. A
snapshot would throw the other 5 away — and worse, it would misreport: one missed
frame right after a good detection would log as "no detection", hiding the detection
entirely. So each record carries, for the whole interval SINCE THE LAST SENT RECORD:

  * counts     - how many frames arrived with no viable target, and how many
                 iterations got no frame at all;
  * a history  - the per-frame outcome of the last 12 frames, packed into 2 slots;
  * the LAST VIABLE DETECTION seen in the interval, not merely the last frame's.

Two rules govern everything here, because this sits on the control path:

  * `note()` never blocks. It accumulates, and on the send tick fires the send as a
    detached task.
  * Nothing raises into the control loop. A dead link, an unsupported plugin, a call
    from a thread with no event loop — all degrade to "no trace" and are counted. A
    stalled control loop can wreck the airframe; a *debug* feature must never be the
    thing that stalls it.

PX4 only writes `debug_array` into the log when the `SDLOG_PROFILE` param has its
Debug bit (32) set, and reads that param at boot. `DroneMover` warns at startup when
it is clear. Measured ceiling: the FC's logger persists only ~195 records/s and lies
about it (`dropouts: 0` while discarding the rest), so keep the rate modest.

Slot layout, FORMAT VERSION 2 (see the design doc; v1 had no history and momentary
counters, and is not upward-compatible — hence the version bump):

     0  format version (=2)              8  commanded roll     <- deg OR deg/s
     1  frame id (most recent)           9  commanded pitch    <- ditto
     2  frames with no viable detection 10  commanded thrust
     3  iterations with no frame at all
     4  frame history, newest 6 frames  11  detection centre x
     5  frame history, previous 6       12  detection centre y
     6  pd_coeff_p.x                    13  detection width
     7  pd_coeff_p.y                    14  detection height
                                        15  detection confidence
                                     16-57  reserved (zero)

The detection block is LAST so that a no-detection interval is all-zero from slot 11
on and MAVLink 2 trims it off the wire.
"""

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Awaitable, Callable

from helpers import XY, Detection

logger = logging.getLogger(__name__)


# Bump when the slot layout changes, so a decoder can tell the layouts apart from the
# log alone. Slot 0 carries it; being non-zero it also stops a message from ever
# truncating away to nothing on the wire.
#
#   v1 - momentary snapshot, no frame history.
#   v2 - interval summary: counts since the last sent record, a packed per-frame
#        history, and the last VIABLE detection of the interval.
TRACE_FORMAT_VERSION = 2

# MAVLink DEBUG_FLOAT_ARRAY carries data[58]; `name` is char[10].
TRACE_SLOTS    = 58
TRACE_NAME     = "BDD"
TRACE_ARRAY_ID = 0


class FrameOutcome(IntEnum):
    """What became of one control-loop iteration. Packed 4 bits at a time into the
    history slots, so the values must stay in 0..15.

    NO_DATA is why this is 4 bits wide and not 2. Four *outcomes* would fit in 2 bits,
    but then a partially-filled history slot would pad with zeros that are
    indistinguishable from genuine NO_FRAME starvation. A distinct "nothing here"
    value costs a bit and removes the ambiguity — and leaves 11 spare outcomes.
    """
    NO_DATA        = 0   # padding: the interval had fewer frames than the history holds
    NO_FRAME       = 1   # the control loop got no frame at all — the pipeline starved
    NO_DETECTIONS  = 2   # a frame arrived, the detector found nothing
    NO_VIABLE      = 3   # a frame arrived with detections, none good enough to act on
    VIABLE         = 4   # a frame arrived and we picked a target


# History packing, FIXED BY THE FORMAT VERSION. A decoder reads these off the version
# in slot 0, so they must never change without bumping it.
HISTORY_BITS_PER_FRAME  = 4
HISTORY_SLOTS           = 2
HISTORY_FRAMES_PER_SLOT = 24 // HISTORY_BITS_PER_FRAME              # 6
HISTORY_FRAMES          = HISTORY_SLOTS * HISTORY_FRAMES_PER_SLOT   # 12

# 24 bits per slot, deliberately: a float32 represents integers exactly only up to
# 2**24, so a 24-bit field is the widest that survives the trip intact. 6 frames * 4
# bits = 24. Anything wider would silently round in the log.
_MAX_SLOT_VALUE = (1 << 24) - 1

# Slot indices, so the packer, the tests and any decoder agree on one definition.
SLOT_FORMAT_VERSION       = 0
SLOT_FRAME_ID             = 1
SLOT_FRAMES_NO_DETECTION  = 2
SLOT_FRAMES_WITHOUT_FRAME = 3
SLOT_HISTORY_0            = 4
SLOT_HISTORY_1            = 5
SLOT_PD_P_X               = 6
SLOT_PD_P_Y               = 7
SLOT_CMD_ROLL             = 8
SLOT_CMD_PITCH            = 9
SLOT_CMD_THRUST           = 10
SLOT_DET_X                = 11
SLOT_DET_Y                = 12
SLOT_DET_W                = 13
SLOT_DET_H                = 14
SLOT_DET_CONF             = 15

# How many failures to swallow silently between log lines, once the first has been
# reported. A dead link must not turn into a per-tick log flood.
_FAILURE_LOG_EVERY = 100


def _finite(value) -> float:
    """Non-finite values become 0.0.

    MAVSDK serialises the message fields as JSON, and `json.dumps` emits a bare `NaN` /
    `Infinity` token, which its own C++ parser then REJECTS — costing us the whole
    message ("Failed to parse JSON", mavlink_direct_impl.cpp). Verified against a real
    mavsdk_server. So one stray NaN from anywhere upstream would silently delete a trace
    sample; clamping it here means we lose one field instead.
    """
    value = float(value)
    return value if math.isfinite(value) else 0.0


def pack_history(outcomes) -> list[float]:
    """Pack per-frame outcomes, NEWEST FIRST, into HISTORY_SLOTS float32 slots.

    `outcomes[0]` is the frame this record was emitted on, `outcomes[i]` the frame i
    before it. Within a slot, frame i sits at bits `4*i .. 4*i+3`. Entries beyond the
    interval's length are NO_DATA (0), so a decoder can tell "nothing here" from a real
    NO_FRAME.

    More frames than the history holds: the OLDEST are dropped. The counts in slots 2
    and 3 still cover the whole interval, so no aggregate information is lost — only
    the per-frame detail of the frames furthest from the record, which are the least
    interesting ones after a crash.
    """
    slots = [0.0] * HISTORY_SLOTS
    for i, outcome in enumerate(list(outcomes)[:HISTORY_FRAMES]):
        packed = int(outcome) & 0xF
        slot, within = divmod(i, HISTORY_FRAMES_PER_SLOT)
        slots[slot] = float(int(slots[slot]) | (packed << (HISTORY_BITS_PER_FRAME * within)))
    for v in slots:
        assert 0 <= v <= _MAX_SLOT_VALUE, "history slot must stay inside float32's exact-integer range"
    return slots


def unpack_history(slots) -> list[FrameOutcome]:
    """Inverse of pack_history. Provided so the tests — and anyone reading a real ulog —
    have one authoritative decoder rather than re-deriving the bit layout by hand."""
    out = []
    for slot_index in range(HISTORY_SLOTS):
        raw = int(slots[slot_index])
        for within in range(HISTORY_FRAMES_PER_SLOT):
            out.append(FrameOutcome((raw >> (HISTORY_BITS_PER_FRAME * within)) & 0xF))
    return out


@dataclass(slots=True, frozen=True)
class UlogTraceSample:
    """One record: a summary of the interval since the previous SENT record.

    Absent values are 0.0 throughout, and are unambiguous in both blocks because a
    zeroed reading is not something the live system can produce:

      * detection: a width, height and confidence of exactly zero is a far more
        reliable signature of "nothing viable was seen in this interval" than of a real
        target sitting dead-centre at zero confidence.
      * command: a real command always carries non-zero thrust (THRUST_CRUISE, and even
        idle() uses IDLE_THRUST / 2), so an all-zero roll/pitch/thrust means no command
        has been issued.

    NaN is NOT used as the absent sentinel: MAVSDK's JSON parser rejects it and drops
    the whole message. See _finite().
    """
    # The most recent frame in the interval.
    frame_id : int = 0

    # Counts SINCE THE LAST SENT RECORD, not momentary values.
    frames_no_detection  : int = 0   # frames that arrived carrying no viable target
    frames_without_frame : int = 0   # iterations where no frame arrived at all

    # Per-frame outcomes, newest first. Packed into HISTORY_SLOTS slots.
    history : tuple = ()

    # The gain in force at send time, after the final clamp into [p_min, p_max].
    pd_p : XY = field(default_factory=XY)

    # Verbatim, as handed to DroneMover: before the upside-down veto and before any
    # clamping. This is the decision the vision pipeline made; what the FC then did with
    # it is already in the ulog's own attitude topics.
    #
    # UNITS DEPEND ON drone.config.use_set_attitude, because DroneMover forwards these
    # to two different MAVSDK calls:
    #   use_set_attitude: true  -> set_attitude      -> DEGREES  (an angle)
    #   use_set_attitude: false -> set_attitude_rate -> DEG/SEC  (a rate)  <- rig default
    # So do not read a value of 230 as an impossible bank angle: on the current rig it is
    # 230 deg/s of roll rate, and is entirely normal.
    #
    # Sticky, and legitimately so: an offboard setpoint stays in force in PX4 until it is
    # replaced, so on a tick that issues no command the last one is still what the FC is
    # flying. That is a fact about the aircraft, not a stale reading.
    cmd_roll   : float = 0.0
    cmd_pitch  : float = 0.0
    cmd_thrust : float = 0.0

    # The LAST VIABLE detection of the interval — NOT the last frame's. A detection on
    # the first of six frames followed by five misses must still be logged: the whole
    # point of the trace is what we saw, and a snapshot would report "nothing".
    # Normalised [0, 1]. All-zero means nothing viable in the interval at all.
    det_x    : float = 0.0
    det_y    : float = 0.0
    det_w    : float = 0.0
    det_h    : float = 0.0
    det_conf : float = 0.0

    def to_floats(self) -> list[float]:
        data = [0.0] * TRACE_SLOTS
        data[SLOT_FORMAT_VERSION]       = float(TRACE_FORMAT_VERSION)
        data[SLOT_FRAME_ID]             = _finite(self.frame_id)
        data[SLOT_FRAMES_NO_DETECTION]  = _finite(self.frames_no_detection)
        data[SLOT_FRAMES_WITHOUT_FRAME] = _finite(self.frames_without_frame)

        history = pack_history(self.history)
        data[SLOT_HISTORY_0] = history[0]
        data[SLOT_HISTORY_1] = history[1]

        data[SLOT_PD_P_X]    = _finite(self.pd_p.x)
        data[SLOT_PD_P_Y]    = _finite(self.pd_p.y)
        data[SLOT_CMD_ROLL]   = _finite(self.cmd_roll)
        data[SLOT_CMD_PITCH]  = _finite(self.cmd_pitch)
        data[SLOT_CMD_THRUST] = _finite(self.cmd_thrust)
        data[SLOT_DET_X]    = _finite(self.det_x)
        data[SLOT_DET_Y]    = _finite(self.det_y)
        data[SLOT_DET_W]    = _finite(self.det_w)
        data[SLOT_DET_H]    = _finite(self.det_h)
        data[SLOT_DET_CONF] = _finite(self.det_conf)
        return data


class TraceAccumulator:
    """Gathers the interval between two SENT records.

    Kept separate from UlogTracer, and free of asyncio, so the aggregation rules can be
    tested directly — they are the part with the interesting behaviour.

    Reset happens when a record is DISPATCHED, not when one is merely due. If a send is
    skipped (rate limit, or a previous send still in flight) the interval simply keeps
    growing, so nothing is ever silently dropped: the next record that does go out
    accounts for every frame since the last one that did.
    """

    def __init__(self):
        self._history = deque(maxlen=HISTORY_FRAMES)   # newest first
        self.reset()

    def reset(self) -> None:
        self.frames_no_detection = 0
        self.frames_without_frame = 0
        self._history.clear()
        # Cleared on every dispatch: if the next interval sees nothing viable, the record
        # must say so with zeros rather than repeat a detection from a window that has
        # already been logged.
        self._detection: Detection | None = None
        self._frame_id = 0
        self._pd_p = XY()
        self._cmd = (0.0, 0.0, 0.0)
        self._notes = 0

    @property
    def empty(self) -> bool:
        return self._notes == 0

    def note(self, *, frame_id: int, outcome: FrameOutcome, pd_p: XY,
             cmd: tuple, detection: Detection | None = None) -> None:
        """Fold one control-loop iteration into the interval."""
        self._notes += 1
        self._history.appendleft(outcome)       # newest first
        self._frame_id = frame_id
        self._pd_p = pd_p
        self._cmd = cmd

        if outcome is FrameOutcome.NO_FRAME:
            self.frames_without_frame += 1
        elif outcome is FrameOutcome.VIABLE:
            # Last viable WINS: a later miss must never erase it.
            if detection is not None:
                self._detection = detection
        else:
            self.frames_no_detection += 1

    def build(self) -> UlogTraceSample:
        d = self._detection
        bbox = d.bbox if d is not None else None
        roll, pitch, thrust = self._cmd
        return UlogTraceSample(
            frame_id             = self._frame_id,
            frames_no_detection  = self.frames_no_detection,
            frames_without_frame = self.frames_without_frame,
            history              = tuple(self._history),
            pd_p                 = self._pd_p,
            cmd_roll             = roll,
            cmd_pitch            = pitch,
            cmd_thrust           = thrust,
            det_x    = bbox.center.x if bbox else 0.0,
            det_y    = bbox.center.y if bbox else 0.0,
            det_w    = bbox.size.x   if bbox else 0.0,
            det_h    = bbox.size.y   if bbox else 0.0,
            det_conf = d.confidence  if d is not None else 0.0,
        )


# (name, array_id, data) -> awaitable. DroneMover.send_debug_array satisfies it.
SendFn = Callable[[str, int, list[float]], Awaitable[None]]


class UlogTracer:
    """Accumulates every control-loop iteration and ships a summary at `rate_hz`,
    fire-and-forget.

    `send_fn` is passed explicitly rather than pulled out of a config dict: it is a live
    object, and config here carries values only.

    Pass `send_fn=None` to disable — `note()` then costs one attribute test.
    """

    def __init__(self,
                 send_fn  : SendFn | None,
                 *,
                 rate_hz  : float = 5.0,
                 name     : str   = TRACE_NAME,
                 array_id : int   = TRACE_ARRAY_ID,
                 now_fn   : Callable[[], float] = time.monotonic) -> None:
        self._send_fn  = send_fn
        self._name     = name
        self._array_id = array_id
        self._now      = now_fn
        self._period_s = (1.0 / rate_hz) if rate_hz > 0 else 0.0

        self._acc = TraceAccumulator()
        self._last_sent_at : float | None = None
        self._inflight : asyncio.Task | None = None

        self.sent    = 0   # records handed to the link
        self.skipped = 0   # a record was due, but the previous send was still in flight
        self.failed  = 0   # the send itself blew up

    @property
    def enabled(self) -> bool:
        return self._send_fn is not None

    def note(self, *, frame_id: int, outcome: FrameOutcome, pd_p: XY,
             cmd: tuple, detection: Detection | None = None) -> None:
        """Hot path, called once per control-loop iteration. Synchronous, cheap, and it
        does not raise. Accumulates always; sends only when the period has elapsed."""
        if self._send_fn is None:
            return

        self._acc.note(frame_id=frame_id, outcome=outcome, pd_p=pd_p,
                       cmd=cmd, detection=detection)

        now = self._now()
        if self._last_sent_at is not None and (now - self._last_sent_at) < self._period_s:
            return

        if self._inflight is not None and not self._inflight.done():
            # A stalled link must never build a backlog. Keep accumulating instead: the
            # interval grows, and the next record that goes out covers all of it.
            self.skipped += 1
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop on this thread. Nothing we can do, and it is not worth taking
            # the control loop down over a trace.
            self._note_failure("no running event loop to send the ulog trace from")
            return

        data = self._acc.build().to_floats()
        self._acc.reset()                       # reset on DISPATCH — see TraceAccumulator
        self._inflight = loop.create_task(self._send(data))
        self._last_sent_at = now

    async def _send(self, data: list[float]) -> None:
        try:
            await self._send_fn(self._name, self._array_id, data)
            self.sent += 1
        except Exception as e:
            self._note_failure(f"sending the ulog trace failed: {e}")

    def _note_failure(self, message: str) -> None:
        self.failed += 1
        # Report the first one, then thin out: a link that is down would otherwise log on
        # every single tick.
        if self.failed == 1 or self.failed % _FAILURE_LOG_EVERY == 0:
            logger.warning("ulog trace: %s (%d failure(s) so far)", message, self.failed)

    def summary(self) -> str:
        return (f"ulog trace: {self.sent} sent, {self.skipped} skipped "
                f"(send still in flight), {self.failed} failed")
