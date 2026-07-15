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
  * a history  - the per-frame outcome of up to 21 frames, packed into 2 slots with an
                 explicit frame count;
  * the LAST VIABLE DETECTION seen in the interval, not merely the last frame's.

Set `rate_hz: 0` for UNCAPPED mode: one record per control-loop iteration, each
covering exactly one frame. Measured safe on the rig (see the design doc).

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

The FORMAT VERSION rides in the MAVLink name ("BDD" + version = "BDD1"), not in a data
slot. Slot layout (see the design doc):

     0  frame id (most recent)           6  commanded roll     <- deg OR deg/s
     1  frames: no detection at all      7  commanded pitch    <- ditto
     2  frames: detections, none viable  8  commanded thrust
     3  frames: no frame at all
     4  pd_coeff_p.x                     9  detection centre x
     5  pd_coeff_p.y                    10  detection centre y
                                        11  detection width
                                        12  detection height
                                        13  detection confidence
                                        14  frame history: count + frames 0..8
                                        15  frame history: frames 9..20
                                     16-57  reserved (zero)

The frame history is LAST of the used slots. When it is empty — uncapped mode, where
a record covers one frame and the history is redundant (the three counts + detection
already pin down the frame's outcome) — the record is all-zero from slot 14 on, so
MAVLink 2 trims it off the wire and a gzipped .ulg compresses it away. When the history
IS present it is non-zero (the frame count fills the low bits), so it must go last or
nothing below it could trim.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Awaitable, Callable

from helpers import XY, Detection

logger = logging.getLogger(__name__)


# The format version rides in the MAVLink message NAME, not in a data slot: the name is
# `TRACE_NAME_PREFIX + str(version)` = "BDD1". Two reasons. It costs no data float — the
# `name` field (char[10]) is part of every DEBUG_FLOAT_ARRAY regardless — and, because on
# the wire `name` sits BEFORE `data` (order: time_usec, array_id, name, data), a non-zero
# name does not block MAVLink 2 from trimming the zero tail of `data`. A decoder reads the
# version off the name suffix: `int(name[len(TRACE_NAME_PREFIX):])`.
#
# Bump when the slot layout changes, so a decoder can tell the layouts apart. Earlier
# in-development formats (a momentary snapshot, then 4-bit and 2-bit histories, then the
# history-last reorder) never shipped, so the version is reset to 1 for the first real
# format rather than carrying that history forward.
TRACE_FORMAT_VERSION = 1

# MAVLink DEBUG_FLOAT_ARRAY carries data[58]; `name` is char[10].
TRACE_SLOTS        = 58
TRACE_NAME_PREFIX  = "BDD"
TRACE_NAME         = f"{TRACE_NAME_PREFIX}{TRACE_FORMAT_VERSION}"   # "BDD1"
TRACE_ARRAY_ID     = 0


def version_from_name(name: str) -> int | None:
    """Read the format version out of a record's MAVLink name, or None if it is not one
    of ours. The authoritative decode, so tooling never re-derives the convention."""
    if not name.startswith(TRACE_NAME_PREFIX):
        return None
    suffix = name[len(TRACE_NAME_PREFIX):]
    return int(suffix) if suffix.isdigit() else None


class FrameOutcome(IntEnum):
    """What became of one control-loop iteration. Packed 2 bits at a time into the
    history, so the values must stay in 0..3.

    Two bits is enough because the history stores an explicit frame COUNT. Without it a
    partially-filled slot's padding zeros would be indistinguishable from genuine
    NO_FRAME starvation, and we would need a fifth "no data here" value (v2 did, at
    4 bits/frame). The count removes the ambiguity and buys back the depth.
    """
    NO_FRAME       = 0   # the control loop got no frame at all — the pipeline starved
    NO_DETECTIONS  = 1   # a frame arrived, the detector found nothing
    NO_VIABLE      = 2   # a frame arrived with detections, none good enough to act on
    VIABLE         = 3   # a frame arrived and we picked a target


# History packing, FIXED BY THE FORMAT VERSION (which rides in the name). None of it may
# change without bumping the version.
#
# 24 bits per slot, deliberately: a float32 represents integers exactly only up to 2**24,
# so 24 bits is the widest field that survives the trip intact. Anything wider would
# silently round in the log — and a rounded bitfield decodes to garbage, not to an
# obviously-wrong number.
#
#   history slot A:  bits  0..5   frame count (how many outcomes this history holds)
#                    bits  6..23  9 outcomes, 2 bits each   -> interval frames 0..8
#   history slot B:  bits  0..23  12 outcomes, 2 bits each  -> interval frames 9..20
#
# No outcome straddles the slot boundary, so a decoder never has to stitch two floats.
_SLOT_BITS              = 24
HISTORY_BITS_PER_FRAME  = 2
HISTORY_COUNT_BITS      = 6                                       # 0..63, capacity fits
HISTORY_SLOTS           = 2
HISTORY_FRAMES_SLOT_0   = (_SLOT_BITS - HISTORY_COUNT_BITS) // HISTORY_BITS_PER_FRAME  # 9
HISTORY_FRAMES_SLOT_1   = _SLOT_BITS // HISTORY_BITS_PER_FRAME                          # 12
HISTORY_FRAMES          = HISTORY_FRAMES_SLOT_0 + HISTORY_FRAMES_SLOT_1                 # 21

_MAX_SLOT_VALUE   = (1 << _SLOT_BITS) - 1
_OUTCOME_MASK     = (1 << HISTORY_BITS_PER_FRAME) - 1
_COUNT_MASK       = (1 << HISTORY_COUNT_BITS) - 1

# Slot indices, so the packer, the tests and any decoder agree on one definition. The
# format version is NOT here — it rides in the name (above), freeing slot 0 for data.
#
# The three starvation counts are separate, not merged. Splitting NO_DETECTIONS (detector
# saw nothing) from NO_VIABLE (detections present, none good enough) makes an UNCAPPED
# record — one frame, history dropped — fully reconstructable: each of the four outcomes
# now maps to a distinct (count, detection) signature, so no information is lost when the
# history is omitted.
SLOT_FRAME_ID                  = 0
SLOT_FRAMES_NO_DETECTION       = 1   # NO_DETECTIONS: the detector returned nothing
SLOT_FRAMES_NO_VIABLE          = 2   # NO_VIABLE: detections present, none good enough
SLOT_FRAMES_WITHOUT_FRAME      = 3   # NO_FRAME: no frame reached the loop at all
SLOT_PD_P_X               = 4
SLOT_PD_P_Y               = 5
SLOT_CMD_ROLL             = 6
SLOT_CMD_PITCH            = 7
SLOT_CMD_THRUST           = 8
SLOT_DET_X                = 9
SLOT_DET_Y                = 10
SLOT_DET_W                = 11
SLOT_DET_H                = 12
SLOT_DET_CONF             = 13
# The frame history is LAST of the used slots so that when it is empty (uncapped
# mode, below) the record is all-zero from slot 14 on and MAVLink 2 trims it off the
# wire — and a later gzip of the .ulg compresses those runs to almost nothing. (The
# raw record on the FC's SD card is fixed-size regardless; that is a property of the
# uORB struct, not something a layout can change.)
SLOT_HISTORY_0            = 14
SLOT_HISTORY_1            = 15

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


def _outcome_shift(index: int) -> tuple[int, int]:
    """Where interval frame `index` lives: (slot, bit offset within that slot)."""
    if index < HISTORY_FRAMES_SLOT_0:
        return 0, HISTORY_COUNT_BITS + HISTORY_BITS_PER_FRAME * index
    return 1, HISTORY_BITS_PER_FRAME * (index - HISTORY_FRAMES_SLOT_0)


def pack_history(outcomes) -> list[float]:
    """Pack per-frame outcomes, OLDEST FIRST, into HISTORY_SLOTS float32 slots, with the
    number of frames stored in the low bits of slot 0.

    `outcomes[0]` is the FIRST frame of the interval. The explicit count is what lets a
    decoder know where the data ends, so no padding value is needed and an outcome fits
    in 2 bits.

    More frames than the history holds: the LATEST that do not fit are simply not
    recorded, and the count reports what was. The counts in slots 2 and 3 still cover the
    WHOLE interval, so nothing aggregate is lost — only per-frame detail, and only when
    the send rate is set so low that an interval exceeds 21 frames.
    """
    stored = list(outcomes)[:HISTORY_FRAMES]
    raw = [len(stored) & _COUNT_MASK, 0]
    for index, outcome in enumerate(stored):
        slot, shift = _outcome_shift(index)
        raw[slot] |= (int(outcome) & _OUTCOME_MASK) << shift

    for v in raw:
        assert 0 <= v <= _MAX_SLOT_VALUE, (
            "a history slot must stay inside float32's exact-integer range, "
            "or it decodes to garbage rather than to an obviously-wrong number")
    return [float(v) for v in raw]


def unpack_history(slots) -> list[FrameOutcome]:
    """Inverse of pack_history: returns exactly the frames the history holds, oldest
    first. Provided so the tests — and anyone reading a real ulog — have one
    authoritative decoder rather than re-deriving the bit layout by hand."""
    raw = [int(slots[0]), int(slots[1])]
    count = min(raw[0] & _COUNT_MASK, HISTORY_FRAMES)

    out = []
    for index in range(count):
        slot, shift = _outcome_shift(index)
        out.append(FrameOutcome((raw[slot] >> shift) & _OUTCOME_MASK))
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

    # Counts SINCE THE LAST SENT RECORD, not momentary values. Three separate starvation
    # modes — kept apart so an uncapped (history-less) record stays fully reconstructable.
    frames_no_detection  : int = 0   # NO_DETECTIONS: the detector returned nothing
    frames_no_viable     : int = 0   # NO_VIABLE: detections present, none good enough
    frames_without_frame : int = 0   # NO_FRAME: no frame reached the loop at all

    # Per-frame outcomes, oldest first. Packed into the LAST used slots. Empty in
    # uncapped mode, where a one-frame history would be redundant.
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
        # The format version is NOT written here — it rides in the MAVLink name.
        data = [0.0] * TRACE_SLOTS
        data[SLOT_FRAME_ID]             = _finite(self.frame_id)
        data[SLOT_FRAMES_NO_DETECTION]  = _finite(self.frames_no_detection)
        data[SLOT_FRAMES_NO_VIABLE]     = _finite(self.frames_no_viable)
        data[SLOT_FRAMES_WITHOUT_FRAME] = _finite(self.frames_without_frame)
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

        # LAST of the used slots. An empty history packs to (0, 0), so an uncapped
        # record (one frame, history intentionally dropped) is all-zero from here on and
        # the payload trims. When the history IS present it is non-zero — the frame
        # count occupies the low bits — so nothing below it can trim, which is why it
        # goes last.
        history = pack_history(self.history)
        data[SLOT_HISTORY_0] = history[0]
        data[SLOT_HISTORY_1] = history[1]
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
        self._history : list[FrameOutcome] = []        # oldest first, capped
        self.reset()

    def reset(self) -> None:
        self.frames_no_detection = 0
        self.frames_no_viable = 0
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
        # Oldest first, and once the history is full the LATEST frames are simply not
        # recorded. The COUNTS below still take every frame, so the aggregate stays
        # exact no matter how long the interval runs.
        if len(self._history) < HISTORY_FRAMES:
            self._history.append(outcome)
        self._frame_id = frame_id
        self._pd_p = pd_p
        self._cmd = cmd

        if outcome is FrameOutcome.NO_FRAME:
            self.frames_without_frame += 1
        elif outcome is FrameOutcome.NO_DETECTIONS:
            self.frames_no_detection += 1
        elif outcome is FrameOutcome.NO_VIABLE:
            self.frames_no_viable += 1
        elif outcome is FrameOutcome.VIABLE:
            # Last viable WINS: a later miss must never erase it.
            if detection is not None:
                self._detection = detection

    def build(self, *, include_history: bool = True) -> UlogTraceSample:
        """Snapshot the interval as a sample.

        `include_history=False` drops the per-frame history. Used for uncapped mode,
        where the interval is a single frame: that frame's outcome is already fully
        implied by the three separate counts and the detection (NO_FRAME / NO_DETECTIONS
        / NO_VIABLE each have their own count; VIABLE shows as a present detection), so
        the history is redundant and dropping it lets the record trim to nothing on the
        wire and compress away in a gzipped .ulg. Nothing is lost.
        """
        d = self._detection
        bbox = d.bbox if d is not None else None
        roll, pitch, thrust = self._cmd
        return UlogTraceSample(
            frame_id             = self._frame_id,
            frames_no_detection  = self.frames_no_detection,
            frames_no_viable     = self.frames_no_viable,
            frames_without_frame = self.frames_without_frame,
            history              = tuple(self._history) if include_history else (),
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

    `rate_hz = 0` means UNCAPPED: emit a record on every control-loop iteration. Each
    record then covers exactly one frame, which is the finest resolution the ulog can
    hold. Measured safe on the rig — at 20-30 fps this is far below the FC logger's
    ~195 records/s ceiling, costs ~5 kB/s of log, and does not perturb the control loop
    or the telemetry streams.

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
        # 0 => uncapped: the period test below can never hold anything back, and each
        # record covers a single frame — so its history is redundant and gets dropped,
        # letting the record trim away on the wire.
        self._uncapped = (rate_hz <= 0)
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

        data = self._acc.build(include_history=not self._uncapped).to_floats()
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
