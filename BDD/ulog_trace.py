#! /usr/bin/env python3
"""Mirror a slice of the decision pipeline into the PX4's own persistent ulog.

After a crash the Pi's SD card may be unreadable, taking every `_DEBUG/` log and
video with it. The flight controller's log survives far more reliably, so a small
trace of what we saw and what we commanded goes there too, as a MAVLink
DEBUG_FLOAT_ARRAY that PX4 republishes as the `debug_array` uORB topic.

Two rules govern everything here, because this sits on the control path:

  * `submit()` never blocks. It rate-limits and fires the send as a detached task.
  * Nothing raises into the control loop. A dead link, an unsupported plugin, a
    call from a thread with no event loop — all degrade to "no trace" and are
    counted. A stalled control loop can wreck the airframe; a *debug* feature
    must never be the thing that stalls it.

PX4 only writes `debug_array` into the log when the `SDLOG_PROFILE` param has its
Debug bit (32) set, and reads that param at boot. `DroneMover` warns at startup
when it is clear.

Slot layout (see docs/superpowers/specs/2026-07-12-px4-ulog-trace-design.md):

     0  format version        7  commanded pitch, deg
     1  frame id              8  commanded thrust
     2  frames since a usable target
     3  iterations with no frame at all
     4  pd_coeff_p.x          9  detection centre x
     5  pd_coeff_p.y         10  detection centre y
     6  commanded roll, deg  11  detection width
                             12  detection height
                             13  detection confidence
                          14-57  reserved (zero)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from helpers import XY

logger = logging.getLogger(__name__)


# Bump when the slot layout changes, so a decoder can tell the layouts apart from
# the log alone. Slot 0 carries it; being non-zero it also stops a message from
# ever truncating away to nothing on the wire.
TRACE_FORMAT_VERSION = 1

# MAVLink DEBUG_FLOAT_ARRAY carries data[58]; `name` is char[10].
TRACE_SLOTS    = 58
TRACE_NAME     = "BDD"
TRACE_ARRAY_ID = 0

NAN = float('nan')

# How many failures to swallow silently between log lines, once the first has
# been reported. A dead link must not turn into a per-tick log flood.
_FAILURE_LOG_EVERY = 100


@dataclass(slots=True, frozen=True)
class UlogTraceSample:
    """One tick of the decision pipeline, as it goes into the ulog.

    The detection fields come LAST on the wire and default to 0.0. MAVLink 2 trims
    trailing zero bytes from the payload, so a tick with no detection physically
    shrinks. Zero is unambiguous here: a width, height and confidence of exactly
    zero is a far more reliable signature of "nothing was detected" than of a real
    target sitting dead-centre at zero confidence.

    The command fields default to NaN instead, because that argument does not hold
    for them — a zero roll or zero thrust is a perfectly legal command, so 0.0
    could not mean "none issued". They sit before the detection block, so a NaN
    there costs nothing: it cannot block the trailing-zero trim.
    """
    frame_id               : int   = 0     # controller's monotonic counter
    frames_since_detection : int   = 0     # frames arrived, vision found no target
    frames_without_frame   : int   = 0     # no frame arrived at all: pipeline starved

    # The gain actually in force, after the final clamp into [p_min, p_max].
    pd_p : XY = field(default_factory=XY)

    # Verbatim, as handed to DroneMover: before the upside-down veto and before
    # any clamping. This is the decision the vision pipeline made; what the FC then
    # did with it is already in the ulog's own attitude topics.
    cmd_roll_deg  : float = NAN
    cmd_pitch_deg : float = NAN
    cmd_thrust    : float = NAN

    # Normalised [0, 1].
    det_x    : float = 0.0
    det_y    : float = 0.0
    det_w    : float = 0.0
    det_h    : float = 0.0
    det_conf : float = 0.0

    def to_floats(self) -> list[float]:
        data = [0.0] * TRACE_SLOTS
        data[0]  = float(TRACE_FORMAT_VERSION)
        data[1]  = float(self.frame_id)
        data[2]  = float(self.frames_since_detection)
        data[3]  = float(self.frames_without_frame)
        data[4]  = float(self.pd_p.x)
        data[5]  = float(self.pd_p.y)
        data[6]  = float(self.cmd_roll_deg)
        data[7]  = float(self.cmd_pitch_deg)
        data[8]  = float(self.cmd_thrust)
        data[9]  = float(self.det_x)
        data[10] = float(self.det_y)
        data[11] = float(self.det_w)
        data[12] = float(self.det_h)
        data[13] = float(self.det_conf)
        return data


# (name, array_id, data) -> awaitable. DroneMover.send_debug_array satisfies it.
SendFn = Callable[[str, int, list[float]], Awaitable[None]]


class UlogTracer:
    """Rate-limits samples and fires them at the flight controller, fire-and-forget.

    `send_fn` is passed explicitly rather than pulled out of a config dict: it is a
    live object, and config here carries values only.

    Pass `send_fn=None` to disable — `submit()` then costs one attribute test.
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

        self._last_sent_at : float | None = None
        self._inflight : asyncio.Task | None = None

        self.sent    = 0   # handed to the link
        self.dropped = 0   # rate-limited, or a previous send still in flight
        self.failed  = 0   # the send itself blew up

    @property
    def enabled(self) -> bool:
        return self._send_fn is not None

    def submit(self, sample: UlogTraceSample) -> None:
        """Hot path, called once per control iteration. Synchronous, cheap, and it
        does not raise. Drops the sample if the period has not elapsed, or if the
        previous send is still in flight — a stalled link can then never build a
        backlog behind it."""
        if self._send_fn is None:
            return

        now = self._now()
        if self._last_sent_at is not None and (now - self._last_sent_at) < self._period_s:
            self.dropped += 1
            return

        if self._inflight is not None and not self._inflight.done():
            self.dropped += 1
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop on this thread. Nothing we can do, and it is not worth
            # taking the control loop down over a trace.
            self._note_failure("no running event loop to send the ulog trace from")
            return

        self._inflight = loop.create_task(self._send(sample.to_floats()))
        self._last_sent_at = now

    async def _send(self, data: list[float]) -> None:
        try:
            await self._send_fn(self._name, self._array_id, data)
            self.sent += 1
        except Exception as e:
            self._note_failure(f"sending the ulog trace failed: {e}")

    def _note_failure(self, message: str) -> None:
        self.failed += 1
        # Report the first one, then thin out: a link that is down would otherwise
        # log on every single tick.
        if self.failed == 1 or self.failed % _FAILURE_LOG_EVERY == 0:
            logger.warning("ulog trace: %s (%d failure(s) so far)", message, self.failed)

    def summary(self) -> str:
        return (f"ulog trace: {self.sent} sent, {self.dropped} dropped "
                f"(rate-limit/in-flight), {self.failed} failed")
