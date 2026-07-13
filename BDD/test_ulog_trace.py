#! /usr/bin/env python3
"""Tests for ulog_trace: the slot layout, and the guarantee that the tracer can
neither block nor break the control loop it hangs off."""

import asyncio
import math
import unittest

from helpers import XY
from ulog_trace import (
    TRACE_FORMAT_VERSION,
    TRACE_SLOTS,
    UlogTracer,
    UlogTraceSample,
)


def _detected(**overrides) -> UlogTraceSample:
    """A sample as it looks on a tick that selected a target."""
    fields = dict(
        frame_id=42,
        frames_since_detection=0,
        frames_without_frame=0,
        pd_p=XY(3.0, 4.0),
        cmd_roll_deg=-1.5,
        cmd_pitch_deg=2.5,
        cmd_thrust=0.7,
        det_x=0.5,
        det_y=0.25,
        det_w=0.1,
        det_h=0.2,
        det_conf=0.9,
    )
    fields.update(overrides)
    return UlogTraceSample(**fields)


class TestSampleLayout(unittest.TestCase):

    def test_every_field_lands_in_its_documented_slot(self):
        data = _detected().to_floats()

        self.assertEqual(len(data), TRACE_SLOTS)
        self.assertEqual(data[0], float(TRACE_FORMAT_VERSION))
        self.assertEqual(data[1], 42.0)     # frame_id
        self.assertEqual(data[2], 0.0)      # frames_since_detection
        self.assertEqual(data[3], 0.0)      # frames_without_frame
        self.assertEqual(data[4], 3.0)      # pd_p.x
        self.assertEqual(data[5], 4.0)      # pd_p.y
        self.assertEqual(data[6], -1.5)     # cmd_roll_deg
        self.assertEqual(data[7], 2.5)      # cmd_pitch_deg
        self.assertEqual(data[8], 0.7)      # cmd_thrust
        self.assertEqual(data[9], 0.5)      # det_x
        self.assertEqual(data[10], 0.25)    # det_y
        self.assertEqual(data[11], 0.1)     # det_w
        self.assertEqual(data[12], 0.2)     # det_h
        self.assertEqual(data[13], 0.9)     # det_conf

    def test_reserved_slots_are_zero(self):
        data = _detected().to_floats()
        self.assertEqual(data[14:], [0.0] * (TRACE_SLOTS - 14))

    def test_no_detection_zeroes_the_whole_tail(self):
        # The detection block is last precisely so that a no-detection tick has
        # nothing but zeros from slot 9 on: MAVLink 2 then trims those bytes off
        # the wire. If a future field is inserted after the detection block this
        # test is what catches the lost saving.
        data = UlogTraceSample(frame_id=7, frames_since_detection=3).to_floats()
        self.assertEqual(data[9:], [0.0] * (TRACE_SLOTS - 9))

    def test_absent_command_is_nan_not_zero(self):
        # A zero roll/thrust is a legal command, so 0.0 could not mean "none issued".
        data = UlogTraceSample(frame_id=7).to_floats()
        for slot in (6, 7, 8):
            self.assertTrue(math.isnan(data[slot]), f"slot {slot} should be NaN")

    def test_defaults_are_the_no_detection_case(self):
        sample = UlogTraceSample()
        self.assertEqual(sample.det_conf, 0.0)
        self.assertTrue(math.isnan(sample.cmd_thrust))


class _FakeSender:
    """Records what the tracer sent. `gate`, when set, holds a send open so the
    in-flight path can be exercised."""

    def __init__(self, *, raises: Exception | None = None):
        self.calls: list[tuple[str, int, list[float]]] = []
        self.raises = raises
        self.gate: asyncio.Event | None = None

    async def __call__(self, name: str, array_id: int, data: list[float]) -> None:
        self.calls.append((name, array_id, data))
        if self.gate is not None:
            await self.gate.wait()
        if self.raises is not None:
            raise self.raises


class _FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        # Rounded so that repeated +0.05 lands exactly on 0.2 rather than a hair
        # under it, which would make the rate-limit test flaky for no good reason.
        self.now = round(self.now + seconds, 9)


class TestTracer(unittest.IsolatedAsyncioTestCase):

    def _tracer(self, sender, **kwargs) -> tuple[UlogTracer, _FakeClock]:
        clock = _FakeClock()
        tracer = UlogTracer(sender, rate_hz=5.0, now_fn=clock, **kwargs)
        return tracer, clock

    async def _drain(self):
        """Let the detached send tasks run."""
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    async def test_sends_the_named_array(self):
        sender = _FakeSender()
        tracer, _ = self._tracer(sender, name="BDD", array_id=0)

        tracer.submit(_detected())
        await self._drain()

        self.assertEqual(len(sender.calls), 1)
        name, array_id, data = sender.calls[0]
        self.assertEqual(name, "BDD")
        self.assertEqual(array_id, 0)
        self.assertEqual(data[0], float(TRACE_FORMAT_VERSION))
        self.assertEqual(tracer.sent, 1)

    async def test_rate_limits_to_the_configured_hz(self):
        sender = _FakeSender()
        tracer, clock = self._tracer(sender)   # 5 Hz -> one send per 200 ms

        # 20 submits at 50 ms apart = 1 s of a 20 Hz control loop.
        for _ in range(20):
            tracer.submit(_detected())
            await self._drain()
            clock.advance(0.05)

        self.assertEqual(len(sender.calls), 5)
        self.assertEqual(tracer.sent, 5)
        self.assertEqual(tracer.dropped, 15)

    async def test_drops_while_a_send_is_still_in_flight(self):
        # A stalled link must never let submits queue up behind it.
        sender = _FakeSender()
        sender.gate = asyncio.Event()
        tracer, clock = self._tracer(sender)

        tracer.submit(_detected())
        await self._drain()
        self.assertEqual(len(sender.calls), 1)

        # Period has elapsed, but the first send has not finished.
        clock.advance(10.0)
        tracer.submit(_detected())
        await self._drain()

        self.assertEqual(len(sender.calls), 1, "second send must not have started")
        self.assertEqual(tracer.dropped, 1)

        # Once it completes, the next submit goes through again.
        sender.gate.set()
        await self._drain()
        clock.advance(10.0)
        tracer.submit(_detected())
        await self._drain()
        self.assertEqual(len(sender.calls), 2)

    async def test_a_failing_send_is_swallowed_and_counted(self):
        # The control loop must not see a link failure. Degrade, never hard-fail.
        sender = _FakeSender(raises=RuntimeError("link down"))
        tracer, clock = self._tracer(sender)

        for _ in range(3):
            tracer.submit(_detected())   # must not raise
            await self._drain()
            clock.advance(1.0)

        self.assertEqual(tracer.failed, 3)
        self.assertEqual(tracer.sent, 0)

    async def test_disabled_tracer_never_sends(self):
        tracer, clock = self._tracer(None)

        self.assertFalse(tracer.enabled)
        for _ in range(5):
            tracer.submit(_detected())
            await self._drain()
            clock.advance(1.0)

        self.assertEqual(tracer.sent, 0)
        self.assertEqual(tracer.dropped, 0)

    async def test_submit_outside_an_event_loop_does_not_raise(self):
        # The replay harness and the unit tests can call into the controller from
        # a plain thread; a tracer that raised there would take the loop down.
        sender = _FakeSender()
        tracer, _ = self._tracer(sender)

        def _from_a_thread():
            tracer.submit(_detected())

        await asyncio.to_thread(_from_a_thread)
        self.assertEqual(tracer.sent, 0)
        self.assertEqual(tracer.failed, 1)


if __name__ == "__main__":
    unittest.main()
