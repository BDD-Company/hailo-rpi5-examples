#! /usr/bin/env python3
"""Tests for ulog_trace: the slot layout, the INTERVAL AGGREGATION rules, and the
guarantee that the tracer can neither block nor break the control loop it hangs off."""

import asyncio
import math
import unittest

from helpers import XY, Rect, Detection
from ulog_trace import (
    HISTORY_FRAMES,
    SLOT_CMD_ROLL,
    SLOT_CMD_THRUST,
    SLOT_DET_CONF,
    SLOT_DET_X,
    SLOT_FORMAT_VERSION,
    SLOT_FRAMES_NO_DETECTION,
    SLOT_FRAMES_WITHOUT_FRAME,
    SLOT_FRAME_ID,
    SLOT_HISTORY_0,
    SLOT_HISTORY_1,
    SLOT_PD_P_X,
    TRACE_FORMAT_VERSION,
    TRACE_SLOTS,
    FrameOutcome,
    TraceAccumulator,
    UlogTracer,
    UlogTraceSample,
    pack_history,
    unpack_history,
)


def _detection(x=0.5, y=0.25, w=0.1, h=0.2, conf=0.9) -> Detection:
    return Detection(bbox=Rect.from_xywh(x - w / 2, y - h / 2, w, h), confidence=conf)


def _sample(**overrides) -> UlogTraceSample:
    fields = dict(
        frame_id=42,
        frames_no_detection=0,
        frames_without_frame=0,
        history=(FrameOutcome.VIABLE,),
        pd_p=XY(3.0, 4.0),
        cmd_roll=-1.5,
        cmd_pitch=2.5,
        cmd_thrust=0.7,
        det_x=0.5, det_y=0.25, det_w=0.1, det_h=0.2, det_conf=0.9,
    )
    fields.update(overrides)
    return UlogTraceSample(**fields)


class TestSampleLayout(unittest.TestCase):

    def test_every_field_lands_in_its_documented_slot(self):
        data = _sample().to_floats()

        self.assertEqual(len(data), TRACE_SLOTS)
        self.assertEqual(data[SLOT_FORMAT_VERSION], float(TRACE_FORMAT_VERSION))
        self.assertEqual(data[SLOT_FRAME_ID], 42.0)
        self.assertEqual(data[SLOT_PD_P_X], 3.0)
        self.assertEqual(data[SLOT_CMD_ROLL], -1.5)
        self.assertEqual(data[SLOT_CMD_THRUST], 0.7)
        self.assertEqual(data[SLOT_DET_X], 0.5)
        self.assertEqual(data[SLOT_DET_CONF], 0.9)

    def test_the_format_version_is_2(self):
        # v1 was a momentary snapshot with no history. A decoder must be able to tell
        # the layouts apart from the log alone, so the version MUST move with the layout.
        self.assertEqual(TRACE_FORMAT_VERSION, 2)

    def test_reserved_slots_are_zero(self):
        data = _sample().to_floats()
        self.assertEqual(data[16:], [0.0] * (TRACE_SLOTS - 16))

    def test_no_detection_zeroes_the_whole_tail(self):
        # The detection block is last precisely so a no-detection interval is nothing
        # but zeros from slot 11 on: MAVLink 2 then trims those bytes off the wire.
        data = UlogTraceSample(frame_id=7, frames_no_detection=3).to_floats()
        self.assertEqual(data[SLOT_DET_X:], [0.0] * (TRACE_SLOTS - SLOT_DET_X))

    def test_non_finite_values_are_zeroed(self):
        # MAVSDK serialises the fields as JSON and its parser REJECTS a bare NaN /
        # Infinity token, costing the WHOLE message. Verified against a real
        # mavsdk_server. Lose one field, not the sample.
        data = _sample(pd_p=XY(float('nan'), 3.0),
                       cmd_roll=float('inf'),
                       cmd_thrust=float('-inf'),
                       det_conf=float('nan')).to_floats()

        self.assertTrue(all(math.isfinite(v) for v in data))
        self.assertEqual(data[SLOT_PD_P_X], 0.0)
        self.assertEqual(data[SLOT_CMD_ROLL], 0.0)
        self.assertEqual(data[SLOT_DET_CONF], 0.0)


class TestHistoryPacking(unittest.TestCase):

    def test_round_trips_newest_first(self):
        outcomes = [FrameOutcome.VIABLE, FrameOutcome.NO_FRAME, FrameOutcome.NO_VIABLE,
                    FrameOutcome.NO_DETECTIONS, FrameOutcome.VIABLE, FrameOutcome.NO_FRAME,
                    FrameOutcome.VIABLE]
        decoded = unpack_history(pack_history(outcomes))

        self.assertEqual(decoded[:len(outcomes)], outcomes)

    def test_short_intervals_pad_with_NO_DATA_not_NO_FRAME(self):
        # THE reason the field is 4 bits and not 2. With 2 bits the padding zeros would
        # be indistinguishable from real NO_FRAME starvation, and every record would
        # look like the pipeline had died.
        decoded = unpack_history(pack_history([FrameOutcome.VIABLE, FrameOutcome.VIABLE]))

        self.assertEqual(decoded[0], FrameOutcome.VIABLE)
        self.assertEqual(decoded[1], FrameOutcome.VIABLE)
        self.assertTrue(all(o is FrameOutcome.NO_DATA for o in decoded[2:]))
        self.assertNotEqual(FrameOutcome.NO_DATA, FrameOutcome.NO_FRAME)

    def test_a_full_history_stays_exactly_representable_as_float32(self):
        # 6 frames * 4 bits = 24 bits per slot, and float32 holds integers exactly only
        # to 2**24. A wider field would silently round in the log.
        worst = [FrameOutcome(15 if False else 4)] * HISTORY_FRAMES
        slots = pack_history([FrameOutcome.VIABLE] * HISTORY_FRAMES)
        for v in slots:
            self.assertEqual(v, float(int(v)))
            self.assertLessEqual(v, float(2 ** 24))
        self.assertEqual(unpack_history(slots), worst)

    def test_overflow_drops_the_OLDEST_frames(self):
        # More frames than the history holds: keep the ones nearest the record. The
        # counts still cover the whole interval, so only per-frame detail is lost.
        outcomes = [FrameOutcome.VIABLE] + [FrameOutcome.NO_FRAME] * (HISTORY_FRAMES + 5)
        decoded = unpack_history(pack_history(outcomes))

        self.assertEqual(len(decoded), HISTORY_FRAMES)
        self.assertEqual(decoded[0], FrameOutcome.VIABLE)   # newest survived


class TestAccumulator(unittest.TestCase):
    """The aggregation rules — the part that actually changed."""

    def setUp(self):
        self.acc = TraceAccumulator()

    def _note(self, outcome, frame_id=1, detection=None, p=XY(1.0, 1.0), cmd=(0.0, 0.0, 0.5)):
        self.acc.note(frame_id=frame_id, outcome=outcome, pd_p=p, cmd=cmd, detection=detection)

    def test_the_users_worked_example(self):
        # "if since last actual send there was 2 missed detections and 1 missed frames,
        #  but this one we have a detection, then the record is:
        #  frames_no_detection = 2, frames_without_frame = 1"
        self._note(FrameOutcome.NO_DETECTIONS, frame_id=1)
        self._note(FrameOutcome.NO_FRAME,      frame_id=1)
        self._note(FrameOutcome.NO_VIABLE,     frame_id=2)
        self._note(FrameOutcome.VIABLE,        frame_id=3, detection=_detection(conf=0.8))

        s = self.acc.build()
        self.assertEqual(s.frames_no_detection, 2)     # NO_DETECTIONS + NO_VIABLE
        self.assertEqual(s.frames_without_frame, 1)
        self.assertEqual(s.frame_id, 3)
        self.assertEqual(s.det_conf, 0.8)

    def test_a_detection_on_the_FIRST_frame_survives_five_later_misses(self):
        # The whole reason for aggregating. A snapshot would report "no detection" here
        # and the detection would vanish from the log entirely.
        self._note(FrameOutcome.VIABLE, frame_id=10, detection=_detection(x=0.7, conf=0.85))
        for i in range(5):
            self._note(FrameOutcome.NO_DETECTIONS, frame_id=11 + i)

        s = self.acc.build()
        self.assertEqual(s.det_conf, 0.85)
        self.assertAlmostEqual(s.det_x, 0.7, places=5)
        self.assertEqual(s.frames_no_detection, 5)
        self.assertEqual(s.frame_id, 15, "frame_id is still the most recent frame")

    def test_the_LAST_viable_detection_wins(self):
        self._note(FrameOutcome.VIABLE, frame_id=1, detection=_detection(conf=0.6))
        self._note(FrameOutcome.NO_DETECTIONS, frame_id=2)
        self._note(FrameOutcome.VIABLE, frame_id=3, detection=_detection(conf=0.95))

        self.assertEqual(self.acc.build().det_conf, 0.95)

    def test_history_is_newest_first_and_covers_every_iteration(self):
        self._note(FrameOutcome.NO_FRAME,      frame_id=1)
        self._note(FrameOutcome.NO_DETECTIONS, frame_id=1)
        self._note(FrameOutcome.VIABLE,        frame_id=2, detection=_detection())

        decoded = unpack_history(pack_history(self.acc.build().history))
        self.assertEqual(decoded[0], FrameOutcome.VIABLE)         # most recent
        self.assertEqual(decoded[1], FrameOutcome.NO_DETECTIONS)
        self.assertEqual(decoded[2], FrameOutcome.NO_FRAME)
        self.assertEqual(decoded[3], FrameOutcome.NO_DATA)

    def test_reset_clears_the_detection_so_an_empty_interval_reports_zeros(self):
        # If the detection survived a reset, a window with nothing viable in it would
        # keep re-reporting an old target as though it were still being seen.
        self._note(FrameOutcome.VIABLE, frame_id=1, detection=_detection(conf=0.9))
        self.acc.reset()
        self._note(FrameOutcome.NO_DETECTIONS, frame_id=2)

        s = self.acc.build()
        self.assertEqual(s.det_conf, 0.0)
        self.assertEqual(s.det_x, 0.0)
        self.assertEqual(s.frames_no_detection, 1)

    def test_pd_p_and_command_are_the_values_in_force_at_the_end(self):
        self._note(FrameOutcome.VIABLE, frame_id=1, detection=_detection(),
                   p=XY(4.0, 2.0), cmd=(1.0, 2.0, 0.5))
        self._note(FrameOutcome.NO_DETECTIONS, frame_id=2,
                   p=XY(0.5, 0.25), cmd=(9.0, 8.0, 0.7))

        s = self.acc.build()
        self.assertEqual((s.pd_p.x, s.pd_p.y), (0.5, 0.25))
        self.assertEqual((s.cmd_roll, s.cmd_pitch, s.cmd_thrust), (9.0, 8.0, 0.7))


class _FakeSender:
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
        # Rounded so that repeated +0.05 lands exactly on 0.2 rather than a hair under
        # it, which would make the rate-limit test flaky for no good reason.
        self.now = round(self.now + seconds, 9)


class TestTracer(unittest.IsolatedAsyncioTestCase):

    def _tracer(self, sender, **kwargs):
        clock = _FakeClock()
        return UlogTracer(sender, rate_hz=5.0, now_fn=clock, **kwargs), clock

    async def _drain(self):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    def _note(self, tracer, outcome, frame_id=1, detection=None):
        tracer.note(frame_id=frame_id, outcome=outcome, pd_p=XY(1.0, 1.0),
                    cmd=(0.0, 0.0, 0.5), detection=detection)

    async def test_sends_the_named_array(self):
        sender = _FakeSender()
        tracer, _ = self._tracer(sender, name="BDD", array_id=0)

        self._note(tracer, FrameOutcome.VIABLE, detection=_detection())
        await self._drain()

        self.assertEqual(len(sender.calls), 1)
        name, array_id, data = sender.calls[0]
        self.assertEqual((name, array_id), ("BDD", 0))
        self.assertEqual(data[SLOT_FORMAT_VERSION], float(TRACE_FORMAT_VERSION))
        self.assertEqual(tracer.sent, 1)

    async def test_one_record_per_period_summarising_every_frame_between(self):
        # 30 fps loop, 5 Hz trace: ~6 frames per record, and every one of them must be
        # accounted for in the record that follows.
        sender = _FakeSender()
        tracer, clock = self._tracer(sender)

        self._note(tracer, FrameOutcome.VIABLE, frame_id=0, detection=_detection())  # sends immediately
        await self._drain()
        self.assertEqual(len(sender.calls), 1)

        for i in range(6):                       # 6 frames at 30 fps = 200 ms
            # Set the time absolutely: six += 1/30 steps accumulate to 0.199999998 and
            # would land a hair inside the period, which is float drift, not behaviour.
            clock.now = round((i + 1) / 30, 9)
            self._note(tracer, FrameOutcome.NO_DETECTIONS, frame_id=i + 1)
            await self._drain()

        self.assertEqual(len(sender.calls), 2, "exactly one more record after 200 ms")
        data = sender.calls[1][2]
        self.assertEqual(data[SLOT_FRAMES_NO_DETECTION], 6.0)
        self.assertEqual(data[SLOT_DET_CONF], 0.0, "nothing viable in that interval")

    async def test_the_interval_keeps_growing_while_a_send_is_in_flight(self):
        # A skipped send must NOT lose the frames it would have covered: the next record
        # that goes out has to account for them too. Otherwise a slow link silently eats
        # the very data we are trying to preserve.
        sender = _FakeSender()
        sender.gate = asyncio.Event()
        tracer, clock = self._tracer(sender)

        self._note(tracer, FrameOutcome.NO_DETECTIONS, frame_id=1)
        await self._drain()
        self.assertEqual(len(sender.calls), 1)     # first record away, send now stuck

        clock.advance(10.0)
        for i in range(3):                          # these are due, but cannot be sent
            self._note(tracer, FrameOutcome.NO_FRAME, frame_id=2)
            await self._drain()
        self.assertEqual(len(sender.calls), 1)
        self.assertEqual(tracer.skipped, 3)

        sender.gate.set()                           # link recovers
        await self._drain()
        clock.advance(10.0)
        self._note(tracer, FrameOutcome.NO_FRAME, frame_id=2)
        await self._drain()

        self.assertEqual(len(sender.calls), 2)
        data = sender.calls[1][2]
        self.assertEqual(data[SLOT_FRAMES_WITHOUT_FRAME], 4.0,
                         "all 4 starved iterations must survive the stalled send")

    async def test_history_reaches_the_wire(self):
        sender = _FakeSender()
        tracer, clock = self._tracer(sender)

        self._note(tracer, FrameOutcome.VIABLE, frame_id=0, detection=_detection())
        await self._drain()

        clock.advance(1 / 30)
        self._note(tracer, FrameOutcome.NO_FRAME, frame_id=1)
        clock.advance(1 / 30)
        self._note(tracer, FrameOutcome.VIABLE, frame_id=2, detection=_detection(conf=0.7))
        clock.advance(0.2)
        self._note(tracer, FrameOutcome.NO_DETECTIONS, frame_id=3)
        await self._drain()

        data = sender.calls[-1][2]
        decoded = unpack_history([data[SLOT_HISTORY_0], data[SLOT_HISTORY_1]])
        self.assertEqual(decoded[0], FrameOutcome.NO_DETECTIONS)   # newest
        self.assertEqual(decoded[1], FrameOutcome.VIABLE)
        self.assertEqual(decoded[2], FrameOutcome.NO_FRAME)
        self.assertEqual(data[SLOT_DET_CONF], 0.7, "the viable detection is not erased")

    async def test_a_failing_send_is_swallowed_and_counted(self):
        sender = _FakeSender(raises=RuntimeError("link down"))
        tracer, clock = self._tracer(sender)

        for i in range(3):
            self._note(tracer, FrameOutcome.NO_FRAME, frame_id=i)   # must not raise
            await self._drain()
            clock.advance(1.0)

        self.assertEqual(tracer.failed, 3)
        self.assertEqual(tracer.sent, 0)

    async def test_disabled_tracer_never_sends(self):
        tracer, clock = self._tracer(None)

        self.assertFalse(tracer.enabled)
        for i in range(5):
            self._note(tracer, FrameOutcome.VIABLE, frame_id=i, detection=_detection())
            await self._drain()
            clock.advance(1.0)

        self.assertEqual(tracer.sent, 0)

    async def test_note_outside_an_event_loop_does_not_raise(self):
        sender = _FakeSender()
        tracer, _ = self._tracer(sender)

        await asyncio.to_thread(lambda: self._note(tracer, FrameOutcome.NO_FRAME))

        self.assertEqual(tracer.sent, 0)
        self.assertEqual(tracer.failed, 1)


if __name__ == "__main__":
    unittest.main()
