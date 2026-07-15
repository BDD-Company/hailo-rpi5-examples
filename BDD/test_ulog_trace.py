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
    SLOT_FRAMES_NO_DETECTION,
    SLOT_FRAMES_NO_VIABLE,
    SLOT_FRAMES_WITHOUT_FRAME,
    SLOT_FRAME_ID,
    SLOT_HISTORY_0,
    SLOT_HISTORY_1,
    SLOT_PD_P_X,
    TRACE_FORMAT_VERSION,
    TRACE_NAME,
    TRACE_NAME_PREFIX,
    TRACE_SLOTS,
    FrameOutcome,
    TraceAccumulator,
    UlogTracer,
    UlogTraceSample,
    pack_history,
    unpack_history,
    version_from_name,
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
        self.assertEqual(data[SLOT_FRAME_ID], 42.0)
        self.assertEqual(data[SLOT_PD_P_X], 3.0)
        self.assertEqual(data[SLOT_CMD_ROLL], -1.5)
        self.assertEqual(data[SLOT_CMD_THRUST], 0.7)
        self.assertEqual(data[SLOT_DET_X], 0.5)
        self.assertEqual(data[SLOT_DET_CONF], 0.9)

    def test_the_version_rides_in_the_name_not_a_data_slot(self):
        # The version is NOT a float in the payload: it is the numeric suffix of the
        # MAVLink name ("BDD1"). That frees a data slot and, because `name` precedes
        # `data` on the wire, does not block the data tail from trimming.
        self.assertEqual(TRACE_NAME, f"{TRACE_NAME_PREFIX}{TRACE_FORMAT_VERSION}")
        self.assertEqual(version_from_name(TRACE_NAME), TRACE_FORMAT_VERSION)
        self.assertEqual(version_from_name("BDD1"), 1)
        self.assertIsNone(version_from_name("OTHER"), "a foreign name is not ours")
        self.assertIsNone(version_from_name("BDD"), "no suffix -> not a versioned record")
        # No data slot equals the version by coincidence-proof: slot 0 is the frame id.
        self.assertEqual(_sample(frame_id=7).to_floats()[0], 7.0)

    def test_the_history_is_the_last_used_block(self):
        # It must sit after the detection block, or an empty (uncapped) history could not
        # trim, and a present history would block the detection block from trimming.
        self.assertGreater(SLOT_HISTORY_0, SLOT_DET_CONF)
        self.assertEqual(SLOT_HISTORY_1, SLOT_HISTORY_0 + 1)

    def test_reserved_slots_are_zero(self):
        data = _sample().to_floats()
        self.assertEqual(data[16:], [0.0] * (TRACE_SLOTS - 16))

    def test_no_detection_zeroes_the_whole_tail(self):
        # With no viable detection and the history empty, everything from the detection
        # block on is zero, so MAVLink 2 trims those bytes off the wire.
        data = UlogTraceSample(frame_id=7, frames_no_detection=3).to_floats()
        self.assertEqual(data[SLOT_DET_X:], [0.0] * (TRACE_SLOTS - SLOT_DET_X))

    def test_the_three_starvation_counts_are_separate(self):
        # Splitting NO_DETECTIONS from NO_VIABLE is what makes an uncapped (history-less)
        # record lossless — each outcome has its own count.
        data = _sample(frames_no_detection=2, frames_no_viable=3,
                       frames_without_frame=1).to_floats()
        self.assertEqual(data[SLOT_FRAMES_NO_DETECTION], 2.0)
        self.assertEqual(data[SLOT_FRAMES_NO_VIABLE], 3.0)
        self.assertEqual(data[SLOT_FRAMES_WITHOUT_FRAME], 1.0)

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

    def test_round_trips_oldest_first(self):
        outcomes = [FrameOutcome.VIABLE, FrameOutcome.NO_FRAME, FrameOutcome.NO_VIABLE,
                    FrameOutcome.NO_DETECTIONS, FrameOutcome.VIABLE, FrameOutcome.NO_FRAME,
                    FrameOutcome.VIABLE]
        self.assertEqual(unpack_history(pack_history(outcomes)), outcomes)

    def test_the_count_is_what_bounds_the_history(self):
        # v3's whole trick: an explicit count, so no padding value is needed and an
        # outcome fits in 2 bits. Decoding must return exactly what was stored — no
        # phantom trailing frames.
        for n in range(0, HISTORY_FRAMES + 1):
            decoded = unpack_history(pack_history([FrameOutcome.VIABLE] * n))
            self.assertEqual(len(decoded), n, f"{n} frames in, {len(decoded)} out")

    def test_all_NO_FRAME_is_not_confused_with_an_empty_history(self):
        # NO_FRAME is 0, so a starved interval packs its outcome bits as zeros. Only the
        # count distinguishes "3 starved frames" from "nothing recorded". This is the
        # exact ambiguity v2 needed a 5th enum value to avoid.
        starved = pack_history([FrameOutcome.NO_FRAME] * 3)
        empty   = pack_history([])

        self.assertEqual(unpack_history(starved), [FrameOutcome.NO_FRAME] * 3)
        self.assertEqual(unpack_history(empty), [])
        self.assertNotEqual(starved, empty)

    def test_a_full_history_stays_exactly_representable_as_float32(self):
        # 24 bits per slot, and float32 holds integers exactly only to 2**24. A wider
        # field would silently round — and a rounded bitfield decodes to garbage, not to
        # an obviously-wrong number.
        slots = pack_history([FrameOutcome.VIABLE] * HISTORY_FRAMES)
        for v in slots:
            self.assertEqual(v, float(int(v)))
            self.assertLess(v, float(2 ** 24))
        self.assertEqual(unpack_history(slots), [FrameOutcome.VIABLE] * HISTORY_FRAMES)

    def test_overflow_drops_the_LATEST_frames_that_do_not_fit(self):
        # As specified: when an interval is longer than the history holds, the frames
        # that cannot fit are simply not recorded. The counts (slots 2/3) still cover the
        # whole interval, so only per-frame detail is lost.
        outcomes = ([FrameOutcome.VIABLE] * HISTORY_FRAMES) + [FrameOutcome.NO_FRAME] * 5
        decoded = unpack_history(pack_history(outcomes))

        self.assertEqual(len(decoded), HISTORY_FRAMES)
        self.assertEqual(decoded, [FrameOutcome.VIABLE] * HISTORY_FRAMES)
        self.assertNotIn(FrameOutcome.NO_FRAME, decoded, "the overflow tail was recorded")


class TestAccumulator(unittest.TestCase):
    """The aggregation rules — the part that actually changed."""

    def setUp(self):
        self.acc = TraceAccumulator()

    def _note(self, outcome, frame_id=1, detection=None, p=XY(1.0, 1.0), cmd=(0.0, 0.0, 0.5)):
        self.acc.note(frame_id=frame_id, outcome=outcome, pd_p=p, cmd=cmd, detection=detection)

    def test_the_users_worked_example(self):
        # "if since last actual send there was 2 missed detections and 1 missed frames,
        #  but this one we have a detection" — now with the counts split, so the two
        #  no-detection kinds are counted separately.
        self._note(FrameOutcome.NO_DETECTIONS, frame_id=1)
        self._note(FrameOutcome.NO_FRAME,      frame_id=1)
        self._note(FrameOutcome.NO_VIABLE,     frame_id=2)
        self._note(FrameOutcome.VIABLE,        frame_id=3, detection=_detection(conf=0.8))

        s = self.acc.build()
        self.assertEqual(s.frames_no_detection, 1)     # the one NO_DETECTIONS
        self.assertEqual(s.frames_no_viable, 1)        # the one NO_VIABLE
        self.assertEqual(s.frames_without_frame, 1)
        self.assertEqual(s.frame_id, 3)
        self.assertEqual(s.det_conf, 0.8)

    def test_uncapped_single_frame_outcome_is_recoverable_from_counts(self):
        # The point of the counter split: with the history dropped (uncapped mode), each
        # of the four outcomes maps to a distinct (counts, detection) signature — so
        # nothing is lost. Build a one-frame interval per outcome and read it back.
        def signature(outcome, detection=None):
            acc = TraceAccumulator()
            acc.note(frame_id=1, outcome=outcome, pd_p=XY(), cmd=(0, 0, 0.5),
                     detection=detection)
            s = acc.build(include_history=False)
            return (s.frames_no_detection, s.frames_no_viable, s.frames_without_frame,
                    s.det_conf > 0)

        sigs = {
            "NO_FRAME":      signature(FrameOutcome.NO_FRAME),
            "NO_DETECTIONS": signature(FrameOutcome.NO_DETECTIONS),
            "NO_VIABLE":     signature(FrameOutcome.NO_VIABLE),
            "VIABLE":        signature(FrameOutcome.VIABLE, _detection()),
        }
        self.assertEqual(len(set(sigs.values())), 4,
                         f"outcomes are NOT distinguishable without history: {sigs}")

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

    def test_history_is_oldest_first_and_covers_every_iteration(self):
        self._note(FrameOutcome.NO_FRAME,      frame_id=1)
        self._note(FrameOutcome.NO_DETECTIONS, frame_id=1)
        self._note(FrameOutcome.VIABLE,        frame_id=2, detection=_detection())

        decoded = unpack_history(pack_history(self.acc.build().history))
        self.assertEqual(decoded, [FrameOutcome.NO_FRAME,          # first of the interval
                                   FrameOutcome.NO_DETECTIONS,
                                   FrameOutcome.VIABLE])           # most recent

    def test_the_history_stops_recording_past_capacity_but_the_counts_do_not(self):
        # A pathologically low rate_hz makes an interval longer than the history holds.
        # Per spec the frames that cannot fit are simply not recorded — but the AGGREGATE
        # must stay exact, or the record lies about how starved the interval was.
        for i in range(HISTORY_FRAMES + 9):
            self._note(FrameOutcome.NO_FRAME, frame_id=1)

        s = self.acc.build()
        self.assertEqual(len(s.history), HISTORY_FRAMES, "history is capped")
        self.assertEqual(s.frames_without_frame, HISTORY_FRAMES + 9,
                         "the COUNT must still cover every frame of the interval")

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

    async def test_sends_the_versioned_name(self):
        sender = _FakeSender()
        tracer, _ = self._tracer(sender)      # default name = TRACE_NAME = "BDD1"

        self._note(tracer, FrameOutcome.VIABLE, detection=_detection())
        await self._drain()

        self.assertEqual(len(sender.calls), 1)
        name, array_id, _data = sender.calls[0]
        self.assertEqual((name, array_id), (TRACE_NAME, 0))
        self.assertEqual(version_from_name(name), TRACE_FORMAT_VERSION)
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
        self.assertEqual(decoded, [FrameOutcome.NO_FRAME,          # oldest first
                                   FrameOutcome.VIABLE,
                                   FrameOutcome.NO_DETECTIONS])
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


class TestRateVsLoopRate(unittest.IsolatedAsyncioTestCase):
    """Frames-per-record must actually track BOTH knobs: the configured send rate
    (explicit) and the control-loop rate (implicit — the loop slows when the hardware
    thermally throttles). If it did not, a throttled flight would silently log a
    different amount of history than we think, and every record would misrepresent how
    much of the flight it covers."""

    async def _run(self, *, loop_fps: float, rate_hz: float, seconds: float = 4.0):
        sender = _FakeSender()
        clock = _FakeClock()
        tracer = UlogTracer(sender, rate_hz=rate_hz, now_fn=clock)

        n_frames = int(loop_fps * seconds)
        for i in range(n_frames):
            clock.now = round((i + 1) / loop_fps, 9)
            tracer.note(frame_id=i, outcome=FrameOutcome.NO_DETECTIONS,
                        pd_p=XY(1.0, 1.0), cmd=(0.0, 0.0, 0.5))
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        records = [d for _n, _a, d in sender.calls]
        frames_in = [len(unpack_history([d[SLOT_HISTORY_0], d[SLOT_HISTORY_1]]))
                     for d in records]
        counted = sum(d[SLOT_FRAMES_NO_DETECTION] + d[SLOT_FRAMES_WITHOUT_FRAME]
                      for d in records)
        return records, frames_in, counted, n_frames

    async def test_lowering_the_send_rate_puts_more_frames_in_each_record(self):
        # Same 30 fps loop, three send rates. Frames per record must scale as fps/rate.
        for rate_hz, expected in ((15.0, 2), (5.0, 6), (3.0, 10)):
            _rec, frames_in, _c, _n = await self._run(loop_fps=30.0, rate_hz=rate_hz)
            typical = sorted(frames_in)[len(frames_in) // 2]
            self.assertAlmostEqual(
                typical, expected, delta=1,
                msg=f"at 30 fps / {rate_hz} Hz a record should cover ~{expected} frames, got {typical}")

    async def test_a_slower_control_loop_puts_fewer_frames_in_each_record(self):
        # Same 5 Hz send rate, three loop rates — this is the THROTTLING case: the
        # hardware slows the loop down and nobody reconfigures anything.
        for loop_fps, expected in ((30.0, 6), (15.0, 3), (10.0, 2)):
            _rec, frames_in, _c, _n = await self._run(loop_fps=loop_fps, rate_hz=5.0)
            typical = sorted(frames_in)[len(frames_in) // 2]
            self.assertAlmostEqual(
                typical, expected, delta=1,
                msg=f"at {loop_fps} fps / 5 Hz a record should cover ~{expected} frames, got {typical}")

    async def test_every_frame_is_accounted_for_at_every_rate(self):
        # The conservation law. Whatever the two rates, the counts across all records
        # must add up to exactly the number of iterations the loop ran — nothing lost
        # between records, nothing double-counted.
        for loop_fps in (10.0, 30.0, 60.0):
            for rate_hz in (0.0, 3.0, 5.0, 15.0):
                _rec, _f, counted, n_frames = await self._run(loop_fps=loop_fps, rate_hz=rate_hz)
                # The final partial interval has not been dispatched yet, so it is not in
                # any record. Everything before it must be.
                self.assertLessEqual(counted, n_frames)
                self.assertGreaterEqual(
                    counted, n_frames - int(loop_fps / rate_hz) - 1 if rate_hz else n_frames - 1,
                    f"frames went missing at {loop_fps} fps / {rate_hz} Hz: "
                    f"{counted} counted of {n_frames}")

    async def test_uncapped_rate_writes_one_record_per_frame_with_no_history(self):
        # rate_hz = 0 => a record per control-loop iteration. The single frame's outcome
        # is redundant with the counts + detection, so the history is DROPPED: the record
        # trims to nothing on the wire. Every frame is still accounted for by the counts.
        records, frames_in, counted, n_frames = await self._run(loop_fps=30.0, rate_hz=0.0,
                                                                seconds=2.0)
        self.assertEqual(len(records), n_frames, "one record per frame")
        self.assertEqual(set(frames_in), {0}, "uncapped records carry no history")
        self.assertEqual(counted, n_frames, "but the counts still cover every frame")

    async def test_uncapped_no_detection_record_trims_to_the_counts(self):
        # The point of moving the history last: a no-detection frame in uncapped mode is
        # all-zero from the detection block on, so the wire payload trims away.
        sender = _FakeSender()
        clock = _FakeClock()
        tracer = UlogTracer(sender, rate_hz=0.0, now_fn=clock)

        tracer.note(frame_id=1, outcome=FrameOutcome.NO_DETECTIONS,
                    pd_p=XY(1.0, 1.0), cmd=(0.0, 0.0, 0.5))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        data = sender.calls[-1][2]
        # Detection block (9-13) and history (14-15) are all zero -> everything from the
        # detection block on trims.
        self.assertEqual(data[SLOT_DET_X:], [0.0] * (TRACE_SLOTS - SLOT_DET_X))
        self.assertEqual(data[SLOT_HISTORY_0], 0.0)
        self.assertEqual(data[SLOT_HISTORY_1], 0.0)


class TestSdlogDebugGate(unittest.IsolatedAsyncioTestCase):
    """DroneMover._check_ulog_debug_logging decides whether to trace at all: PX4 only
    WRITES the trace when SDLOG_PROFILE has the debug bit, so streaming to an FC without
    it is pure waste. The controller reads `ulog_debug_logging_enabled` and, when False,
    hands the tracer send_fn=None — the null-object path."""

    def _mover(self, param_result):
        # A DroneMover with just enough faked out to call the check. param_result is either
        # an int (the SDLOG_PROFILE value) or an Exception to raise.
        from drone import DroneMover

        class _FakeParam:
            async def get_param_int(self, name):
                if isinstance(param_result, Exception):
                    raise param_result
                return param_result

        class _FakeSystem:
            param = _FakeParam()

        mover = DroneMover.__new__(DroneMover)      # skip __init__ / MAVSDK connect
        mover.ulog_debug_logging_enabled = True
        mover.drone = _FakeSystem()
        return mover

    async def test_debug_bit_set_enables_tracing(self):
        from drone import DroneMover
        mover = self._mover(1024 | DroneMover.SDLOG_PROFILE_DEBUG_BIT | 1)   # the rig's 1057
        self.assertTrue(await mover._check_ulog_debug_logging())
        self.assertTrue(mover.ulog_debug_logging_enabled)

    async def test_debug_bit_clear_disables_tracing(self):
        mover = self._mover(1)      # default profile, no debug bit
        self.assertFalse(await mover._check_ulog_debug_logging())
        self.assertFalse(mover.ulog_debug_logging_enabled)

    async def test_unreadable_param_leaves_tracing_enabled(self):
        # A transient param-read glitch (seen on the real link) must NOT silently kill a
        # post-crash forensic feature. Degrade toward keeping it.
        mover = self._mover(RuntimeError("timeout"))
        self.assertTrue(await mover._check_ulog_debug_logging())
        self.assertTrue(mover.ulog_debug_logging_enabled)

    async def test_a_disabled_mover_makes_the_tracer_a_no_op(self):
        # This is the wiring the controller does: send_fn=None when the gate is closed.
        mover = self._mover(1)
        await mover._check_ulog_debug_logging()
        enabled = mover.ulog_debug_logging_enabled
        tracer = UlogTracer(mover.send_debug_array if enabled else None, rate_hz=5.0)
        self.assertFalse(tracer.enabled)


if __name__ == "__main__":
    unittest.main()
