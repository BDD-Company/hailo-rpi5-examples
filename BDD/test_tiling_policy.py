"""Unit tests for the size-driven tiling ladder policy (host-runnable; pure — no
hailo/GStreamer). Pins the tier decision that user_app_callback_class.note_tiling
delegates to, plus build_ladder's validation and empty-ladder handling.

side = max(bbox_w, bbox_h) of the primary tracked target (0..1); None = lost.
now_s is an injected monotonic timestamp so loss timers test without sleeping.

The second half of this file covers TilingSwitchCoordinator, which serializes every
switch (policy worker + --test-switch-s harness) through one entry point, and
BranchStallWatchdog, which reverts to whole-frame when an accepted rung later dies.
"""
import threading
import time

import pytest
from tiling_policy import (
    BranchStallWatchdog,
    LadderTier,
    TilingLadderPolicy,
    TilingSwitchCoordinator,
    build_ladder,
)


# Default 3-rung ladder: index 0 = 3x2 (most tiles) -> 2 = whole.
def ladder3():
    return [
        LadderTier(3, 2, up_side=0.05, down_side=None),
        LadderTier(2, 1, up_side=0.10, down_side=0.02),
        LadderTier(1, 1, up_side=None, down_side=0.05),
    ]


def policy(current_i, l2=1.0, l10=10.0):
    return TilingLadderPolicy(ladder3(), lost_to_2x1_s=l2, lost_to_3x2_s=l10, current_i=current_i)


# --- size hysteresis: one tier per decision, dead-bands don't thrash ----------
def test_whole_descends_to_2x1_below_down_side():
    p = policy(current_i=2)                 # whole
    assert p.note(0.04, now_s=0.0) == 1     # side < 0.05 -> descend to 2x1
    assert p.note(0.03, now_s=0.0) == 1     # not yet committed -> keeps asking


def test_whole_holds_in_deadband():
    p = policy(current_i=2)                 # whole; leaves only at side<0.05
    assert p.note(0.07, now_s=0.0) is None  # 0.05..0.10 dead-band -> stay whole


def test_2x1_climbs_to_whole_at_up_side():
    p = policy(current_i=1)
    assert p.note(0.10, now_s=0.0) == 2     # side >= 0.10 -> climb to whole
    assert p.note(0.20, now_s=0.0) == 2


def test_2x1_descends_to_3x2_below_down_side():
    p = policy(current_i=1)
    assert p.note(0.01, now_s=0.0) == 0     # side < 0.02 -> descend to 3x2


def test_2x1_holds_between_thresholds():
    p = policy(current_i=1)                 # leaves at >=0.10 up or <0.02 down
    assert p.note(0.05, now_s=0.0) is None


def test_one_tier_per_decision_even_on_huge_jump():
    p = policy(current_i=0)                 # 3x2, tiny target suddenly huge
    assert p.note(0.9, now_s=0.0) == 1      # only ONE step (to 2x1), not to whole
    p.committed(1)
    assert p.note(0.9, now_s=0.0) == 2      # next decision climbs one more


# --- loss escalation (time-based, injected now_s) -----------------------------
def test_lost_descends_after_1s_then_after_10s():
    p = policy(current_i=2)                 # whole
    assert p.note(None, now_s=0.0) is None  # just became lost
    assert p.note(None, now_s=0.5) is None  # < 1s
    assert p.note(None, now_s=1.0) == 1     # >= lost_to_2x1_s -> descend one rung
    p.committed(1)
    assert p.note(None, now_s=9.9) is None  # < 10s from lost start, already 2x1
    assert p.note(None, now_s=10.0) == 0    # >= lost_to_3x2_s -> tier 0 (3x2)


def test_lost_never_climbs_toward_whole():
    p = policy(current_i=0)                 # already max tiles
    assert p.note(None, now_s=100.0) is None  # loss only descends; nothing to do


def test_reacquire_resets_lost_timer_and_resumes_size():
    p = policy(current_i=2)                 # whole
    p.note(None, now_s=0.0)                 # lost starts at t=0
    assert p.note(0.5, now_s=0.5) is None   # target back, large -> stay whole
    # a fresh loss must re-time from now, not from the old t=0.
    assert p.note(None, now_s=0.6) is None
    assert p.note(None, now_s=1.6) == 1     # 0.6 -> 1.6 is >= 1s


# --- committed() --------------------------------------------------------------
def test_committed_adopts_tier():
    p = policy(current_i=2)
    p.note(0.04, now_s=0.0)                 # asks for 2x1
    p.committed(1)
    assert p.current_i == 1
    assert p.note(0.10, now_s=0.0) == 2     # reverse threshold from the new rung


# --- build_ladder: validation + empty-ladder ----------------------------------
class _Cfg:  # duck-typed stand-in for Config.Tiling
    def __init__(self, ladder=None, l2=1.0, l10=10.0):
        self.ladder = ladder or []
        self.lost_to_2x1_s, self.lost_to_3x2_s = l2, l10


class _T:  # duck-typed Config.Tiling.Tier
    def __init__(self, tx, ty, up=None, down=None):
        self.tiles_x, self.tiles_y, self.up_side, self.down_side = tx, ty, up, down


def test_build_ladder_uses_explicit_ladder():
    tiers = build_ladder(_Cfg(ladder=[_T(3, 2, up=0.05), _T(2, 1, up=0.10, down=0.02), _T(1, 1, down=0.05)]))
    assert [(t.tiles_x, t.tiles_y) for t in tiers] == [(3, 2), (2, 1), (1, 1)]


def test_build_ladder_empty_returns_empty():
    assert build_ladder(_Cfg(ladder=[])) == []      # whole-frame only, not switchable


def test_build_ladder_rejects_last_not_whole():
    with pytest.raises(ValueError):
        build_ladder(_Cfg(ladder=[_T(3, 2, up=0.05), _T(2, 1, down=0.02)]))  # last is 2x1


def test_build_ladder_rejects_non_decreasing_tile_counts():
    with pytest.raises(ValueError):
        build_ladder(_Cfg(ladder=[_T(2, 1, up=0.05), _T(3, 2, up=0.10, down=0.02), _T(1, 1, down=0.05)]))


# ===========================================================================
# TilingSwitchCoordinator — the ONE serialized entry point for tier switches.
# ===========================================================================
class FakeSwitch:
    """Stand-in for GStreamerApp.switch_to_tier: records calls, can be slow, can fail.

    `overlap` records whether any two calls were ever in flight simultaneously —
    the race the coordinator exists to prevent.
    """
    def __init__(self, result=True, delay=0.0, raises=None):
        self.result = result
        self.delay = delay
        self.raises = raises
        self.calls = []
        self.overlap = False
        self._active = 0
        self._lock = threading.Lock()

    def __call__(self, tier_i):
        with self._lock:
            self._active += 1
            if self._active > 1:
                self.overlap = True
        try:
            self.calls.append(tier_i)
            if self.delay:
                time.sleep(self.delay)
            if self.raises:
                raise self.raises
            return self.result(tier_i) if callable(self.result) else self.result
        finally:
            with self._lock:
                self._active -= 1


class FakeClock:
    def __init__(self):
        self.t = 1000.0
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


def make_coord(clock=None, **kw):
    """Coordinator on the default 3-rung ladder, booted on whole-frame (tier 2)."""
    p = policy(current_i=2)
    sw = FakeSwitch(**kw)
    coord = (TilingSwitchCoordinator(p, sw, now=clock) if clock
             else TilingSwitchCoordinator(p, sw))
    return coord, sw, p


def test_successful_request_commits_the_tier():
    coord, sw, p = make_coord()
    assert coord.request(1) is True
    assert sw.calls == [1]
    assert p.current_i == 1 and coord.current_tier == 1
    assert coord.on_whole is False


def test_request_is_a_noop_when_already_on_target_tier():
    coord, sw, _ = make_coord()
    assert coord.request(2) is True          # already whole-frame
    assert sw.calls == []                    # never touched the pipeline


def test_manual_toggle_keeps_policy_in_sync():
    """THE DESYNC BUG: --test-switch-s used to call the switch directly, leaving the
    policy's current_i stale so it could never command the reverse switch."""
    coord, _, p = make_coord()
    coord.request(1)                         # what the harness thread does
    assert p.current_i == 1

    # Policy now sees a big target on 2x1 -> must be able to ask for whole-frame.
    assert coord.note(0.5, now_s=0.0) == 2
    assert coord.request(2) is True
    assert p.current_i == 2


def test_failed_switch_keeps_the_working_tier_and_backs_off():
    clock = FakeClock()
    coord, sw, p = make_coord(clock=clock, result=False)
    assert coord.request(0) is False
    assert p.current_i == 2                  # still on the rung that was producing
    assert coord.tiles_blocked is True       # backed off instead of retrying per frame

    # A tiled target is refused while the backoff holds...
    assert coord.note(0.001, now_s=0.0) is None
    # ...and it can try again once the backoff expires.
    clock.advance(TilingSwitchCoordinator.FAILED_SWITCH_BACKOFF_S + 0.1)
    assert coord.tiles_blocked is False
    assert coord.note(0.001, now_s=0.0) == 1


def test_a_failed_switch_never_blocks_the_way_back_to_whole_frame():
    clock = FakeClock()
    coord, sw, p = make_coord(clock=clock, result=False)
    coord.request(0)                         # fails, arms the backoff
    assert coord.tiles_blocked is True
    sw.result = True
    assert coord.request(coord.whole_i) is True   # whole-frame is never refused


def test_switch_fn_exception_is_treated_as_a_failed_switch():
    coord, _, p = make_coord(raises=RuntimeError("gst blew up"))
    assert coord.request(1) is False         # does not propagate
    assert p.current_i == 2


def test_note_returns_none_while_a_switch_is_in_flight():
    """Otherwise the streaming thread spawns a worker per frame: the policy keeps
    returning a target for as long as the size/loss condition holds."""
    coord, sw, _ = make_coord(delay=0.2)
    t = threading.Thread(target=coord.request, args=(1,))
    t.start()
    time.sleep(0.05)                         # switch is now in flight
    assert coord.switch_in_flight is True
    assert coord.note(0.001, now_s=0.0) is None
    assert coord.note(0.001, now_s=0.0) is None
    t.join()
    assert sw.calls == [1]


def test_note_never_blocks_while_a_switch_is_in_flight():
    """The lock must NOT be held across the handover: the GStreamer streaming thread
    calls note() every frame and would stall ~30 frames on every switch."""
    coord, _, _ = make_coord(delay=0.5)
    t = threading.Thread(target=coord.request, args=(1,))
    t.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    for _ in range(50):
        coord.note(0.001, now_s=0.0)
    elapsed = time.perf_counter() - t0
    t.join()
    assert elapsed < 0.05, f"note() blocked on the handover ({elapsed:.3f}s for 50 calls)"


def test_concurrent_requests_never_overlap_in_the_handover():
    """THE RACE: two callers interleaving valve/selector set_property calls could leave
    the input-selector on a shut-valve branch."""
    coord, sw, _ = make_coord(delay=0.05)
    threads = [threading.Thread(target=coord.request, args=(i % 3,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sw.overlap is False, "two switch_to_tier calls were in flight at once"


def test_in_flight_request_reports_current_tier_rather_than_queueing():
    coord, sw, _ = make_coord(delay=0.2)
    t = threading.Thread(target=coord.request, args=(1,))
    t.start()
    time.sleep(0.05)
    # A second caller asking for whole-frame while the tier-1 switch settles: we are
    # still on whole-frame, so it is already satisfied; it must not touch the pipeline.
    assert coord.request(2) is True
    t.join()
    assert sw.calls == [1]


def test_policy_and_harness_do_not_leave_pipeline_and_policy_diverged():
    """End-to-end of the two bugs together: a harness thread cycling tiers while the
    policy also drives. Whatever the interleaving, the policy's belief must equal the
    tier the fake pipeline actually ended on."""
    pipeline_tier = {'i': 2}

    def switch(tier_i):
        time.sleep(0.01)
        pipeline_tier['i'] = tier_i
        return True

    p = policy(current_i=2)
    coord = TilingSwitchCoordinator(p, switch)

    stop = threading.Event()

    def harness():
        want = 0
        while not stop.is_set():
            coord.request(want)
            want = (want + 1) % 3
            time.sleep(0.005)

    def streaming():
        while not stop.is_set():
            # Alternate a tiny and a large target so the policy keeps asking to move.
            side = 0.001 if pipeline_tier['i'] == 2 else 0.9
            target = coord.note(side, now_s=time.monotonic())
            if target is not None:
                coord.request(target)
            time.sleep(0.001)

    ts = [threading.Thread(target=harness), threading.Thread(target=streaming)]
    for t in ts: t.start()
    time.sleep(1.0)
    stop.set()
    for t in ts: t.join()

    # Quiesce: no switch in flight, then the two views must agree.
    assert coord.switch_in_flight is False
    assert coord.current_tier == pipeline_tier['i'], (
        f"policy believes tier={coord.current_tier} but pipeline is on "
        f"tier={pipeline_tier['i']}")


# ===========================================================================
# BranchStallWatchdog — recovers from a rung that warmed up and then DIED.
# ===========================================================================
class FakePipeline:
    """A ladder pipeline whose tiled rungs can be 'killed' (stop delivering)."""
    def __init__(self, whole_i=2):
        self.whole_i = whole_i
        self.tier = whole_i
        self.tile_dead = False
        self.frames = 0
        self.switch_calls = []

    def switch(self, tier_i):
        self.switch_calls.append(tier_i)
        self.tier = tier_i
        return True

    def tick(self):
        """One source frame. A dead tiled rung delivers nothing to the callback."""
        if self.tier != self.whole_i and self.tile_dead:
            return
        self.frames += 1


def make_watchdog(stall_timeout_s=2.0, cooldown_s=30.0):
    clock = FakeClock()
    pipe = FakePipeline()
    p = policy(current_i=2)
    coord = TilingSwitchCoordinator(p, pipe.switch, now=clock)
    wd = BranchStallWatchdog(coord, lambda: pipe.frames, stall_timeout_s=stall_timeout_s,
                             cooldown_s=cooldown_s, now=clock)
    return wd, coord, pipe, clock


def test_watchdog_is_quiet_while_frames_flow():
    wd, coord, pipe, clock = make_watchdog()
    coord.request(1)
    for _ in range(20):
        pipe.tick()
        clock.advance(0.25)
        assert wd.poll() is False
    assert coord.current_tier == 1          # never touched a healthy rung


def test_watchdog_reverts_a_tiled_rung_that_died_after_warming_up():
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=2.0)
    coord.request(1)
    pipe.tick(); wd.poll()                  # establish progress
    pipe.tile_dead = True                   # rung dies AFTER being accepted

    clock.advance(1.0); assert wd.poll() is False   # inside the stall window
    clock.advance(1.5)                              # now past stall_timeout_s
    assert wd.poll() is True

    assert coord.current_tier == 2
    assert pipe.tier == 2
    assert pipe.switch_calls == [1, 2]


def test_watchdog_never_switches_when_whole_frame_is_the_stalled_rung():
    """Whole-frame is fed by the same source tee and the same hailonet: if it is starved,
    tiling cannot help. Log, do not thrash."""
    wd, coord, pipe, clock = make_watchdog()
    # never left whole-frame; simply stop delivering frames
    pipe.tick(); wd.poll()
    clock.advance(10.0)
    assert wd.poll() is False
    assert pipe.switch_calls == []          # pipeline untouched


def test_a_handover_in_flight_is_not_mistaken_for_a_stall(caplog):
    """A switch legitimately pauses delivery for up to warmup_timeout_s. Asserting only
    `poll() is False` is not enough: the coordinator's own in-flight guard would make a
    spurious revert a no-op and still return False. Assert nothing was even attempted."""
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0)
    coord.request(1)
    pipe.tick(); wd.poll()

    with coord._lock:
        coord._in_flight = True
    clock.advance(5.0)                          # frames pause for the whole handover
    with caplog.at_level("ERROR"):
        assert wd.poll() is False
    assert not [r for r in caplog.records if "STALL" in r.getMessage()]
    assert coord.tiles_blocked is False         # no cooldown armed
    assert pipe.switch_calls == [1]             # pipeline untouched

    with coord._lock:
        coord._in_flight = False
    # The stall clock was RESET by the handover, not disabled: a fresh timeout is needed...
    assert wd.poll() is False
    clock.advance(1.5)
    assert wd.poll() is True                    # ...and then it does fire


def test_cooldown_blocks_re_entering_the_dead_rung():
    """Without this the loss escalation fires again within ~1s and dives straight back
    into the rung the watchdog just proved dead."""
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0, cooldown_s=30.0)
    coord.request(1)
    pipe.tick(); wd.poll()
    pipe.tile_dead = True
    clock.advance(1.5)
    assert wd.poll() is True                       # reverted
    assert coord.tiles_blocked is True

    # The policy wants tiles again immediately (target tiny / lost every frame).
    assert [coord.note(0.001, now_s=0.0) for _ in range(5)] == [None] * 5
    assert coord.request(1) is False               # and a direct request is refused
    assert pipe.tier == 2
    assert pipe.switch_calls == [1, 2]             # never re-entered

    clock.advance(31.0)                            # cooldown expires
    assert coord.tiles_blocked is False
    assert coord.request(1) is True
    assert pipe.switch_calls == [1, 2, 1]


def test_watchdog_retries_if_the_revert_itself_fails():
    """If whole-frame will not come back either, keep trying — never give up on
    returning to the known-good rung."""
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0)
    coord.request(1)
    pipe.tick(); wd.poll()
    pipe.tile_dead = True

    calls = []
    def failing_switch(tier_i):
        calls.append(tier_i)
        return False                       # revert cannot land
    coord._switch_fn = failing_switch

    clock.advance(1.5)
    assert wd.poll() is False              # reported failure, did not claim success
    assert coord.current_tier == 1         # still stuck on the dead rung, honestly reported

    clock.advance(1.5)
    assert wd.poll() is False
    assert calls == [2, 2]                 # retried rather than latching


def test_whole_frame_stall_is_logged_once_not_every_poll(caplog):
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0)
    pipe.tick(); wd.poll()
    clock.advance(2.0)
    with caplog.at_level("ERROR"):
        for _ in range(5):
            wd.poll()
    stalls = [r for r in caplog.records if "WHOLE-FRAME branch" in r.getMessage()]
    assert len(stalls) == 1, [r.getMessage() for r in stalls]

    # ...and it re-arms once frames come back and stop again.
    pipe.tick(); wd.poll()
    clock.advance(2.0)
    with caplog.at_level("ERROR"):
        wd.poll()
    stalls = [r for r in caplog.records if "WHOLE-FRAME branch" in r.getMessage()]
    assert len(stalls) == 2


def test_watchdog_stays_disarmed_until_the_first_frame_ever_arrives(caplog):
    """Regression: on the rig the watchdog screamed
    "STALL: no callbacks for 2.0s on the WHOLE-FRAME branch" ~2s into every launch, while
    the camera and the Hailo device were still coming up. A watchdog for "warmed up and
    then died" must first observe "warmed up"."""
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0)
    assert pipe.frames == 0

    with caplog.at_level("ERROR"):
        for _ in range(10):                      # 10s of startup, no callbacks yet
            clock.advance(1.0)
            assert wd.poll() is False
    assert not [r for r in caplog.records if "STALL" in r.getMessage()], \
        [r.getMessage() for r in caplog.records]

    # Once the pipeline has proven it can deliver, a later freeze IS a stall.
    pipe.tick(); wd.poll()
    clock.advance(2.0)
    with caplog.at_level("ERROR"):
        wd.poll()
    assert [r for r in caplog.records if "STALL" in r.getMessage()]
