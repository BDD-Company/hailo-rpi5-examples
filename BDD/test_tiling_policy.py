"""Unit tests for the detection-state tiling switch policy (host-runnable; the
policy is pure, so no hailo/GStreamer needed — that's why it was split out of the
un-importable app.py). These pin the whole<->tile switch decision that
user_app_callback_class.note_detection delegates to.

Contract mirrored from app.py: each frame's best target confidence is fed to
``note`` on the streaming thread; when it returns a branch, app.py fires the
(blocking) switch on a worker thread and calls ``committed`` once it lands.

The second half of this file covers TilingSwitchCoordinator, which serializes every
switch (policy worker + --test-switch-s harness) through one entry point, and
BranchStallWatchdog, which reverts to whole-frame when an accepted branch later dies.
"""
import threading
import time

from tiling_policy import (
    BranchStallWatchdog,
    TilingSwitchCoordinator,
    TilingSwitchPolicy,
)


def feed(policy, confs):
    """Feed a sequence of confidences; return the list of note() results."""
    return [policy.note(c) for c in confs]


# ---------------------------------------------------------------------------
# whole -> tile: fires only after N *consecutive* lost frames.
# ---------------------------------------------------------------------------
def test_stays_whole_while_confident():
    p = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=10, locked_frames_to_whole=5)
    assert feed(p, [0.9] * 50) == [None] * 50
    assert p.tiling_on is False


def test_switches_to_tiling_exactly_at_threshold():
    p = TilingSwitchPolicy(lost_frames_to_tile=10)
    results = feed(p, [0.0] * 12)
    # first 9 lost frames: no switch; the 10th crosses the threshold.
    assert results[:9] == [None] * 9
    assert results[9] is True
    # still lost and not yet committed -> keeps asking to switch each frame.
    assert results[10] is True and results[11] is True


def test_lost_streak_needs_consecutive_frames():
    p = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=10)
    # 9 lost, then one confident frame resets the streak, so no switch yet.
    assert feed(p, [0.0] * 9 + [0.8]) == [None] * 10
    # need a *fresh* run of 10 lost frames.
    assert feed(p, [0.0] * 9) == [None] * 9
    assert p.note(0.0) is True


def test_confidence_at_threshold_counts_as_confident():
    # best_conf == switch_conf is "confident" (>=), so it does NOT accrue a lost frame.
    p = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=3)
    assert feed(p, [0.4, 0.4, 0.4, 0.4]) == [None, None, None, None]
    assert p.tiling_on is False


# ---------------------------------------------------------------------------
# tile -> whole: fires only after N *consecutive* confident frames while tiling.
# ---------------------------------------------------------------------------
def test_switches_back_to_whole_after_locked_streak():
    p = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=5, tiling_on=True)
    results = feed(p, [0.9] * 6)
    assert results[:4] == [None] * 4
    assert results[4] is False              # 5th confident frame -> back to whole
    assert results[5] is False              # keeps asking until committed


def test_lock_streak_decays_by_the_penalty_on_a_miss_it_does_not_reset():
    """A miss costs `locked_streak_miss_penalty` frames of progress, not all of it.
    Resetting would demand `locked_frames_to_whole` hits in an unbroken row, which a
    detector that misses even occasionally never delivers — the rig would stay pinned to
    the high-latency tile branch forever."""
    p = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=5,
                           locked_streak_miss_penalty=2, tiling_on=True)
    assert feed(p, [0.9] * 4) == [None] * 4          # streak 4
    assert p.note(0.1) is None                       # miss -> 4 - 2 = 2
    assert p._lock_streak == 2
    assert feed(p, [0.9] * 2) == [None] * 2          # 3, 4
    assert p.note(0.9) is False                      # 5 -> switch to whole-frame


def test_lock_streak_floors_at_zero_so_a_miss_run_banks_no_debt():
    p = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=5,
                           locked_streak_miss_penalty=2, tiling_on=True)
    feed(p, [0.9] * 2)                               # streak 2
    feed(p, [0.0] * 10)                              # long miss run
    assert p._lock_streak == 0                       # not -18
    assert feed(p, [0.9] * 4) == [None] * 4          # so 5 confident frames still suffice
    assert p.note(0.9) is False


def test_a_flaky_detector_still_reaches_whole_frame_below_the_miss_threshold():
    """Drift per frame is (1-m) - m*P, so whole-frame is reached iff m < 1/(1+P).
    P=2 => tolerates just under 1-in-3 misses. This is the whole point of the decay."""
    # 1 miss every 4 frames (m=0.25 < 1/3): converges.
    p = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=10,
                           locked_streak_miss_penalty=2, tiling_on=True)
    assert any(p.note(c) is False for c in [0.9, 0.9, 0.9, 0.0] * 40), \
        "a 25% miss rate must still get back to whole-frame"

    # 1 miss every 2 frames (m=0.5 > 1/3): never converges, stays on tiling.
    q = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=10,
                           locked_streak_miss_penalty=2, tiling_on=True)
    assert all(q.note(c) is None for c in [0.9, 0.0] * 200)
    assert q._lock_streak == 0


def test_penalty_zero_counts_confident_frames_regardless_of_misses():
    p = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=3,
                           locked_streak_miss_penalty=0, tiling_on=True)
    assert feed(p, [0.9, 0.0, 0.9, 0.0]) == [None] * 4    # streak 2, misses ignored
    assert p.note(0.9) is False                            # 3rd confident frame


def test_penalty_at_or_above_the_threshold_reproduces_reset_on_miss():
    """The old behaviour stays expressible, for anyone who wants it back."""
    p = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=5,
                           locked_streak_miss_penalty=5, tiling_on=True)
    feed(p, [0.9] * 4)
    assert p.note(0.0) is None
    assert p._lock_streak == 0                       # fully reset
    assert feed(p, [0.9] * 4) == [None] * 4
    assert p.note(0.9) is False


def test_the_lost_streak_still_resets_on_a_single_confident_frame():
    """Deliberately asymmetric: the decay applies to the LOCK streak only. One real
    sighting means the target is not lost, so do not pay tiling's latency to hunt it."""
    p = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=5, tiling_on=False)
    assert feed(p, [0.0] * 4) == [None] * 4
    assert p.note(0.9) is None                       # one sighting...
    assert p._lost_streak == 0                       # ...wipes the whole lost streak
    assert feed(p, [0.0] * 4) == [None] * 4          # a fresh run of 5 is required
    assert p.note(0.0) is True


def test_no_downswitch_when_already_whole():
    # Confident frames while already on whole-frame never produce a decision.
    p = TilingSwitchPolicy(locked_frames_to_whole=1, tiling_on=False)
    assert feed(p, [0.99] * 5) == [None] * 5


# ---------------------------------------------------------------------------
# committed(): adopt the new branch + reset streaks (as app.py's worker does).
# ---------------------------------------------------------------------------
def test_committed_adopts_branch_and_resets_streaks():
    p = TilingSwitchPolicy(lost_frames_to_tile=10, locked_frames_to_whole=5)
    feed(p, [0.0] * 10)                      # asks to switch to tiling
    p.committed(True)                        # worker reports the switch landed
    assert p.tiling_on is True
    assert p._lost_streak == 0 and p._lock_streak == 0
    # now the *reverse* threshold applies from a clean slate.
    assert feed(p, [0.9] * 4) == [None] * 4
    assert p.note(0.9) is False


def test_full_round_trip():
    p = TilingSwitchPolicy(switch_conf=0.5, lost_frames_to_tile=3, locked_frames_to_whole=2)
    assert p.note(0.0) is None
    assert p.note(0.0) is None
    assert p.note(0.0) is True               # 3 lost -> tile
    p.committed(True)
    assert p.note(0.7) is None
    assert p.note(0.7) is False              # 2 locked -> whole
    p.committed(False)
    assert p.tiling_on is False
    assert p.note(0.7) is None               # clean slate, confident, stay whole


def test_reset_streaks_clears_counters_without_changing_branch():
    p = TilingSwitchPolicy(tiling_on=True)
    feed(p, [0.9, 0.9])
    p.reset_streaks()
    assert p._lock_streak == 0 and p._lost_streak == 0
    assert p.tiling_on is True               # branch unchanged


# ===========================================================================
# TilingSwitchCoordinator — the ONE serialized entry point for branch switches.
# ===========================================================================
class FakeSwitch:
    """Stand-in for GStreamerApp.switch_tiling: records calls, can be slow, can fail.

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

    def __call__(self, to_tiling):
        with self._lock:
            self._active += 1
            if self._active > 1:
                self.overlap = True
        try:
            self.calls.append(to_tiling)
            if self.delay:
                time.sleep(self.delay)
            if self.raises:
                raise self.raises
            return self.result(to_tiling) if callable(self.result) else self.result
        finally:
            with self._lock:
                self._active -= 1


def make_coord(**kw):
    policy = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=3,
                                locked_frames_to_whole=2, tiling_on=False)
    sw = FakeSwitch(**kw)
    return TilingSwitchCoordinator(policy, sw), sw, policy


def test_successful_request_commits_the_branch():
    coord, sw, policy = make_coord()
    assert coord.request(True) is True
    assert sw.calls == [True]
    assert policy.tiling_on is True and coord.tiling_on is True


def test_request_is_a_noop_when_already_on_target_branch():
    coord, sw, _ = make_coord()
    assert coord.request(False) is True      # already whole-frame
    assert sw.calls == []                    # never touched the pipeline


def test_manual_toggle_keeps_policy_in_sync():
    """THE DESYNC BUG: --test-switch-s used to call switch_tiling directly, leaving
    policy.tiling_on stale so the policy could never command the reverse switch."""
    coord, _, policy = make_coord()
    coord.request(True)                      # what the harness thread does
    assert policy.tiling_on is True

    # Policy now sees confident frames while tiling -> must be able to ask for whole.
    assert coord.note(0.9) is None           # locked_frames_to_whole=2, first frame
    assert coord.note(0.9) is False          # streak reached -> switch back to whole
    assert coord.request(False) is True
    assert policy.tiling_on is False


def test_failed_switch_keeps_the_working_branch_and_backs_off():
    coord, sw, policy = make_coord(result=False)
    assert coord.request(True) is False
    assert policy.tiling_on is False         # still on the branch that was producing
    assert policy._lost_streak == 0 and policy._lock_streak == 0   # backed off

    # It must be able to try again once a fresh streak builds.
    assert [coord.note(0.0) for _ in range(3)] == [None, None, True]


def test_switch_fn_exception_is_treated_as_a_failed_switch():
    coord, _, policy = make_coord(raises=RuntimeError("gst blew up"))
    assert coord.request(True) is False      # does not propagate
    assert policy.tiling_on is False


def test_note_returns_none_while_a_switch_is_in_flight():
    """Otherwise the streaming thread spawns a worker per frame: TilingSwitchPolicy.note
    keeps returning a target once its streak threshold is crossed."""
    coord, sw, _ = make_coord(delay=0.2)
    t = threading.Thread(target=coord.request, args=(True,))
    t.start()
    time.sleep(0.05)                         # switch is now in flight
    assert coord.switch_in_flight is True
    assert coord.note(0.0) is None
    assert coord.note(0.0) is None
    t.join()
    assert sw.calls == [True]


def test_note_never_blocks_while_a_switch_is_in_flight():
    """The lock must NOT be held across the handover: the GStreamer streaming thread
    calls note() every frame and would stall ~30 frames on every switch."""
    coord, _, _ = make_coord(delay=0.5)
    t = threading.Thread(target=coord.request, args=(True,))
    t.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    for _ in range(50):
        coord.note(0.0)
    elapsed = time.perf_counter() - t0
    t.join()
    assert elapsed < 0.05, f"note() blocked on the handover ({elapsed:.3f}s for 50 calls)"


def test_concurrent_requests_never_overlap_in_the_handover():
    """THE RACE: two callers interleaving valve/selector set_property calls could leave
    the input-selector on a shut-valve branch."""
    coord, sw, _ = make_coord(delay=0.05)
    threads = [threading.Thread(target=coord.request, args=(bool(i % 2),))
               for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert sw.overlap is False, "two switch_tiling calls were in flight at once"


def test_in_flight_request_reports_current_branch_rather_than_queueing():
    coord, sw, _ = make_coord(delay=0.2)
    t = threading.Thread(target=coord.request, args=(True,))
    t.start()
    time.sleep(0.05)
    # A second caller asking for whole-frame while the tiling switch settles: we are
    # still on whole-frame, so it is already satisfied; it must not touch the pipeline.
    assert coord.request(False) is True
    t.join()
    assert sw.calls == [True]


def test_policy_and_harness_do_not_leave_pipeline_and_policy_diverged():
    """End-to-end of the two bugs together: a harness thread toggling while the policy
    also drives. Whatever the interleaving, the policy's belief must equal the branch
    the fake pipeline actually ended on."""
    pipeline_branch = {'tiling': False}

    def switch(to_tiling):
        time.sleep(0.01)
        pipeline_branch['tiling'] = to_tiling
        return True

    policy = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=2,
                                locked_frames_to_whole=2, tiling_on=False)
    coord = TilingSwitchCoordinator(policy, switch)

    stop = threading.Event()

    def harness():
        want = True
        while not stop.is_set():
            coord.request(want)
            want = not want
            time.sleep(0.005)

    def streaming():
        while not stop.is_set():
            target = coord.note(0.0 if pipeline_branch['tiling'] is False else 0.9)
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
    assert coord.tiling_on == pipeline_branch['tiling'], (
        f"policy believes tiling={coord.tiling_on} but pipeline is on "
        f"tiling={pipeline_branch['tiling']}")


# ===========================================================================
# BranchStallWatchdog — recovers from a branch that warmed up and then DIED.
# ===========================================================================
class FakeClock:
    def __init__(self):
        self.t = 1000.0
    def __call__(self):
        return self.t
    def advance(self, dt):
        self.t += dt


class FakePipeline:
    """A switchable pipeline whose tile branch can be 'killed' (stops delivering)."""
    def __init__(self):
        self.tiling = False
        self.tile_dead = False
        self.frames = 0
        self.switch_calls = []

    def switch(self, to_tiling):
        self.switch_calls.append(to_tiling)
        self.tiling = to_tiling
        return True

    def tick(self):
        """One source frame. A dead tile branch delivers nothing to the callback."""
        if self.tiling and self.tile_dead:
            return
        self.frames += 1


def make_watchdog(stall_timeout_s=2.0, cooldown_s=30.0, lost_frames_to_tile=3):
    clock = FakeClock()
    pipe = FakePipeline()
    policy = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=lost_frames_to_tile,
                                locked_frames_to_whole=2, tiling_on=False)
    coord = TilingSwitchCoordinator(policy, pipe.switch, now=clock)
    wd = BranchStallWatchdog(coord, lambda: pipe.frames, stall_timeout_s=stall_timeout_s,
                             cooldown_s=cooldown_s, now=clock)
    return wd, coord, pipe, clock


def test_watchdog_is_quiet_while_frames_flow():
    wd, coord, pipe, clock = make_watchdog()
    coord.request(True)
    for _ in range(20):
        pipe.tick()
        clock.advance(0.25)
        assert wd.poll() is False
    assert coord.tiling_on is True          # never touched a healthy branch


def test_watchdog_reverts_a_tile_branch_that_died_after_warming_up():
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=2.0)
    coord.request(True)
    pipe.tick(); wd.poll()                  # establish progress
    pipe.tile_dead = True                   # branch dies AFTER being accepted

    clock.advance(1.0); assert wd.poll() is False   # inside the stall window
    clock.advance(1.5)                              # now past stall_timeout_s
    assert wd.poll() is True

    assert coord.tiling_on is False
    assert pipe.tiling is False
    assert pipe.switch_calls == [True, False]


def test_watchdog_never_switches_when_whole_frame_is_the_stalled_branch():
    """Whole-frame is fed by the same source tee and the same hailonet: if it is starved,
    tiling cannot help. Log, do not thrash."""
    wd, coord, pipe, clock = make_watchdog()
    # never switched to tiling; simply stop delivering frames
    pipe.tick(); wd.poll()
    clock.advance(10.0)
    assert wd.poll() is False
    assert pipe.switch_calls == []          # pipeline untouched


def test_a_handover_in_flight_is_not_mistaken_for_a_stall(caplog):
    """A switch legitimately pauses delivery for up to warmup_timeout_s. Asserting only
    `poll() is False` is not enough: the coordinator's own in-flight guard would make a
    spurious revert a no-op and still return False. Assert nothing was even attempted."""
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0)
    coord.request(True)
    pipe.tick(); wd.poll()

    with coord._lock:
        coord._in_flight = True
    clock.advance(5.0)                          # frames pause for the whole handover
    with caplog.at_level("ERROR"):
        assert wd.poll() is False
    assert not [r for r in caplog.records if "STALL" in r.getMessage()]
    assert coord.tiling_blocked is False        # no cooldown armed
    assert pipe.switch_calls == [True]          # pipeline untouched

    with coord._lock:
        coord._in_flight = False
    # The stall clock was RESET by the handover, not disabled: a fresh timeout is needed...
    assert wd.poll() is False
    clock.advance(1.5)
    assert wd.poll() is True                    # ...and then it does fire


def test_cooldown_blocks_re_entering_the_dead_branch():
    """Without this the policy rebuilds its lost-streak in ~0.7s and dives straight back
    into the branch the watchdog just proved dead."""
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0, cooldown_s=30.0)
    coord.request(True)
    pipe.tick(); wd.poll()
    pipe.tile_dead = True
    clock.advance(1.5)
    assert wd.poll() is True                       # reverted
    assert coord.tiling_blocked is True

    # The policy wants tiling again immediately (target lost every frame).
    assert [coord.note(0.0) for _ in range(5)] == [None] * 5   # note() refuses to even ask
    assert coord.request(True) is False                        # and a direct request is refused
    assert pipe.tiling is False
    assert pipe.switch_calls == [True, False]                  # never re-entered

    clock.advance(31.0)                                        # cooldown expires
    assert coord.tiling_blocked is False
    assert coord.request(True) is True
    assert pipe.switch_calls == [True, False, True]


def test_watchdog_retries_if_the_revert_itself_fails():
    """If whole-frame will not come back either, keep trying — never give up on
    returning to the known-good branch."""
    wd, coord, pipe, clock = make_watchdog(stall_timeout_s=1.0)
    coord.request(True)
    pipe.tick(); wd.poll()
    pipe.tile_dead = True

    calls = []
    def failing_switch(to_tiling):
        calls.append(to_tiling)
        return False                       # revert cannot land
    coord._switch_fn = failing_switch

    clock.advance(1.5)
    assert wd.poll() is False              # reported failure, did not claim success
    assert coord.tiling_on is True         # still stuck on the dead branch, honestly reported

    clock.advance(1.5)
    assert wd.poll() is False
    assert calls == [False, False]         # retried rather than latching


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


# ===========================================================================
# start_on_tiling: the policy must be seeded with the branch the pipeline booted on.
# ===========================================================================
def test_policy_booted_on_tiling_switches_to_whole_after_the_locked_streak():
    """Shipped config: start_on_tiling + locked_frames_to_whole=30. The first decision
    must be about the branch we are ACTUALLY on."""
    policy = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=10,
                                locked_frames_to_whole=30, tiling_on=True)
    pipe = FakePipeline(); pipe.tiling = True
    coord = TilingSwitchCoordinator(policy, pipe.switch)

    assert coord.tiling_on is True
    # 29 confident frames: still tiling.
    assert [coord.note(0.81) for _ in range(29)] == [None] * 29
    assert coord.note(0.81) is False              # the 30th crosses the threshold
    assert coord.request(False) is True
    assert pipe.tiling is False and coord.tiling_on is False


def test_one_unconfident_frame_costs_two_frames_not_the_whole_locked_streak():
    """Through the coordinator, end to end: a single miss must not throw away 29 frames
    of progress, or a merely-flaky detector never gets the rig off the tile branch."""
    policy = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=30,
                                locked_streak_miss_penalty=2, tiling_on=True)
    coord = TilingSwitchCoordinator(policy, lambda t: True)
    for _ in range(29):
        coord.note(0.81)
    assert coord.note(0.1) is None                # one miss: 29 -> 27
    assert [coord.note(0.81) for _ in range(2)] == [None, None]   # 28, 29
    assert coord.note(0.81) is False              # 30 -> back to whole-frame


def test_seeding_the_policy_wrong_would_invert_the_first_decision():
    """If the pipeline boots on tiling but the policy is told tiling_on=False (the bug the
    assert in main() guards), confident frames produce NO switch and lost frames ask to
    switch to the branch we are already on."""
    wrong = TilingSwitchPolicy(switch_conf=0.4, lost_frames_to_tile=10,
                               locked_frames_to_whole=30, tiling_on=False)
    assert [wrong.note(0.81) for _ in range(60)] == [None] * 60    # never comes back to whole
    assert wrong.note(0.0) is None
    assert [wrong.note(0.0) for _ in range(9)][-1] is True         # asks for tiling... again
