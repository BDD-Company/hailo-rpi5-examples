"""Unit tests for the detection-state tiling switch policy (host-runnable; the
policy is pure, so no hailo/GStreamer needed — that's why it was split out of the
un-importable app.py). These pin the whole<->tile switch decision that
user_app_callback_class.note_detection delegates to.

Contract mirrored from app.py: each frame's best target confidence is fed to
``note`` on the streaming thread; when it returns a branch, app.py fires the
(blocking) switch on a worker thread and calls ``committed`` once it lands.

The second half of this file covers TilingSwitchCoordinator, which serializes every
switch (policy worker + --test-switch-s harness) through one entry point.
"""
import threading
import time

from tiling_policy import TilingSwitchPolicy, TilingSwitchCoordinator


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


def test_lock_streak_needs_consecutive_frames():
    p = TilingSwitchPolicy(switch_conf=0.4, locked_frames_to_whole=5, tiling_on=True)
    # 4 confident, then a lost frame resets the lock streak.
    assert feed(p, [0.9] * 4 + [0.1]) == [None] * 5
    assert feed(p, [0.9] * 4) == [None] * 4
    assert p.note(0.9) is False


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
