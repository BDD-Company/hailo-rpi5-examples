"""Unit tests for the detection-state tiling switch policy (host-runnable; the
policy is pure, so no hailo/GStreamer needed — that's why it was split out of the
un-importable app.py). These pin the whole<->tile switch decision that
user_app_callback_class.note_detection delegates to.

Contract mirrored from app.py: each frame's best target confidence is fed to
``note`` on the streaming thread; when it returns a branch, app.py fires the
(blocking) switch on a worker thread and calls ``committed`` once it lands.
"""
from tiling_policy import TilingSwitchPolicy


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
