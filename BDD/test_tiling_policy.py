"""Unit tests for the size-driven tiling ladder policy (host-runnable; pure — no
hailo/GStreamer). Pins the tier decision that user_app_callback_class.note_tiling
delegates to, plus build_ladder's validation and empty-ladder handling.

side = max(bbox_w, bbox_h) of the primary tracked target (0..1); None = lost.
now_s is an injected monotonic timestamp so loss timers test without sleeping.
"""
import pytest
from tiling_policy import TilingLadderPolicy, LadderTier, build_ladder


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
