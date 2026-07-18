#!/usr/bin/env python3
"""Host tests for re-acquisition target selection and the static-clutter filter.

These cover the controller-side lock logic exercised by the 2026-07-18 FAST-target
eval: `_pick_target_detection` (follow the locked track; at re-acquisition refuse
persistently-static distractors) and `_TrackMotionTracker` (which ids are static).
See docs/experiments/bytetrack-fast-target-2026-07-18.md.
"""
import pytest

from helpers import Detection, Rect
from drone_controller import _pick_target_detection, _TrackMotionTracker

CONF_MIN = 0.4


def det(x, y, conf, tid, wh=0.02):
    return Detection(bbox=Rect.from_xywh(x - wh / 2, y - wh / 2, wh, wh),
                     confidence=conf, track_id=tid)


# --- _pick_target_detection: following an existing lock -----------------------

def test_follows_locked_track_over_higher_confidence_other():
    dets = [det(0.5, 0.5, 0.6, tid=1), det(0.2, 0.2, 0.95, tid=2)]
    picked = _pick_target_detection(dets, CONF_MIN, locked_track_id=1, use_track_lock=True)
    assert picked.track_id == 1


def test_locked_track_absent_returns_none():
    dets = [det(0.2, 0.2, 0.95, tid=2)]
    picked = _pick_target_detection(dets, CONF_MIN, locked_track_id=1, use_track_lock=True)
    assert picked is None


def test_no_lock_picks_highest_confidence():
    dets = [det(0.5, 0.5, 0.6, tid=1), det(0.2, 0.2, 0.9, tid=2)]
    picked = _pick_target_detection(dets, CONF_MIN, locked_track_id=None, use_track_lock=True)
    assert picked.track_id == 2


def test_confidence_floor_filters_pool():
    dets = [det(0.5, 0.5, 0.3, tid=1)]           # below CONF_MIN
    assert _pick_target_detection(dets, CONF_MIN, None, True) is None


# --- re-acquisition static-clutter rejection ---------------------------------

def test_reacq_rejects_static_distractor_prefers_moving():
    # tid 2 is the higher-confidence detection but is known static -> skip it,
    # re-lock the moving tid 1 even though it scores lower.
    dets = [det(0.5, 0.5, 0.7, tid=1), det(0.2, 0.2, 0.95, tid=2)]
    picked = _pick_target_detection(dets, CONF_MIN, locked_track_id=None,
                                    use_track_lock=True, static_track_ids={2})
    assert picked.track_id == 1


def test_reacq_all_static_returns_none():
    # every visible detection is persistently-static clutter -> no target this frame.
    dets = [det(0.5, 0.5, 0.7, tid=1), det(0.2, 0.2, 0.95, tid=2)]
    picked = _pick_target_detection(dets, CONF_MIN, locked_track_id=None,
                                    use_track_lock=True, static_track_ids={1, 2})
    assert picked is None


def test_reacq_new_track_not_in_static_set_is_eligible():
    # a brand-new track id (never judged static) stays acquirable even if others are static.
    dets = [det(0.5, 0.5, 0.8, tid=9)]
    picked = _pick_target_detection(dets, CONF_MIN, locked_track_id=None,
                                    use_track_lock=True, static_track_ids={2, 3})
    assert picked.track_id == 9


def test_static_set_does_not_affect_following_a_lock():
    # the static filter is re-acquisition only; a locked static track keeps being followed.
    dets = [det(0.5, 0.5, 0.7, tid=1)]
    picked = _pick_target_detection(dets, CONF_MIN, locked_track_id=1,
                                    use_track_lock=True, static_track_ids={1})
    assert picked.track_id == 1


# --- _TrackMotionTracker: static vs moving classification --------------------

def test_static_track_classified_static_after_two_obs():
    mt = _TrackMotionTracker(static_speed=0.02, window=5)
    for f in range(4):
        mt.update([det(0.5, 0.5, 0.8, tid=1)], frame_id=f)   # never moves
    assert 1 in mt.static_ids()


def test_moving_track_not_static():
    mt = _TrackMotionTracker(static_speed=0.02, window=5)
    for f in range(4):
        mt.update([det(0.5 + 0.05 * f, 0.5, 0.8, tid=1)], frame_id=f)  # 0.05/frame
    assert 1 not in mt.static_ids()


def test_single_observation_is_not_judged_static():
    mt = _TrackMotionTracker(static_speed=0.02, window=5)
    mt.update([det(0.5, 0.5, 0.8, tid=1)], frame_id=0)
    assert 1 not in mt.static_ids()          # too new to judge -> eligible


def test_stale_tracks_are_pruned():
    mt = _TrackMotionTracker(static_speed=0.02, window=5)
    mt.update([det(0.5, 0.5, 0.8, tid=1)], frame_id=0)
    # advance well past 2*window on a multiple of 30 (prune tick) with a different id
    mt.update([det(0.1, 0.1, 0.8, tid=2)], frame_id=30)
    assert 1 not in mt._pos and 1 not in mt._last_seen


def test_speed_threshold_boundary():
    # a track moving just under the threshold is static; just over is not.
    slow = _TrackMotionTracker(static_speed=0.02, window=5)
    fast = _TrackMotionTracker(static_speed=0.02, window=5)
    for f in range(3):
        slow.update([det(0.5 + 0.01 * f, 0.5, 0.8, tid=1)], frame_id=f)   # 0.01/frame
        fast.update([det(0.5 + 0.03 * f, 0.5, 0.8, tid=1)], frame_id=f)   # 0.03/frame
    assert 1 in slow.static_ids()
    assert 1 not in fast.static_ids()
