#!/usr/bin/env python3

from frame_order import FrameOrderGuard, resolve_frame_id


def _accepts(guard, camera_id, ids):
    """Return the subset of `ids` the guard accepts, in call order."""
    return [fid for fid in ids if guard.accept(camera_id, fid)]


def test_monotonic_sequence_all_accepted():
    g = FrameOrderGuard()
    assert _accepts(g, 0, [0, 1, 2, 3, 4]) == [0, 1, 2, 3, 4]


def test_forward_skips_are_accepted():
    # Dropping frames (skips) is allowed; only ORDER matters.
    g = FrameOrderGuard()
    assert _accepts(g, 0, [0, 2, 5, 6, 30]) == [0, 2, 5, 6, 30]


def test_duplicate_is_rejected():
    g = FrameOrderGuard()
    assert g.accept(0, 7) is True
    assert g.accept(0, 7) is False  # exact duplicate


def test_reorder_is_rejected():
    g = FrameOrderGuard()
    assert _accepts(g, 0, [0, 2, 1, 3]) == [0, 2, 3]  # the stale "1" is dropped


def test_recovers_after_a_reorder():
    # A single stale frame must not wedge the guard: the next in-order frame passes.
    g = FrameOrderGuard()
    assert g.accept(0, 5) is True
    assert g.accept(0, 4) is False  # stale
    assert g.accept(0, 6) is True   # back in order


def test_cameras_are_independent():
    g = FrameOrderGuard()
    # cam0 far ahead must not mask cam1 starting from its own low ids.
    assert g.accept(0, 100) is True
    assert g.accept(1, 0) is True
    assert g.accept(1, 1) is True
    # ...and a stale within cam1 is still caught.
    assert g.accept(1, 1) is False


def test_camera_switch_back_and_forth():
    # Each camera keeps its own high-water mark across interleaving.
    g = FrameOrderGuard()
    assert g.accept(0, 10) is True
    assert g.accept(1, 3) is True
    assert g.accept(0, 11) is True   # cam0 resumes above its own last (10)
    assert g.accept(1, 2) is False   # cam1 stale (< its own last 3)
    assert g.accept(1, 4) is True


def test_rewind_without_reset_rejects_the_whole_next_pass():
    # Pins WHY reset() exists. If a rewind restarts frame ids from 0, the guard's
    # high-water mark is still at the end of the previous pass, so it rejects every
    # frame of every later pass -- the app stays alive but stops seeing anything.
    # The file path's buffer.offset does not restart (measured), but its fallback
    # buffer.pts does; this is the failure that fallback would produce.
    g = FrameOrderGuard()
    assert _accepts(g, 0, [0, 1, 2, 3]) == [0, 1, 2, 3]
    assert _accepts(g, 0, [0, 1, 2, 3]) == []  # the bug, pinned


def test_reset_lets_a_rewound_stream_replay_the_same_ids():
    # A rewind is a deliberate stream discontinuity, not a reorder: the flush seek
    # drops in-flight buffers, so ids legitimately restart from 0.
    g = FrameOrderGuard()
    _accepts(g, 0, [0, 1, 2, 3])
    g.reset()
    assert _accepts(g, 0, [0, 1, 2, 3]) == [0, 1, 2, 3]


def test_reset_clears_every_camera():
    g = FrameOrderGuard()
    g.accept(0, 10)
    g.accept(1, 20)
    g.reset()
    assert g.last(0) is None
    assert g.last(1) is None


def test_order_is_still_enforced_after_a_reset():
    # Reset must not weaken the invariant within the new stream generation.
    g = FrameOrderGuard()
    g.accept(0, 100)
    g.reset()
    assert g.accept(0, 5) is True
    assert g.accept(0, 4) is False  # stale within the new pass, still rejected


def test_last_reports_high_water_mark():
    g = FrameOrderGuard()
    assert g.last(0) is None
    g.accept(0, 3)
    g.accept(0, 9)
    g.accept(0, 5)  # rejected, must not lower the mark
    assert g.last(0) == 9


# ---------------------------------------------------------------------------
# resolve_frame_id: pick the per-frame id fed to BOTH the guard AND ByteTracker.
# It must be branch-INDEPENDENT (so a tiling branch switch can't blind the guard) and
# increment ~1 per frame (ByteTracker's track_buffer expiry counts frame_id deltas).
# The only buffer property that is both is buffer.pts -- which survives the tile
# aggregator (unlike custom metas) and is the same source frame on every rung (unlike
# the per-branch buffer.offset) -- converted to a frame index via the frame duration.
# Raw pts (nanoseconds) would jump ~3e7/frame and break ByteTracker, hence the divide.
# ---------------------------------------------------------------------------

_DUR = 33_333_333  # ns/frame at 30 fps


def test_producer_frame_meta_wins_over_everything():
    assert resolve_frame_id(500, 33_333_333, 7, _DUR) == 500


def test_pts_becomes_a_one_per_frame_index():
    assert resolve_frame_id(None, 0, None, _DUR) == 0
    assert resolve_frame_id(None, 1 * _DUR, None, _DUR) == 1
    assert resolve_frame_id(None, 2 * _DUR, None, _DUR) == 2
    # tolerates real-CFR pts jitter that isn't an exact multiple of the duration
    assert resolve_frame_id(None, 3 * _DUR + 1000, None, _DUR) == 3


def test_pts_index_is_branch_independent():
    # Same source frame (same pts) resolves to the SAME id regardless of which branch
    # produced the buffer -- that is the whole point.
    assert resolve_frame_id(None, 7 * _DUR, None, _DUR) == resolve_frame_id(None, 7 * _DUR, 999, _DUR)


def test_pts_index_across_a_branch_switch_is_never_rejected_by_the_guard():
    # Frames 0..2 on the whole-frame rung, then a switch and frames 3..5 on a tiled rung.
    # Because the id comes from pts (branch-independent), the sequence stays monotonic and
    # the guard accepts every frame -- the fix for the 53%-blinding bug.
    g = FrameOrderGuard()
    ids = [resolve_frame_id(None, n * _DUR, None, _DUR) for n in range(6)]
    assert all(g.accept(0, fid) for fid in ids)
    assert ids == [0, 1, 2, 3, 4, 5]


def test_falls_back_to_offset_when_pts_cannot_be_indexed():
    # No producer meta and no known frame duration -> we cannot turn pts into a ~1/frame
    # index without breaking ByteTracker, so use the raw offset rather than raw pts.
    assert resolve_frame_id(None, 33_333_333, 42, None) == 42


def test_falls_back_to_offset_when_no_meta_and_no_pts():
    assert resolve_frame_id(None, None, 42, _DUR) == 42


def test_returns_none_when_nothing_identifies_the_frame():
    # Caller substitutes a wallclock last-resort id.
    assert resolve_frame_id(None, None, None, _DUR) is None


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
