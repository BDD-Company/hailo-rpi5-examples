#!/usr/bin/env python3

from frame_order import FrameOrderGuard


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
    # Regression: file input derives frame ids from buffer.offset, which restarts
    # at 0 on the loop rewind. Without a reset the guard's high-water mark is still
    # at the end of the previous pass, so it rejects every frame of every later
    # pass -- the app stays alive but stops seeing anything at all.
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


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
