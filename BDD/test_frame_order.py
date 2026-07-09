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
