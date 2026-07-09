"""Tests for _shared_pts_ns — the buffer.pts baseline shared across camera threads.

app_base imports gi/hailo_apps, so this module skips on a dev host and runs on the Pi.

What is being protected: buffer.pts must be strictly monotonic (splitmuxsink crashes
with "Queued GOP time is negative" otherwise) AND must keep the sensor's even ~fps
cadence. The clamp that guarantees monotonicity is `pts = last + 1`, which is fine for a
one-off wobble but, applied to a SUSTAINED backward shift, pins PTS at 1 ns/frame forever:
splitmuxsink's max-size-time never fires, one RAW_*.mkv grows until the card fills, and the
muxed timestamps become meaningless. Hence the re-baseline.

Nothing on the control path reads buffer.pts (frame ids come from the frame-id ref-meta,
latency from the sensor/unix ref-metas), so this is a recording-robustness invariant.
"""

import pytest

pytest.importorskip("gi", reason="app_base imports gi (Pi only)")
pytest.importorskip("hailo_apps", reason="app_base imports hailo_apps (Pi only)")

import app_base  # noqa: E402


FRAME_NS = 33_333_333          # ~30 fps
BASE = 5_000_000_000_000       # arbitrary CLOCK_BOOTTIME-ish origin


@pytest.fixture(autouse=True)
def _reset_pts_state():
    """_shared_pts_ns keeps process-wide state; isolate every test."""
    app_base._pts_baseline_ns = None
    app_base._pts_last_ns = -1
    yield
    app_base._pts_baseline_ns = None
    app_base._pts_last_ns = -1


def test_even_cadence_is_preserved_and_starts_at_zero():
    pts = [app_base._shared_pts_ns(BASE + k * FRAME_NS) for k in range(5)]
    assert pts[0] == 0
    assert [b - a for a, b in zip(pts, pts[1:])] == [FRAME_NS] * 4


def test_small_backward_wobble_is_clamped_then_self_heals():
    app_base._shared_pts_ns(BASE)
    app_base._shared_pts_ns(BASE + FRAME_NS)

    # 1 ms backwards: jitter, not a clock change. Clamp keeps it strictly monotonic.
    assert app_base._shared_pts_ns(BASE + FRAME_NS - 1_000_000) == FRAME_NS + 1
    # As soon as the source passes the high-water mark again, cadence is exact.
    assert app_base._shared_pts_ns(BASE + 2 * FRAME_NS) == 2 * FRAME_NS


def test_sustained_backward_jump_rebaselines_instead_of_stalling_at_1ns():
    """The regression this guards: a 1 ns/frame PTS creep that never recovers."""
    app_base._shared_pts_ns(BASE)
    app_base._shared_pts_ns(BASE + FRAME_NS)

    jumped = BASE + FRAME_NS - 10_000_000_000      # 10 s back == a different clock
    first = app_base._shared_pts_ns(jumped)
    assert first == FRAME_NS + 1                   # one frame of cadence is sacrificed

    after = [app_base._shared_pts_ns(jumped + k * FRAME_NS) for k in (1, 2, 3)]
    deltas = [b - a for a, b in zip([first] + after, after)]
    assert deltas == [FRAME_NS] * 3, (
        f"expected the new clock's cadence after re-baselining, got {deltas}; "
        "1ns deltas mean splitmuxsink will never cut a segment")


def test_pts_is_strictly_monotonic_under_adversarial_sources():
    """Whatever the source does, PTS must never step back or repeat."""
    sources = [BASE, BASE + FRAME_NS, BASE - 50_000_000_000, BASE + 2 * FRAME_NS,
               BASE - 1, BASE + 3 * FRAME_NS, BASE + 3 * FRAME_NS]
    pts = [app_base._shared_pts_ns(s) for s in sources]
    assert all(b > a for a, b in zip(pts, pts[1:])), pts
