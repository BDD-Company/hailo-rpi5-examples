"""Phase-0 aim-quality metrics for the target noise-reduction work.

The whole noise-reduction effort is an A/B against two numbers, reported as a
pair (see BDD/experiments/2026-07-12-target-estimate-noise-reduction-design.md):

- **Jitter** — RMS of the *second difference* of the final aim point. Blind to a
  constant aim offset and a constant aim velocity (the two things the estimator
  legitimately produces), maximally sensitive to frame-to-frame shake.
- **Filter-added lag** — the sample delay between a reference aim series and a
  filtered one, by cross-correlation. Measures exactly "did this change delay its
  own input", needing NO ground truth.

Pure math on an aim-point series. Deliberately imports nothing from the replay
harness (flight_debugger → PyQt6) so it stays unit-testable without a GUI stack.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def rms_second_difference(values: Sequence[float]) -> float:
    """RMS of the second difference of *values*.

    ``d2[n] = values[n] - 2*values[n-1] + values[n-2]``; result is
    ``sqrt(mean(d2**2))``. Zero for a constant or constant-velocity series.
    Returns NaN when there are fewer than 3 points (no second difference exists) —
    an honest "not enough data", never a fake 0.
    """
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return float("nan")
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return float(np.sqrt(np.mean(d2 * d2)))


def cross_correlation_lag(reference: Sequence[float],
                          signal: Sequence[float],
                          max_lag: int | None = None) -> int:
    """Integer sample lag of *signal* relative to *reference*.

    Returns the shift ``k`` maximising the cross-correlation of ``signal`` against
    ``reference``: a positive ``k`` means ``signal`` LAGS ``reference`` by ``k``
    samples (its features arrive later), negative means it leads. Zero for
    identical series. This is the filter-added-lag metric.

    ``max_lag`` bounds the search to ``[-max_lag, +max_lag]``; default is
    ``len - 1``.
    """
    r = np.asarray(reference, dtype=float)
    s = np.asarray(signal, dtype=float)
    n = min(r.size, s.size)
    if n == 0:
        return 0
    r = r[:n] - r[:n].mean()
    s = s[:n] - s[:n].mean()
    limit = n - 1 if max_lag is None else min(max_lag, n - 1)

    best_lag = 0
    best_corr = -np.inf
    for k in range(-limit, limit + 1):
        # k>0: signal delayed by k — compare signal[k:] against reference[:n-k].
        if k >= 0:
            a, b = r[: n - k], s[k:] if k else s
        else:
            a, b = r[-k:], s[: n + k]
        if a.size == 0:
            continue
        corr = float(np.dot(a, b))
        if corr > best_corr:
            best_corr, best_lag = corr, k
    return best_lag


def aim_series_from_commands(commands: Sequence[tuple]) -> list[tuple[int, float, float]]:
    """Extract the aim direction time series from ``MockDroneMover.commands``.

    Only ``move_to_target_zenith`` commands carry an aim direction; idle,
    standstill, ned and abort do not and are skipped. Each kept row is
    ``(frame_id, roll_degree, pitch_degree)`` in issue order.
    """
    series: list[tuple[int, float, float]] = []
    for cmd in commands:
        if cmd and cmd[0] == "move_to_target_zenith":
            _, frame_id, roll, pitch, _thrust = cmd
            series.append((frame_id, float(roll), float(pitch)))
    return series


def jitter_by_axis(series: Sequence[tuple[int, float, float]]) -> dict:
    """Jitter (RMS second difference) of an aim series, per axis.

    ``series`` is ``(frame_id, roll, pitch)`` rows as returned by
    :func:`aim_series_from_commands`. Returns ``{"roll": ..., "pitch": ...}``.
    """
    roll = [row[1] for row in series]
    pitch = [row[2] for row in series]
    return {
        "roll": rms_second_difference(roll),
        "pitch": rms_second_difference(pitch),
    }


def filter_added_lag(reference: Sequence[tuple[int, float, float]],
                     filtered: Sequence[tuple[int, float, float]],
                     max_lag: int | None = None) -> dict:
    """Per-axis filter-added lag between two aim series, aligned on frame id.

    Both series are ``(frame_id, roll, pitch)`` rows. Only frames present in
    BOTH runs are compared (a filter can change which frames emit an aim), taken
    in frame order, then cross-correlated per axis. Returns
    ``{"roll": lag, "pitch": lag}`` in samples of the common subsequence;
    positive means *filtered* lags *reference*.
    """
    by_id_ref = {row[0]: row for row in reference}
    common_ids = sorted(fid for fid, *_ in filtered if fid in by_id_ref)
    by_id_filt = {row[0]: row for row in filtered}

    ref_roll = [by_id_ref[i][1] for i in common_ids]
    filt_roll = [by_id_filt[i][1] for i in common_ids]
    ref_pitch = [by_id_ref[i][2] for i in common_ids]
    filt_pitch = [by_id_filt[i][2] for i in common_ids]

    return {
        "roll": cross_correlation_lag(ref_roll, filt_roll, max_lag),
        "pitch": cross_correlation_lag(ref_pitch, filt_pitch, max_lag),
    }


def summarize_run(commands: Sequence[tuple]) -> dict:
    """Jitter summary of one replay, from raw ``MockDroneMover.commands``."""
    series = aim_series_from_commands(commands)
    return {"aim_frames": len(series), "jitter": jitter_by_axis(series)}


def summarize_ab(reference_commands: Sequence[tuple],
                 filtered_commands: Sequence[tuple]) -> dict:
    """A/B summary of two replays: jitter of each run + the filter-added lag.

    The two arguments are raw ``MockDroneMover.commands`` from a baseline run and
    a filtered run of the SAME log. Reports both jitters (is it quieter?) and the
    lag the filter added (at what cost?) — the design's non-negotiable pair.
    """
    ref_series = aim_series_from_commands(reference_commands)
    filt_series = aim_series_from_commands(filtered_commands)
    return {
        "reference": {"aim_frames": len(ref_series),
                      "jitter": jitter_by_axis(ref_series)},
        "filtered": {"aim_frames": len(filt_series),
                     "jitter": jitter_by_axis(filt_series)},
        "lag": filter_added_lag(ref_series, filt_series),
    }
