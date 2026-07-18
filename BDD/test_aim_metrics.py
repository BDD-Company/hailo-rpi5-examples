"""Tests for the Phase-0 aim-quality metrics (jitter + filter-added lag).

These are the numbers every noise-reduction phase is A/B'd against
(see BDD/experiments/2026-07-12-target-estimate-noise-reduction-design.md §3.1).

Kept deliberately free of the replay-harness import chain (flight_debugger →
PyQt6): the metric core is pure math over an aim-point series, so it must be
unit-testable without a GUI stack on the machine.
"""

import math

import aim_metrics as am


# ── rms_second_difference: the jitter kernel ────────────────────────────
# Jitter = RMS of the second difference of the aim point. Blind to a constant
# offset (aim bias) AND a constant velocity (legit estimator output), maximally
# sensitive to frame-to-frame shake.

def test_constant_offset_has_zero_jitter():
    # A perfectly still aim point: no shake.
    assert am.rms_second_difference([5.0, 5.0, 5.0, 5.0, 5.0]) == 0.0


def test_constant_velocity_has_zero_jitter():
    # A steadily moving aim point (constant slope) is legit estimator output,
    # not shake — second difference is identically zero.
    assert am.rms_second_difference([0.0, 2.0, 4.0, 6.0, 8.0]) == 0.0


def test_constant_acceleration_second_difference_is_the_accel():
    # x[n] = n^2 → second difference is a constant 2.0 everywhere → RMS 2.0.
    assert math.isclose(am.rms_second_difference([0.0, 1.0, 4.0, 9.0, 16.0]), 2.0)


def test_alternating_shake_matches_hand_computation():
    # [0,1,0,1,0]: second diffs are (0-2+0)=-2? compute against the definition.
    x = [0.0, 1.0, 0.0, 1.0, 0.0]
    # d2[n] = x[n] - 2 x[n-1] + x[n-2] for n=2..4:
    #  n2: 0 - 2*1 + 0 = -2 ; n3: 1 - 0 + 1 = 2 ; n4: 0 - 2 + 0 = -2
    expected = math.sqrt((4 + 4 + 4) / 3)
    assert math.isclose(am.rms_second_difference(x), expected)


def test_fewer_than_three_points_is_nan():
    # Not enough data to form a single second difference — say so, don't fake a 0.
    assert math.isnan(am.rms_second_difference([1.0, 2.0]))
    assert math.isnan(am.rms_second_difference([1.0]))
    assert math.isnan(am.rms_second_difference([]))


# ── cross_correlation_lag: filter-added lag, no ground truth ─────────────
# The delay (in samples) between a reference aim series and a filtered one.
# Positive => the filtered signal LAGS the reference by that many samples.

def test_identical_series_have_zero_lag():
    ref = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 0.0, 2.0]
    assert am.cross_correlation_lag(ref, ref) == 0


def test_delayed_signal_reports_positive_lag():
    ref = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 0.0, 2.0]
    # signal is ref shifted 2 samples later (lags by 2)
    signal = [0.0, 0.0] + ref[:-2]
    assert am.cross_correlation_lag(ref, signal) == 2


def test_advanced_signal_reports_negative_lag():
    ref = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 0.0, 2.0]
    signal = ref[3:] + [0.0, 0.0, 0.0]  # leads by 3
    assert am.cross_correlation_lag(ref, signal) == -3


# ── aim_series_from_commands: pull the aim signal out of a replay ───────
# MockDroneMover.commands is the per-frame record. Only zenith commands carry
# an aim direction (roll, pitch); idle/standstill/ned/abort do not.

def test_extracts_only_zenith_aim_rows_in_order():
    commands = [
        ("idle", 10),
        ("move_to_target_zenith", 11, 1.5, -2.0, 0.4),
        ("standstill", 12, 0.3),
        ("move_to_target_zenith", 13, 1.7, -1.8, 0.4),
        ("ABORT", 14),
    ]
    series = am.aim_series_from_commands(commands)
    assert series == [(11, 1.5, -2.0), (13, 1.7, -1.8)]


def test_empty_commands_give_empty_series():
    assert am.aim_series_from_commands([]) == []


# ── jitter_by_axis: report roll and pitch jitter of one run ─────────────

def test_jitter_by_axis_reports_each_axis_independently():
    # roll shakes, pitch is a clean ramp (zero jitter).
    series = [
        (1, 0.0, 0.0),
        (2, 1.0, 1.0),
        (3, 0.0, 2.0),
        (4, 1.0, 3.0),
        (5, 0.0, 4.0),
    ]
    out = am.jitter_by_axis(series)
    assert math.isclose(out["roll"], math.sqrt((4 + 4 + 4) / 3))
    assert out["pitch"] == 0.0


def test_jitter_by_axis_nan_when_too_short():
    out = am.jitter_by_axis([(1, 0.0, 0.0), (2, 1.0, 1.0)])
    assert math.isnan(out["roll"]) and math.isnan(out["pitch"])


# ── filter_added_lag: sample delay a filtered run adds vs a reference ────

def test_filter_added_lag_zero_for_identical_runs():
    series = [(i, float(i % 3), float((2 * i) % 5)) for i in range(12)]
    out = am.filter_added_lag(series, series)
    assert out["roll"] == 0 and out["pitch"] == 0


def test_filter_added_lag_detects_a_two_frame_delay():
    # Same frames in both runs; the filtered roll is the reference roll delayed 2.
    ref = [(i, float([0, 1, 3, 2, 5, 4, 6, 1, 0, 2][i]), 0.0) for i in range(10)]
    delayed_roll = [0.0, 0.0] + [r[1] for r in ref][:-2]
    filt = [(i, delayed_roll[i], 0.0) for i in range(10)]
    out = am.filter_added_lag(ref, filt)
    assert out["roll"] == 2


def test_filter_added_lag_aligns_on_common_frame_ids():
    # The filtered run skipped frame 3 (no aim there); lag must align on the
    # frames both runs emitted, not blindly by position.
    ref = [(1, 0.0, 0.0), (2, 1.0, 0.0), (3, 2.0, 0.0), (4, 3.0, 0.0), (5, 4.0, 0.0)]
    filt = [(1, 0.0, 0.0), (2, 1.0, 0.0), (4, 3.0, 0.0), (5, 4.0, 0.0)]
    out = am.filter_added_lag(ref, filt)
    assert out["roll"] == 0


# ── summarize_run / summarize_ab: what the harness main() prints ────────
# These take raw MockDroneMover.commands (what the replay produces) so the
# harness glue is a thin I/O shell over tested logic.

def _zenith(frame_id, roll, pitch, thrust=0.4):
    return ("move_to_target_zenith", frame_id, roll, pitch, thrust)


def test_summarize_run_reports_jitter_and_aim_frame_count():
    commands = [
        ("idle", 0),
        _zenith(1, 0.0, 0.0),
        _zenith(2, 1.0, 1.0),
        _zenith(3, 0.0, 2.0),
        _zenith(4, 1.0, 3.0),
        _zenith(5, 0.0, 4.0),
    ]
    rep = am.summarize_run(commands)
    assert rep["aim_frames"] == 5
    assert math.isclose(rep["jitter"]["roll"], math.sqrt((4 + 4 + 4) / 3))
    assert rep["jitter"]["pitch"] == 0.0


def test_summarize_ab_pairs_jitter_and_reports_added_lag():
    ref_rolls = [0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 1.0, 0.0, 2.0]
    ref = [_zenith(i, ref_rolls[i], 0.0) for i in range(10)]
    delayed = [0.0, 0.0] + ref_rolls[:-2]
    filt = [_zenith(i, delayed[i], 0.0) for i in range(10)]

    rep = am.summarize_ab(ref, filt)
    assert rep["lag"]["roll"] == 2
    assert "jitter" in rep["reference"] and "jitter" in rep["filtered"]
