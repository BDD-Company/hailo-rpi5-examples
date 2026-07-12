#!/usr/bin/env python3
"""Host tests for the pure control-math helpers in drone_controller.

No drone, no hardware, no GStreamer — these exercise the P-coefficient math and
the telemetry parsing directly.
"""

import math

import pytest

from config import Config
from helpers import XY
from drone_controller import speed_from_telemetry, speed_reduced_p

SpeedReduction = Config.PDCoeff.SpeedReduction


def _cfg(start_speed_ms=10.0, coeff=0.9, speed_step_ms=1.0) -> SpeedReduction:
    return SpeedReduction(
        start_speed_ms=start_speed_ms,
        coeff=coeff,
        speed_step_ms=speed_step_ms,
    )


def _telemetry(x_m_s, y_m_s, z_m_s) -> dict:
    return {"odometry": {"velocity_body": {"x_m_s": x_m_s, "y_m_s": y_m_s, "z_m_s": z_m_s}}}


# --- speed_reduced_p: the worked examples from TODO.md -----------------------

def test_todo_example_step_1_ms():
    # rs=10, ds=20, rc=0.9, rss=1, P=10  ->  10 * 0.9**10 = 3.4867...
    p = speed_reduced_p(XY(10.0, 10.0), 20.0, _cfg(speed_step_ms=1.0))
    assert p.x == pytest.approx(3.4867844, rel=1e-6)
    assert p.y == pytest.approx(3.4867844, rel=1e-6)


def test_todo_example_step_2_ms():
    # same, but rss=2  ->  10 * 0.9**5 = 5.9049
    p = speed_reduced_p(XY(10.0, 10.0), 20.0, _cfg(speed_step_ms=2.0))
    assert p.x == pytest.approx(5.9049, rel=1e-6)
    assert p.y == pytest.approx(5.9049, rel=1e-6)


# --- speed_reduced_p: when NOT to reduce -------------------------------------

def test_below_start_speed_leaves_p_untouched():
    p = speed_reduced_p(XY(10.0, 4.0), 9.99, _cfg(start_speed_ms=10.0))
    assert (p.x, p.y) == (10.0, 4.0)


def test_at_exactly_start_speed_leaves_p_untouched():
    p = speed_reduced_p(XY(10.0, 4.0), 10.0, _cfg(start_speed_ms=10.0))
    assert (p.x, p.y) == (10.0, 4.0)


def test_disabled_when_config_section_absent():
    # An absent `speed_reduction` section (None) means the feature is off.
    p = speed_reduced_p(XY(10.0, 4.0), 100.0, None)
    assert (p.x, p.y) == (10.0, 4.0)


def test_coeff_of_one_is_a_no_op_at_any_speed():
    p = speed_reduced_p(XY(10.0, 4.0), 90.0, _cfg(coeff=1.0))
    assert p.x == pytest.approx(10.0)
    assert p.y == pytest.approx(4.0)


# --- speed_reduced_p: shape of the reduction ---------------------------------

def test_reduction_is_monotonic_in_speed():
    cfg = _cfg()
    ps = [speed_reduced_p(XY(10.0, 10.0), speed, cfg).x for speed in (10, 15, 20, 30, 40)]
    assert ps == sorted(ps, reverse=True), f"P must fall as speed rises, got {ps}"


def test_both_axes_scaled_by_the_same_factor():
    # Speed is a scalar property of the airframe, so the per-axis P ratio survives.
    p = speed_reduced_p(XY(8.0, 2.0), 20.0, _cfg())
    assert p.x / p.y == pytest.approx(4.0)


def test_zero_coeff_collapses_p_above_threshold():
    p = speed_reduced_p(XY(10.0, 10.0), 20.0, _cfg(coeff=0.0))
    assert p.x == pytest.approx(0.0)


def test_does_not_mutate_the_input_xy():
    original = XY(10.0, 10.0)
    speed_reduced_p(original, 20.0, _cfg())
    assert (original.x, original.y) == (10.0, 10.0)


# --- speed_from_telemetry ----------------------------------------------------

def test_speed_is_the_3d_magnitude_of_velocity_body():
    assert speed_from_telemetry(_telemetry(3.0, 4.0, 12.0)) == pytest.approx(13.0)


def test_speed_includes_the_vertical_component():
    # A pure dive is fast even with zero horizontal velocity.
    assert speed_from_telemetry(_telemetry(0.0, 0.0, -20.0)) == pytest.approx(20.0)


def test_speed_is_none_when_odometry_missing():
    assert speed_from_telemetry({}) is None
    assert speed_from_telemetry({"odometry": None}) is None


def test_speed_is_none_when_velocity_body_missing():
    assert speed_from_telemetry({"odometry": {}}) is None


def test_speed_is_none_when_velocity_is_malformed():
    # A partial or non-numeric velocity must degrade, never raise: a telemetry
    # glitch may not be allowed to kill the control loop.
    assert speed_from_telemetry({"odometry": {"velocity_body": {"x_m_s": 1.0}}}) is None
    assert speed_from_telemetry({"odometry": {"velocity_body": {
        "x_m_s": 1.0, "y_m_s": None, "z_m_s": 2.0}}}) is None
    assert speed_from_telemetry({"odometry": {"velocity_body": {
        "x_m_s": 1.0, "y_m_s": "nope", "z_m_s": 2.0}}}) is None


def test_speed_is_none_when_telemetry_is_none():
    assert speed_from_telemetry(None) is None


def test_speed_is_none_on_nan_velocity():
    # NaN would silently poison P (nan ** k = nan) — reject it like any other
    # unusable reading so the caller falls back to the last known good speed.
    assert speed_from_telemetry(_telemetry(float("nan"), 0.0, 0.0)) is None
    assert speed_from_telemetry(_telemetry(float("inf"), 0.0, 0.0)) is None
