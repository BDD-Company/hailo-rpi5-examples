import math
from guidance.lead import lead_setpoint_ned


def test_uses_closest_approach_when_velocity_trustworthy():
    # |v| = 10 (in [5,40]) -> horizontal = closest-approach foot.
    n, e, d = lead_setpoint_ned(0, 0, 0, 0, 20, 0, 10, 0, 0, 0, 0,
                                lead_max_lat=60, lead_max_alt_m=70, lead_alt_offset=0.0)
    assert math.isclose(n, 0.0, abs_tol=1e-6)   # foot onto a +north path from drone at origin
    assert math.isclose(e, 20.0, abs_tol=1e-6)


def test_holds_horizontal_when_velocity_untrustworthy():
    # |v| = 1 (<5) -> hold drone's current horizontal (10, 5), just set altitude.
    n, e, d = lead_setpoint_ned(10, 5, -3, 0, 20, -40, 1, 0, 0, 0, 0,
                                lead_max_lat=60, lead_max_alt_m=70, lead_alt_offset=0.0)
    assert (n, e) == (10, 5)


def test_lateral_clamped_to_home():
    # huge foot, clamp to home +/- lead_max_lat.
    n, e, d = lead_setpoint_ned(0, 0, 0, 0, 1000, 0, 10, 0, 0, 0, 0,
                                lead_max_lat=15, lead_max_alt_m=70, lead_alt_offset=0.0)
    assert e <= 15 + 1e-6


def test_altitude_capped():
    # target very high (down=-200) -> setpoint down capped at -lead_max_alt_m.
    n, e, d = lead_setpoint_ned(0, 0, 0, 0, 0, -200, 10, 0, 0, 0, 0,
                                lead_max_lat=60, lead_max_alt_m=70, lead_alt_offset=0.0)
    assert math.isclose(d, -70.0, abs_tol=1e-6)
