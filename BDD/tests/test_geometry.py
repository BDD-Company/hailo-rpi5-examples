import math
from guidance.geometry import NedVel, lead_intercept_point, closest_approach_point


def test_closest_approach_foot_is_perpendicular():
    # target at (0,10,0) moving +north at 5 m/s; drone at origin.
    # the foot of the perpendicular onto the path is the target's current point.
    n, e, d = closest_approach_point(0, 0, 0, 0, 10, 0, 5, 0, 0)
    assert math.isclose(n, 0.0, abs_tol=1e-6)
    assert math.isclose(e, 10.0, abs_tol=1e-6)


def test_lead_point_leads_along_target_velocity():
    # target ahead, moving +east; the intercept point is shifted +east of it.
    n, e, d = lead_intercept_point(0, 0, 0, 20, 0, 0, 0, 8, 0, v_drone=20, t_max=4.0)
    assert e > 0.0          # led toward where the target is going
    assert math.isclose(n, 20.0, abs_tol=1e-6)


def test_lead_point_capped_by_t_max():
    # uncatchable: target far and fast -> lead bounded by t_max, not infinite.
    n, e, d = lead_intercept_point(0, 0, 0, 200, 0, 0, 0, 50, 0, v_drone=10, t_max=2.0)
    assert e <= 50 * 2.0 + 1e-6


def test_nedvel_fields():
    v = NedVel(1.0, 2.0, 3.0, 90.0)
    assert (v.north_m_s, v.east_m_s, v.down_m_s, v.yaw_deg) == (1.0, 2.0, 3.0, 90.0)
