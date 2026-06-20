import math
from guidance.pronav import pronav_velocity_ned


def test_closes_along_los_for_stationary_target():
    # target due north at 10 m, not moving -> command points due north at v_close.
    v = pronav_velocity_ned(0, 0, 0, 10, 0, 0, 0, 0, 0, v_close=15, n=1.0, v_max=25)
    assert v.north_m_s > 0 and abs(v.east_m_s) < 1e-6
    assert math.isclose(math.hypot(v.north_m_s, v.east_m_s), 15.0, rel_tol=1e-3)


def test_vz_capped():
    # target high above -> down command (climb) capped at -vz_max.
    v = pronav_velocity_ned(0, 0, 0, 0, 0, -100, 0, 0, 0, v_close=20, n=1.0, v_max=30, vz_max=8.0)
    assert v.down_m_s >= -8.0 - 1e-6


def test_v_max_capped():
    v = pronav_velocity_ned(0, 0, 0, 5, 0, 0, 0, 30, 0, v_close=25, n=2.0, v_max=20)
    assert math.sqrt(v.north_m_s**2 + v.east_m_s**2 + v.down_m_s**2) <= 20.0 + 1e-6
