import math
from guidance.phased import select_phase, lead_pursuit_velocity_ned


def test_select_phase_bands():
    assert select_phase(50, 35, 12) == "FAR"
    assert select_phase(20, 35, 12) == "MID"
    assert select_phase(8,  35, 12) == "CLOSE"
    assert select_phase(35, 35, 12) == "FAR"     # boundary inclusive -> FAR
    assert select_phase(12, 35, 12) == "MID"     # boundary inclusive -> MID


def test_far_command_points_at_lead_point_at_vmax():
    # target due north, moving +east -> lead point is north & a bit east; cmd at v_max.
    v = lead_pursuit_velocity_ned(0, 0, 0, 40, 0, 0, 0, 10, 0,
                                  v_max=25, v_drone=25, t_max=3.0, vz_max=10, lead_max_alt_m=120)
    assert v.north_m_s > 0 and v.east_m_s > 0
    assert math.isclose(math.sqrt(v.north_m_s**2+v.east_m_s**2+v.down_m_s**2), 25.0, rel_tol=1e-3)


def test_vz_capped():
    v = lead_pursuit_velocity_ned(0, 0, 0, 0, 0, -200, 0, 0, 0,
                                  v_max=25, v_drone=25, t_max=3.0, vz_max=8, lead_max_alt_m=300)
    assert v.down_m_s >= -8.0 - 1e-6


def test_alt_guard_blocks_climb_above_cap():
    # drone already above the cap (down=-130 < -120) and target higher -> no further climb.
    v = lead_pursuit_velocity_ned(0, 0, -130, 0, 0, -200, 0, 0, 0,
                                  v_max=25, v_drone=25, t_max=3.0, vz_max=10, lead_max_alt_m=120)
    assert v.down_m_s >= 0.0    # climb (negative down) suppressed
