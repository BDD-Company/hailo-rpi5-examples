import math
from guidance.visual import NedVelCmd, phase_from_size, visual_velocity_ned, FAR, MID, NEAR


def test_phase_from_size_bands():
    assert phase_from_size(0.01, 0.06, 0.20) == FAR
    assert phase_from_size(0.10, 0.06, 0.20) == MID
    assert phase_from_size(0.30, 0.06, 0.20) == NEAR


def test_far_chases_along_los():
    los = (0.0, 0.0, -1.0)  # straight up
    cmd = visual_velocity_ned(los, None, 0.0, 0.01, v_far=12, v_close=14, n_gain=8,
                              term_gain=16, mid_thresh=0.06, near_thresh=0.20,
                              v_max=30, climb_min=3.0)
    assert cmd.phase == FAR
    assert cmd.down_m_s < 0  # climbing toward an overhead target


def test_v_max_capped():
    los = (1.0, 0.0, 0.0)
    cmd = visual_velocity_ned(los, (0.9, 0.4, 0.0), 0.05, 0.30, v_far=12, v_close=14,
                              n_gain=8, term_gain=16, mid_thresh=0.06, near_thresh=0.20,
                              v_max=10, climb_min=3.0)
    assert math.sqrt(cmd.north_m_s**2 + cmd.east_m_s**2 + cmd.down_m_s**2) <= 10.0 + 1e-6
