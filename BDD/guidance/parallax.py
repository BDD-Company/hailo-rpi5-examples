"""Size-independent parallax range: triangulate the target NED position from two
line-of-sight rays taken from two drone positions (ego-motion stereo), with the
target's constant-velocity motion compensated. Closed-form, plain floats, O(1) —
no numpy, no allocation beyond small tuples, never raises (None on degenerate)."""
import math


def closest_point_two_rays(q1, d1, q2, d2):
    """Closest point of two rays q_i + s_i*d_i (d_i UNIT). NED (n,e,d) float tuples.
    Returns (point, sin2, miss) or None if near-parallel or behind a camera."""
    rn, re, rd = q1[0] - q2[0], q1[1] - q2[1], q1[2] - q2[2]
    b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]          # a=c=1 (unit)
    d = d1[0] * rn + d1[1] * re + d1[2] * rd
    e = d2[0] * rn + d2[1] * re + d2[2] * rd
    denom = 1.0 - b * b                                        # sin^2(angle)
    if denom < 1e-9:
        return None
    s1 = (b * e - d) / denom
    s2 = (e - b * d) / denom
    if s1 <= 0.0 or s2 <= 0.0:
        return None
    p1 = (q1[0] + s1 * d1[0], q1[1] + s1 * d1[1], q1[2] + s1 * d1[2])
    p2 = (q2[0] + s2 * d2[0], q2[1] + s2 * d2[1], q2[2] + s2 * d2[2])
    point = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (p1[2] + p2[2]))
    miss = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return point, denom, miss


def triangulate_parallax(p_now, los_now, t_now, p_old, los_old, t_old, v_target,
                         *, min_sin2, min_baseline_m, max_miss_m):
    """Velocity-compensated two-ray triangulation. Returns (point_now_ned, prange)
    or None. p_*: drone NED tuples; los_*: unit LOS tuples; t_*: seconds; v_target:
    (vn,ve,vd) target velocity (from the Kalman)."""
    # perpendicular baseline gate (parallax magnitude vs the current LOS)
    bn, be, bd = p_now[0] - p_old[0], p_now[1] - p_old[1], p_now[2] - p_old[2]
    along = bn * los_now[0] + be * los_now[1] + bd * los_now[2]
    perp2 = (bn * bn + be * be + bd * bd) - along * along
    if perp2 < min_baseline_m * min_baseline_m:
        return None
    # Motion-compensate the OLD ray to t_now: target at t_old was v*(t_old-t_now)
    # behind its t_now position, so shift the old origin by -v*dt (dt<0) to solve
    # for x(t_now) directly. q_now uses the current frame as the time origin.
    dt = t_old - t_now
    q_now = p_now
    q_old = (p_old[0] - v_target[0] * dt, p_old[1] - v_target[1] * dt, p_old[2] - v_target[2] * dt)
    res = closest_point_two_rays(q_now, los_now, q_old, los_old)
    if res is None:
        return None
    pt, sin2, miss = res
    if sin2 < min_sin2 or miss > max_miss_m:
        return None
    prange = math.sqrt((pt[0] - p_now[0]) ** 2 + (pt[1] - p_now[1]) ** 2 + (pt[2] - p_now[2]) ** 2)
    return pt, prange
