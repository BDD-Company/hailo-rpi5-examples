"""Pure intercept geometry: lead point + closest-approach foot, and the NedVel result type."""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NedVel:
    north_m_s: float
    east_m_s: float
    down_m_s: float
    yaw_deg: float


def lead_intercept_point(
    drone_n: float, drone_e: float, drone_d: float,
    tgt_n: float, tgt_e: float, tgt_d: float,
    tvn: float, tve: float, tvd: float,
    *, v_drone: float, t_max: float,
):
    """Predicted intercept POINT (NED) for position-setpoint guidance.

    Solves time-to-go t* (iteratively) where the drone, at speed v_drone, just
    reaches the target's future position; capped at t_max so an uncatchable fast
    target gives a bounded lead instead of running to infinity. Returns the point
    target_pos + target_vel*t*. As the target nears, t*->0 and the point converges
    on the target, so commanding it as a POSITION setpoint lets the controller
    decelerate onto the crossing point (no blast-through / fly-off)."""
    t_go = 0.0
    for _ in range(6):
        fn = tgt_n + tvn * t_go
        fe = tgt_e + tve * t_go
        fd = tgt_d + tvd * t_go
        dist = math.sqrt((fn - drone_n) ** 2 + (fe - drone_e) ** 2 + (fd - drone_d) ** 2)
        t_go = min(t_max, dist / max(1e-3, v_drone))
    return (tgt_n + tvn * t_go, tgt_e + tve * t_go, tgt_d + tvd * t_go)


def closest_approach_point(
    drone_n: float, drone_e: float, drone_d: float,
    tgt_n: float, tgt_e: float, tgt_d: float,
    tvn: float, tve: float, tvd: float,
):
    """Perpendicular foot from the drone onto the target's velocity line — the
    point on the target's predicted PATH nearest the drone. For an uncatchable
    crossing target (target faster than the drone), pre-position here and hold:
    the target sweeps THROUGH this point. No lead-chase, so no fly-off. For a
    straight-flying target this is the natural closest-approach intercept."""
    vv = tvn * tvn + tve * tve + tvd * tvd
    if vv < 1e-9:
        return (tgt_n, tgt_e, tgt_d)
    s = ((drone_n - tgt_n) * tvn + (drone_e - tgt_e) * tve + (drone_d - tgt_d) * tvd) / vv
    return (tgt_n + tvn * s, tgt_e + tve * s, tgt_d + tvd * s)
