"""LEAD law: pre-position on the target's closest-approach point and hold.

Pure extraction of the inline LEAD branch from drone_controller.py — the math is
unchanged. Returns the position setpoint (n, e, down) for move_to_target_ned."""
import math

from guidance.geometry import closest_approach_point


def lead_setpoint_ned(
    drone_n, drone_e, drone_d,
    tgt_n, tgt_e, tgt_d,
    tvn, tve, tvd,
    home_n, home_e,
    *, lead_max_lat, lead_max_alt_m, lead_alt_offset,
):
    ftn, fte, ftd = closest_approach_point(
        drone_n, drone_e, drone_d, tgt_n, tgt_e, tgt_d, tvn, tve, tvd)
    vmag = math.sqrt(tvn * tvn + tve * tve + tvd * tvd)
    if 5.0 <= vmag <= 40.0:
        fn, fe = ftn, fte
    else:
        fn, fe = drone_n, drone_e
    fn = max(home_n - lead_max_lat, min(home_n + lead_max_lat, fn))
    fe = max(home_e - lead_max_lat, min(home_e + lead_max_lat, fe))
    down = tgt_d + lead_alt_offset
    down = max(down, -lead_max_alt_m)
    return (fn, fe, down)
