"""3-stage phased intercept guidance.

Range bands (from the monocular range estimate) select the law:
  FAR  (range >= mid_dist) : lead-pursuit -- fly at v_max toward the predicted
                             intercept point. Velocity command bypasses the cruise
                             limit, so this is the speed lever that lifts the ceiling.
  MID  (close_dist..mid_dist): ProNav (handled by the caller via pronav_velocity_ned).
  CLOSE(range < close_dist) : visual terminal servo (handled by the caller).

A crossing target must be LED, not centred (range-free centring tail-chases and
never closes -- see the reverted INVERTED-hybrid). lead_intercept_point provides the
lead; the alt guard mirrors LEAD's lead_max_alt_m to prevent a runaway climb."""
import math

from guidance.geometry import NedVel, lead_intercept_point


def select_phase(est_range, mid_dist, close_dist):
    if est_range >= mid_dist:
        return "FAR"
    if est_range >= close_dist:
        return "MID"
    return "CLOSE"


def lead_pursuit_velocity_ned(
    drone_n, drone_e, drone_d, tgt_n, tgt_e, tgt_d, tvn, tve, tvd,
    *, v_max, v_drone, t_max, vz_max, lead_max_alt_m,
):
    pn, pe, pd = lead_intercept_point(
        drone_n, drone_e, drone_d, tgt_n, tgt_e, tgt_d, tvn, tve, tvd,
        v_drone=v_drone, t_max=t_max)
    dn, de, dd = pn - drone_n, pe - drone_e, pd - drone_d
    dist = math.sqrt(dn * dn + de * de + dd * dd)
    if dist < 1e-6:
        return NedVel(0.0, 0.0, 0.0, 0.0)
    ux, uy, uz = dn / dist, de / dist, dd / dist
    vn, ve, vd = v_max * ux, v_max * uy, v_max * uz
    # vertical cap
    if vd > vz_max:
        vd = vz_max
    elif vd < -vz_max:
        vd = -vz_max
    # anti-runaway: never climb past the altitude cap
    if drone_d < -lead_max_alt_m and vd < 0:
        vd = 0.0
    yaw_deg = math.degrees(math.atan2(de, dn))
    return NedVel(vn, ve, vd, yaw_deg)
