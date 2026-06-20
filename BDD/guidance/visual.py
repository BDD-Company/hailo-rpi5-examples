"""Range-free image-based interception guidance (3-phase).

Every phase works from the target's BEARING in the image + the drone's attitude —
never from the (noisy) monocular range. The line-of-sight (LOS) unit vector in NED
is reliable: it needs only the camera ray (cx, cy) and the body->NED quaternion, no
distance. Phases switch on the target's apparent size in the frame (a robust range
proxy):

  FAR  (small box) : visual servoing / chase — close along the LOS, keep the target
                     in frame. Goal: don't lose it, steadily approach.
  MID  (medium)    : image-based proportional navigation — null the LOS rotation
                     rate so the closing line stays straight (constant bearing,
                     decreasing range = collision course).
  NEAR (large box) : terminal visual correction — high-gain null of the LOS drift to
                     minimise the final in-frame miss.

Output is an NED velocity setpoint (north, east, down) + yaw. Flying along the LOS
naturally climbs toward a high target (LOS points up) and stops climbing once the
target is level, so altitude needs no range estimate.
"""
import math
from dataclasses import dataclass

from telemetry_position import rotate_frd_to_ned

FAR, MID, NEAR = "FAR", "MID", "NEAR"


@dataclass(frozen=True)
class NedVelCmd:
    north_m_s: float
    east_m_s: float
    down_m_s: float
    yaw_deg: float
    phase: str


def los_unit_ned(cx, cy, aim_x, aim_y, fov_h_deg, fov_v_deg, quaternion):
    """Unit line-of-sight vector drone->target in NED, range-free.

    Same camera ray convention as telemetry_position.project_camera_to_ned (camera
    looks up = -Z body), but normalised — distance is never used."""
    tx = aim_x - cx
    ty = aim_y - cy
    half_h = math.tan(math.radians(fov_h_deg / 2.0))
    half_v = math.tan(math.radians(fov_v_deg / 2.0))
    rx = -ty * 2.0 * half_v   # FRD forward (frame top = body rear)
    ry = -tx * 2.0 * half_h   # FRD right  (frame left = body left)
    rz = -1.0                 # FRD down   (optical axis points up)
    n = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
    dn, de, dd = rotate_frd_to_ned(quaternion, rx / n, ry / n, rz / n)
    m = math.sqrt(dn * dn + de * de + dd * dd) or 1.0
    return (dn / m, de / m, dd / m)


def phase_from_size(box_frac, mid_thresh, near_thresh):
    if box_frac >= near_thresh:
        return NEAR
    if box_frac >= mid_thresh:
        return MID
    return FAR


def visual_velocity_ned(los, los_prev, dt, box_frac, *,
                        v_far, v_close, n_gain, term_gain,
                        mid_thresh, near_thresh, v_max, climb_min):
    """3-phase image-based velocity command (NED).

    los, los_prev : unit LOS vectors (NED) this frame and the previous one
    dt            : seconds since the previous LOS (for the LOS rate)
    box_frac      : target apparent size (normalised, max of w/h) -> selects phase
    """
    lx, ly, lz = los
    phase = phase_from_size(box_frac, mid_thresh, near_thresh)

    # perpendicular component of the LOS rate = the rotation we want to null (PN)
    perp_n = perp_e = perp_d = 0.0
    if los_prev is not None and dt > 1e-3:
        dn = (los[0] - los_prev[0]) / dt
        de = (los[1] - los_prev[1]) / dt
        dd = (los[2] - los_prev[2]) / dt
        dot = dn * lx + de * ly + dd * lz       # along-LOS part (range rate dir)
        perp_n, perp_e, perp_d = dn - dot * lx, de - dot * ly, dd - dot * lz

    if phase == FAR:
        # chase: close along the LOS (already points up at a high target), which
        # keeps the target centred as the drone flies toward it.
        vn, ve, vd = v_far * lx, v_far * ly, v_far * lz
    else:
        # MID image-ProNav / NEAR terminal: close along LOS + null the LOS rotation;
        # the terminal gain is higher to drive the final in-frame miss to zero.
        g = term_gain if phase == NEAR else n_gain
        vn = v_close * lx + g * perp_n
        ve = v_close * ly + g * perp_e
        vd = v_close * lz + g * perp_d

    # while the target is clearly above (lz<0 = up), keep at least a minimum climb so
    # the drone never stalls/sinks under it (bounded: stops once LOS is level).
    if lz < -0.2 and vd > -climb_min:
        vd = -climb_min

    speed = math.sqrt(vn * vn + ve * ve + vd * vd)
    if speed > v_max and speed > 0.0:
        s = v_max / speed
        vn, ve, vd = vn * s, ve * s, vd * s

    yaw_deg = math.degrees(math.atan2(ly, lx))
    return NedVelCmd(vn, ve, vd, yaw_deg, phase)
