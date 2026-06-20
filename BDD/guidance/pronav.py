"""Collision-course (proportional-navigation / CBDR) velocity guidance."""
import math
from guidance.geometry import NedVel


def pronav_velocity_ned(
    drone_n: float, drone_e: float, drone_d: float,
    tgt_n: float, tgt_e: float, tgt_d: float,
    tgt_vn: float, tgt_ve: float, tgt_vd: float,
    *, v_close: float, n: float, v_max: float, vz_max: float = 10.0,
) -> NedVel:
    rn = tgt_n - drone_n
    re = tgt_e - drone_e
    rd = tgt_d - drone_d
    R = math.sqrt(rn * rn + re * re + rd * rd)
    if R < 1e-6:
        return NedVel(0.0, 0.0, 0.0, 0.0)
    lx, ly, lz = rn / R, re / R, rd / R               # LOS unit vector (NED)

    # target velocity component perpendicular to the line of sight
    v_dot_l = tgt_vn * lx + tgt_ve * ly + tgt_vd * lz
    pvn = tgt_vn - v_dot_l * lx
    pve = tgt_ve - v_dot_l * ly
    pvd = tgt_vd - v_dot_l * lz

    # collision-course command: close along LOS + match perpendicular drift
    vn = v_close * lx + n * pvn
    ve = v_close * ly + n * pve
    vd = v_close * lz + n * pvd

    # cap the vertical command: a steep line-of-sight to a high target would
    # otherwise drive a runaway climb, since offboard set_velocity_ned bypasses
    # MPC_Z_VEL_MAX_UP. Clamp before the overall v_max cap.
    if vd > vz_max:
        vd = vz_max
    elif vd < -vz_max:
        vd = -vz_max

    speed = math.sqrt(vn * vn + ve * ve + vd * vd)
    if speed > v_max and speed > 0.0:
        s = v_max / speed
        vn, ve, vd = vn * s, ve * s, vd * s

    yaw_deg = math.degrees(math.atan2(re, rn))        # face the target
    return NedVel(vn, ve, vd, yaw_deg)
