"""Extract 3D position and orientation from drone telemetry dict."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class PositionNED:
    north_m: float
    east_m: float
    down_m: float

    @property
    def altitude_m(self) -> float:
        return -self.down_m


@dataclass(frozen=True, slots=True)
class VelocityNED:
    north_m_s: float
    east_m_s: float
    down_m_s: float


@dataclass(frozen=True, slots=True)
class EulerAngles:
    pitch_deg: float
    roll_deg: float
    yaw_deg: float


@dataclass(frozen=True, slots=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def to_euler(self) -> EulerAngles:
        # Roll (x-axis)
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis)
        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        # Yaw (z-axis)
        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return EulerAngles(
            pitch_deg=math.degrees(pitch),
            roll_deg=math.degrees(roll),
            yaw_deg=math.degrees(yaw),
        )


@dataclass(frozen=True, slots=True)
class AccelerationFRD:
    forward_m_s2: float
    right_m_s2: float
    down_m_s2: float


@dataclass(frozen=True, slots=True)
class MagneticFieldFRD:
    forward_gauss: float
    right_gauss: float
    down_gauss: float


@dataclass(frozen=True, slots=True)
class Pose:
    position: PositionNED
    orientation: EulerAngles
    velocity: VelocityNED
    quaternion: Quaternion
    timestamp_us: int
    acceleration: AccelerationFRD | None = None
    magnetic_field: MagneticFieldFRD | None = None


def get_position_ned(telemetry: dict) -> PositionNED:
    pos = telemetry["odometry"]["position_body"]
    return PositionNED(north_m=pos["x_m"], east_m=pos["y_m"], down_m=pos["z_m"])


def get_velocity_ned(telemetry: dict) -> VelocityNED:
    vel = telemetry["odometry"]["velocity_body"]
    return VelocityNED(north_m_s=vel["x_m_s"], east_m_s=vel["y_m_s"], down_m_s=vel["z_m_s"])


def get_orientation_euler(telemetry: dict) -> EulerAngles:
    att = telemetry["attitude_euler"]
    return EulerAngles(pitch_deg=att["pitch_deg"], roll_deg=att["roll_deg"], yaw_deg=att["yaw_deg"])


def get_orientation_quaternion(telemetry: dict) -> Quaternion:
    q = telemetry["odometry"]["q"]
    return Quaternion(w=q["w"], x=q["x"], y=q["y"], z=q["z"])


def get_acceleration_frd(telemetry: dict) -> AccelerationFRD | None:
    imu = telemetry.get("imu")
    if not imu:
        return None
    a = imu["acceleration_frd"]
    return AccelerationFRD(forward_m_s2=a["forward_m_s2"], right_m_s2=a["right_m_s2"], down_m_s2=a["down_m_s2"])


def get_magnetic_field_frd(telemetry: dict) -> MagneticFieldFRD | None:
    imu = telemetry.get("imu")
    if not imu:
        return None
    m = imu["magnetic_field_frd"]
    return MagneticFieldFRD(forward_gauss=m["forward_gauss"], right_gauss=m["right_gauss"], down_gauss=m["down_gauss"])


def rotate_frd_to_ned(q: Quaternion, fwd: float, right: float, down: float) -> tuple[float, float, float]:
    """Rotate a vector from body frame (FRD) to world frame (NED) using quaternion."""
    # Quaternion rotation: v' = q * v * q_conjugate
    # q = (w, x, y, z), v as pure quaternion (0, vx, vy, vz)
    # Body FRD axes map to NED when drone is level/north-facing
    vx, vy, vz = fwd, right, down
    # q * v
    tw = -q.x * vx - q.y * vy - q.z * vz
    tx = q.w * vx + q.y * vz - q.z * vy
    ty = q.w * vy + q.z * vx - q.x * vz
    tz = q.w * vz + q.x * vy - q.y * vx
    # (q * v) * q_conjugate
    north = -tw * q.x + tx * q.w - ty * q.z + tz * q.y
    east = -tw * q.y + ty * q.w - tz * q.x + tx * q.z
    down_out = -tw * q.z + tz * q.w - tx * q.y + ty * q.x
    return north, east, down_out


def get_pose(telemetry: dict) -> Pose:
    return Pose(
        position=get_position_ned(telemetry),
        orientation=get_orientation_euler(telemetry),
        velocity=get_velocity_ned(telemetry),
        quaternion=get_orientation_quaternion(telemetry),
        timestamp_us=telemetry["odometry"]["time_usec"],
        acceleration=get_acceleration_frd(telemetry),
        magnetic_field=get_magnetic_field_frd(telemetry),
    )


def project_detection_to_ned(
    detection_center_x: float,
    detection_center_y: float,
    aim_point_x: float,
    aim_point_y: float,
    fov_h_deg: float,
    fov_v_deg: float,
    distance_m: float,
    quaternion: Quaternion,
    drone_pos: PositionNED,
) -> PositionNED:
    """Project a 2D detection + distance estimate into an absolute NED position.

    Camera is mounted pointing up (-Z in FRD body frame).
    Frame bottom = drone front, frame left = drone left.

    Parameters
    ----------
    detection_center_x, detection_center_y : float
        Detection centre in normalised image coordinates (0-1, origin top-left).
    aim_point_x, aim_point_y : float
        Reference point in normalised image coordinates (usually 0.5, 0.5).
    fov_h_deg, fov_v_deg : float
        Camera horizontal / vertical field-of-view in degrees.
    distance_m : float
        Estimated distance from camera to target (metres).
    quaternion : Quaternion
        Body-to-NED orientation quaternion from telemetry.
    drone_pos : PositionNED
        Drone position in NED frame from telemetry.

    Returns
    -------
    PositionNED
        Estimated absolute target position in the NED world frame.
    """
    # Target offset in normalised camera frame (same convention as drone_controller):
    #   tx > 0  ⟹  target is LEFT  of aim point
    #   ty > 0  ⟹  target is ABOVE aim point (toward frame top = body rear)
    tx = aim_point_x - detection_center_x
    ty = aim_point_y - detection_center_y

    # Angular half-extents at unit distance along optical axis
    half_h = math.tan(math.radians(fov_h_deg / 2.0))
    half_v = math.tan(math.radians(fov_v_deg / 2.0))

    # 3-D ray in FRD body frame (camera optical axis = -Z_body = up).
    # Matches flight_debugger.py projection:
    #   frd_x (forward) = -ty * 2 * half_v   (frame top = rear ⟹ negate)
    #   frd_y (right)   = -tx * 2 * half_h   (frame left = body left ⟹ negate)
    #   frd_z (down)    = -1                  (optical axis points up)
    ray_x = -ty * 2.0 * half_v
    ray_y = -tx * 2.0 * half_h
    ray_z = -1.0

    # Normalise and scale by estimated distance
    ray_len = math.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z)
    scale = distance_m / ray_len
    offset_frd_x = ray_x * scale
    offset_frd_y = ray_y * scale
    offset_frd_z = ray_z * scale

    # Rotate offset from FRD body frame to NED world frame
    dn, de, dd = rotate_frd_to_ned(quaternion, offset_frd_x, offset_frd_y, offset_frd_z)

    return PositionNED(
        north_m=drone_pos.north_m + dn,
        east_m=drone_pos.east_m + de,
        down_m=drone_pos.down_m + dd,
    )


if __name__ == "__main__":
    example = {
        "attitude_euler": {
            "pitch_deg": 4.369004726409912,
            "roll_deg": 0.507519006729126,
            "timestamp_us": 465552000,
            "yaw_deg": -147.0821533203125,
        },
        "odometry": {
            "angular_velocity_body": {
                "pitch_rad_s": -0.024692663922905922,
                "roll_rad_s": -0.02139773592352867,
                "yaw_rad_s": 0.009160407818853855,
            },
            "child_frame_id": "1 (BODY_NED)",
            "frame_id": "1 (BODY_NED)",
            "pose_covariance": {
                "covariance_matrix": (
                    0.0009145613294094801, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    0.0009191993158310652, float("nan"), float("nan"), float("nan"), float("nan"),
                    0.08319202810525894, float("nan"), float("nan"), float("nan"),
                    0.00018126762006431818, float("nan"), float("nan"),
                    0.00015400342817883939, float("nan"),
                    0.005896727554500103,
                )
            },
            "position_body": {"x_m": 135.6093292236328, "y_m": 9.63675594329834, "z_m": 0.11730130016803741},
            "q": {
                "timestamp_us": 0,
                "w": -0.2828364074230194,
                "x": -0.037743814289569855,
                "y": -0.006644157692790031,
                "z": 0.9584023356437683,
            },
            "time_usec": 465522189,
            "velocity_body": {
                "x_m_s": 0.05169222131371498,
                "y_m_s": -0.004100066609680653,
                "z_m_s": -0.040390852838754654,
            },
            "velocity_covariance": {
                "covariance_matrix": (
                    0.003294615540653467, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                    0.003316952381283045, float("nan"), float("nan"), float("nan"), float("nan"),
                    0.00677329720929265, float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"),
                    float("nan"),
                )
            },
        },
        "landed_state": None,
        "imu": {
            "acceleration_frd": {
                "down_m_s2": -9.49996280670166,
                "forward_m_s2": 0.385750949382782,
                "right_m_s2": 0.00011855748016387224,
            },
            "angular_velocity_frd": {
                "down_rad_s": 0.0061968485824763775,
                "forward_rad_s": 0.0411582812666893,
                "right_rad_s": -0.0010773935355246067,
            },
            "magnetic_field_frd": {
                "down_gauss": 0.13252323865890503,
                "forward_gauss": -0.2951774597167969,
                "right_gauss": 0.3240545094013214,
            },
            "temperature_degc": 15.0,
            "timestamp_us": 465552189,
        },
    }

    pose = get_pose(example)
    print("3D Pose (NED frame):")
    print(f"  Position: N={pose.position.north_m:.3f}m  E={pose.position.east_m:.3f}m  D={pose.position.down_m:.3f}m")
    print(f"  Altitude: {pose.position.altitude_m:.3f}m")
    print(f"  Orientation: pitch={pose.orientation.pitch_deg:.2f}°  roll={pose.orientation.roll_deg:.2f}°  yaw={pose.orientation.yaw_deg:.2f}°")
    print(f"  Timestamp: {pose.timestamp_us} µs")
    print(f"  Velocity: vN={pose.velocity.north_m_s:.3f}m/s  vE={pose.velocity.east_m_s:.3f}m/s  vD={pose.velocity.down_m_s:.3f}m/s")

    euler_from_q = pose.quaternion.to_euler()
    print(f"  Quaternion->Euler: roll={euler_from_q.roll_deg:.2f}°  pitch={euler_from_q.pitch_deg:.2f}°  yaw={euler_from_q.yaw_deg:.2f}°")
