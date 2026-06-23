#!/usr/bin/env python3
"""Betaflight backend for the BDD pipeline — a drop-in alternative to `DroneMover`.

This mirrors the public surface of `drone.DroneMover` (the MAVSDK/PX4 backend) so
`drone_controller.py` can drive a Betaflight flight controller instead of PX4. It
talks to the **BetaFlightSim** stand (Betaflight SITL + Gazebo) exactly the way
`/media/Pets/BDD/BetaFlightSim/scripts/fly.py` does:

  * Control  — RC stick channels injected over **UDP** (default 127.0.0.1:9004).
               Packet = `<d16H` (double `time.time()` + 16×uint16 µs channels),
               values 1000–2000, AETR order:  ch0=Roll ch1=Pitch ch2=Throttle
               ch3=Yaw ch4=AUX1(ARM).  Must be sent continuously (~100 Hz) or the
               FC trips failsafe — a background thread handles the keepalive.
  * Telemetry — MSP over **TCP** (default 127.0.0.1:5761).  OPTIONAL and OFF by
               default: while an MSP client is connected Betaflight may refuse to
               arm (`ARMING_DISABLED_MSP`, the "configurator connected" guard) and
               the SITL only offers one TCP slot.  Enable via config
               `msp_telemetry: true` once you've confirmed arming still works.
               NB: in the current SITL build `MSP_RAW_IMU` returns zeros, so the
               IMU-dependent features (belly-down yaw, lift clamp) can't be
               validated in sim — attitude/altitude are fine.

What works vs PX4 (see the 2026-05-18 analysis):
  * move_to_target_zenith_async  — WORKS.  roll/pitch/thrust map straight to
    sticks; ACRO (rates) and ANGLE (use_set_attitude) both supported.
  * upside-down monitor          — WORKS (attitude only).
  * belly-down yaw / lift clamp  — needs real IMU accel (zero in this SITL).
  * move_to_target_ned           — NOT SUPPORTED.  Betaflight has no world-frame
    position controller and no GPS-free NED state.  Keep FOLLOW_TARGET_POSITION_NED
    and ESTIMATION_3D OFF; the call is a logged no-op that holds the last attitude.

Connection string (passed positionally, same slot as the MAVSDK URL):
  - "127.0.0.1"            -> RC udp 127.0.0.1:9004, MSP tcp 127.0.0.1:5761
  - "127.0.0.1:9004"      -> RC port overridden
  - ""/None               -> all defaults
Per-channel scales and ports are also overridable via the config dict.
"""
from __future__ import annotations

import asyncio
import logging
import math
import socket
import struct
import threading
import time

from helpers import dotdict

logger = logging.getLogger(__name__)

# --- RC wire format (must match BetaFlightSim/scripts/fly.py) ---------------
RC_NUM_CHANNELS = 16
RC_PACKET_FMT = "<d16H"            # double timestamp + 16×uint16 µs
US_MIN, US_MID, US_MAX = 1000, 1500, 2000
DEFAULT_RC_PORT = 9004
DEFAULT_MSP_PORT = 5761
DEFAULT_RC_RATE_HZ = 100           # keepalive cadence; <50 Hz risks failsafe

# AETR + AUX channel indices
CH_ROLL, CH_PITCH, CH_THROTTLE, CH_YAW, CH_AUX1_ARM = 0, 1, 2, 3, 4

# Arm switch (AUX1) endpoints — config default.diff puts ARM on AUX1, range ~1700+
AUX_LOW, AUX_HIGH = 1000, 2000


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _to_us(value: float, full_scale: float) -> int:
    """Map a signed command (±full_scale) onto an RC channel (1000–2000 µs)."""
    if full_scale <= 0:
        frac = 0.0
    else:
        frac = _clamp(value / full_scale, -1.0, 1.0)
    return int(round(US_MID + frac * (US_MAX - US_MID)))


def _euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> dict:
    """ZYX (yaw-pitch-roll) Euler -> quaternion, matching the PX4 dict's odometry.q."""
    r, p, y = math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return {
        "w": cr * cp * cy + sr * sp * sy,
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "timestamp_us": 0,
    }


class BetaflightDroneMover:
    """RC-over-UDP / MSP-over-TCP backend with the DroneMover public surface."""

    def __init__(self, drone_connection_string: str | None = None, config: dict | None = None) -> None:
        self.config = {} if config is None else config

        host, rc_port = self._parse_connection(drone_connection_string)
        self.rc_addr = (host, int(self.config.get("rc_port", rc_port)))
        self.msp_addr = (self.config.get("msp_host", host),
                         int(self.config.get("msp_port", DEFAULT_MSP_PORT)))

        self.rc_rate_hz = float(self.config.get("rc_rate_hz", DEFAULT_RC_RATE_HZ))
        self.use_set_attitude = self.config.get("use_set_attitude", False)

        # Stick scaling. In ACRO (rate) mode roll/pitch are deg/s; in ANGLE mode
        # they are degrees. Full-scale stick = these values.
        self.max_rate_deg_s = float(self.config.get("max_rate_deg_s", 360.0))
        self.max_angle_deg = float(self.config.get("max_angle_deg", 55.0))
        self.max_yaw_rate_deg_s = float(self.config.get("max_yaw_rate_deg_s", 200.0))
        # thrust ∈ [0,1] -> throttle µs. Hover ≈ 0.65 in this SITL (fly.py: ~1650).
        self.throttle_min_us = int(self.config.get("throttle_min_us", US_MIN))
        self.throttle_max_us = int(self.config.get("throttle_max_us", US_MAX))
        self.idle_thrust = float(self.config.get("idle_thrust", 0.0))

        self.upside_down_angle_deg = float(self.config.get("upside_down_angle_deg", 90.0))

        # Telemetry (MSP) is opt-in — see module docstring on the arming guard.
        self.msp_telemetry = bool(self.config.get("msp_telemetry", False))

        # --- runtime state ---
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._channels = [US_MID, US_MID, US_MIN, US_MID, AUX_LOW] + [US_MIN] * (RC_NUM_CHANNELS - 5)
        self._ch_lock = threading.Lock()
        self._tx_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._armed = False
        self.aborted = False
        self.upside_down_state = False

        self._telemetry: dict = {}
        self._tele_lock = threading.Lock()
        self._msp = None  # MSPClient when enabled

    # -- connection-string parsing -----------------------------------------
    @staticmethod
    def _parse_connection(s: str | None) -> tuple[str, int]:
        if not s:
            return "127.0.0.1", DEFAULT_RC_PORT
        s = s.split("://", 1)[-1]          # tolerate udp://host:port
        if ":" in s:
            host, port = s.rsplit(":", 1)
            try:
                return host or "127.0.0.1", int(port)
            except ValueError:
                return s, DEFAULT_RC_PORT
        return s, DEFAULT_RC_PORT

    # -- RC channel plumbing -----------------------------------------------
    def _set_channels(self, **named: int) -> None:
        idx = {"roll": CH_ROLL, "pitch": CH_PITCH, "throttle": CH_THROTTLE,
               "yaw": CH_YAW, "aux1": CH_AUX1_ARM}
        with self._ch_lock:
            for name, value in named.items():
                self._channels[idx[name]] = int(_clamp(value, US_MIN, US_MAX))

    def _tx_loop(self) -> None:
        period = 1.0 / self.rc_rate_hz
        while not self._stop.is_set():
            with self._ch_lock:
                ch = tuple(self._channels)
            try:
                self._sock.sendto(struct.pack(RC_PACKET_FMT, time.time(), *ch), self.rc_addr)
            except OSError as e:
                logger.warning("RC send failed: %s", e)
            time.sleep(period)

    # -- lifecycle ----------------------------------------------------------
    async def startup_sequence(self, arm_attempts: int = 100, force_arm: bool = False) -> None:
        """Start the RC keepalive, prime the link (AUX1 LOW), then arm (AUX1 HIGH).

        `force_arm` has no Betaflight analogue (arming preconditions are enforced
        FC-side); it is accepted for API parity and only shortens the link prime.
        """
        if self._tx_thread is None:
            self._stop.clear()
            self._tx_thread = threading.Thread(target=self._tx_loop, name="bf-rc-tx", daemon=True)
            self._tx_thread.start()

        if self.msp_telemetry:
            self._start_msp()

        # 1) RC link must appear with the arm switch LOW, else BF latches
        #    ARMING_DISABLED_ARM_SWITCH. Mirrors fly.py's "link/warmup (disarmed)".
        prime_s = 0.5 if force_arm else 2.0
        self._set_channels(throttle=US_MIN, aux1=AUX_LOW)
        await asyncio.sleep(prime_s)

        # 2) Arm: AUX1 HIGH, throttle at idle. There is no SET-and-confirm; we set
        #    the switch and hold. If MSP telemetry is on we could read STATUS flags
        #    to confirm, but by default we fly open-loop like fly.py.
        self._set_channels(throttle=US_MIN, aux1=AUX_HIGH)
        await asyncio.sleep(1.5)
        self._armed = True
        logger.info("Betaflight RC link up on %s; armed (open-loop).", self.rc_addr)

    # -- telemetry ----------------------------------------------------------
    def get_telemetry_dict_cached(self) -> dotdict:
        """Latest telemetry in the same dict shape the PX4 backend produces.

        Empty when `msp_telemetry` is off. Downstream consumers already guard on
        missing aspects (e.g. `(telemetry or {}).get('attitude_euler', {})`)."""
        with self._tele_lock:
            return dotdict(dict(self._telemetry))

    async def get_telemetry_dict(self, wait: bool = False) -> dict:
        return self.get_telemetry_dict_cached()

    # -- commands -----------------------------------------------------------
    async def move_to_target_zenith_async(self, roll_degree: float, pitch_degree: float,
                                          thrust: float = 0.0, current_telemetry=None) -> None:
        """Map a tilt+thrust setpoint onto RC sticks.

        PX4 backend sends either AttitudeRate (ACRO) or Attitude (ANGLE) depending
        on `use_set_attitude`; Betaflight's two native modes line up 1:1:
          ACRO  : roll/pitch are deg/s  -> scaled by max_rate_deg_s
          ANGLE : roll/pitch are deg    -> scaled by max_angle_deg
        """
        if self.upside_down_state:
            logger.warning("drone is UPSIDE-DOWN, ignoring command")
            return

        full_scale = self.max_angle_deg if self.use_set_attitude else self.max_rate_deg_s
        roll_us = _to_us(roll_degree, full_scale)
        pitch_us = _to_us(pitch_degree, full_scale)
        throttle_us = int(round(self.throttle_min_us + _clamp(thrust, 0.0, 1.0)
                                * (self.throttle_max_us - self.throttle_min_us)))

        # Yaw: belly-down assist needs body-frame accel (zero in this SITL build),
        # so it degrades to centred yaw until real IMU data is available.
        yaw_us = _to_us(self._belly_down_yaw_rate(current_telemetry), self.max_yaw_rate_deg_s)

        self._set_channels(roll=roll_us, pitch=pitch_us, throttle=throttle_us, yaw=yaw_us)
        logger.debug("zenith[%s] roll=%.1f pitch=%.1f thrust=%.2f -> rc(R=%d P=%d T=%d Y=%d)",
                     "ANGLE" if self.use_set_attitude else "ACRO",
                     roll_degree, pitch_degree, thrust, roll_us, pitch_us, throttle_us, yaw_us)

    async def move_to_target_ned(self, target_position_ned, current_telemetry=None) -> None:
        """No Betaflight equivalent — world-frame position control needs a Pi-side
        VIO loop. Hold the last attitude and log, so a mis-configured controller
        (FOLLOW_TARGET_POSITION_NED on) fails loud rather than silently."""
        logger.warning("move_to_target_ned unsupported on Betaflight backend; "
                       "disable FOLLOW_TARGET_POSITION_NED/ESTIMATION_3D. Holding attitude.")

    def _belly_down_yaw_rate(self, current_telemetry) -> float:
        """Port of DroneMover._belly_down_yaw_rate: drive the nose toward earth-down
        using the body-frame accelerometer. Returns 0 without IMU accel."""
        if not self.config.get("belly_down_yaw", True) or not current_telemetry:
            return 0.0
        try:
            accel = current_telemetry["imu"]["acceleration_frd"]
            down_fwd = -accel["forward_m_s2"]
            down_rgt = -accel["right_m_s2"]
        except (TypeError, KeyError):
            return 0.0
        if math.hypot(down_fwd, down_rgt) < float(self.config.get("belly_down_min_horizontal_g_mss", 2.0)):
            return 0.0
        kp = float(self.config.get("belly_down_yaw_kp", 1.5))
        max_rate = float(self.config.get("belly_down_yaw_max_rate_deg_s", 90.0))
        return _clamp(kp * math.degrees(math.atan2(down_rgt, down_fwd)), -max_rate, max_rate)

    async def standstill(self, thrust: float = 0.3) -> None:
        """Level sticks, hold the given throttle (still armed)."""
        throttle_us = int(round(self.throttle_min_us + _clamp(thrust, 0.0, 1.0)
                                * (self.throttle_max_us - self.throttle_min_us)))
        self._set_channels(roll=US_MID, pitch=US_MID, yaw=US_MID, throttle=throttle_us)
        logger.debug("standstill thrust=%.2f", thrust)

    async def idle(self) -> None:
        """Level sticks at idle throttle (armed, minimal lift)."""
        await self.standstill(self.idle_thrust)

    def ABORT(self) -> None:
        """Disarm: throttle to min, arm switch LOW. Stops sticks via failsafe."""
        self.aborted = True
        self._armed = False
        self._set_channels(roll=US_MID, pitch=US_MID, yaw=US_MID, throttle=US_MIN, aux1=AUX_LOW)
        logger.info("ABORT: disarming (aux1=LOW, throttle=min)")

    # -- upside-down monitor ------------------------------------------------
    async def start_upside_down_monitor(self) -> asyncio.Task:
        async def _monitor():
            while not self._stop.is_set():
                tele = self.get_telemetry_dict_cached()
                att = (tele or {}).get("attitude_euler") or {}
                roll, pitch = att.get("roll_deg"), att.get("pitch_deg")
                if roll is not None and pitch is not None:
                    tilt = max(abs(roll), abs(pitch))
                    self.upside_down_state = tilt > self.upside_down_angle_deg
                await asyncio.sleep(0.1)
        task = asyncio.create_task(_monitor())
        return task

    # -- MSP telemetry reader ----------------------------------------------
    def _start_msp(self) -> None:
        try:
            self._msp = _MSPClient(self.msp_addr)
            self._msp.connect()
        except OSError as e:
            logger.warning("MSP connect to %s failed (%s); telemetry disabled.", self.msp_addr, e)
            self._msp = None
            return
        t = threading.Thread(target=self._msp_loop, name="bf-msp-rx", daemon=True)
        t.start()

    def _msp_loop(self) -> None:
        accel_lsb_per_g = float(self.config.get("accel_lsb_per_g", 2048.0))
        mag_lsb_per_gauss = float(self.config.get("mag_lsb_per_gauss", 1090.0))
        period = 1.0 / float(self.config.get("msp_rate_hz", 50.0))
        while not self._stop.is_set() and self._msp is not None:
            try:
                att = self._msp.attitude()      # roll/pitch deci-deg, yaw deg
                imu = self._msp.raw_imu()        # raw counts (zeros in this SITL)
                alt = self._msp.altitude()       # cm, cm/s
            except OSError as e:
                logger.warning("MSP read error: %s", e)
                time.sleep(0.2)
                continue

            roll_deg, pitch_deg, yaw_deg = att
            ax, ay, az = (c / accel_lsb_per_g * 9.81 for c in imu[0:3])
            mx, my, mz = (c / mag_lsb_per_gauss for c in imu[6:9])
            alt_m, vario_m_s = alt[0] / 100.0, alt[1] / 100.0

            tele = {
                "attitude_euler": {
                    "roll_deg": roll_deg, "pitch_deg": pitch_deg, "yaw_deg": yaw_deg,
                    "timestamp_us": 0,
                },
                "imu": {
                    "acceleration_frd": {"forward_m_s2": ax, "right_m_s2": ay, "down_m_s2": az},
                    "magnetic_field_frd": {"forward_gauss": mx, "right_gauss": my, "down_gauss": mz},
                },
                # No world-frame position from Betaflight: x/y are 0 (DO NOT use for
                # NED nav). z from baro is drifty. q reconstructed from Euler so the
                # attitude-only consumers keep working.
                "odometry": {
                    "position_body": {"x_m": 0.0, "y_m": 0.0, "z_m": -alt_m},
                    "velocity_body": {"x_m_s": 0.0, "y_m_s": 0.0, "z_m_s": -vario_m_s},
                    "q": _euler_to_quaternion(roll_deg, pitch_deg, yaw_deg),
                    "time_usec": 0,
                },
                "landed_state": None,  # no FC equivalent; infer from throttle if needed
            }
            with self._tele_lock:
                self._telemetry = tele
            time.sleep(period)

    # -- teardown -----------------------------------------------------------
    def close(self) -> None:
        self._stop.set()
        if self._msp is not None:
            self._msp.close()
        try:
            self._sock.close()
        except OSError:
            pass


class _MSPClient:
    """Minimal MSP v1 client over TCP (request/response). Hand-rolled framing so
    the backend has no extra dependency; swap for `yamspy`/`pyMultiWii` if richer
    coverage is needed."""

    MSP_RAW_IMU = 102
    MSP_ATTITUDE = 108
    MSP_ALTITUDE = 109

    def __init__(self, addr: tuple[str, int]):
        self.addr = addr
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection(self.addr, timeout=2.0)

    def close(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None

    def _request(self, cmd: int, payload: bytes = b"") -> bytes:
        assert self.sock is not None
        # $M< <size> <cmd> <payload> <checksum>
        header = struct.pack("<BB", len(payload), cmd)
        checksum = 0
        for b in header + payload:
            checksum ^= b
        self.sock.sendall(b"$M<" + header + payload + bytes([checksum]))
        return self._read_response(cmd)

    def _read_response(self, expect_cmd: int) -> bytes:
        assert self.sock is not None

        def recvn(n: int) -> bytes:
            buf = b""
            while len(buf) < n:
                chunk = self.sock.recv(n - len(buf))
                if not chunk:
                    raise OSError("MSP connection closed")
                buf += chunk
            return buf

        # sync on '$M>'
        while recvn(1) != b"$":
            pass
        if recvn(1) != b"M":
            raise OSError("bad MSP magic")
        recvn(1)  # direction '>'
        size, cmd = struct.unpack("<BB", recvn(2))
        payload = recvn(size)
        recvn(1)  # checksum (not verified)
        if cmd != expect_cmd:
            raise OSError(f"MSP cmd mismatch: want {expect_cmd}, got {cmd}")
        return payload

    def attitude(self) -> tuple[float, float, float]:
        p = self._request(self.MSP_ATTITUDE)
        roll, pitch, yaw = struct.unpack("<hhh", p[:6])
        return roll / 10.0, pitch / 10.0, float(yaw)  # deci-deg, deci-deg, deg

    def raw_imu(self) -> tuple[int, ...]:
        p = self._request(self.MSP_RAW_IMU)
        return struct.unpack("<9h", p[:18])  # acc[3], gyro[3], mag[3] raw counts

    def altitude(self) -> tuple[int, int]:
        p = self._request(self.MSP_ALTITUDE)
        alt_cm = struct.unpack("<i", p[:4])[0]
        vario = struct.unpack("<h", p[4:6])[0] if len(p) >= 6 else 0
        return alt_cm, vario


# --- standalone smoke test: replicate fly.py's takeoff→hover→land -----------
async def _demo() -> None:
    logging.basicConfig(level=logging.INFO)
    drone = BetaflightDroneMover("127.0.0.1")
    await drone.startup_sequence(arm_attempts=1, force_arm=True)
    try:
        await drone.move_to_target_zenith_async(0, 0, thrust=0.80, current_telemetry=None)  # takeoff
        await asyncio.sleep(2.5)
        await drone.standstill(thrust=0.65)                                                  # hover
        await asyncio.sleep(10.0)
        await drone.standstill(thrust=0.45)                                                  # descend
        await asyncio.sleep(6.0)
        await drone.standstill(thrust=0.0)
        await asyncio.sleep(1.5)
    finally:
        drone.ABORT()
        await asyncio.sleep(1.0)
        drone.close()


if __name__ == "__main__":
    asyncio.run(_demo())
