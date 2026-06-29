#!/usr/bin/env python3
"""Betaflight backend for the BDD pipeline — a drop-in alternative to `DroneMover`.

This mirrors the public surface of `drone.DroneMover` (the MAVSDK/PX4 backend) so
`drone_controller.py` can drive a Betaflight flight controller instead of PX4.

It supports two transports, chosen by the connection string:

  * `ip://HOST[:PORT]`  — the BetaFlightSim stand (Betaflight SITL + Gazebo).
      RC over **UDP** (default :9004, packet `<d16H` = double `time.time()` + 16×
      uint16 µs channels, exactly like BetaFlightSim/scripts/fly.py); telemetry
      over **MSP/TCP** (default :5761).
  * `uart`              — real hardware on a Raspberry Pi.
      RC over **CRSF/serial** into the FC's RX UART (default /dev/ttyAMA2 @ 420000,
      16×11-bit channels in a 0x16 frame, like _TMP/crsf_bridge_msp_watch.py);
      telemetry over **MSP/serial** on a separate UART (default /dev/ttyAMA3 @
      115200). Needs `pyserial`, imported lazily so the sim path has no dependency.
      Optional **pilot pass-through** (the bridge's 3rd UART, default /dev/ttyAMA0):
      with `pilot_passthrough: true` the Pi also reads the pilot's CRSF receiver
      and the pilot keeps ultimate authority — their sticks pass straight through
      to the FC unless they flip the control-switch (`pilot_control_channel`,
      default ch6/AUX3) at/above `pilot_control_threshold_us` (1500) to hand
      control to this code. A stale/absent pilot link falls back to companion
      control. This is the human-override path; without it a Pi crash = the FC's
      only RC link is gone (failsafe), with no way for a pilot to take over.

Either way the internal channel state is kept in **microseconds** (1000–2000,
AETR+AUX: ch0=Roll ch1=Pitch ch2=Throttle ch3=Yaw ch4=AUX1/ARM); each transport
converts to its own wire format. RC must be sent continuously (~100 Hz UDP /
~250 Hz CRSF) or the FC trips failsafe — a background thread handles the keepalive.

Telemetry (MSP) is OPT-IN (`msp_telemetry: true`): on the SITL an active MSP
client can latch `ARMING_DISABLED_MSP` (the "configurator connected" guard), and
in that build `MSP_RAW_IMU` returns zeros (attitude/altitude are fine).

What works vs PX4 (see the 2026-05-18 analysis):
  * move_to_target_zenith_async  — WORKS.  roll/pitch/thrust map straight to
    sticks; ACRO (rates) and ANGLE (use_set_attitude) both supported.
  * upside-down monitor          — WORKS (attitude only).
  * belly-down yaw / lift clamp  — needs real IMU accel (zero in the SITL).
  * move_to_target_ned           — NOT SUPPORTED.  Betaflight has no world-frame
    position controller and no GPS-free NED state.  Keep FOLLOW_TARGET_POSITION_NED
    and ESTIMATION_3D OFF; the call is a logged no-op that holds the last attitude.

Connection string (passed positionally, same slot as the MAVSDK URL):
  - "ip://127.0.0.1"        -> UDP RC :9004 + MSP/TCP :5761 (sim)
  - "ip://127.0.0.1:9005"   -> RC port overridden
  - "uart"                  -> CRSF/serial RC + MSP/serial (real hardware)
  - ""/None / bare host     -> ip mode (sim), back-compat
Ports, serial devices, baud and per-channel scales are overridable via config.
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

# --- RC channel model (transport-independent, microseconds) -----------------
RC_NUM_CHANNELS = 16
US_MIN, US_MID, US_MAX = 1000, 1500, 2000

# UDP (sim) wire format — must match BetaFlightSim/scripts/fly.py
RC_UDP_PACKET_FMT = "<d16H"        # double timestamp + 16×uint16 µs
DEFAULT_RC_PORT = 9004
DEFAULT_MSP_PORT = 5761

# CRSF (real hardware) wire format — must match _TMP/crsf_bridge_msp_watch.py
CRSF_SYNC = 0xC8
CRSF_FRAMETYPE_RC = 0x16
DEFAULT_CRSF_UART = "/dev/ttyAMA2"      # CRSF OUT -> FC RX (we drive the FC)
DEFAULT_CRSF_IN_UART = "/dev/ttyAMA0"   # CRSF IN  <- pilot radio receiver (override)
DEFAULT_CRSF_BAUD = 420000
DEFAULT_MSP_UART = "/dev/ttyAMA3"
DEFAULT_MSP_BAUD = 115200

DEFAULT_RC_RATE_UDP_HZ = 100       # keepalive cadence; <50 Hz risks failsafe
DEFAULT_RC_RATE_CRSF_HZ = 250      # CRSF comfortably runs faster than UDP sim

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


def _us_to_crsf(us: int) -> int:
    """RC µs (1000–2000, centre 1500) -> CRSF 11-bit tick (≈191–1792, centre 992).

    Inverse of Betaflight's `us = (crsf - 992) * 5/8 + 1500`."""
    return _clamp(int(round((us - US_MID) * 8 / 5)) + 992, 0, 0x7FF)


def _crsf_to_us(tick: int) -> int:
    """CRSF 11-bit tick -> RC µs (inverse of _us_to_crsf), clamped to [1000, 2000]."""
    return _clamp(int(round((tick - 992) * 5 / 8)) + US_MID, US_MIN, US_MAX)


def _parse_crsf_rc_channels(payload: bytes) -> list[int]:
    """Decode a CRSF RC payload (16×11-bit LE) into 16 channel µs. Full bit
    accumulator (reads every byte) — the lossy 2-byte unpack in the example
    drops the high bits of channels whose 11 bits straddle a third byte."""
    bits = 0
    nbits = 0
    out = []
    it = iter(payload)
    for _ in range(RC_NUM_CHANNELS):
        while nbits < 11:
            bits |= next(it) << nbits
            nbits += 8
        out.append(_crsf_to_us(bits & 0x7FF))
        bits >>= 11
        nbits -= 11
    return out


def _crsf_crc8(data: bytes) -> int:
    """CRSF CRC-8/DVB-S2 (poly 0xD5) over type+payload."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _build_crsf_rc_frame(channels_us) -> bytes:
    """Pack 16 µs channels into a CRSF RC frame (0x16): 16×11-bit little-endian."""
    bits = 0
    nbits = 0
    payload = bytearray()
    for us in channels_us:
        bits |= (_us_to_crsf(us) & 0x7FF) << nbits
        nbits += 11
        while nbits >= 8:
            payload.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    if nbits:
        payload.append(bits & 0xFF)
    length = len(payload) + 2          # type + payload + crc
    frame = bytearray([CRSF_SYNC, length, CRSF_FRAMETYPE_RC])
    frame += payload
    frame.append(_crsf_crc8(frame[2:]))
    return bytes(frame)


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


# ===========================================================================
# Transports — RC injection links and MSP byte-stream links. Each pair (UDP/TCP
# for sim, CRSF/serial for hardware) is selected from the connection string.
# ===========================================================================
class _UDPRCLink:
    """RC sticks as `<d16H` UDP packets (BetaFlightSim)."""
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock: socket.socket | None = None

    def open(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, channels_us) -> None:
        self.sock.sendto(struct.pack(RC_UDP_PACKET_FMT, time.time(), *channels_us), self.addr)

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __repr__(self):
        return f"udp://{self.addr[0]}:{self.addr[1]}"


class _CRSFSerialRCLink:
    """RC sticks as CRSF 0x16 frames on a serial UART into the FC's RX port."""
    def __init__(self, device: str, baud: int):
        self.device, self.baud = device, baud
        self.ser = None

    def open(self) -> None:
        import serial  # lazy: only the hardware path needs pyserial
        self.ser = serial.Serial(self.device, self.baud, timeout=0)

    def send(self, channels_us) -> None:
        self.ser.write(_build_crsf_rc_frame(channels_us))

    def close(self) -> None:
        if self.ser:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def __repr__(self):
        return f"crsf://{self.device}@{self.baud}"


class _CRSFSerialRCReader:
    """Reads CRSF RC frames from a serial UART fed by the pilot's radio receiver
    (the example's /dev/ttyAMA0). `poll()` drains whatever bytes are available,
    parses any complete RC (0x16) frames, and returns the most recent channels as
    16 µs values (or None if no full frame arrived). CRC-checked; bad frames skipped.
    """
    def __init__(self, device: str, baud: int):
        self.device, self.baud = device, baud
        self.ser = None
        self._buf = bytearray()

    def open(self) -> None:
        import serial  # lazy
        self.ser = serial.Serial(self.device, self.baud, timeout=0.005)

    def poll(self) -> list[int] | None:
        data = self.ser.read(512)
        if data:
            self._buf += data
        latest = None
        buf = self._buf
        while len(buf) >= 3:
            if buf[0] != CRSF_SYNC:
                del buf[0]
                continue
            length = buf[1]
            if length < 2 or length > 62:      # implausible -> resync
                del buf[0]
                continue
            if len(buf) < length + 2:
                break                          # frame not fully arrived yet
            frame = bytes(buf[:length + 2])
            del buf[:length + 2]
            if _crsf_crc8(frame[2:-1]) != frame[-1]:
                continue                       # bad CRC -> drop
            if frame[2] == CRSF_FRAMETYPE_RC and len(frame) - 4 >= 22:
                latest = _parse_crsf_rc_channels(frame[3:3 + 22])
        return latest

    def close(self) -> None:
        if self.ser:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def __repr__(self):
        return f"crsf-in://{self.device}@{self.baud}"


class _TCPMSPLink:
    """MSP byte stream over TCP (BetaFlightSim UART1 bridge, :5761)."""
    def __init__(self, host: str, port: int):
        self.addr = (host, port)
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection(self.addr, timeout=2.0)

    def write(self, data: bytes) -> None:
        self.sock.sendall(data)

    def read(self, n: int) -> bytes:
        chunk = self.sock.recv(n)
        if not chunk:
            raise OSError("MSP connection closed")
        return chunk

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __repr__(self):
        return f"tcp://{self.addr[0]}:{self.addr[1]}"


class _SerialMSPLink:
    """MSP byte stream over a serial UART (real hardware)."""
    def __init__(self, device: str, baud: int):
        self.device, self.baud = device, baud
        self.ser = None

    def connect(self) -> None:
        import serial  # lazy
        self.ser = serial.Serial(self.device, self.baud, timeout=0.1)

    def write(self, data: bytes) -> None:
        self.ser.write(data)

    def read(self, n: int) -> bytes:
        chunk = self.ser.read(n)        # up to n bytes, blocks up to `timeout`
        if not chunk:
            raise OSError("MSP serial read timeout")
        return chunk

    def close(self) -> None:
        if self.ser:
            try:
                self.ser.close()
            finally:
                self.ser = None

    def __repr__(self):
        return f"serial://{self.device}@{self.baud}"


class BetaflightDroneMover:
    """Betaflight backend (UDP+TCP for sim, CRSF+MSP serial for hardware)."""

    def __init__(self, drone_connection_string: str | None = None, config: dict | None = None) -> None:
        self.config = {} if config is None else config

        self.rc_rate_hz = self.config.get("rc_rate_hz")  # resolved per-transport below
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

        # Pilot pass-through (uart only): read the pilot's CRSF receiver on a 3rd
        # UART and let the pilot keep ultimate authority. The pilot's control-switch
        # channel decides who drives the FC — below threshold the pilot's sticks
        # pass straight through; at/above it the companion (this code) commands.
        # Opt-in so the 2-UART path (and the sim) are unaffected.
        self.pilot_passthrough = bool(self.config.get("pilot_passthrough", False))
        self.pilot_control_channel = int(self.config.get("pilot_control_channel", 6))   # AUX3
        self.pilot_control_threshold_us = float(self.config.get("pilot_control_threshold_us", 1500))
        self.pilot_link_timeout_s = float(self.config.get("pilot_link_timeout_s", 0.5))

        # Pick transports from the connection string.
        self.mode, self.rc_link, self.rc_in_link, self.msp_link = self._build_links(drone_connection_string)
        if self.rc_rate_hz is None:
            self.rc_rate_hz = (DEFAULT_RC_RATE_CRSF_HZ if self.mode == "uart"
                               else DEFAULT_RC_RATE_UDP_HZ)
        else:
            self.rc_rate_hz = float(self.rc_rate_hz)

        # --- runtime state ---
        self._channels = [US_MID, US_MID, US_MIN, US_MID, AUX_LOW] + [US_MIN] * (RC_NUM_CHANNELS - 5)
        self._ch_lock = threading.Lock()
        self._tx_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._armed = False
        self.aborted = False
        self.upside_down_state = False

        # Pilot input state (populated by the CRSF-in reader thread).
        self._pilot_channels: list[int] | None = None
        self._pilot_ts = 0.0
        self._pilot_in_command = False
        self._pilot_lock = threading.Lock()
        self._pilot_thread: threading.Thread | None = None

        self._telemetry: dict = {}
        self._tele_lock = threading.Lock()
        self._msp = None  # _MSPClient when enabled

    # -- connection-string parsing -----------------------------------------
    @staticmethod
    def _parse_connection(s: str | None) -> tuple[str, str, int]:
        """Return (mode, host, rc_port). mode is 'uart' or 'ip'."""
        if s and s.strip().lower().startswith("uart"):
            return "uart", "", 0
        host, port = "127.0.0.1", DEFAULT_RC_PORT
        if s:
            rest = s.split("://", 1)[-1]          # tolerate ip://, udp://, bare host
            if rest:
                if ":" in rest:
                    h, p = rest.rsplit(":", 1)
                    try:
                        host, port = (h or host), int(p)
                    except ValueError:
                        host = rest
                else:
                    host = rest
        return "ip", host, port

    def _build_links(self, s: str | None):
        mode, host, rc_port = self._parse_connection(s)
        c = self.config
        rc_in = None
        if mode == "uart":
            rc = _CRSFSerialRCLink(c.get("crsf_uart", DEFAULT_CRSF_UART),
                                   int(c.get("crsf_baud", DEFAULT_CRSF_BAUD)))
            msp = _SerialMSPLink(c.get("msp_uart", DEFAULT_MSP_UART),
                                 int(c.get("msp_baud", DEFAULT_MSP_BAUD)))
            if self.pilot_passthrough:   # 3rd UART: pilot radio receiver
                rc_in = _CRSFSerialRCReader(c.get("crsf_in_uart", DEFAULT_CRSF_IN_UART),
                                            int(c.get("crsf_in_baud", DEFAULT_CRSF_BAUD)))
        else:
            rc = _UDPRCLink(host, int(c.get("rc_port", rc_port)))
            msp = _TCPMSPLink(c.get("msp_host", host), int(c.get("msp_port", DEFAULT_MSP_PORT)))
        logger.info("Betaflight transport: %s  (RC %r, MSP %r%s)", mode, rc, msp,
                    f", pilot-in {rc_in!r}" if rc_in else "")
        return mode, rc, rc_in, msp

    # -- RC channel plumbing -----------------------------------------------
    def _set_channels(self, **named: int) -> None:
        idx = {"roll": CH_ROLL, "pitch": CH_PITCH, "throttle": CH_THROTTLE,
               "yaw": CH_YAW, "aux1": CH_AUX1_ARM}
        with self._ch_lock:
            for name, value in named.items():
                self._channels[idx[name]] = int(_clamp(value, US_MIN, US_MAX))

    def _select_output_channels(self) -> list[int]:
        """Decide which sticks reach the FC this tick: the pilot's (pass-through)
        or this code's (companion). The pilot keeps authority — only when their
        control-switch is at/above threshold (and their link is fresh) do we send
        our own channels; otherwise their sticks pass straight through. A stale or
        absent pilot link falls back to companion control."""
        if self.rc_in_link is not None:
            with self._pilot_lock:
                pilot = self._pilot_channels
                ts = self._pilot_ts
            if pilot is not None and (time.monotonic() - ts) < self.pilot_link_timeout_s:
                ch = self.pilot_control_channel
                pi_command = ch >= len(pilot) or pilot[ch] >= self.pilot_control_threshold_us
                self._pilot_in_command = not pi_command
                if not pi_command:
                    return list(pilot)              # pilot flies; pass sticks through
            else:
                self._pilot_in_command = False
        with self._ch_lock:
            return list(self._channels)

    def _tx_loop(self) -> None:
        try:
            self.rc_link.open()
        except Exception as e:   # missing serial device / pyserial / socket
            logger.error("RC link open failed (%r): %s; no sticks will be sent.", self.rc_link, e)
            return
        period = 1.0 / self.rc_rate_hz
        while not self._stop.is_set():
            try:
                self.rc_link.send(self._select_output_channels())
            except OSError as e:
                logger.warning("RC send failed: %s", e)
            time.sleep(period)
        self.rc_link.close()

    def _pilot_loop(self) -> None:
        """Read the pilot's CRSF receiver and cache the latest channels for
        _select_output_channels()."""
        link = self.rc_in_link
        if link is None:
            return
        try:
            link.open()
        except Exception as e:
            logger.error("pilot CRSF-in open failed (%r): %s; pass-through off, "
                         "companion stays in command.", link, e)
            return
        logger.info("pilot CRSF pass-through active on %r (control switch ch%d @ %dus)",
                    link, self.pilot_control_channel, int(self.pilot_control_threshold_us))
        while not self._stop.is_set():
            try:
                ch = link.poll()
            except OSError as e:
                logger.warning("pilot CRSF read error: %s", e)
                time.sleep(0.05)
                continue
            if ch:
                with self._pilot_lock:
                    self._pilot_channels = ch
                    self._pilot_ts = time.monotonic()
            time.sleep(0.002)
        link.close()

    def pilot_channels(self) -> list[int] | None:
        """Latest pilot RC channels (µs) from the receiver, or None. For logging."""
        with self._pilot_lock:
            return None if self._pilot_channels is None else list(self._pilot_channels)

    # -- lifecycle ----------------------------------------------------------
    async def startup_sequence(self, arm_attempts: int = 100, force_arm: bool = False,
                               arm: bool = True) -> None:
        """Start the RC keepalive, prime the link (AUX1 LOW), then arm (AUX1 HIGH).

        `force_arm` has no Betaflight analogue (arming preconditions are enforced
        FC-side); it is accepted for API parity and only shortens the link prime.
        `arm=False` brings the link up but never toggles the arm switch — for
        bench/telemetry verification where the motors must stay disarmed.
        """
        if self._tx_thread is None:
            self._stop.clear()
            self._tx_thread = threading.Thread(target=self._tx_loop, name="bf-rc-tx", daemon=True)
            self._tx_thread.start()

        if self.rc_in_link is not None and self._pilot_thread is None:
            self._pilot_thread = threading.Thread(target=self._pilot_loop, name="bf-crsf-in", daemon=True)
            self._pilot_thread.start()

        if self.msp_telemetry:
            self._start_msp()

        # 1) RC link must appear with the arm switch LOW, else BF latches
        #    ARMING_DISABLED_ARM_SWITCH. Mirrors fly.py's "link/warmup (disarmed)".
        prime_s = 0.5 if force_arm else 2.0
        self._set_channels(throttle=US_MIN, aux1=AUX_LOW)
        await asyncio.sleep(prime_s)

        if not arm:
            logger.info("Betaflight RC link up (%s, %r); staying DISARMED (arm=False).",
                        self.mode, self.rc_link)
            return

        # 2) Arm: AUX1 HIGH, throttle at idle. There is no SET-and-confirm; we set
        #    the switch and hold. If MSP telemetry is on we could read STATUS flags
        #    to confirm, but by default we fly open-loop like fly.py.
        self._set_channels(throttle=US_MIN, aux1=AUX_HIGH)
        await asyncio.sleep(1.5)
        self._armed = True
        logger.info("Betaflight RC link up (%s, %r); armed (open-loop).", self.mode, self.rc_link)

    # -- telemetry ----------------------------------------------------------
    def get_telemetry_dict_cached(self) -> dotdict:
        """Latest telemetry in the same dict shape the PX4 backend produces.

        Empty when `msp_telemetry` is off. Downstream consumers already guard on
        missing aspects (e.g. `(telemetry or {}).get('attitude_euler', {})`)."""
        with self._tele_lock:
            return dotdict(dict(self._telemetry))

    async def get_telemetry_dict(self, wait: bool = False) -> dict:
        return self.get_telemetry_dict_cached()

    async def get_cached_attitude(self, wait_for_first: bool = True) -> dict | None:
        """Latest Euler attitude, for parity with DroneMover.get_cached_attitude.

        PX4 returns a MAVSDK EulerAngle; here we return the `attitude_euler`
        sub-dict (roll_deg/pitch_deg/yaw_deg) from cached MSP telemetry, or None
        when telemetry is unavailable. `wait_for_first` is accepted for signature
        parity but there is nothing to await — MSP telemetry is poll-cached."""
        with self._tele_lock:
            att = self._telemetry.get("attitude_euler")
        return dict(att) if att else None

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
            self.msp_link.connect()
            self._msp = _MSPClient(self.msp_link)
        except Exception as e:   # connect refused / missing device / pyserial
            logger.warning("MSP connect failed (%r): %s; telemetry disabled.", self.msp_link, e)
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
        for link in (self.rc_link, self.rc_in_link):   # idempotent; loops also close on stop
            try:
                if link is not None:
                    link.close()
            except Exception:
                pass


class _MSPClient:
    """Minimal MSP v1 client over an abstract byte link (TCP or serial). Hand-rolled
    framing so the backend has no extra dependency; swap for `yamspy`/`pyMultiWii`
    if richer coverage is needed."""

    MSP_RAW_IMU = 102
    MSP_ATTITUDE = 108
    MSP_ALTITUDE = 109

    def __init__(self, link):
        self.link = link

    def close(self) -> None:
        self.link.close()

    def _request(self, cmd: int, payload: bytes = b"") -> bytes:
        # $M< <size> <cmd> <payload> <checksum>
        header = struct.pack("<BB", len(payload), cmd)
        checksum = 0
        for b in header + payload:
            checksum ^= b
        self.link.write(b"$M<" + header + payload + bytes([checksum]))
        return self._read_response(cmd)

    def _read_response(self, expect_cmd: int) -> bytes:
        def recvn(n: int) -> bytes:
            buf = b""
            while len(buf) < n:
                buf += self.link.read(n - len(buf))   # raises OSError on close/timeout
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
async def _demo(connection: str = "ip://127.0.0.1") -> None:
    logging.basicConfig(level=logging.INFO)
    drone = BetaflightDroneMover(connection)
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
    import sys
    asyncio.run(_demo(sys.argv[1] if len(sys.argv) > 1 else "ip://127.0.0.1"))
