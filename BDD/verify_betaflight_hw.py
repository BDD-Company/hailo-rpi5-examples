#!/usr/bin/env python3
"""Bench verification for the Betaflight UART backend against a REAL flight controller.

Run this ON the Raspberry Pi that has the FC wired up (CRSF RX UART + MSP UART).
It exercises betaflight_drone.BetaflightDroneMover end-to-end and reports whether
the real FC is talking back, WITHOUT spinning the motors.

SAFETY
  * Default run is DISARMED: throttle held at minimum, AUX1 (arm) LOW the whole
    time. Nothing should spin. Safe with props on, but remove props anyway.
  * `--arm` additionally toggles the arm switch HIGH while keeping throttle at
    MINIMUM (no thrust command) to confirm the FC accepts our RC link and arms.
    REMOVE PROPELLERS before using --arm.
  * Ctrl+C disarms and exits.

What it checks
  1. MSP telemetry IN  — polls MSP_ATTITUDE; if roll/pitch track when you tilt the
     craft, the MSP UART -> FC path works (proves telemetry).
  2. RC injection OUT  — sends a distinctive stick pattern over CRSF and reads it
     back via MSP_RC; if the values appear, the CRSF UART -> FC path works.

Usage (on the Pi):
  python3 verify_betaflight_hw.py                 # uart, disarmed, telemetry+RC check
  python3 verify_betaflight_hw.py --seconds 15
  python3 verify_betaflight_hw.py --arm           # also test arming (props OFF!)
  python3 verify_betaflight_hw.py ip://127.0.0.1  # against the SITL instead
  CRSF_UART=/dev/ttyAMA2 MSP_UART=/dev/ttyAMA3 python3 verify_betaflight_hw.py

Device paths / baud come from env (CRSF_UART, CRSF_BAUD, MSP_UART, MSP_BAUD) or the
betaflight_drone defaults (/dev/ttyAMA2@420000, /dev/ttyAMA3@115200).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import struct
import time

import betaflight_drone as bf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("verify-bf")

MSP_RC = 105


def _config_from_env() -> dict:
    cfg = {"msp_telemetry": True}
    for env, key, cast in (
        ("CRSF_UART", "crsf_uart", str), ("CRSF_BAUD", "crsf_baud", int),
        ("MSP_UART", "msp_uart", str), ("MSP_BAUD", "msp_baud", int),
    ):
        if os.environ.get(env):
            cfg[key] = cast(os.environ[env])
    return cfg


def _read_msp_rc(drone: bf.BetaflightDroneMover) -> list[int] | None:
    """Read the FC's view of the RC channels (MSP_RC, cmd 105) -> list of µs."""
    msp = drone._msp
    if msp is None:
        return None
    try:
        payload = msp._request(MSP_RC)
    except OSError as e:
        log.warning("MSP_RC read failed: %s", e)
        return None
    n = len(payload) // 2
    return list(struct.unpack(f"<{n}H", payload[: n * 2]))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("connection", nargs="?", default="uart",
                    help="connection string: 'uart' (default) or 'ip://HOST'")
    ap.add_argument("--seconds", type=float, default=10.0, help="telemetry watch duration")
    ap.add_argument("--arm", action="store_true",
                    help="also toggle the arm switch (throttle stays MIN). PROPS OFF!")
    args = ap.parse_args()

    cfg = _config_from_env()
    drone = bf.BetaflightDroneMover(args.connection, cfg)
    log.info("backend=%s mode=%s  RC=%r  MSP=%r",
             type(drone).__name__, drone.mode, drone.rc_link, drone.msp_link)

    # Start the RC keepalive + MSP reader. arm=False keeps AUX1 LOW the whole time
    # (no arming) for the safe telemetry/RC run; --arm toggles it (throttle stays min).
    if args.arm:
        log.warning("ARMING (throttle stays at minimum). Ensure PROPELLERS ARE OFF.")
    await drone.startup_sequence(arm_attempts=1, force_arm=True, arm=args.arm)

    msp_ok = rc_ok = False
    t_end = time.time() + args.seconds
    first_att = None
    while time.time() < t_end:
        tele = drone.get_telemetry_dict_cached()
        att = (tele or {}).get("attitude_euler")
        if att:
            msp_ok = True
            if first_att is None:
                first_att = att
            log.info("ATT  roll=%6.1f  pitch=%6.1f  yaw=%6.1f  (tilt the craft to see it move)",
                     att["roll_deg"], att["pitch_deg"], att["yaw_deg"])
        else:
            log.info("ATT  <no MSP telemetry yet>")
        # When pilot pass-through is configured, show the pilot's sticks + who's in command.
        if drone.rc_in_link is not None:
            p = drone.pilot_channels()
            log.info("PILOT %s  in_command=%s",
                     p[:7] if p else "<no CRSF from receiver>", drone._pilot_in_command)
        await asyncio.sleep(1.0)

    # RC injection check: send a distinctive pattern and read it back via MSP_RC.
    log.info("--- RC injection check: setting roll=1600 pitch=1700 yaw=1400 (throttle stays min) ---")
    drone._set_channels(roll=1600, pitch=1700, yaw=1400)
    await asyncio.sleep(0.5)
    seen = _read_msp_rc(drone)
    if seen:
        log.info("MSP_RC (FC's view of channels, µs): %s", seen[:8])
        rc_ok = any(v in seen for v in (1600, 1700, 1400)) or \
                any(abs(v - t) <= 8 for v in seen for t in (1600, 1700, 1400))
    else:
        log.info("MSP_RC: <no response>")

    drone.ABORT()
    await asyncio.sleep(0.5)
    drone.close()

    print("\n================ VERIFICATION RESULT ================")
    print(f"  transport            : {drone.mode}  (RC {drone.rc_link!r}, MSP {drone.msp_link!r})")
    print(f"  MSP telemetry IN     : {'OK — FC attitude received' if msp_ok else 'FAIL — no MSP attitude'}")
    print(f"  CRSF RC injection OUT: {'OK — sticks seen in MSP_RC' if rc_ok else 'UNCONFIRMED — check wiring/map'}")
    print("=====================================================")
    return 0 if (msp_ok or rc_ok) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\ninterrupted")
