#! /usr/bin/env python3
"""One-time provisioning: enable PX4 debug-topic logging so the BDD ulog trace is written.

The trace (see ulog_trace.py) is a MAVLink DEBUG_FLOAT_ARRAY that PX4 republishes as the
`debug_array` uORB topic. PX4 only WRITES that topic to its log when the `SDLOG_PROFILE`
parameter has the Debug bit (32) set — and it reads that parameter ONLY AT BOOT. So this
cannot be done from the flight app for the current session; it must be run, and the flight
controller rebooted, BEFORE flying.

    RUN THIS ONCE (before or after deploying, but BEFORE running the app), then REBOOT the
    flight controller. It is idempotent — safe to run again; it no-ops if already enabled.

    ./enable_px4_ulog_debug.py                 # auto-discover the USB flight controller
    ./enable_px4_ulog_debug.py --connection serial:///dev/ttyACM0:1000000
    ./enable_px4_ulog_debug.py --dry-run       # report only, change nothing

Exit code 0 = the debug bit is set (now, or already was). Non-zero = it is not.
"""
import argparse
import asyncio
import sys

from drone import DroneMover

# Kept identical to DroneMover.SDLOG_PROFILE_DEBUG_BIT; imported from there so there is one
# definition, not two that can drift.
DEBUG_BIT = DroneMover.SDLOG_PROFILE_DEBUG_BIT


async def _connect(connection: str):
    """Resolve + connect a bare MAVSDK System, reusing DroneMover's USB auto-discovery.
    We do NOT call startup_sequence: no arming, no offboard — this only touches a param."""
    mover = DroneMover(connection)
    resolved = mover._resolve_connection_string()
    print(f"connecting to {resolved} ...")
    await mover.drone.connect(system_address=resolved)

    async def _wait():
        async for state in mover.drone.core.connection_state():
            if state.is_connected:
                return
    await asyncio.wait_for(_wait(), timeout=30)
    print("  connected")
    return mover


async def run(connection: str, dry_run: bool) -> int:
    try:
        mover = await _connect(connection)
    except asyncio.TimeoutError:
        print("FAILED: no flight controller heartbeat within 30 s", file=sys.stderr)
        return 2

    try:
        profile = await mover.drone.param.get_param_int("SDLOG_PROFILE")
    except Exception as e:
        print(f"FAILED: could not read SDLOG_PROFILE: {e}", file=sys.stderr)
        return 2

    if profile & DEBUG_BIT:
        print(f"SDLOG_PROFILE={profile}: the debug bit ({DEBUG_BIT}) is already set. "
              f"Nothing to do — the ulog trace will be logged.")
        return 0

    target = profile | DEBUG_BIT
    if dry_run:
        print(f"SDLOG_PROFILE={profile}: debug bit CLEAR. Would set it to {target}. "
              f"(--dry-run: no change made.)")
        return 1

    print(f"SDLOG_PROFILE={profile}: debug bit clear -> setting to {target} ...")
    try:
        await mover.drone.param.set_param_int("SDLOG_PROFILE", target)
        readback = await mover.drone.param.get_param_int("SDLOG_PROFILE")
    except Exception as e:
        print(f"FAILED: could not set SDLOG_PROFILE: {e}", file=sys.stderr)
        return 2

    if not (readback & DEBUG_BIT):
        print(f"FAILED: set did not stick (read back {readback})", file=sys.stderr)
        return 2

    print(f"OK: SDLOG_PROFILE is now {readback}.")
    print("\n>>> REBOOT THE FLIGHT CONTROLLER before flying — PX4 reads SDLOG_PROFILE only")
    print(">>> at boot, so the change is not active until it restarts.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--connection", default="usb",
                   help="MAVSDK connection string, or 'usb' to auto-discover (default).")
    p.add_argument("--dry-run", action="store_true",
                   help="Report the current SDLOG_PROFILE and what would change; write nothing.")
    args = p.parse_args()
    return asyncio.run(run(args.connection, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
