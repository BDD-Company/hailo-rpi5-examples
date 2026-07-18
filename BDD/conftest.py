"""Shared pytest fixtures / import shims for the BDD host test-suite.

`drone.py` (and everything that imports it, e.g. `drone_controller`) imports the
`mavsdk` SDK at module load. mavsdk is a flight/runtime dependency that is present
on the Pi but not on a plain dev host, so importing those modules to unit-test their
PURE helpers would fail at collection. When mavsdk is genuinely absent we register a
lightweight stand-in so the pure-Python logic is host-testable; when the real mavsdk
IS installed (the Pi, CI with the full env) we touch nothing and the real package is
used. The stand-in is import-time only — no test exercises a mavsdk call.
"""
import sys
from unittest.mock import MagicMock

try:
    import mavsdk  # noqa: F401  -- real SDK present, use it
except ImportError:
    for _mod in (
        "mavsdk",
        "mavsdk.offboard",
        "mavsdk.telemetry",
        "mavsdk.mavlink_direct",
        "mavsdk.action",
        "mavsdk.server_utility",
    ):
        sys.modules.setdefault(_mod, MagicMock())
