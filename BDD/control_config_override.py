"""Apply a JSON override file onto the in-code control_config dict.

Used by the SITL tuning harness (PX4-Autopilot/scripts/tune.py): each trial
writes a JSON of scalar params; app.py loads it via --control-config and merges
it over the defaults. Only keys that already exist in the base are allowed, so a
typo fails loudly instead of silently doing nothing."""
import json


def apply_overrides(base: dict, path) -> dict:
    if path is None:
        return base
    with open(path) as f:
        overrides = json.load(f)
    for key, value in overrides.items():
        if key not in base:
            raise KeyError(
                f"--control-config key {key!r} is not in control_config")
        base[key] = value
    return base
