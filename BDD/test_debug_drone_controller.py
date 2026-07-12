"""Tests for the off-rig replay harness (`debug_drone_controller`).

The regression these guard against: `main()` used to hand `drone_controlling_thread`
the pre-refactor *flat config dict* scraped out of the log, while the controller had
been refactored to take the `Config` dataclass. It died on the first line with
`AttributeError: 'dict' object has no attribute 'DEBUG'` and nothing noticed, because
nothing exercised it.

`test_replay_runs_through_the_real_control_thread` is the one that matters: it drives
the REAL `drone_controlling_thread` end to end. If the config wiring rots again, it fails.

Everything here asserts *structure*, never values, from `config.yaml` — that file carries
live flight tuning and its numbers change constantly.
"""

import math
from pathlib import Path

import pytest

from config import Config
from parse_config import ConfigError

import debug_drone_controller as ddc


CONFIG_YAML = Path(__file__).resolve().parent / "config.yaml"


# ───────────────────────────────────────────────────────────────────────
# Synthetic log
# ───────────────────────────────────────────────────────────────────────

# Copied verbatim from a real flight log (BDD_20260427-160029.log) so the harness's
# regexes and eval() namespace are exercised exactly as they are in production.
_TELEMETRY_TEMPLATE = (
    "{{'attitude_euler': {{'pitch_deg': 3.47, 'roll_deg': 0.69, 'timestamp_us': 304806000, "
    "'yaw_deg': -132.96}}, 'odometry': {{'angular_velocity_body': {{'pitch_rad_s': -0.006, "
    "'roll_rad_s': -0.064, 'yaw_rad_s': -0.007}}, 'child_frame_id': '1 (BODY_NED)', "
    "'frame_id': '1 (BODY_NED)', 'position_body': {{'x_m': -225.5, 'y_m': -458.5, 'z_m': 0.99}}, "
    "'q': {{'timestamp_us': 0, 'w': -0.398, 'x': -0.030, 'y': -0.006, 'z': 0.916}}, "
    "'time_usec': 304806018, 'velocity_body': {{'x_m_s': {vx}, 'y_m_s': 0.018, 'z_m_s': -0.080}}}}, "
    "'landed_state': '2 (IN_AIR)', "
    "'imu': {{'acceleration_frd': {{'down_m_s2': -8.52, 'forward_m_s2': 1.66, 'right_m_s2': -0.11}}, "
    "'angular_velocity_frd': {{'down_rad_s': -0.007, 'forward_rad_s': -0.064, 'right_rad_s': -0.006}}, "
    "'magnetic_field_frd': {{'down_gauss': 0.14, 'forward_gauss': -0.37, 'right_gauss': 0.54}}, "
    "'temperature_degc': 15.0, 'timestamp_us': 304806018}}, 'flight_mode': '7 (OFFBOARD)'}}"
)

_PREFIX = "2026-04-27 16:00:{sec:02d}.{ms:03d} [Drone] @ {{ drone_controller.py:545 : x() }} <DEBUG> :\t"


def _synthetic_log(n_frames: int = 12) -> str:
    """A minimal but format-faithful flight log: a target drifting across frame while
    the drone accelerates from 0 to ~33 m/s (so speed-gated code actually triggers)."""
    lines = []
    for i in range(n_frames):
        sec, ms = 10 + (i * 40) // 1000, (i * 40) % 1000
        prefix = _PREFIX.format(sec=sec, ms=ms)
        x = 0.30 + i * 0.01                      # target drifts right
        det = (f"Detection(bbox=Rect(x={x:.3f}, y=0.419, w=0.023, h=0.041), "
               f"confidence=0.75, track_id=None, class_id=None)")
        lines.append(
            f"{prefix}frame=#{i:04d} !!! GOT DETECTIONS, objects detected: 1 ([{det}]), "
            f"detection delay: 66.4ms, total delay: 105.4ms"
        )
        lines.append(
            f"{prefix}frame=#{i:04d} telemetry: "
            + _TELEMETRY_TEMPLATE.format(vx=round(i * 3.0, 3))
        )
    return "\n".join(lines) + "\n"


@pytest.fixture
def synthetic_log(tmp_path: Path) -> Path:
    log = tmp_path / "BDD_synthetic.log"
    log.write_text(_synthetic_log())
    return log


# ───────────────────────────────────────────────────────────────────────
# load_replay_config
# ───────────────────────────────────────────────────────────────────────

def test_loads_the_real_config_yaml_despite_the_hef_not_existing_here():
    """inference.hef_model_path is an ExistingFile and the HEF only exists on the Pi,
    so a naive load of ANY of our configs raises ConfigError on a dev host. The harness
    must stub the path itself — replay never runs inference."""
    cfg = ddc.load_replay_config(CONFIG_YAML)

    assert isinstance(cfg, Config)
    assert cfg.inference.hef_model_path.is_file()   # stubbed to something that exists


def test_stubbing_warns_loudly(caplog):
    with caplog.at_level("WARNING"):
        ddc.load_replay_config(CONFIG_YAML)

    assert any("hef_model_path" in r.message for r in caplog.records), \
        "silently swapping the model path would be a lie; it must warn"


def test_debug_is_forced_on():
    # DEBUG is a runtime-only field (never settable from the file); replay always wants it.
    assert ddc.load_replay_config(CONFIG_YAML).DEBUG is True


def test_dotted_overrides_land_on_the_nested_config():
    cfg = ddc.load_replay_config(
        CONFIG_YAML,
        overrides={"pd_coeff.p": [4, 5], "thrust.max": 0.75, "confidence_min": 0.9},
    )

    assert (cfg.pd_coeff.p.x, cfg.pd_coeff.p.y) == (4, 5)
    assert cfg.thrust.max == 0.75
    assert cfg.confidence_min == 0.9


def test_an_override_means_exactly_what_it_would_mean_in_the_yaml_file():
    # An XY field needs [x, y] in config.yaml, so it needs [x, y] here too. Quietly
    # broadcasting a scalar would make --params a second, subtly different config dialect.
    with pytest.raises(ConfigError):
        ddc.load_replay_config(CONFIG_YAML, overrides={"pd_coeff.p": 4})


def test_overrides_are_validated_like_any_other_config_value():
    # confidence_min is Range(0.0, 1.0): an out-of-range override must not sneak through
    # just because it came from the command line.
    with pytest.raises(ConfigError):
        ddc.load_replay_config(CONFIG_YAML, overrides={"confidence_min": 7.0})


def test_old_flat_keys_fail_loudly_rather_than_silently_doing_nothing():
    # The whole bug this harness died of: pre-refactor flat keys. `pd_coeff_p` is not
    # `pd_coeff.p`, and quietly ignoring it would make an A/B run look like a no-op change.
    with pytest.raises(ConfigError):
        ddc.load_replay_config(CONFIG_YAML, overrides={"pd_coeff_p": 4})


def test_a_dotted_path_into_a_disabled_section_is_rejected_not_invented():
    with pytest.raises(ConfigError):
        ddc.load_replay_config(CONFIG_YAML, overrides={"pd_coeff.no_such_knob": 1})


# ───────────────────────────────────────────────────────────────────────
# Replay
# ───────────────────────────────────────────────────────────────────────

def test_parses_the_synthetic_log(synthetic_log):
    _config_dict, frames, base_ns = ddc.parse_log(synthetic_log)

    assert len(frames) == 12
    assert frames[0]["detections"], "detections should survive the eval() namespace"
    assert frames[11]["telemetry"]["odometry"]["velocity_body"]["x_m_s"] == pytest.approx(33.0)
    assert base_ns > 0


def test_replay_queue_auto_advance_does_not_wait_for_a_keypress():
    q = ddc.ReplayQueue([], auto_advance=True)
    assert q.get() is ddc.STOP      # would block forever without auto_advance


def test_replay_runs_through_the_real_control_thread(synthetic_log):
    """THE regression test. Drives the real `drone_controlling_thread` with a real
    `Config`, headless. This is what was broken: it died instantly on a flat dict."""
    cfg = ddc.load_replay_config(CONFIG_YAML)

    _config_dict, frames, base_ns = ddc.parse_log(synthetic_log)
    detections_list = ddc.build_detections_list(frames, None)    # no video -> blank frames
    replay_queue = ddc.ReplayQueue(detections_list, auto_advance=True)

    drone = ddc.run_replay(cfg, replay_queue, frames, base_ns)

    assert replay_queue.qsize() == 0, "replay should consume every frame"
    assert drone.commands, "the control loop issued no commands at all"


def test_replay_leaves_no_monkeypatches_behind(synthetic_log):
    """run_replay patches drone_controller's module globals (DroneMover, time). If it
    doesn't put them back, it poisons every test that runs after it."""
    import drone_controller as dc

    before = (dc.DroneMover, dc.time)

    _config_dict, frames, base_ns = ddc.parse_log(synthetic_log)
    ddc.run_replay(
        ddc.load_replay_config(CONFIG_YAML),
        ddc.ReplayQueue(ddc.build_detections_list(frames, None), auto_advance=True),
        frames,
        base_ns,
    )

    assert (dc.DroneMover, dc.time) == before


def test_the_replayed_speeds_reach_the_control_loop(synthetic_log):
    """Speed-dependent code (e.g. the PD speed reduction) is the main reason this harness
    exists — on the bench everything sits at 0 m/s. Prove the telemetry actually lands."""
    cfg = ddc.load_replay_config(CONFIG_YAML)
    _config_dict, frames, base_ns = ddc.parse_log(synthetic_log)
    replay_queue = ddc.ReplayQueue(ddc.build_detections_list(frames, None), auto_advance=True)

    drone = ddc.run_replay(cfg, replay_queue, frames, base_ns)

    speeds = [
        math.dist((0, 0, 0), tuple(t["odometry"]["velocity_body"].values()))
        for t in drone.telemetry_seen if t.get("odometry")
    ]
    assert max(speeds) > 30.0, "the whole speed range should have been replayed"
