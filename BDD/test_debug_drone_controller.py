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


def test_the_ulog_trace_is_emitted_through_the_real_control_thread(synthetic_log):
    """The ulog trace only earns its keep if it is actually wired into the loop. Drive
    the real control thread and check the records that reached DroneMover.send_debug_array:
    right name, right format version, and a command that matches what the loop commanded."""
    from ulog_trace import (
        SLOT_CMD_ROLL, SLOT_CMD_PITCH, SLOT_CMD_THRUST,
        SLOT_HISTORY_0, SLOT_HISTORY_1,
        TRACE_FORMAT_VERSION, TRACE_NAME, FrameOutcome, unpack_history,
    )

    cfg = ddc.load_replay_config(CONFIG_YAML)
    assert cfg.ulog_trace is not None, "config.yaml should ship with the trace enabled"

    _config_dict, frames, base_ns = ddc.parse_log(synthetic_log)
    replay_queue = ddc.ReplayQueue(ddc.build_detections_list(frames, None), auto_advance=True)

    drone = ddc.run_replay(cfg, replay_queue, frames, base_ns)

    assert drone.debug_arrays, "the control loop emitted no ulog trace at all"

    for name, array_id, data in drone.debug_arrays:
        assert name == TRACE_NAME
        assert array_id == 0
        assert len(data) == 58
        assert data[0] == float(TRACE_FORMAT_VERSION)
        # A NaN here would be silently rejected by MAVSDK's JSON parser, costing the
        # whole message. Nothing non-finite may ever reach the wire.
        assert all(math.isfinite(v) for v in data)

    # The trace is rate-limited on the control loop's clock, which the replay drives from
    # the log's timestamps — so the count should track 5 Hz of FLIGHT time, not wall time.
    stamps = sorted(fd["timestamp_ns"] for fd in frames.values() if fd.get("timestamp_ns"))
    flight_s = (stamps[-1] - stamps[0]) / 1e9
    expected = flight_s * cfg.ulog_trace.rate_hz
    assert 0.5 * expected <= len(drone.debug_arrays) <= 1.5 * expected + 2, (
        f"{len(drone.debug_arrays)} traces over {flight_s:.1f}s of flight, "
        f"expected ~{expected:.0f} at {cfg.ulog_trace.rate_hz} Hz"
    )

    # Each record must summarise a real interval, not a single frame: at ~20 fps and
    # 5 Hz there should be several frames of history behind every record.
    histories = [
        unpack_history([d[SLOT_HISTORY_0], d[SLOT_HISTORY_1]])
        for _n, _a, d in drone.debug_arrays
    ]
    assert max(len(h) for h in histories) > 1, \
        "every record covered a single frame — aggregation is not happening"
    assert any(FrameOutcome.VIABLE in h for h in histories), "no viable frame ever recorded"

    # THE conservation law: every control-loop iteration must appear exactly once, either
    # in a count or as a VIABLE frame in a history. Frames lost BETWEEN records would be
    # invisible in the log, which is the failure this whole design exists to prevent.
    #
    # The only iterations legitimately missing are those of the final interval, which is
    # still sitting in the accumulator when the loop stops — it was never dispatched. (In
    # the case that actually matters, a crash, the process dies mid-interval and that tail
    # is lost regardless; there is nothing to flush it to.)
    counted = sum(d[2] + d[3] for _n, _a, d in drone.debug_arrays)
    viable  = sum(1 for h in histories for o in h if o is FrameOutcome.VIABLE)
    accounted = counted + viable
    tail = len(frames) - accounted

    assert tail >= 0, f"{accounted:.0f} iterations accounted for, but the loop only ran {len(frames)} — double-counted"
    assert tail <= max(len(h) for h in histories) + 1, (
        f"{tail} iterations went missing; only the final undispatched interval may be absent"
    )

    # The commanded attitude in the trace must be a command the loop really issued.
    commanded = {
        (round(roll, 4), round(pitch, 4), round(thrust, 4))
        for kind, _frame, roll, pitch, thrust in
        (c for c in drone.commands if c[0] == "move_to_target_zenith")
    }
    traced = {
        (round(d[SLOT_CMD_ROLL], 4), round(d[SLOT_CMD_PITCH], 4), round(d[SLOT_CMD_THRUST], 4))
        for _n, _a, d in drone.debug_arrays
    }
    traced.discard((0.0, 0.0, 0.0))     # "no command issued yet" — before the first one
    assert traced, "every traced command was empty"
    assert traced <= commanded, f"traced commands the loop never issued: {traced - commanded}"


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
