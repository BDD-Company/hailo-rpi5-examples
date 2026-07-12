# HANDOFF: `debug_drone_controller.py` is stale and cannot run

Written 2026-07-12, while verifying the speed-dependant PD coeff feature.
Status: **BROKEN, unfixed.** Not caused by that feature — it was already dead.

## TL;DR

`debug_drone_controller.py` replays a recorded flight through the real
`drone_controlling_thread`. It is our only way to exercise the control loop off-rig, and
for anything speed- or telemetry-dependent it is the ONLY way at all (you cannot fly the
rig on the bench, and at 0 m/s most of the interesting code never triggers).

It dies immediately on any log:

```
AttributeError: 'dict' object has no attribute 'DEBUG'
```

**The cause is one line.** `main()` still hands `drone_controlling_thread` the *old flat
config dict* scraped out of the log, but `drone_controller` was refactored to take the
`Config` dataclass:

```python
# debug_drone_controller.py:658 — in main()
drone_controller_module.drone_controlling_thread(
    "replay://mock",
    drone_config,
    replay_queue,
    control_config=dict(config_dict),   # <-- flat dict, e.g. {'pd_coeff_p': 4, ...}
    output_queue=output_queue,
)
```

```python
# drone_controller.py:278 — first thing the thread touches
DEBUG = control_config.DEBUG            # <-- expects Config, gets dict -> AttributeError
```

Everything *else* in the harness still works. I reused `parse_log`,
`build_detections_list`, `ReplayQueue`, `MockDroneMover` and `MockMonotonicNs` verbatim in
a throwaway driver, passed a real `Config`, and replayed 1655 frames of a real flight
through the real control thread with zero exceptions. **Only the config wiring is rotten.**

## Why it's not a one-line fix

The obvious patch — "eval the config out of the log into a real `Config`" — does not work.
The harness's config path is built end-to-end on the pre-refactor flat dict, in three places:

1. **`parse_log`** scrapes `!!! Config: ...` (regex `CONFIG_RE`, line 93) and `eval()`s it
   with `_eval_namespace()` (line 96). That namespace knows `XY`, `Rect`, `Detection`,
   `DistanceClass` — it does **not** know `Config` or any of its nested classes.
2. **The no-config-in-log fallback** (line ~516) hardcodes another flat dict
   (`'thrust_takeoff'`, `'estimation_use_3d'`, …) — also pre-refactor.
3. **`--params`** applies overrides with `config_dict.update(args.params)` (line 564), i.e.
   flat keys.

And the two log eras are not interchangeable:

| Log era | `!!! Config:` line contains | Eval-able? |
|---|---|---|
| **Old** (e.g. the 2026-04 UAE logs) | flat dict: `{'confidence_min': 0.4, 'pd_coeff_p': 4, ...}` | yes, but it's a **dict**, and its keys are flat (`pd_coeff_p`), not nested (`pd_coeff.p`) — there is no `Config` to build from it without a hand-written flat→nested translation table |
| **New** (post-refactor; `app.py:1039` still logs `!!! Config: %s`, but `config` is now a `Config`) | dataclass repr: `Config(confidence_min=0.4, ..., pd_coeff=Config.PDCoeff(...))` | **no** — verified 2026-07-12: the repr contains `<VelocityMethod.NUMPY_REGRESSION: 'numpy'>`, which `eval()` cannot parse. Adding `Config` to the namespace is necessary but not sufficient |

So "fix the eval namespace" gets you a `SyntaxError` on new logs and a flat dict on old ones.

## Recommended fix

**Stop reconstructing the config from the log.** Load it the same way the app does:

- Add a `--config PATH` argument defaulting to `config.yaml`, and build the real object with
  `parse_config.loads_config(Config, text)`.
- Keep `--params` as overrides, but apply them to the YAML text / parsed object, not to a
  flat dict.
- Log a loud warning that the replayed config is **not** the config the flight actually flew
  with (the old flat dict in the log is the only record of that, and it is not mechanically
  convertible). For most debugging this is fine and often what you want — you are usually
  replaying an old flight *against today's config* to see what today's code would have done.
- Delete the hardcoded flat fallback dict (item 2 above) — it is dead weight and wrong.

Gotcha when loading `config.yaml` on a dev host: `inference.hef_model_path` has an
`ExistingFile` constraint and the HEF only exists on the Pi, so a bare
`load_config(Config, 'config.yaml')` raises `ConfigError`. Either point `--config` at a bench
config, or string-replace the HEF path before parsing (what I did).

Optional, while in there: `main()` also needs a display. It uses `InteractiveDisplaySink`
(OpenCV `imshow`); there is no `xvfb` on the dev box, so headless runs need
`QT_QPA_PLATFORM=offscreen`, and `--autoplay` still blocks on the advance event unless
`ReplayQueue.get` is wrapped to self-advance. Consider a `--headless` mode that skips the
sink entirely.

## Proof the rest of the harness is sound

Working driver (ran headless, 1655 frames, exit 0, zero exceptions). This is the shape the
fixed `main()` should have:

```python
import debug_drone_controller as ddc     # stubs mavsdk on import
import drone_controller as dc_mod
from parse_config import loads_config
from config import Config

# 1. REAL config — the one thing main() gets wrong
text = Path("config.yaml").read_text().replace(HEF_PATH, "config.yaml")
cfg = loads_config(Config, text)

# 2. everything below is the harness's own code, unmodified
config_dict, frames, base_ns = ddc.parse_log(log_path)
detections_list = ddc.build_detections_list(frames, None)   # None video -> blank frames

mock_monotonic = ddc.MockMonotonicNs(base_ns)
mock_drone = [None]

def on_frame(frame_id, idx):
    fd = frames.get(frame_id, {})
    mock_monotonic.set_frame(fd.get("timestamp_ns", base_ns))
    if mock_drone[0]:
        mock_drone[0].set_frame_data(frame_id, fd.get("telemetry", {}))

replay_queue = ddc.ReplayQueue(detections_list)
replay_queue.set_on_frame_callback(on_frame)

_orig_get = replay_queue.get                      # autoplay: don't wait for a keypress
replay_queue.get = lambda timeout=None: (replay_queue.advance(), _orig_get(timeout))[1]

dc_mod.DroneMover = ddc.MockDroneMover            # capture the instance in __init__
mock_time = types.ModuleType("mock_time")         # freeze the clock to log timestamps
for a in dir(real_time):
    if not a.startswith("_"): setattr(mock_time, a, getattr(real_time, a))
mock_time.monotonic_ns = mock_monotonic
dc_mod.time = mock_time

dc_mod.drone_controlling_thread(
    "replay://mock",
    {"upside_down_angle_deg": 130, "upside_down_hold_s": 0.2},
    replay_queue,
    control_config=cfg,                           # <-- Config, not dict
    output_queue=OverwriteQueue(maxsize=200),
)
```

Run with `QT_QPA_PLATFORM=offscreen`.

## Why you want this working

It is a genuinely good verification surface, and it caught real things:

- Replaying `/media/BDD/_BACKUPS/UAE/2026-04/2026-04-27/_DEBUG_13/BDD_20260427-160029.log`
  (1655 frames, speeds p50 10.4 / p90 30.6 / **max 39.1 m/s**) exercised the speed-dependent
  P reduction across its whole range — impossible on the bench.
- Module-level helpers can be fault-injected by monkeypatching the module attribute (the loop
  looks them up as globals at call time). Killing `dc_mod.speed_from_telemetry` mid-dive
  proved the last-known-speed fallback holds a reduced P instead of snapping back to full gain.
- A/B by flipping one config flag gives a clean control: with the feature off, P was a flat
  constant, so every variation was attributable to the change.

Good replay logs live under `/media/BDD/_BACKUPS/UAE/2026-04/2026-04-27/_DEBUG_13/`
(logs + matching `debug_*.mkv` fragments).

## Also worth knowing

`main()` is unreachable-by-design in places anyway: it builds an `InteractiveDisplaySink` and a
`debug_output_thread` before it ever reaches the control thread, so any fix should be tested
both with and without a display.

See also memory `replay-flight-log-through-control-loop`.
