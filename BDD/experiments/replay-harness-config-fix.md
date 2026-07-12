# Making the replay harnesses load a real `Config`

Branch: `fix/replay-harness-real-config`. **Done 2026-07-12.**
Supersedes the diagnosis in [debug-drone-controller-stale-handoff.md](debug-drone-controller-stale-handoff.md).

## How to use it now

```bash
cd BDD
python debug_drone_controller.py <log> --headless                     # replay, no display
python debug_drone_controller.py <log> --headless \
    --params "{'pd_coeff.speed_reduction.enabled': False}"            # A/B control
python debug_drone_controller.py <log> --video <RAW_*.mkv>            # step through frames
```

`--params` keys are DOTTED (`pd_coeff.p`, `thrust.max`), take the same values the YAML does
(`pd_coeff.p` needs `[x, y]`, not a scalar), and are validated by the real parser: an unknown
key is an error, not a silent no-op. `--config` defaults to `config.yaml`.

`debug_app_callback.py` takes the same flags, but additionally needs the GStreamer
introspection typelibs on a dev host (`GstApp`, from `gir1.2-gst-plugins-base-1.0`) — it
imports `app.py`. `debug_drone_controller.py` does not, and is the one to reach for.

## Outcome

Both tools replay clean, headless, on the whole corpus: exit 0, zero exceptions, and the dive
issues 1646 commands through *either* tool (they agree). The speed-dependent P is exercised
across its full range — 157 distinct P values on the dive, collapsing to 2 with
`speed_reduction.enabled: False`, which is the clean A/B control the handoff wanted. The
1655-frame dive replays in ~3 seconds headless.

**Running it found three more dead things that static reading did not.** All three were the
same species as the original bug — a hardcoded mirror of something that moved on:

1. `debug_app_callback`'s `pipelines` stub enumerated six symbols; `app_base` had grown a
   seventh (`QUEUE`). Because the stub shadows the real `pipelines.py`, the import died before
   `main()` with a baffling `cannot import name 'QUEUE' from 'pipelines' (unknown location)`.
   Now a PEP 562 `__getattr__` that answers to any name — replay builds no GStreamer pipeline,
   so nothing imported from there is ever called, and it cannot rot this way again.
2. `seen_frames` no longer exists in `app.py` (`FrameOrderGuard` replaced the last-10 dedup
   set). A fresh `app.frame_order_guard` is now rebound per replay: a replay restarts at frame
   0, and a stale guard would reject every frame as a reorder.
3. The ByteTracker was built from flat `bytetrack_*` log keys with `.get(default)` — so it
   never crashed, it just silently tracked with defaults. The nested `bytetrack` section has
   **nine** parameters where the scraper knew six: `recovery_max_dist`, `nms_thresh` and
   `nms_dist_thresh` were never passed at all.

The lesson for the next harness: `.get(key, default)` against a config that has moved on is
worse than a crash. The crash is what got `debug_drone_controller` fixed; the silent default is
what let `debug_app_callback` lie for months.

---

Everything below is the plan as written before implementation. It held up, except that
`debug_app_callback` needed the three extra fixes above.

## Problem

`debug_drone_controller.py` and `debug_app_callback.py` both replay a recorded flight
through the real `drone_controlling_thread`. Both die instantly:

```
AttributeError: 'dict' object has no attribute 'DEBUG'
```

Both still scrape a **pre-refactor flat config dict** out of the log and hand it to
`drone_controlling_thread`, which was refactored to take the `Config` dataclass
(`drone_controller.py:278`: `DEBUG = control_config.DEBUG`).

The handoff covers `debug_drone_controller`. `debug_app_callback.py:666` has the identical
bug and the identical dead fallback dict; it is fixed here too.

## Three caveats the handoff missed

1. **`parse_log` is shared.** `debug_tracker.py:340` and `debug_app_callback.py:503` both
   unpack `config_dict, frames, base_ns = parse_log(...)`. Its signature must NOT change.
   The fix therefore *ignores* its `config_dict` rather than reshaping it. `debug_tracker`
   keeps working untouched.
2. **`debug_app_callback` is stale in a second way.** It builds `BYTETracker(**bytetrack_config)`
   from flat `bytetrack_*` keys popped out of the log. `Config` now has a nested `bytetrack`
   section with `tracker_kwargs()` (`app.py:865`). Silent staleness — `.get(default)` means it
   never crashed, it just tracked with wrong parameters.
3. **The HEF gotcha can't be solved with `--config`.** The handoff suggests pointing `--config`
   at a bench config, but all three checked-in `config.test-*.yaml` files also reference
   `/home/bdd/models/...`, absent on a dev host. `inference.hef_model_path` is an `ExistingFile`,
   so *every* config raises `ConfigError` here. The harness must handle this itself.

## Design

### One shared helper, in `debug_drone_controller.py`

`debug_app_callback.py` already imports `parse_log`, `find_files_in_dir`, `MockDroneMover`,
`MockMonotonicNs` from it. No new module.

```python
def load_replay_config(path: Path, overrides: dict | None = None) -> Config
```

1. `yaml.safe_load(path.read_text())` → plain dict.
2. **Stub un-openable inference files.** If `inference.hef_model_path` (or `labels_json`)
   does not exist on this host, repoint it at the config file itself and log a loud warning.
   Replay never runs inference; this is what makes the tool work on a dev box at all.
3. **Apply `overrides` as dotted paths** into the nested dict — `{'pd_coeff.p': 4,
   'thrust.max': 0.6}` — *before* validation.
4. `parse_config(Config, data, source=str(path))` → a real `Config`.
   Overrides are validated for free. An old flat key (`pd_coeff_p`) now fails loudly as an
   unknown key instead of silently doing nothing.
5. `replace(cfg, DEBUG=True)` — both tools always forced `DEBUG`; it is a runtime-only field
   (`metadata={'runtime': True}`), so it cannot come from the file.
6. Log a **loud warning**: the replayed config is the one named by `--config`, NOT the config
   the flight actually flew with. Print the raw `!!! Config:` line from the log next to it so
   the difference is visible. (The logged flat dict is the only record of what flew, and it is
   not mechanically convertible — see the handoff's log-era table.)

### `main()` in both tools

- New `--config PATH`, default `config.yaml` next to the script (mirrors `app.py:677`).
- `--params` keeps its `{...}` literal syntax but keys become **dotted paths**. Breaking, and
  deliberately so: the old flat keys are silently ineffective today.
- Delete the hardcoded flat fallback dict, the `!!! OVERWRITE config` block, and the
  `_stale_keys` pop list. All three are dead weight and wrong.
- `drone_config` = `asdict(config.drone.config)`, mirroring `app.py:1079`, replacing the
  hardcoded `{"upside_down_angle_deg": 130, "upside_down_hold_s": 0.2}`.
- `debug_app_callback` only: `BYTETracker(**config.bytetrack.tracker_kwargs())
  if config.bytetrack is not None else None`, mirroring `app.py:865`.

### `--headless`

- `ReplayQueue` gains `auto_advance: bool = False`; when set, `get()` does not wait on the
  advance event. (The handoff's throwaway driver monkeypatched `get`; a flag is the honest fix.)
- Headless skips the `InteractiveDisplaySink` and the `debug_output_thread` and passes
  `output_queue=None` — already guarded at `drone_controller.py:1305`. No display, so no
  `QT_QPA_PLATFORM=offscreen` needed.
- Headless does not load video unless `--video` is given explicitly: blank frames feed the
  controller fine and video decode is the slow part. If `optical_refinement` is enabled while
  frames are blank, warn — that path reads pixels and will not behave as it did in flight.
- Missing/unreadable video degrades to blank frames with a warning instead of `sys.exit(-1)`
  (project principle: degrade, never hard-fail).

### Testability

Extract from `main()`:

```python
def run_replay(config, frames, base_ns, detections_list, *, output_queue=None) -> None
```

It owns the monkeypatching (`DroneMover` → `MockDroneMover`, `time.monotonic_ns` → frozen
clock) and the `drone_controlling_thread` call. `main()` calls it; so does the test. This is
the unit the regression actually lives in.

## Verification

New `BDD/test_debug_drone_controller.py`:

1. `load_replay_config` stubs a non-existent HEF path (and warns).
2. Dotted overrides land on the nested `Config` (`{'pd_coeff.p': 4}` → `cfg.pd_coeff.p`).
3. An old flat key (`pd_coeff_p`) raises `ConfigError` — no silent no-op.
4. `DEBUG` is forced True.
5. **Headless smoke test**: a synthetic few-frame log → `parse_log` → `run_replay` through the
   *real* `drone_controlling_thread`, asserting it completes with no exception and issues drone
   commands. This is exactly the regression that rotted; it now fails in CI if it comes back.

Tests load the repo's real `config.yaml`, so they also fail if `config.yaml` and `Config` drift
apart again. They assert **structure, never values** — `config.yaml` carries local flight tuning
and its numbers change constantly.

Acceptance (manual, off-rig): replay the **whole standard corpus** headless, expecting exit 0
and zero exceptions on each. This same set is what every future control-loop change gets
verified against, so the fixed harness must handle all three regimes, not just the dive.

`/media/BDD/_BACKUPS/UAE/2026-04/2026-04-27/_DEBUG_13/`:

| Log | Frames | Speed p50 / p90 / max (m/s) | Why |
|---|---|---|---|
| `BDD_20260427-160029.log` | 1655 | 10.4 / 30.5 / **39.1** | Full dive — the only log that exercises the whole speed range |
| `BDD_20260427-155753.log` | 1080 | 11.4 / 17.2 / 21.2 | Second, independent flight — cruise speeds |
| `BDD_20260427-160006.log` | 98 | 0.3 / 0.6 / 0.7 | Near-stationary (the bench condition); fast smoke test |
| `BDD_20260427-155731.log` | 0 | — | **Junk, no frames.** Do not use |

Traps: (a) do NOT pass the `_DEBUG_13` directory — `find_files_in_dir` picks
`sorted(glob("*.log"))[0]`, i.e. the *empty* `155731` log, and globs videos from all three
flights together; pass an explicit log path. (b) Video stamps are off by one second from the log
stem (`155753.log` ↔ `debug_20260427-15575`**`4`**`_*.mkv`).

Then re-run `160029` with `--params "{'pd_coeff.speed_reduction.enabled': False}"` to confirm the
A/B control the handoff describes still works.

## Out of scope

- `debug_tracker.py` still reads flat `bytetrack_*` keys from the log via `parse_log`'s
  `config_dict`. It degrades to defaults rather than crashing, so it is left alone; noted here
  so it is not mistaken for correct.
- The `!!! Config:` line `app.py` writes is now a `Config` repr containing
  `<VelocityMethod.NUMPY_REGRESSION: 'numpy'>` and is not `eval()`-able. Making the app log a
  round-trippable config (e.g. YAML) is a separate, worthwhile change.
