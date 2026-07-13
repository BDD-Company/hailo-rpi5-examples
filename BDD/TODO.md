# Notes on how to read list

Here in this list priority is signified by number of exclamation signs BEFORE the item name, like "! task 1" is more urgent/imporant than "task 2" and both less important than "!!! task 3".

Tasks that are already done are marked as `+`.
`#` is a comment, as usual.

# List of tasks

+ speed-dependant PD coeff — DONE 2026-07-12 (branch `feat/speed-dependant-pd-coeff`). P decays geometrically above a threshold: `P *= coeff ** ((speed - start_speed_ms) / speed_step_ms)`, speed = 3D magnitude of `odometry.velocity_body`. Applied LAST of all P modifications, then clamped into `[p_min, p_max]` — that final clamp now runs unconditionally, which also fixed a latent bug where the NEAR/MEDIUM `*= 1.1` boosts landed AFTER `pd_coeff_p_for_target_size()`'s clamp and could push P past `p_max`. Config: optional `pd_coeff.speed_reduction` (standard `enabled` flag; defaults 15 m/s / 0.947 / 1 m/s). A telemetry dropout REUSES the last known speed rather than reverting to full gain (holding the calmer P mid-dive is the safe side to fail on). 20 host tests + replayed against the 2026-04-27 UAE flight log (39 m/s peak): P tracked speed down to the p_min floor, and killing telemetry at 29.6 m/s held P at 1.806 instead of snapping back to 4.0. Remaining work is TUNING (coeff/threshold on the rig), not implementation.
- !!! implement noise reduction on target position and speed estimation
- !! targets below the drone must have lower priority than targets above the drone. Adjust for drone nose attitude, ideally anything below horizon should be considered a lesser priority target, especially at the early stages of the flight (1s). Make it configurable via .yaml: enabled/disabled switch, confidence multiplier (float, [0.5 .. 1], deafult = 0.8), early stage multiplier (float, [0.5 .. 1], deafult = 0.8), early stage duration in seconds ([0 .. inf], deafult = 1.0)
+ write extra logs into the px4's persistent ulog — DONE 2026-07-13 (branch `feat/px4-ulog-trace`). One MAVLink `DEBUG_FLOAT_ARRAY` named `BDD` per tick at 5 Hz (`ulog_trace:` config section, absent/`enabled: false` = off), which PX4 republishes as the `debug_array` uORB topic and writes into its own log. Carries format version, frame id, BOTH starvation counters, the post-clamp PD `p`, the commanded roll/pitch/thrust, and the selected detection — 14 of the 58 float32 slots. New `ulog_trace.py` (`UlogTraceSample` + `UlogTracer`); `DroneMover.send_debug_array()` + `last_attitude_command()`; two submit sites in `drone_controller` (after the command has gone out, so ZERO added capture→command latency; and from the STARVED path, so the trace keeps beating with a rising `frames_without_frame` when the vision pipeline dies instead of just falling silent).
  Four things only measurement revealed, each of which would have shipped as a bug:
  * **MAVSDK rejects NaN.** It serialises message fields as JSON, `json.dumps` emits a bare `NaN` token, and its own C++ parser then throws `Failed to parse JSON` — one NaN silently costs the ENTIRE message. The original design used NaN as the "absent" sentinel. Everything absent is now `0.0` (unambiguous anyway: a real command always carries non-zero thrust), and `to_floats()` clamps any non-finite value.
  * **`cmd_roll`/`cmd_pitch` are deg/SEC, not degrees**, whenever `use_set_attitude: false` (the rig default) — `DroneMover` forwards them to `set_attitude_rate`. The UAE dive replay shows ±235, absurd as an angle, normal as a rate. Named `cmd_roll`/`cmd_pitch`, never `..._deg`.
  * **No FC provisioning needed.** The rig's PX4 already has `SDLOG_PROFILE = 1057` (= 1024 + **32** + 1), so the debug bit is set. The app still checks and WARNS at startup, but never writes FC params. Note `SDLOG_MODE = 0` — PX4 logs only while ARMED, which is exactly the crash window.
  * The detection block goes LAST and absent = 0, so MAVLink 2's trailing-zero trim shrinks a no-detection tick: **76 B/msg measured, vs 264 B untrimmed** (~335 B/s at 5 Hz). The float16/bit-packing the task sketched is unnecessary — 58 slots is far more than the ~14 needed.
  Verified: 169 host tests (12 for the tracer, mutation-checked) on py3.12 AND the Pi's py3.11; a real `mavsdk_server` + fake PX4 decoding all 58 slots; the 39 m/s UAE dive replayed through the REAL control thread (366 traces / 82.7 s = 4.43 Hz, 0 non-finite); and the real Auterion PX4 on the rig (3/3 direct sends, 25 tracer sends, 0 failures over the 1 Mbps serial link). Design: `docs/superpowers/specs/2026-07-12-px4-ulog-trace-design.md`.
  Remaining: `frame_id` is a float32, so `+1` steps stop being distinguishable past 2^24 = 16,777,216 frames — ~65 days at 3 fps, and it resets each process start. Left as-is deliberately.
- Figure out how to test the code on CI/CD ti make sure that merged PRs are viable.
- break code into proper modules, so the directory structure of "BDD" folder is not flat, but split into the 'src', 'tests', 'utils'
- !! UI for launches
- !! figure out video capture without cropping (i.e. capture full frame)
- ! right now probes are commented out, add CLI argument that enables them (default OFF)
+ rework tiling mode-switch conditions & handover robustness (see tiling_switch_rework below) — ALL THREE DONE 2026-07-09: #3 handover-abort (switch_tiling reverts to the working branch on a dead-branch timeout), #4 serialization (all switches go through tiling_policy.TilingSwitchCoordinator; `--plus-one`/`enable_plus_one()` deleted), #4c post-switch watchdog (tiling_policy.BranchStallWatchdog reverts to whole-frame when an accepted branch later dies, with a cooldown against thrash). Remaining work here is tuning, not robustness.
- ! CONTROL-PATH FRAME TRUST (robustness review 2026-07, item #1, deferred): is `Detections.frame` — the image `OpticalObjectInfo` segments in drone_controller.py — stale w.r.t. its own detections? Commit 3bc99a6's stated root cause ("the aggregator reorders the bypass image stream while the metadata stays in order") cannot be literally true: back then ONE tee fed both the recorder and the callback, and a tee fans buffers to all src pads synchronously and in order. Leading theory is pooled bypass memory recycled under the then-120-deep recording queue, which the callback never saw because get_numpy_from_buffer copies synchronously inside the pad probe — that would mean the control path was always fine. Until settled, `optical_refinement` (ON by default) is on unverified pixels. Step 0 costs nothing: watch an existing `_DEBUG/debug_*.mkv` and see whether the bbox tracks the ball or lags it, and grep a run log for `Dropping out-of-order/duplicate frame`. Full context + the decisive experiment: memory `handoff-aggregator-bypass-image`.
- ! SIGINT does not stop the app. **ROOT-CAUSED 2026-07-09 (the old "the GLib loop never quits" explanation was WRONG).** `shutdown()` DOES quit the loop — it ends with `GLib.idle_add(self.loop.quit)` (app_base.py, in `shutdown`). The hang is *after* `loop.run()` returns, in the cleanup `for t in self.threads: t.join()`: `self.threads` are the **picamera producer threads**, created **without `daemon=True`** and given **no stop signal**. `picamera_thread` only leaves its loop when `picam2.capture_request(wait=capture_timeout_s)` raises `TimeoutError` — i.e. when the camera actually stalls. In a healthy run frames keep arriving, so it never exits and the join never returns. Evidence: the main thread's `/proc/<pid>/wchan` is `futex_wait_queue` (a join), not `poll`/`epoll` where `loop.run()` would sit.
  Two independent bugs ride along:
  (a) `shutdown()` drives the pipeline straight `PAUSED -> READY -> NULL` **without sending EOS**, so `splitmuxsink` never finalizes the last `RAW_*.mkv` fragment. That truncation has nothing to do with the hang.
  (b) `GStreamerApp.run()` ends in `sys.exit(0)`/`sys.exit(1)`, so `main()`'s tail — `print("Done !!!")`, `detections_queue.put(STOP)`, `action_thread.join()`, the `output_thread` drain — is **unreachable even on a clean exit**. The shutdown callbacks are the only teardown that ever runs. (This is why the `debug_*.mkv` overlay recorder had to be finalized from a shutdown callback; it works, verified on-rig — the last segment ffprobes clean and decodes while the process still hangs.)
  Fix sketch: give `picamera_thread` a stop `Event` set by `shutdown()` before the join (its `capture_request` already has a bounded wait, so it exits within a frame period); `join(timeout=...)` and log stragglers; send EOS and wait for it on the bus before going to NULL so `splitmuxsink` finalizes; and let `run()` return instead of `sys.exit` so `main()`'s teardown is reachable. Add a hard `os._exit` backstop — hailo VDevice teardown can deadlock on set-NULL mid-inference.
  NOTE for debugging: during teardown the process releases /dev/hailo0 and /dev/video0 while still alive, so `fuser /dev/hailo0` reports "free" for a live process — use `kill -0 <pid>`. Python 3.11 does not set OS thread names, so every `/proc/<pid>/task/*/comm` reads "Hailo Detection" (inherited from `setproctitle`) — comm is useless for identifying threads here; use `wchan`.
- rotate camera 90 degrees (to maximize lead without leading visual)
- deprecated CLI alias `--switch-test-s` (renamed to `--test-switch-s` on 2026-07-09): DELETE the alias and `_pop_deprecated_value()` after **2026-09-30**. Kept so runbooks/skills pinned to older checkouts keep running. Test/verification-harness flags are `--test-*`; run-mode flags (`--DEBUG`, `--no-record`, `--preview`, `--tiles`, `--switch-tiles`, `--vision-only`) are not.
- NV12 capture assumes `stride == width` (low priority — unreachable on the current rig). `picamera_thread` takes the NV12 path via `request.make_buffer("main")`, which returns the raw ISP buffer INCLUDING row padding; picamera2 aligns stride (typically 32/64 B). At imx477 @ 1280 wide the stride IS 1280, so it is correct today, and we now only use imx477. Any width that is not stride-aligned (e.g. an imx708 1296-wide mode) would push padded rows into an appsrc declaring `width=W` -> sheared image and `len(frame_bytes) != W*H*3/2`. Fix when a new sensor/mode lands: read `config['main']['stride']` and either assert equality at startup (fail fast, it is pre-launch) or de-pad. The comment above the `make_buffer` call still describes `make_array` and is wrong.
+ replay harnesses (`debug_drone_controller.py`, `debug_app_callback.py`) — FIXED 2026-07-12 (branch `fix/replay-harness-real-config`). Both handed `drone_controlling_thread` the pre-refactor FLAT config dict scraped from the log and died instantly with `AttributeError: 'dict' object has no attribute 'DEBUG'`. They now load a real `Config` from `config.yaml` via `load_replay_config()` (shared), never from the log — the logged config is a flat dict on old logs and a non-eval-able repr on new ones, and is now only echoed for comparison. `--params` overrides became DOTTED paths (`pd_coeff.p`) validated by the real parser, so an old flat key fails loudly instead of silently doing nothing. New `--config PATH` and `--headless` (no window, no video decode). `debug_app_callback` was stale in three further ways, all found only by running it: its ByteTracker came from flat `bytetrack_*` keys (the nested section has NINE params where the scraper knew six — `recovery_max_dist`/`nms_thresh`/`nms_dist_thresh` were never passed); its `pipelines` stub enumerated six symbols and `app_base` had grown a seventh (`QUEUE`), killing the import before `main()`; and `seen_frames` no longer exists (`FrameOrderGuard` replaced it). Guarded by `test_debug_drone_controller.py`, which drives the REAL control thread. Verified headless on all three UAE logs (0.7 / 21 / 39 m/s peak): exit 0, zero exceptions, 1646 commands on the dive from both tools. Note `debug_app_callback` needs GStreamer typelibs on a dev host (`gir1.2-gst-plugins-base-1.0`, for `GstApp`). Writeups: `experiments/replay-harness-config-fix.md`, `experiments/debug-drone-controller-stale-handoff.md`.
- `platform_controller.py` is STALE — mark for removal. It reads the same `Detections` queue as `drone_controller` but never got the `FrameOrderGuard` defense-in-depth added in 2b0d457, and line ~298 binds `frame` and never uses it (a trap now that `.frame` is a `Frame`, not an ndarray). Do not add features to it; delete once nothing imports `platform_controlling_thread`.
- `test_OverwriteQueue.py::test_fifo_when_not_overwritten` fails in the FULL suite and passes in isolation — pre-existing on `origin/main` (verified 2026-07-09), not a regression. It asserts strict FIFO with `maxsize=128` while a producer races a consumer; leaked daemon threads from earlier tests starve the consumer, the deque overwrites the oldest, and `len(out) == N` fails (~8700-9100 of 10000). Either bound the producer (semaphore) or assert "no reorder among what survived" instead of "nothing dropped".
- consolidate the near-duplicate bench configs config.test-single-imx477.yaml / config.test-single-imx477-nv12.yaml (PR#9 review #8, low priority): they differ only by `video_format`, which main() now overrides from the model input via detect_hef_video_format(), so the two are functionally identical when pointed at the same HEF. Parametrize or drop one; also propagate new keys like `record_videos` into the remaining test config(s).

## Experiments

- !! runtime tiling / branch switching (see runtime_tiling) on a shared Hailo-8: change tile size+count at runtime and/or switch between detection branches (same HEF on every branch) with minimal latency and frames lost.

## Further refinement of tasks

### write extra logs into the px4's persistent ulog
To analyze drone behaviour post-crash, when SD card might be severely damaged, write small traces of what was going on in the decision pipeline to the more robust and crash-tolerant ulog of px4. Things worth writing (at 5Hz?) rate:
  - format version
  - number frames with no detection (zero in case of detection)
  - Selected detection: x,y,w,h,confidence
  - computed value of p of pd
  - control decision: x,y,thrust
Ideally those must be tightly-packed, if values must be of uniform types, then float16 should be enough. If there is a way to log arbitrary binary, then there are even tighter packaging possible to reduce strain.

### speed-dependant PD coeff

make P inverse-propoprtional to speed, i.e. starting from 100kmh reduce p, higher speed --> smaller p.
This scaling must be applied last of all P modification, and final values MUST not exceed min and max values of P from config.
#### Config
It all must be configurable under a separate optional section in config.
Config field:
  - reduction starting speed meters per second: float [0..100] = 15
  - p reduction coeffecient: float [0..1] = 0.947
  - p reduction speed step meters per second : float [1..10] = 1

### Example
reduction start speed rs = 10 ms, drone speed ds = 20mps, reduction coefficient rc= 0.9 and reduction speed step rss= 1mps, P=10.
Then new value of P is computed as follows:
P *= rc ** ((ds - rs) / rss)
which is:
P = 10 * (0.9 ** ((20 - 10) / 1)) = 3.48
same but rss=2mps
P = 10 * (0.9 ** ((20 - 10) / 2)) = 5.9


### runtime_tiling : runtime tiling / branch switching

BLOCKED ON the camera Stage-A task above. FULL HANDOFF (architecture + switch-latency estimate + tile budget + Hailo-8-vs-8L + measurements) in `experiments/runtime-tiling-branch-switch.md` — read its TL;DR first. No prototyping yet. Next-session order:
  1. (prerequisite) the camera Stage-A latency task above — Fixes 2 & 3 (buffer_count, exposure pin/auto-pin) DONE on branch `camera-stage-a-latency`. STILL TO DO HERE: **Fix 1** = capture 30fps + drop surplus frames newest-first & CHEAPLY (request-level, skip make_array/numpy). Deferred to this task on purpose (only pays off once 2x2 tiling pushes inference below capture). See memory `fix1-drop-surplus-deferred-to-tiling`. Validate delivered-fps, Stage-A p50/p99, libcamera "no buffers" warnings, thermal.
  2. bench: do two same-HEF hailonets share one network group or two? (decides per-branch vs shared hailonet). Measure real t_inf/tile + 8L throughput ceiling (A-alone vs 2x2, read `detection delay`).
  3. consider model swap yolo11n -> yolov8n (faster on the existing 8L, free) before/instead of any Hailo-8 hardware upgrade.
  4. only then prototype: parameterize cropper builder in pipelines.py (whole `hailocropper` OR `hailotilecropper`) + valve per branch + input-selector; add switch_branch() 4-step hot handover in app_base.py. Realistic tile ceiling on current rig ~2x2.

### tiling_switch_rework : hot-switch policy + handover robustness (PR#9 review #3, #4)

The runtime whole<->tile switching landed (branch `nv12_tiling`), but two rough
edges to fix when we revisit the switch CONDITIONS (we'll rework them anyway, so
deferred, not patched inline):

- **#3 — warmup-timeout stranding the pipeline: FIXED via abort-and-revert.**
  `switch_tiling` used to log "switching anyway" on timeout and flip the selector
  onto the incoming branch that produced NO buffer while idling the working one ->
  app_callback stops firing, DET FPS -> 0, control loop starves (which on this rig
  can wreck hardware). Now (option a): on timeout it re-closes the incoming valve,
  leaves the selector + outgoing valve on the branch that IS producing, returns
  False; `_fire_switch` skips committed() and the policy backs off / retries. Still
  OPTIONAL (option c, not done): a post-switch watchdog that flips back to
  whole-frame if N callbacks are missed AFTER a switch that initially succeeded but
  then the branch dies.

+ **#4 — manual `--test-switch-s` desynced the policy and raced switch_tiling.** DONE 2026-07-09.
  Was: the harness thread called `app.switch_tiling()` directly and never updated
  `policy.tiling_on`, so with `auto_switch` ALSO on, the policy's belief went stale and
  it could never command the reverse switch (the rig latches onto the >200 ms tiling
  branch). And `switch_tiling` held no lock, so two callers could interleave their
  valve/selector `set_property` calls and leave the selector on a shut-valve branch.

  Fix, as designed and reviewed:
    * New `tiling_policy.TilingSwitchCoordinator` — the ONE entry point. `request()`
      is serialized, and updates the policy (`committed()` / `reset_streaks()`) on the
      way out. `note()` is the streaming-thread hot path and returns a target only when
      a switch should START, never while one is in flight (so no worker-per-frame).
    * TWO locks, never nested. `switch_tiling` blocks up to `warmup_timeout_s` (1 s) in
      `ready.wait()`, so one shared lock would stall the streaming thread ~30 frames per
      switch — worse than the desync. `GStreamerApp._switch_lock` (RLock) guards only the
      valve/selector handover; the coordinator's lock guards only policy state and is
      never held while `switch_fn` runs. Deadlock-free by construction.
    * `_tiling_active` is initialized and exposed as `GStreamerApp.tiling_active`.
    * Both drivers — the policy worker and the `--test-switch-s` thread — call
      `user_data.request_tiling()`. Nothing calls `app.switch_tiling()` directly anymore.
  Covered by 10 tests in `test_tiling_policy.py`, mutation-checked: reverting the lock,
  the `committed()` call, or the lock-scope each makes the intended tests fail.
  `--plus-one` / `enable_plus_one()` were deleted earlier the same day, removing a third
  unlocked caller.

+ **#4c — post-switch watchdog.** DONE 2026-07-09.
  Was: `switch_tiling` reverts when the incoming branch produces no buffer within the
  warmup timeout, but nothing recovered from a branch that warmed up fine and died
  *afterwards* — app_callback stops firing, DET FPS -> 0, the control loop starves, and
  nothing notices because every component downstream is *waiting* rather than failing.

  `tiling_policy.BranchStallWatchdog` polls the callback's frame counter from a daemon
  thread. If it freezes for `tiling.stall_timeout_s` (default 2.0) while TILING is active,
  it reverts to whole-frame — the branch that was known good at startup.
    * A handover in flight legitimately pauses delivery and does NOT count as a stall.
    * After a revert it arms a `tiling.stall_cooldown_s` (default 30) cooldown, because
      the policy otherwise rebuilds its lost-streak in ~0.7 s and dives straight back into
      the branch just proven dead. `note()` stops even asking during the cooldown; a direct
      `request(True)` is refused. Requests to WHOLE-FRAME are never blocked — the watchdog
      must always be able to get back to the known-good branch.
    * It deliberately does NOT act when WHOLE-FRAME is the stalled branch: that branch is
      fed by the same source tee and infers on the same hailonet, so switching to tiling
      cannot help. It logs once (not per poll) and leaves the pipeline alone.
    * If the revert itself fails, it reports failure honestly and retries after another
      timeout rather than latching.
  Fault injection `--test-kill-tile-after-s N` shuts `valve_tile` while the selector is on
  it, which is the only way to produce this failure on demand. 7 host tests, mutation-checked.