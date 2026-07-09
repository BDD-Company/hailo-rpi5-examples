# Notes on how to read list

Here in this list priority is signified by number of exclamation signs BEFORE the item name, like "! task 1" is more urgent/imporant than "task 2" and less important than "!!! task 3".

Tasks that are already done are marked as `+`.
`#` is a comment, as usual.

# List of tasks

- !!! on RPI box 2 cores are used in realtime mode, hence the whole system only uses two remaining ones. Consider either disable realtime mode for those 2 cores OR pin our app to the those 2 realtime cores so it doesn't conflict with anything else. FIRST try to pin our app, that's a minor fix in bdd.ssh and it is easily verifieable.
- !!! PD coeff: make P inverse-propoprtional to speed, i.e. starting from 100kmh reduce p, higher speed --> smaller p. (there should be a coeff)

+ !! break configuration into the config-file (YAML) and run-time config object (dataclass). Parsing of the config file and validation of the fields must be based on types of the fields of dataclass.
- Figure out how to test the code on CI/CD ti make sure that merged PRs are viable.
- break code into proper modules, so the directory structure of "BDD" folder is not flat, but split into the 'src', 'tests', 'utils'
+ !!!! merge other two-camera and latency related commts from `detection-latency-bench-2026-06-03` branch.
+ !!! logs and videos percistency -- both should survive power down event so that whatever debug information we have after a crash is not lost. It is Ok to change video encoder/file format.
- !! implement noise reduction on target position log and estimation
- !! UI for launches
- !! figure out video capture without cropping (i.e. capture full frame)
- ! right now probes are commented out, add CLI argument that enables them (default OFF)
- ! rework tiling mode-switch conditions & handover robustness (see tiling_switch_rework below) — PR#9 review #3 handover-abort DONE (switch_tiling reverts to the working branch on a dead-branch timeout); #4 serialization DONE 2026-07-09 (all switches go through tiling_policy.TilingSwitchCoordinator; `--plus-one`/`enable_plus_one()` deleted). STILL OPEN: #4c, the post-switch watchdog for a branch that warms up and then dies.
- ! CONTROL-PATH FRAME TRUST (robustness review 2026-07, item #1, deferred): is `Detections.frame` — the image `OpticalObjectInfo` segments in drone_controller.py — stale w.r.t. its own detections? Commit 3bc99a6's stated root cause ("the aggregator reorders the bypass image stream while the metadata stays in order") cannot be literally true: back then ONE tee fed both the recorder and the callback, and a tee fans buffers to all src pads synchronously and in order. Leading theory is pooled bypass memory recycled under the then-120-deep recording queue, which the callback never saw because get_numpy_from_buffer copies synchronously inside the pad probe — that would mean the control path was always fine. Until settled, `optical_refinement` (ON by default) is on unverified pixels. Step 0 costs nothing: watch an existing `_DEBUG/debug_*.mkv` and see whether the bbox tracks the ball or lags it, and grep a run log for `Dropping out-of-order/duplicate frame`. Full context + the decisive experiment: memory `handoff-aggregator-bypass-image`.
- ! SIGINT does not stop the app. `shutdown()` (app_base.py:141) runs its callbacks, but the GLib main loop never quits, so `app.run()` never returns, `main()` never reaches `Done !!!`, and the operator must `kill -TERM` then `-9`. Re-confirmed on-rig 2026-07-09. Consequences: (a) the main pipeline's `splitmuxsink` cannot finalize the **last** `RAW_*.mkv` segment; (b) `main()`'s bounded `output_thread.join()` is unreachable in practice. Since 2026-07-09 the `debug_*.mkv` overlay recorder DOES finalize at SIGINT (its thread takes a sentinel from a shutdown callback, exits, and runs `sink.stop()` -> EOS -> NULL; verified: last segment ffprobes clean and decodes while the process still hangs). Fix: make `shutdown()` actually quit the GLib loop, then verify `Done !!!` prints and the last RAW segment ffprobes clean. NOTE for debugging: during teardown the process releases /dev/hailo0 and /dev/video0 while still alive, so `fuser /dev/hailo0` reports "free" for a live process — use `kill -0 <pid>`.
- rotate camera 90 degrees (to maximize lead without leading visual)
- deprecated CLI alias `--switch-test-s` (renamed to `--test-switch-s` on 2026-07-09): DELETE the alias and `_pop_deprecated_value()` after **2026-09-30**. Kept so runbooks/skills pinned to older checkouts keep running. Test/verification-harness flags are `--test-*`; run-mode flags (`--DEBUG`, `--no-record`, `--preview`, `--tiles`, `--switch-tiles`, `--vision-only`) are not.
- NV12 capture assumes `stride == width` (low priority — unreachable on the current rig). `picamera_thread` takes the NV12 path via `request.make_buffer("main")`, which returns the raw ISP buffer INCLUDING row padding; picamera2 aligns stride (typically 32/64 B). At imx477 @ 1280 wide the stride IS 1280, so it is correct today, and we now only use imx477. Any width that is not stride-aligned (e.g. an imx708 1296-wide mode) would push padded rows into an appsrc declaring `width=W` -> sheared image and `len(frame_bytes) != W*H*3/2`. Fix when a new sensor/mode lands: read `config['main']['stride']` and either assert equality at startup (fail fast, it is pre-launch) or de-pad. The comment above the `make_buffer` call still describes `make_array` and is wrong.
- `platform_controller.py` is STALE — mark for removal. It reads the same `Detections` queue as `drone_controller` but never got the `FrameOrderGuard` defense-in-depth added in 2b0d457, and line ~298 binds `frame` and never uses it (a trap now that `.frame` is a `Frame`, not an ndarray). Do not add features to it; delete once nothing imports `platform_controlling_thread`.
- `test_OverwriteQueue.py::test_fifo_when_not_overwritten` fails in the FULL suite and passes in isolation — pre-existing on `origin/main` (verified 2026-07-09), not a regression. It asserts strict FIFO with `maxsize=128` while a producer races a consumer; leaked daemon threads from earlier tests starve the consumer, the deque overwrites the oldest, and `len(out) == N` fails (~8700-9100 of 10000). Either bound the producer (semaphore) or assert "no reorder among what survived" instead of "nothing dropped".
- consolidate the near-duplicate bench configs config.test-single-imx477.yaml / config.test-single-imx477-nv12.yaml (PR#9 review #8, low priority): they differ only by `video_format`, which main() now overrides from the model input via detect_hef_video_format(), so the two are functionally identical when pointed at the same HEF. Parametrize or drop one; also propagate new keys like `record_videos` into the remaining test config(s).

## Experiments

+ !!! camera Stage-A latency (see camera_Stage-A_latency) reduction (sensor->appsrc)
  # DONE 2026-06-23 (branch camera-stage-a-latency): config-driven exposure pin + buffer_count(floor 2);
  # measured Stage-A p99/max ~halved (41->22ms) on bench. Fix 1 (drop surplus) deferred to the tiling task
  # (only pays off when inference < capture rate). See experiments/camera-stage-a-latency.md "Results".

- !! runtime tiling / branch switching (see runtime_tiling) on a shared Hailo-8: change tile size+count at runtime and/or switch between detection branches (same HEF on every branch) with minimal latency and frames lost.

## Further refinement of tasks

### camera_Stage-A_latency

the DOMINANT remaining latency (~135-170ms of ~234ms e2e; target 50-120ms) and the prerequisite that GATES the tiling task below. FULL HANDOFF in `experiments/camera-stage-a-latency.md` (read its TL;DR first) — do NOT lose details. Core fix: capture 30fps + drop surplus frames newest-first & CHEAPLY (request-level, skip make_array/numpy) + buffer_count 3->2 (NOT 1, breaks double-buffering) + pin exposure short (enables 30fps, needs enough light). Expected ~150ms -> ~60-70ms. Validate delivered-fps, detection-delay p50 AND p99, libcamera dropped-frame warnings.

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

- ! **#4c — post-switch watchdog (still open, the remaining piece of handover robustness).**
  `switch_tiling` reverts when the incoming branch produces no buffer within the warmup
  timeout, but nothing recovers from a branch that warms up fine and dies *afterwards*:
  app_callback stops firing, DET FPS -> 0, the control loop starves. Add a watchdog that
  flips back to whole-frame if N callbacks are missed after a switch that initially
  succeeded. Deferred deliberately — it is a new failure-detection mechanism, not part of
  the serialization fix.