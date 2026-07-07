# Notes on how to read list

Here in this list priority is signified by number of exclamation signs BEFORE the item name, like "! task 1" is more urgent/imporant than "task 2" and less important than "!!! task 3".

Tasks that are already done are marked as `+`.
`#` is a comment, as usual.

# List of tasks

+ !! break configuration into the config-file (YAML) and run-time config object (dataclass). Parsing of the config file and validation of the fields must be based on types of the fields of dataclass.
- Figure out how to test the code on CI/CD ti make sure that merged PRs are viable.
- break code into proper modules, so the directory structure of "BDD" folder is not flat, but split into the 'src', 'tests', 'utils'
+ !!!! merge other two-camera and latency related commts from `detection-latency-bench-2026-06-03` branch.
+ !!! logs and videos percistency -- both should survive power down event so that whatever debug information we have after a crash is not lost. It is Ok to change video encoder/file format.
- !! PD coeff: make P inverse-propoprtional to speed, i.e. starting from 100kmh reduce p, higher speed --> smaller p. (there should be a coeff)
- !! implement noise reduction on target position log and estimation
- !! UI for launches
- !! figure out video capture without cropping (i.e. capture full frame)
- ! right now probes are commented out, add CLI argument that enables them (default OFF)
- ! rework tiling mode-switch conditions & handover robustness (see tiling_switch_rework below) — PR#9 review #3 handover-abort DONE (switch_tiling now reverts to the working branch on a dead-branch timeout); still open: #4 (--switch-test-s desyncs the policy + races the unlocked switch_tiling) and the optional post-switch watchdog.
- rotate camera 90 degrees (to maximize lead without leading visual)
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

- **#4 — manual `--switch-test-s` desyncs the policy and races switch_tiling.**
  The test thread calls `app.switch_tiling()` directly and never updates
  `policy.tiling_on`, so with `auto_switch` ALSO on the policy's belief goes stale
  and it can't command the reverse switch. `switch_tiling` also holds no lock, so
  the test thread and the policy worker can interleave valve/selector set_property
  calls -> selector can land on a closed-valve branch (stall). When reworking:
  route ALL switches (manual + policy) through ONE serialized entry point that
  updates policy.committed() and holds a lock; or simply forbid `--switch-test-s`
  together with `auto_switch`.