# ByteTrack target-locking evaluation — 2026-07-15

Branch: `feat/bytetrack-locking-eval`

## Goal

Enable the app's ByteTrack tracker (`bytetrack.enabled: true`) and measure how
often the controller's locked target **switches from the one true target (the
pink ball that is always in frame) to a noise ball**. Per the brief, switching
*onto* the true target (acquisition/recovery) does **not** count — the displayed
video is looped and starts at a random phase, so the run may begin mid-noise.

## Setup

- Box: **192.168.8.89** (`bdd@`, hostname `bdd-sd9`, the one whose camera films a
  *live* looping scene). Note: there is a second box reachable over Tailscale as
  `bdd-sd9-mandarin` that shares the hostname but whose display was **frozen** —
  the first two runs landed there and were discarded (static ball ⇒ trivial 0
  switches). Always target the live box **by IP**.
- Model: `2026-04-09_11n_ball_v3_nv12_h8.hef` (verified **HAILO8 + NV12** ball
  detector; the red-plate model on `main` is HAILO8L/NHWC — wrong arch for this
  box and wrong target class for a ball scene).
- Run mode: `--vision-only --no-record --tiles 1x1`.
  - `--vision-only` because this box has **no USB flight controller** — the drone
    control thread crashes with `No viable USB devices found`. Vision-only runs
    the identical `app_callback` + `BYTETracker` and skips only the control loop
    (which we replay offline anyway).
  - `--tiles 1x1` (whole-frame) to isolate tracker behaviour from tiling-switch
    perturbations.
- Instrumentation (this branch): `app.py` logs a per-frame `!!! RAWDETS` line
  (all pre-tracker detections, xyxy+conf) so vision-only runs still capture the
  full detection stream for faithful offline analysis and param sweeps.
- ByteTrack params under test (current on-box config):
  `track_thresh 0.3, det_thresh 0.35, match_thresh 0.3, track_buffer 30,
  frame_rate 30, match_max_dist 0.2, recovery_max_dist None, nms 0.3/0.06,
  target_lock true`.

## Method

The controller's lock (`drone_controller.py`) works like this: `BYTETracker`
assigns stable `track_id`s; `_pick_target_detection` follows **only the locked
id** while it is visible, else nothing; the lock is **cleared after
`clear_estimator_history_after_frames` (=3) frames** with no confident pick, then
re-acquired as the highest-confidence detection. So a "target switch" is a lock
moving from id A to a different id B, almost always through a short loss gap.

Analysis is fully offline and **validated**: a harness re-runs the real
`BYTETracker` over the captured RAWDETS and reproduces the on-box track_ids
**100.0%** (RUN A 4758/4758, RUN B 6380/6380). It then replays the exact lock
state machine, counts A→B transitions, and classifies each by the **lifespan** of
the involved ids — the true ball is always present, so its track(s) are
persistent (>10 % of frames); noise balls are transient (tens of frames).

Two live runs were captured:

| run | duration | frames | distinct track ids | locked |
|-----|----------|--------|--------------------|--------|
| A   | ~163 s   | 4759   | 192                | 98.3 % |
| B   | ~216 s   | 6380   | 6380 → 48 ids      | 98.7 % |

## Results — switch statistics (with `target_lock`, current params)

| run | lock switches | **FALSE true→noise** | same-ball re-id (frag) | noise→true (benign) |
|-----|--------------:|---------------------:|-----------------------:|--------------------:|
| A   | 3 | **1** | 1 | 1 |
| B   | 1 | **0** | 1 | 0 |
| **total (~6.3 min)** | 4 | **1** | 2 | 1 |

- **The one FALSE switch (RUN A, frame 2276→2314) is a video-loop / scene-cut
  artifact.** The detector went **fully blank (n=0) for 39 frames**, then 6–8
  balls appeared at once scattered across the frame. The old true-ball track
  (id 1, last at bottom-left (0.27,0.84)) could not re-associate; the lock
  cleared and grabbed the highest-confidence ball — a transient noise ball
  (id 82, conf 0.79, lived 59 fr) — for ~2 s, then settled onto the true ball
  (id 90). Per the brief's looped-video caveat this is essentially the
  "random-start" case, not a genuine tracking failure.
- The other lock changes are the **true ball being re-id'd by ByteTrack**
  (id 1→90→191 in A; 1→11 in B) after detector dropouts — the lock stays on the
  same physical ball.

### The lock is doing almost all the work

Without `target_lock` (pick highest-confidence each frame), the *same footage*
switches away from the true ball constantly:

| run | true→noise switches (no lock) | rate |
|-----|------------------------------:|------|
| A   | 181 | 67 /min |
| B   |  82 | 23 /min |

`target_lock` collapses **263 → 1**. It is the single most important mechanism
for target stability in this scene.

### The real limiter is detector blackouts, not the tracker

Every lock disruption coincides with a stretch where the detector reports **zero
balls**, even though a ball is "always in frame":

| run | blackouts ≥5 fr | total blank frames | longest |
|-----|----------------:|-------------------:|--------:|
| A   | 5 | 88 (1.8 %) | 39 fr (~1.3 s) |
| B   | 3 | 86 (1.3 %) | 62 fr (~2.1 s) |

The tracker's job across a blackout is to **preserve the id** so the lock
re-lands on the same ball. Whether it can depends on where the ball reappears:
- RUN B: ball reappears **near** its last position → bridgeable (see below).
- RUN A: ball reappears **elsewhere** (scene cut) → not bridgeable by any tracker
  param, and a noise ball may be grabbed in the meantime.

## Offline param sweep (same RAWDETS, real BYTETracker, validated harness)

`#ids` = distinct track ids (lower = less fragmentation); `true%` = coverage of
the dominant/true track; `FALSE` = true→noise locks; `frag` = same-ball re-ids.

```
RUN A                                     #ids  true%  switch  FALSE  frag
baseline on-box (buf30,m0.3,mmd0.2)        192    46%       3      1     1
track_buffer 90                            147    46%       3      1     1
buf90 + match_max_dist 0.4                 105    78%       3      1     1   <- better re-assoc
buf90 + rec0.2 + mmd0.4                     100    78%       3      1     1

RUN B                                      #ids  true%  switch  FALSE  frag
baseline on-box (buf30,m0.3,mmd0.2)         48    50%       1      0     1
track_buffer 60                             44    50%       1      0     1
track_buffer 90                             43    98%       0      0     0   <- FIX
buf90 + rec0.2 + mmd0.4                      38    98%       0      0     0
```

## Hypotheses — ByteTrack params to tweak

1. **`track_buffer` 30 → ~90 (primary).** On-box `track_buffer=30` (1 s) is
   *shorter than the observed detector dropouts* (up to ~62 fr / 2 s), so every
   dropout expires the lost track and mints a **new id**, clearing the lock.
   Raising it to 90 (3 s) covers the dropouts: **proven on RUN B** — true-track
   coverage 50 %→98 %, fragmentation and the lock switch eliminated (1→0). Helps
   only when the ball reappears near its predicted path (fails across scene cuts,
   RUN A). Low risk; the only downside is a longer window in which a lost track
   could re-bind to a *different* ball in dense noise.
2. **`match_max_dist` 0.2 → ~0.4 (secondary).** Loosens the centroid gate so the
   *fast* ball (it crosses the whole frame; frame-to-frame IoU often ~0) stays
   associated. RUN A true-track coverage 46 %→78 %. Pair with (1). Density risk:
   a wider gate can also attach a nearby noise ball — validate on the rig.
   `recovery_max_dist: 0.2` adds a recovery association pass but gave little
   extra here.
3. **Not a ByteTrack knob, but dominant:** the blackouts are a **detection-recall**
   problem (whole-frame 320×640 loses a fast/small/edge ball for up to 2 s).
   Tiling (`--switch-tiles`) or a better model would cut blackouts at the source
   and remove most of the re-acquisition risk. Controller-side, the lock could
   **prefer re-locking a pre-existing persistent track over a freshly-spawned
   one**, and/or hold the lock through longer dropouts
   (`target_lost.clear_estimator_history_after_frames > 3`) so it coasts on the
   estimator instead of grabbing a noise ball right after a blackout — this is
   what would have prevented RUN A's one false switch.

## Applied change

`config.yaml`: `bytetrack.track_buffer 30 → 90` (offline-validated primary
recommendation; comment records the evidence). All other params unchanged.
`match_max_dist 0.4` is left commented as the secondary recommendation pending
on-rig validation.

**On-rig confirmation still owed:** the sweep is offline (faithful, 100 %
id-repro), but a fresh live run at `track_buffer=90` on 192.168.8.89 has not yet
been captured.

## Reproduce

```
cd docs/experiments/bytetrack_eval
python3 harness.py logs/runA.txt.gz "RUN A"   # per-run switch analysis
python3 sweep.py                               # param sweep, runs A & B
python3 compare.py "[('logs/runA.txt.gz','OLD-A'),('logs/runC.txt.gz','NEW-C')]"  # hardware compare
```

Runs A/B were captured on the box's **old CPU config (2 cores + an SC16IS752 IRQ
storm)**; runs C/D after the fix (see next section).

---

# Addendum — hardware comparison (2-core → 4-core), 2026-07-15

The box's CPU config was fixed between run sets: `isolcpus` dropped (2 → **4
usable cores**) and the SC16IS752 SPI-UART IRQ storm (which had eaten ~87 % of a
core) removed. I recaptured two live runs (C, D) with the identical setup and
compared the **detection stream** — the clean hardware signal — plus locking.

| run | cores | fps | frame-drops | blackout blank% | longest blackout |
|-----|-------|-----|-------------|-----------------|------------------|
| A | 2 | 27.4 | **2.70 %** | 1.8 % | 1.4 s |
| B | 2 | 28.4 | **1.47 %** | 1.3 % | 2.2 s |
| C | 4 | 30.0 | **0.04 %** | 0.5 % | 0.6 s |
| D | 4 | 30.0 | **0.00 %** | 2.2 % | 0.7 s |

- **fps** rises to the 30 fps camera cap (was 27–28, CPU-starved).
- **Frame drops** — captured frames that never reached inference — collapse from
  1.5–2.7 % to ~0 %. This is the cleanest, most robust hardware effect: the fix
  removed the CPU-pressure frame loss.
- **Worst-case blackout** (longest run of frames with zero detections) shrinks
  from 1.4–2.2 s to 0.6–0.7 s. Overall blackout % stays scene-dependent (D hit a
  busier segment).

**Locking counts are NOT directly comparable across the two sets** — the filmed
video loops at a random phase, so each run samples a different scene density
(C hit a crowded 6–13-balls/frame segment → 15 reacquisitions / 3 false; the
calmer A/B → 1–3 / 0–1). The switch metric is dominated by how many noise balls
happen to be present, which swamps the hardware effect. Treat the detection-stream
metrics above as the hardware verdict; use the offline param sweep (same footage,
params varied) for locking conclusions.

**Caveat that motivated the next step:** because the filmed scene is a looped
video at a random phase, neither the scene content nor the exact frame timing is
controlled between runs — so run-to-run locking numbers carry scene noise. That
is exactly what deterministic **file input** would remove.

---

# File-input methodology — attempted, blocked (see handoff)

To make measurements faithful and repeatable (and to get exact ground truth from
the noise-free video), we tried feeding the source `.mp4` to the app instead of
the camera. It is **blocked** by two independent walls — full write-up and
resume plan in
[`file-input-handoff-2026-07-15.md`](file-input-handoff-2026-07-15.md):

1. **Detector domain gap (fundamental).** The model was tuned on the camera's
   view of a *screen in a room* with washed-out colors
   ([handoff_assets/camera_view_of_screen.png](bytetrack_eval/handoff_assets/camera_view_of_screen.png)).
   Fed pristine digital frames it detects ~0.5 % of frames vs ~100 % on camera;
   zoom and resolution don't help.
2. **Flaky file decode.** Playback intermittently wedges at frame 1
   (`libav: co located POCs unavailable` → EOS → the app's loop-rewind stalls);
   box-state-dependent, not clip-dependent.

The file path *does* ingest `-i file.mp4` with **deterministic sequential
frame_ids** and auto-loops — so once a clean-domain detector (or in-domain
footage + a decode fix) is available, the ground-truth plan is ready to run.
