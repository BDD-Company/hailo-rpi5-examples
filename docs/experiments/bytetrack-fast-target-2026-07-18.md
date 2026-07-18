# ByteTrack config validation for a FAST-moving target (file input)

Date: 2026-07-18
Branch: `feat/bytetrack-fast-target` (off `main`)
Box: `bdd-sd9-testrig1` @ 100.91.134.66 (the box formerly Tailscale-named `-mandarin`;
hostname still `bdd-sd9`). File input — the camera is bypassed, so the old
"frozen camera display" caveat does not apply.

## Goal

Validate the ByteTracker config for a **fast-moving** target using file input, and
tune the config so the controller:

1. Locks correctly onto the true ball in both the clean and the noisy clip.
2. Stays locked in the noisy clip with a minimal number of switches onto distractor
   balls.
3. Re-locks onto the true target quickly after any loss/switch.

Scope: **config params only** (the `bytetrack:` block in `config.yaml`). No tracker
or controller code changes. If config alone cannot meet the targets, report the
floor and what a code change would buy — do not make one.

## Test data

- `sky_red_ball_10-200px_10m_FAST.mp4` — clean, no noise. Reference trajectory.
- `sky_red_ball_10-200px_10m_FAST_W_NOISE_0.75.mp4` — same footage + distractor balls.

Both: 4K (3840×2160), 30 fps, 600 s, 18000 frames, MPEG-4 Simple Profile, no B-frames.

**Downscaled to 1280×720 H.264 baseline (no B-frames), 30 fps** before use, because:
- The pipeline runs at 1280×720 (`app_base.py: video_width=1280, video_height=720`),
  which is also the camera's native capture resolution — so a 1280×720 source is the
  *most faithful* input (the detector sees exactly the 640×640 it derives from 1280×720,
  same as live capture). Feeding 4K would just make the pipeline do the same downscale.
- Pi 5 has no hardware decode for 4K MPEG-4; software-decoding 4K at 30 fps risks
  starving the paced source. 720p decode is trivial.
- Baseline profile / no B-frames keeps `identity sync=true` pacing sound (PTS==DTS),
  per the file-input fix ([[file-input-domain-gap]]).

## Success targets

Measured against exact per-frame ground truth (clean run = truth):

- Clean: ~100% on-true, 0 false switches.
- Noisy: **≥95% on-true**, **≤2 false switches** over the run, **re-lock ≤15 frames**
  (~0.5 s) after the true ball reappears.

## Method — capture → offline sweep → on-box confirm

The pre-tracker detection stream is params-independent, so we capture it once per
video and sweep tracker params offline against the *real* `BYTETracker`
(100% id-reproducible vs on-box, validated by the prior eval).

1. **Capture (2 on-box passes, ~10 min each).** Deploy `main`-equivalent app to the
   box. Run each 720p video once:
   `--input <file> --vision-only --no-record --tiles 1x1`, log `!!! RAWDETS`.
   Verify each pass is healthy (~30 fps, low drops) before trusting it. `--tiles 1x1`
   gives a deterministic whole-frame stream so clean/noisy align frame-exact.
2. **Ground truth.** `groundtruth.py`: clean run's detection at frame F = true ball at
   F; `find_shift` corrects the ±1-frame skew; noisy dets labelled true/noise by
   proximity (size-scaled tolerance).
3. **Offline sweep** (`sweep_gt.py`, FAST cases) over the config-only levers that bite
   for a fast target:
   - `nms_dist_thresh`: {0.06, 0.04, 0.02, 0.015} — confirm the 0.06→0.02 fix holds.
   - `recovery_max_dist`: {null, 0.05, 0.1, 0.15} — Stage-1.5 re-acquire by last raw
     position; the key knob when a fast ball outruns its Kalman prediction.
   - `match_max_dist`: {0.15, 0.2, 0.3} — association gate vs large per-frame jumps.
   - `track_buffer`: {30, 60, 90} — keeps the true id alive across detector blackouts.
   - Re-verify `track_thresh`/`det_thresh`/`match_thresh` are inert before trusting.
   Metrics per case: on-true %, on-noise %, lost %, FALSE (true→noise) switches,
   fragmentations, and re-lock latency (frames from a true→noise/loss to back-on-true).
4. **On-box confirm (1–2 passes).** Put the winning params in `config.yaml`, re-run the
   noisy video on-box, confirm the harness reproduces the same track ids and the
   metrics hold under real box execution.
5. **Commit** the winning `bytetrack:` block + this writeup to the branch.

## Tooling

`docs/experiments/bytetrack_eval/`: `harness.py` (real BYTETracker over RAWDETS +
controller-lock replay), `groundtruth.py` (exact labels + `find_shift`),
`sweep_gt.py` (param sweep vs ground truth). Ported from the prior
`feat/bytetrack-locking-eval` work; API-compatible with `main`'s `bytetrack.py`.

## Known traps (from the prior eval)

- **First pass only.** `buffer.offset` counts through the loop rewind, so ids ≥ clip
  frames are a second pass over the same footage. Keep the run to one clip length
  (18000 frames) and filter to `id < nb_frames`.
- **±1 frame skew** between clean and noisy despite identical cuts — `find_shift`
  detects it; a small skew hides inside any tolerance and biases every label.
- Track-lifespan heuristics are inferior to exact labels; use `groundtruth.py`.

## Results

(To be filled in after the sweep + confirm.)
