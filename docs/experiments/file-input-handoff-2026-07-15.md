# HANDOFF — deterministic file-input for ByteTrack locking eval

Branch: `feat/bytetrack-locking-eval`  ·  Status: **blocked**, resume-ready.
Companion: [`bytetrack-locking-eval-2026-07-15.md`](bytetrack-locking-eval-2026-07-15.md).

## Why we want file input

The current measurement films a **looped video on a screen** with the RPi camera.
That has three problems the deterministic path removes:
1. Random loop phase → each run samples a different scene → locking numbers carry
   scene noise and aren't run-to-run comparable.
2. No exact ground truth for "which detection is the true ball" (we fall back to a
   track-persistence heuristic).
3. Not reproducible after the display state changes (it was even *frozen* once —
   see [[two-boxes-same-hostname-live-vs-frozen]]).

Feeding the source `.mp4` directly fixes all three, and the **noise-free** video
(`sky_red_ball_10-200px_10m.mp4`) gives exact true-ball positions per frame — the
noisy videos share the same frame IDs (a `N/18000` counter is even burned into
every frame's corners).

## What already works

- The app accepts `-i /path/to.mp4` (source_type "file"); it builds
  `filesrc → qtdemux → decodebin → videoconvert(NV12) → 1280x720 → letterbox
  hailocropper → hailonet` — same NV12 format and downstream as the camera path.
- **Frame ids are deterministic**: for file input `normalized_frame_id` falls to
  `buffer.offset` (sequential 0,1,2,…). Clean and noisy runs therefore assign the
  **same id to the same video frame** → frame-exact alignment, even across drops.
- The pipeline **auto-loops** the file (`on_eos → rewind`).
- Per-frame `!!! RAWDETS n=K [(x1,y1,x2,y2,conf),…]` logging (all pre-tracker
  detections) is already committed in `app.py` — vision-only runs capture the
  full detection stream. Parse + analyze with
  `docs/experiments/bytetrack_eval/harness.py` (re-runs the real BYTETracker +
  lock offline, validated 100 % id-repro vs on-box).

## The two blockers

### 1. Detector domain gap (the hard one)
The HEF `2026-04-09_11n_ball_v3_nv12_h8.hef` was tuned on the **camera's view of a
laptop screen in a room**: screen bezel + desk visible, washed-out/desaturated
colors, the ball rendered coral-red (not the source's bright magenta). See
[`bytetrack_eval/handoff_assets/camera_view_of_screen.png`](bytetrack_eval/handoff_assets/camera_view_of_screen.png).

Measured detection rate (ball present every frame):

| input | detect rate |
|-------|-------------|
| live camera (same scene) | ~100 % |
| clean video 720p full-frame | ~0.5 % |
| clean video 1080p | ~5 % |
| clean video 2× center-zoom | ~0.4 % |

Ruled out: **size** (zoom didn't help; a big obvious ball at frame 151 still
`n=0`), **resolution**, and a quick desaturate/soft-contrast domain-match
(couldn't even be scored — see blocker 2). The pipeline graph is correct. This is
an appearance/training-domain gap, not a format bug.

### 2. Flaky H.264 decode / loop
Playback intermittently wedges at frame 1: `libav :0:: co located POCs
unavailable` → immediate `End-of-stream` → `on_eos()` rewind stalls. It is
**box-state-dependent, not clip-dependent** — a clip that ran 1401 frames earlier
later stalled at frame 1 twice in a row. So even in-domain camera *recordings*
(which would detect ~100 %) couldn't be fed back reliably.

## Resume plan (in priority order)

1. **Fix the detector domain gap — the unblocker.** Options:
   - Best: obtain / train a **domain-randomized or clean-frame** ball detector
     (HEF, HAILO8, NV12 320×640 input to match the pipeline). Then file input just
     works and gives exact ground truth. Confirm with `hailortcli parse-hef`
     (want `HAILO8` + `NV12`).
   - Cheaper probe first: verify it's really domain and not a solvable color issue
     — run the **color-range** test properly (it stalled last time). Encode with
     an explicit range/primaries and a decode-stable recipe (below) and compare
     detection. If forcing full-range NV12 lifts detection materially, it was
     color, not domain.
2. **Fix / sidestep the decode stall.** Once detection works, make playback
   reliable:
   - Try all-intra / small closed GOP: `-x264-params keyint=15:min-keyint=15:scenecut=0:open-gop=0:bframes=0 -profile:v baseline`.
   - Or a **reboot** of 192.168.8.89 first (the stall correlated with accumulated
     box state after many runs) — note: reboot wipes the tmpfs deploy, redeploy
     via `git archive`.
   - Or bypass the app's loop/rewind: feed a clip long enough (≥ the capture
     window) so no EOS/rewind happens during measurement.
3. **Ground-truth measurement** (once 1 & 2 done), first ~2 min only:
   - Run **clean** video → RAWDETS = true-ball position per frame_id (≈1 det/frame).
   - Run **W_NOISE** and **W_NOISE_0.5** → RAWDETS with noise.
   - Align by frame_id; label each noisy detection true/noise by proximity to the
     clean position at that frame_id → **exact** ground truth (no persistence
     heuristic).
   - Feed through `harness.py` (real BYTETracker + lock replay) → switch stats and
     the param sweep, now perfectly repeatable and comparable across noise levels.

## Concrete commands / recipes

Encode a Pi-decodable clip (the recipe that ran furthest; still not 100 % reliable):
```
ffmpeg -y -ss 0 -t 120 -i sky_red_ball_10-200px_10m.mp4 -vf scale=1280:720 \
  -c:v libx264 -preset veryfast -crf 20 -an clean_2min.mp4
```
Source videos: `/media/Pets/BDD/UTILS/sky_red_ball_10-200px_10m{,_W_NOISE,_W_NOISE_0.5}.mp4`
(4K, 30 fps, 600 s, frame counter burned in).

Feed it (deterministic, vision-only, whole-frame):
```
ssh bdd@192.168.8.89   # NOT the Tailscale bdd-sd9-mandarin (frozen display, different box)
scp clean_2min.mp4 bdd@192.168.8.89:/tmp/
tmux new-session -d -s bdd-app \
  "cd /home/bdd && exec bash hailo-rpi5-examples/scripts/bdd.sh -i /tmp/clean_2min.mp4 --vision-only --no-record --tiles 1x1"
# pull:  grep -aE 'RAWDETS|ByteTracker output' _DEBUG/BDD_*.log
```
Deploy the branch first (tmpfs is wiped on reboot): `git archive --format=tar HEAD
| gzip | ssh … tar -xzf`, plus scp the HEF to `/home/bdd/models/`. Camera-liveness
check no longer needed for file input, but confirm detection rate on the CLEAN
clip is high before trusting any noisy run.

## Key files
- `app.py` — RAWDETS instrumentation (committed on this branch).
- `docs/experiments/bytetrack_eval/harness.py` — real-tracker + lock replay from RAWDETS.
- `docs/experiments/bytetrack_eval/compare.py` — detection-stream + drop-rate stats.
- `docs/experiments/bytetrack_eval/sweep.py` — bytetrack param sweep.
- `docs/experiments/bytetrack_eval/handoff_assets/camera_view_of_screen.png` — the domain-gap evidence.
