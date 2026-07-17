# RESOLVED — deterministic file-input for ByteTrack locking eval

Branch: `feat/file-input-fix` (off `main`) · Status: **working**, on-rig verified 2026-07-17.
Companion: [`bytetrack-locking-eval-2026-07-15.md`](bytetrack-locking-eval-2026-07-15.md).

Supersedes the 2026-07-15 handoff, which recorded this as blocked on two things. Both
were the same single bug and neither original diagnosis was correct. The file-input path
now detects the ball on **99%** of frames, drops **zero** frames, and gives exact
per-frame ground truth.

## What was actually wrong

Nothing paced the file source. `filesrc` raced to EOF — a 12MB clip in ~60ms — so the
`leaky=downstream`, 1-deep decode queue shed **compressed** H.264 packets, gutting
reference frames, and `decodebin` emitted torn partial pictures. Both reported blockers
fall out of that:

- **"Detector domain gap" (~0.5% detection).** There is no domain gap. The detector was
  scoring torn frames. Paced, the same HEF on the same clip detects 99% — level with the
  ~100% it gets off the live camera. The training-domain story, the camera-vs-source
  colour difference, the resolution and zoom experiments: all chasing an artifact.
- **"Flaky H.264 decode / loop."** EOS arrived ~60ms in, `on_eos` rewound, and it
  repeated: **13 EOS/s for an entire 80s run**, seek-thrashing the pipeline while the
  queues fed inference a mix of overlapping passes. That is what "wedges at frame 1 then
  EOS" was. It was never the codec, which is why it looked box-state-dependent rather
  than clip-dependent.

Measured on the clean clip, before → after pacing:

| | before | after |
|---|---|---|
| detection rate | 0.5% | **99.0%** (camera ≈100%) |
| frames/s | ~64 | 29.8 (the clip's true 30fps) |
| frames dropped | ~18% | **0** (ids 0..2383, contiguous) |
| EOS per 80s run | 1075 | 0 |
| rewinds per 145s run | ~2000 | 1 (one clean loop at 120s) |

## The fix

`identity sync=true` on the file source, in `get_source_pipeline_string`
([`BDD/pipelines.py`](../../BDD/pipelines.py)). Placement is load-bearing in both
directions and each alternative was measured, so don't "tidy" either:

- **Before the decode queue, not after `decodebin`.** That queue is leaky and 1 deep, so
  pacing downstream of it still sheds compressed packets → 0 frames in 80s.
- **At the source, not the sink.** The whole tail is deliberately live-tuned to never
  wait on a clock (`leaky=downstream`, `max-size-buffers=1`, `sync=false`). Clocking
  `detection_sink` fights that → stalled at 2 frames in 80s.

Two supporting changes:
- The `--no-record` terminator is now `fakesink sync=false async=false`. A bare fakesink
  prerolls, holding the pipeline in PAUSED waiting for a buffer `identity` can't release
  until the clock runs in PLAYING — a deadlock that sat in PAUSED forever.
- `on_eos` flags an EOS under 2s as a decode failure rather than a loop. It no longer
  fires, but it is what made the thrash visible instead of looking like healthy looping.

**Live capture is untouched** — `source_type != 'file'` takes none of this path.

## Frame ids and ground truth

With the source paced, ids are contiguous from 0 with nothing dropped, so **id == frame
index** and clean/noisy encodes of the same source give the same id to the same video
frame. Verified end-to-end on 100s clean + noisy clips (ids 0..3136, zero gaps in both):

```
true-ball detections : 2885
noise detections     : 128
frames with no det   :  14   <- detector blackouts, the real locking limiter
```

That is exact labelling by proximity to the clean ball position at the same id — the
persistence heuristic is gone.

**Caveat: first pass only.** `buffer.offset` keeps counting across the loop rewind rather
than restarting (verified: one clean rewind at 120s, ids ran on to 4335, no restart), so
after a loop `id == clip_frames + frame_index`. Keep measurement runs under one clip
length — the eval plan only wants the first ~2 min anyway — or filter to `id < nb_frames`.
The 2026-07-15 note's claim that ids align frame-exactly *across* loops is wrong.

## Recipes

Build a clip (closed-GOP / baseline profile is **required** — pacing sits on compressed
buffers and needs PTS == DTS, so no B-frames):
```
docs/experiments/bytetrack_eval/make_clips.sh <source.mp4> <outdir> [seconds]
```

Run it:
```
scp clip.mp4 bdd@192.168.8.89:/tmp/
ssh bdd@192.168.8.89 'cd /home/bdd && bash hailo-rpi5-examples/scripts/bdd.sh \
    -i /tmp/clip.mp4 --vision-only --no-record --tiles 1x1'
# pull: grep -aE 'RAWDETS|ByteTracker output' _DEBUG/BDD_*.log
```
Deploy first (`git archive --format=tar HEAD | gzip | ssh … tar -xzf -`) and put the HEF
at `/home/bdd/models/`.

## Environment notes (2026-07-17)

- Source clips live at `/media/enmk/BDD/_TEST_DATA/` (was `/media/Pets/BDD/UTILS/`).
  There is now also a `_W_NOISE_0.75` variant.
- Models are at `/media/enmk/BDD/_MODELS/`; the config's ball HEF is the one under
  `experimental/_2/`. Verified `HAILO8` + `NV12(320x640x3)` via `hailortcli parse-hef`.
- `bdd-sd9-mandarin` no longer resolves. The Tailscale box is now `bdd-sd9-testrig1`
  (`100.91.134.66`). `192.168.8.89` is the box used here; it needed the HEF copied over.
- `main`'s `config.yaml` points at a **red-plate** model, so this branch also carries
  `b8e2de0` (ByteTrack + target_lock + the ball HEF). Without it the app won't start —
  config validation rejects the missing file before `--hef-path` can override.

## Next

The original blockers are gone, so the eval can proceed straight to the payoff: run the
noisy clips through [`bytetrack_eval/harness.py`](bytetrack_eval/harness.py) with exact
labels instead of the persistence heuristic, and redo the param sweep. Worth re-checking
the earlier camera-based conclusions too — the 2026-07-15 survey's detector-blackout
claims were measured against the camera path and are probably still sound, but anything
that compared *file* input against it was comparing against torn frames.
