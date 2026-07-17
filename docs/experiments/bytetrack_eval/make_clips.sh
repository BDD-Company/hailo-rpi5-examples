#!/usr/bin/env bash
# Build an eval clip the RPi file-input path can play deterministically.
#
#   ./make_clips.sh <source.mp4> <outdir> [seconds]
#
# Encodes 1280x720, closed-GOP, no B-frames, baseline profile. Two reasons, one of
# them load-bearing:
#
#   1. PTS == DTS. The file source is paced by an identity sync=true sitting on
#      COMPRESSED buffers, ahead of the decode queue (see get_source_pipeline_string
#      in BDD/pipelines.py -- it has to be there, or that leaky 1-deep queue sheds
#      H.264 packets). identity waits on the buffer timestamp, so with B-frames
#      present -- buffers arriving in DTS order while pacing reads PTS -- the pacing
#      would be wrong. Baseline profile removes B-frames, so the two agree. Do not
#      "optimise" this back to a B-frame profile.
#
#   2. Seek safety. Every frame is an IDR or depends only on earlier frames, so the
#      loop's flush seek can never land mid-GOP on a frame whose references were
#      flushed. keyint=15 puts an IDR twice a second, making the rewind cheap. This
#      also encodes out the "co located POCs unavailable" decode failure the handoff
#      reported -- though that was a symptom of the unpaced source, not a codec fault.
#
# The bitrate cost is irrelevant for a 2-minute eval clip.
#
# Frame ids: with the source paced, ids are contiguous from 0 with nothing dropped,
# so id == frame index within a pass -- clean and noisy encodes of the same source
# give the same id to the same video frame, which is what the ground-truth alignment
# needs. This holds for the FIRST PASS ONLY. buffer.offset keeps counting across the
# loop rewind rather than restarting (verified: one clean rewind at 120s, ids ran on
# to 4335 with no restart), so after a loop id == 3600 + frame index. Keep measurement
# runs under one clip length, which is what the eval plan does anyway.

set -euo pipefail

SRC="${1:?usage: make_clips.sh <source.mp4> <outdir> [seconds]}"
OUTDIR="${2:?usage: make_clips.sh <source.mp4> <outdir> [seconds]}"
SECONDS_LEN="${3:-120}"

[ -r "$SRC" ] || { echo "make_clips.sh: cannot read source: $SRC" >&2; exit 1; }
mkdir -p "$OUTDIR"

NAME="$(basename "${SRC%.*}")"
OUT="$OUTDIR/${NAME}_${SECONDS_LEN}s.mp4"

echo "==> $(basename "$OUT")"
ffmpeg -y -v error -ss 0 -t "$SECONDS_LEN" -i "$SRC" \
  -vf "scale=1280:720" \
  -c:v libx264 -preset veryfast -crf 20 -profile:v baseline -pix_fmt yuv420p \
  -x264-params keyint=15:min-keyint=15:scenecut=0:open-gop=0:bframes=0 \
  -movflags +faststart -an \
  "$OUT"

echo "Wrote: $OUT"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,pix_fmt,nb_frames,has_b_frames \
  -of default=nw=1 "$OUT" | sed 's/^/    /'
