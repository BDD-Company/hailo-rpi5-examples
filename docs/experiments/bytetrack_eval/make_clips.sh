#!/usr/bin/env bash
# Build decode-stable eval clips for the RPi file-input path, plus the colour variants
# that test WHY the ball detector barely fires on source video (~0.5%) while hitting
# ~100% on the live camera filming that same video on a screen.
#
#   ./make_clips.sh <source.mp4> <outdir> [seconds]
#
# Every clip is closed-GOP, no B-frames, baseline profile. The Pi's libav decoder
# intermittently died on the original clips with "co located POCs unavailable" and
# reported instant EOS; that failure needs open GOPs / B-pyramids to occur at all, so
# encoding it out is cheaper than chasing decoder state. Costs bitrate efficiency,
# which does not matter for a 2-minute eval clip.
#
# Three variants come out, scored by detection rate in one run each:
#
#   <name>_limited.mp4    Baseline. The handoff's recipe, inheriting the source's
#                         unspecified colour metadata (the sources are
#                         color_range=unknown, color_primaries=unknown), so decoders
#                         fall back to guessing limited-range BT.601 on yuv420p.
#
#   <name>_full.mp4       Range probe. Tags range/primaries/transfer/matrix explicitly
#                         and scales full-range in and out. This is the test the
#                         handoff asked for, kept because it is nearly free -- but see
#                         "what the pixels say" below: it is expected to FAIL, and a
#                         failure here is the point, since it retires the hypothesis.
#
#   <name>_camdomain.mp4  Hue probe, and the one with evidence behind it. Maps the
#                         file's ball colour onto the camera's, see below.
#
# What the pixels say (measured, frame 151, ball body = redness > 0.75*max):
#
#     file 720p as encoded    median ball RGB = (254,  0, 149)   B/R = 0.59
#     camera view of screen   median ball RGB = (226, 10,  73)   B/R = 0.32
#
# The ball is NOT washed out in the file -- it is vivid, and if anything MORE saturated
# than the camera's. So limited-range decoding is not crushing it and the range
# hypothesis is already on thin ice (forcing full range measurably lowers redness,
# i.e. moves AWAY from the camera). The real gap is HUE: the file's ball is magenta
# (B=149), the camera's is coral-red (B=73). The screen's gamut plus the camera's white
# balance roughly halve the blue channel, and the detector only ever trained on the
# latter. Hence bb=0.49 (73/149) and rr=0.89 (226/254), which land the file's ball
# almost exactly on the camera's measured colour.
#
# Reading the result:
#   _camdomain >> _limited  -> it was colour/hue, and file input is fixable by encoding.
#   all three ~equal & low  -> the gap is not colour at all but appearance/training
#                              domain (screen bezel, desk, optics, focus). No encoding
#                              saves that: retrain the detector, or replay camera
#                              recordings instead of source video.

set -euo pipefail

SRC="${1:?usage: make_clips.sh <source.mp4> <outdir> [seconds]}"
OUTDIR="${2:?usage: make_clips.sh <source.mp4> <outdir> [seconds]}"
SECONDS_LEN="${3:-120}"

[ -r "$SRC" ] || { echo "make_clips.sh: cannot read source: $SRC" >&2; exit 1; }
mkdir -p "$OUTDIR"

NAME="$(basename "${SRC%.*}")"

# Closed GOP, no B-frames, no scenecut, baseline profile: every frame is an IDR or
# depends only on earlier frames, so a flush seek can never land mid-GOP on a frame
# whose references were flushed. keyint=15 puts an IDR twice a second, which also makes
# the loop rewind cheap.
DECODE_STABLE=(-c:v libx264 -preset veryfast -crf 20 -profile:v baseline -pix_fmt yuv420p
               -x264-params keyint=15:min-keyint=15:scenecut=0:open-gop=0:bframes=0
               -movflags +faststart -an)

echo "==> ${NAME}_limited.mp4 (baseline)"
ffmpeg -y -v error -ss 0 -t "$SECONDS_LEN" -i "$SRC" \
  -vf "scale=1280:720" \
  "${DECODE_STABLE[@]}" \
  "$OUTDIR/${NAME}_limited.mp4"

echo "==> ${NAME}_full.mp4 (range probe: explicit full-range BT.709)"
ffmpeg -y -v error -ss 0 -t "$SECONDS_LEN" -i "$SRC" \
  -vf "scale=1280:720:in_range=full:out_range=full" \
  -color_range pc -color_primaries bt709 -color_trc bt709 -colorspace bt709 \
  "${DECODE_STABLE[@]}" \
  "$OUTDIR/${NAME}_full.mp4"

echo "==> ${NAME}_camdomain.mp4 (hue probe: match the camera's measured ball colour)"
ffmpeg -y -v error -ss 0 -t "$SECONDS_LEN" -i "$SRC" \
  -vf "scale=1280:720,colorchannelmixer=rr=0.89:bb=0.49" \
  "${DECODE_STABLE[@]}" \
  "$OUTDIR/${NAME}_camdomain.mp4"

echo
echo "Wrote:"
for v in limited full camdomain; do
    f="$OUTDIR/${NAME}_${v}.mp4"
    printf '  %s\n' "$f"
    ffprobe -v error -select_streams v:0 \
      -show_entries stream=width,height,pix_fmt,color_range,nb_frames \
      -of default=nw=1 "$f" | sed 's/^/      /'
done
