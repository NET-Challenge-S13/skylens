#!/usr/bin/env bash
# Stage 1 -- video files to evenly spaced frames.
#
#   ./01_extract_frames.sh <scene> <video>:<count> [<video>:<count> ...]
#   ./01_extract_frames.sh corridor IMG_2037.MP4:135 IMG_2038.MP4:270 IMG_2039.MP4:135
#
# Frames are named cam<stem>_%04d.jpg. That prefix is what subset_model.py
# filters on later, so the per-capture identity has to survive into the filename.
#
# Frame count is chosen per video, not as a fixed fps: a longer pass over the
# same space should not dominate the reconstruction just for being longer. Rate
# is derived from each clip's duration so every capture contributes its share.
#
# More frames is not automatically better. Exhaustive matching in stage 2 grows
# with the square of the frame count -- going 360 -> 540 frames cost 5x the
# matching time. And frames shot from nearly the same spot add no parallax,
# which is what actually resolves depth.

set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

SCENE="${1:?usage: 01_extract_frames.sh <scene> <video>:<count> ...}"
shift

OUT="$DATA_DIR/$SCENE/images"
rm -rf "$OUT"
mkdir -p "$OUT"

for spec in "$@"; do
  VIDEO="${spec%:*}"
  COUNT="${spec##*:}"
  STEM="$(basename "$VIDEO" | sed 's/\.[^.]*$//' | tr -cd '0-9')"

  DURATION="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")"
  FPS="$(python3 -c "print(round($COUNT / $DURATION, 4))")"

  log "$VIDEO -> $COUNT frames (${FPS} fps over ${DURATION}s)"
  ffmpeg -hide_banner -loglevel error -i "$VIDEO" \
    -vf "fps=$FPS,scale=w=$FRAME_LONG_EDGE:h=$FRAME_LONG_EDGE:force_original_aspect_ratio=decrease:flags=lanczos" \
    -q:v 2 -start_number 0 "$OUT/cam${STEM}_%04d.jpg"
done

log "total $(find "$OUT" -name '*.jpg' | wc -l) frames in $OUT ($(du -sh "$OUT" | cut -f1))"
