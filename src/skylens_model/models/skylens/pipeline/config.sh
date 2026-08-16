#!/usr/bin/env bash
# Shared configuration for the reconstruction pipeline.
# Source this from every stage script; override any value from the environment.
#
#   SKYLENS_WORK=/data/skylens ./01_extract_frames.sh corridor IMG_2037.MP4:135 ...

set -euo pipefail

# Where captures, sparse models and training output live. Never inside the repo:
# .gitignore excludes data/ and *.pt, but keeping multi-GB artifacts out of the
# working tree entirely avoids accidental adds.
SKYLENS_WORK="${SKYLENS_WORK:-$HOME/skylens-work}"

# COLMAP binary. A CUDA build is strongly preferred -- see INSTALL.md.
COLMAP_BIN="${COLMAP_BIN:-colmap}"

# Checkout of nerfstudio-project/gsplat, matching the installed gsplat version.
# The examples/ trainer lives here; the pip package alone does not ship it.
GSPLAT_EXAMPLES="${GSPLAT_EXAMPLES:-$HOME/gsplat/examples}"

# Capture resolution for frame extraction. 2048 is the long edge.
FRAME_LONG_EDGE="${FRAME_LONG_EDGE:-2048}"

# Camera intrinsics for the OPENCV model, as COLMAP expects them:
#   fx,fy,cx,cy,k1,k2,p1,p2
# MUST match FRAME_LONG_EDGE. Supplying these and letting the mapper refine them
# is the correct combination; see INSTALL.md "focal length" for what happens
# otherwise.
CAMERA_PARAMS="${CAMERA_PARAMS:-1683.5,1640.3,1024,576,0.0365,-0.0001,-0.0003,0.0009}"

DATA_DIR="$SKYLENS_WORK/data"
RESULT_DIR="$SKYLENS_WORK/results"
LOG_DIR="$SKYLENS_WORK/logs"
mkdir -p "$DATA_DIR" "$RESULT_DIR" "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date +%T)" "$*"; }
