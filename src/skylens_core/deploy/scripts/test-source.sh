#!/usr/bin/env bash
# Push a test video into MediaMTX over SRT, to exercise the pipeline without a
# real drone.
#
# Usage: ./test-source.sh [host] [port] [path] [publish_pass]
#   Daejeon, no auth  : ./test-source.sh 127.0.0.1 10890 drone1
#   bridge, with auth : ./test-source.sh 175.126.98.44 10890 drone1 <PUBLISH_PASS>
set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-10890}"
PATHNAME="${3:-drone1}"   # the core names them drone1..drone3
PASS="${4:-}"

STREAMID="publish:${PATHNAME}"
[ -n "$PASS" ] && STREAMID="publish:${PATHNAME}:drone:${PASS}"

exec ffmpeg -hide_banner -loglevel warning -re \
  -f lavfi -i "testsrc2=size=1280x720:rate=30" \
  -c:v libx264 -preset veryfast -tune zerolatency -profile:v baseline \
  -pix_fmt yuv420p -g 60 -keyint_min 60 -b:v 2500k \
  -an \
  -f mpegts "srt://${HOST}:${PORT}?streamid=${STREAMID}&pkt_size=1316"
