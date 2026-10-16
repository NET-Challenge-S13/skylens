#!/usr/bin/env bash
# 테스트 영상을 SRT 로 MediaMTX 에 밀어넣는다 (실제 드론 대신 파이프라인 검증용).
#
# 사용: ./test-source.sh [host] [port] [path] [publish_pass]
#   대전 로컬 (인증 없음):  ./test-source.sh 127.0.0.1 10890 drone
#   브리지 (인증 있음):     ./test-source.sh 175.126.98.44 10890 drone <PUBLISH_PASS>
set -euo pipefail

HOST="${1:-127.0.0.1}"
PORT="${2:-10890}"
PATHNAME="${3:-drone}"
PASS="${4:-}"

STREAMID="publish:${PATHNAME}"
[ -n "$PASS" ] && STREAMID="publish:${PATHNAME}:drone:${PASS}"

exec ffmpeg -hide_banner -loglevel warning -re \
  -f lavfi -i "testsrc2=size=1280x720:rate=30" \
  -c:v libx264 -preset veryfast -tune zerolatency -profile:v baseline \
  -pix_fmt yuv420p -g 60 -keyint_min 60 -b:v 2500k \
  -an \
  -f mpegts "srt://${HOST}:${PORT}?streamid=${STREAMID}&pkt_size=1316"
