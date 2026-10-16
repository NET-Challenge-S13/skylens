#!/usr/bin/env bash
# Daejeon VM installer: MediaMTX, the distribution node.
#
# Usage:
#   sudo ./setup-daejeon.sh local     # stage 1: take a local ffmpeg test source (mediamtx.yml)
#   sudo ./setup-daejeon.sh bridge    # stage 2: pull from the bridge SRT (mediamtx-pull.yml)
#
# The bridge dials the tunnel itself (autossh); on this side we only register
# the bridge public key:  ./register-bridge-key.sh '<bridge public key>'
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MODE="${1:-local}"
VER=v1.21.0

[ "$(id -u)" -eq 0 ] || { echo "sudo 로 실행하세요" >&2; exit 1; }

case "$MODE" in
  local)  CFG=mediamtx.yml ;;
  bridge) CFG=mediamtx-pull.yml ;;
  *) echo "mode 는 local 또는 bridge" >&2; exit 1 ;;
esac

echo "== MediaMTX $VER ($MODE: $CFG) =="
./scripts/install-mediamtx.sh "$CFG" "$VER"
install -m 0644 mediamtx.service /etc/systemd/system/mediamtx.service
systemctl daemon-reload
systemctl enable --now mediamtx >/dev/null 2>&1
systemctl restart mediamtx
sleep 1
systemctl is-active --quiet mediamtx && echo "   mediamtx 활성" || { journalctl -u mediamtx -n 20 --no-pager; exit 1; }

echo
echo "== 열린 포트 =="
ss -tulnp 2>/dev/null | grep -E 'mediamtx' | awk '{print "   "$1, $5}' | sort -u
echo
echo "== 브리지 터널 (브리지가 걸어옴) =="
if ss -tn 2>/dev/null | grep -qE ':26022\s+175\.126\.98\.44'; then
  echo "   175.126.98.44 에서 SSH 세션 있음 — 터널 연결됨"
else
  echo "   아직 없음. 브리지 공개키를 받아 ./register-bridge-key.sh 로 등록하면 붙는다"
fi
