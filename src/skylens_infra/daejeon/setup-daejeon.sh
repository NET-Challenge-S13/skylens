#!/usr/bin/env bash
# 대전 VM 설치 — MediaMTX (배포 노드)
#
# 사용:
#   sudo ./setup-daejeon.sh local     # 1단계: 로컬 ffmpeg 테스트 소스를 받는다 (mediamtx.yml)
#   sudo ./setup-daejeon.sh bridge    # 2단계: 브리지 SRT 에서 당겨온다     (mediamtx-pull.yml)
#
# 브리지 → 대전 터널은 브리지 쪽(autossh)이 건다. 대전에서는 브리지 공개키만 등록한다:
#   ./register-bridge-key.sh '<브리지 공개키>'
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
../scripts/install-mediamtx.sh "$CFG" "$VER"
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
