#!/usr/bin/env bash
# 대전 VM 설치 — MediaMTX(배포 노드) + 브리지로 가는 SSH 리버스 터널
#
# 사용:
#   sudo ./setup-daejeon.sh local                          # 1단계: 로컬 테스트 소스
#   sudo ./setup-daejeon.sh bridge <ssh_port> [user]       # 2단계: 브리지에서 당겨오기 + 터널
#
# 전제: ~/.ssh/id_tunnel(.pub) 이 ubuntu 계정에 있고, 공개키가 브리지 tunnel 계정에 등록됨.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

MODE="${1:-local}"
BRIDGE_HOST=175.126.98.44
BRIDGE_PORT="${2:-22}"
BRIDGE_USER="${3:-tunnel}"
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

if [ "$MODE" = bridge ]; then
  echo "== 리버스 터널 → $BRIDGE_USER@$BRIDGE_HOST:$BRIDGE_PORT =="
  command -v autossh >/dev/null || apt-get install -y -qq autossh >/dev/null
  [ -f /home/ubuntu/.ssh/id_tunnel ] || { echo "/home/ubuntu/.ssh/id_tunnel 없음" >&2; exit 1; }
  printf 'BRIDGE_HOST=%s\nBRIDGE_PORT=%s\nBRIDGE_USER=%s\n' \
    "$BRIDGE_HOST" "$BRIDGE_PORT" "$BRIDGE_USER" > /etc/default/whep-tunnel
  chmod 0600 /etc/default/whep-tunnel
  install -m 0644 whep-tunnel.service /etc/systemd/system/whep-tunnel.service
  systemctl daemon-reload
  systemctl enable --now whep-tunnel >/dev/null 2>&1
  systemctl restart whep-tunnel
  sleep 3
  if systemctl is-active --quiet whep-tunnel; then
    echo "   whep-tunnel 활성"
  else
    echo "   whep-tunnel 실패:"; journalctl -u whep-tunnel -n 15 --no-pager; exit 1
  fi
fi

echo
echo "== 열린 포트 =="
ss -tulnp 2>/dev/null | grep -E 'mediamtx|autossh|ssh ' | awk '{print "   "$1, $5}' | sort -u
