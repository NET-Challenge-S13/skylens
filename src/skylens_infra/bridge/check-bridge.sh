#!/usr/bin/env bash
# 브리지 상태 점검 — 설치 후, 그리고 문제가 있을 때 실행한다. root 불필요.
set -u
cd "$(dirname "$(readlink -f "$0")")"

HTTP_PORT=80; SRT_PORT=10890
[ -f bridge.env ] && . ./bridge.env
TUNNEL_PORT=20889

pass() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; }
info() { printf '  \033[33m·\033[0m %s\n' "$*"; }

echo "== 서비스 =="
systemctl is-active --quiet nginx    && pass "nginx 실행 중"    || fail "nginx 죽음  → sudo systemctl status nginx"
systemctl is-active --quiet mediamtx && pass "mediamtx 실행 중" || fail "mediamtx 죽음  → sudo journalctl -u mediamtx -n 30"

echo "== 포트 =="
ss -tln 2>/dev/null | grep -qE ":$HTTP_PORT\b" && pass "TCP $HTTP_PORT (nginx) 리스닝" || fail "TCP $HTTP_PORT 안 열림"
ss -uln 2>/dev/null | grep -qE ":$SRT_PORT\b"  && pass "UDP $SRT_PORT (SRT) 리스닝"    || fail "UDP $SRT_PORT 안 열림"
if ss -tln 2>/dev/null | grep -qE "127\.0\.0\.1:$TUNNEL_PORT\b"; then
  pass "127.0.0.1:$TUNNEL_PORT — 대전 터널 연결됨"
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$HTTP_PORT/healthz" || echo 000)
  case "$code" in
    2*|3*|404) pass "nginx → 터널 → 대전 MediaMTX 응답 (HTTP $code)";;
    *)         fail "nginx 는 떠 있으나 대전 응답 없음 (HTTP $code)";;
  esac
else
  info "127.0.0.1:$TUNNEL_PORT 없음 — 대전이 아직 터널을 안 걸었음 (대전 쪽에서 whep-tunnel 서비스 확인)"
fi

echo "== 계정 =="
id tunnel >/dev/null 2>&1 && pass "tunnel 계정 있음" || fail "tunnel 계정 없음 → setup-bridge.sh 재실행"
[ -s /home/tunnel/.ssh/authorized_keys ] && pass "대전 공개키 등록됨" || fail "authorized_keys 비어 있음"
sudo -n sshd -T 2>/dev/null | grep -q "permitlisten 127.0.0.1:$TUNNEL_PORT" \
  && pass "sshd PermitListen 적용" || info "sshd 설정 확인 불가(root 필요) 또는 미적용"

echo "== 방화벽 =="
if command -v ufw >/dev/null && sudo -n ufw status 2>/dev/null | grep -q '^Status: active'; then
  sudo -n ufw status | grep -E "$HTTP_PORT/tcp|$SRT_PORT/udp" | sed 's/^/    /'
else
  info "ufw 비활성 또는 확인 불가"
fi

echo "== 공인 IP =="
info "$(curl -4 -s -m 5 ifconfig.me || echo '확인 실패')"
info "밖에서 확인:  휴대폰 5G 로  http://<공인IP>:$HTTP_PORT/healthz  →  응답이 오면 포트포워딩 OK"
