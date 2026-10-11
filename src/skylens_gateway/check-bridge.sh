#!/usr/bin/env bash
# 브리지 상태 점검 — 설치 후, 그리고 문제가 있을 때 실행한다. root 불필요.
set -u
cd "$(dirname "$(readlink -f "$0")")"

HTTP_PORT=80; SRT_PORT=10890; DAEJEON_HOST=116.89.187.181; DAEJEON_PORT=26022
[ -f bridge.env ] && . ./bridge.env
TUNNEL_PORT=20889

pass() { printf '  \033[32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$*"; }
info() { printf '  \033[33m·\033[0m %s\n' "$*"; }

echo "== 서비스 =="
systemctl is-active --quiet nginx          && pass "nginx 실행 중"          || fail "nginx 죽음  → sudo systemctl status nginx"
systemctl is-active --quiet mediamtx       && pass "mediamtx 실행 중"       || fail "mediamtx 죽음  → sudo journalctl -u mediamtx -n 30"
systemctl is-active --quiet daejeon-tunnel && pass "daejeon-tunnel 실행 중" || fail "daejeon-tunnel 죽음  → sudo journalctl -u daejeon-tunnel -n 30"

echo "== 포트 =="
ss -tln 2>/dev/null | grep -qE ":$HTTP_PORT\b" && pass "TCP $HTTP_PORT (nginx) 리스닝" || fail "TCP $HTTP_PORT 안 열림"
ss -uln 2>/dev/null | grep -qE ":$SRT_PORT\b"  && pass "UDP $SRT_PORT (SRT) 리스닝"    || fail "UDP $SRT_PORT 안 열림"

echo "== 대전 터널 =="
if ss -tln 2>/dev/null | grep -qE "127\.0\.0\.1:$TUNNEL_PORT\b"; then
  pass "127.0.0.1:$TUNNEL_PORT 열림 — 대전과 연결됨"
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 3 "http://127.0.0.1:$HTTP_PORT/healthz" || echo 000)
  case "$code" in
    2*|3*|404) pass "nginx → 터널 → 대전 MediaMTX 응답 (HTTP $code)";;
    *)         fail "터널은 열렸는데 대전 MediaMTX 응답 없음 (HTTP $code)";;
  esac
else
  info "127.0.0.1:$TUNNEL_PORT 없음 — 대전에 아직 키가 등록되지 않았거나 접속 실패"
  if sudo -n journalctl -u daejeon-tunnel -n 3 --no-pager -o cat 2>/dev/null | grep -qiE 'Permission denied|publickey'; then
    info "원인: 대전이 이 서버의 키를 아직 모름. 아래 공개키를 정준모에게 보내세요:"
  else
    info "최근 로그:"; sudo -n journalctl -u daejeon-tunnel -n 3 --no-pager -o cat 2>/dev/null | sed 's/^/      /'
  fi
  [ -r /etc/skylens/tunnel_key.pub ] && sed 's/^/      /' /etc/skylens/tunnel_key.pub \
    || sudo -n cat /etc/skylens/tunnel_key.pub 2>/dev/null | sed 's/^/      /'
fi
if command -v nc >/dev/null; then
  nc -z -w 3 "$DAEJEON_HOST" "$DAEJEON_PORT" 2>/dev/null && pass "대전 SSH $DAEJEON_HOST:$DAEJEON_PORT 도달" || fail "대전 SSH 도달 불가 — 이 서버 IP 가 KOREN 에 등록돼 있는지 확인"
fi

echo "== 방화벽 =="
if command -v ufw >/dev/null && sudo -n ufw status 2>/dev/null | grep -q '^Status: active'; then
  sudo -n ufw status | grep -E "$HTTP_PORT/tcp|$SRT_PORT/udp" | sed 's/^/    /'
else
  info "ufw 비활성 또는 확인 불가"
fi

echo "== 공인 IP =="
info "$(curl -4 -s -m 5 ifconfig.me || echo '확인 실패')"
info "밖에서 확인:  휴대폰 5G 로  http://<공인IP>:$HTTP_PORT/healthz  →  응답이 오면 포트포워딩 OK"
