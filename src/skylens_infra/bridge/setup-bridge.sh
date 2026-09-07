#!/usr/bin/env bash
# SkyLens 브리지 설치 — 175.126.98.44 (Ubuntu 20.04+, amd64)
#
# 이 서버가 맡는 일:
#   1. 대전 VM 의 SSH 리버스 터널을 받아준다   (tunnel 계정, 로컬 :20889)
#   2. nginx 가 브라우저 요청을 받아 그 터널로 넘긴다 (:HTTP_PORT → 127.0.0.1:20889)
#   3. MediaMTX 가 드론 게이트웨이의 SRT 영상을 받아둔다 (:SRT_PORT/udp)
#
# 기존 서비스(WireGuard 등)는 건드리지 않는다. 추가만 한다.
# 여러 번 실행해도 안전하다(멱등).
#
# 사용:  sudo ./setup-bridge.sh
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

if [ "$(id -u)" -ne 0 ]; then
  echo "root 로 실행하세요:  sudo ./setup-bridge.sh" >&2; exit 1
fi

# ---------- 설정 ----------
SSH_PORT=22
HTTP_PORT=80
PUBLISH_PASS=""
SRT_PORT=10890
MEDIAMTX_VERSION=v1.21.0
if [ -f bridge.env ]; then
  # shellcheck disable=SC1091
  . ./bridge.env
fi
if [ -z "$PUBLISH_PASS" ]; then
  PUBLISH_PASS="$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 20)"
  GENERATED_PASS=1
else
  GENERATED_PASS=0
fi

TUNNEL_USER=tunnel
TUNNEL_PORT=20889

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }

# ---------- 0. 패키지 ----------
step "패키지 설치"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx curl ca-certificates >/dev/null
ok "nginx, curl"

# ---------- 1. 터널 계정 ----------
step "터널 계정 ($TUNNEL_USER)"
if ! id "$TUNNEL_USER" >/dev/null 2>&1; then
  useradd -r -m -d "/home/$TUNNEL_USER" -s /usr/sbin/nologin "$TUNNEL_USER"
  ok "계정 생성"
else
  ok "계정 있음"
fi
install -d -m 0700 -o "$TUNNEL_USER" -g "$TUNNEL_USER" "/home/$TUNNEL_USER/.ssh"
# 키에 restrict 를 걸어 포트포워딩만 허용한다 (셸·X11·에이전트 불가)
{
  while read -r line; do
    [ -z "$line" ] && continue
    echo "restrict,port-forwarding $line"
  done < tunnel.pub
} > "/home/$TUNNEL_USER/.ssh/authorized_keys"
chown "$TUNNEL_USER:$TUNNEL_USER" "/home/$TUNNEL_USER/.ssh/authorized_keys"
chmod 0600 "/home/$TUNNEL_USER/.ssh/authorized_keys"
ok "대전 공개키 등록 (restrict,port-forwarding)"

# sshd: 이 계정만 127.0.0.1:20889 리스닝 허용
SSHD_DROPIN=/etc/ssh/sshd_config.d/skylens-tunnel.conf
SSHD_BLOCK="Match User $TUNNEL_USER
    AllowTcpForwarding yes
    PermitListen 127.0.0.1:$TUNNEL_PORT
    GatewayPorts no
    PermitTTY no
    X11Forwarding no
    AllowAgentForwarding no
    PasswordAuthentication no"
if grep -qE '^\s*Include\s+/etc/ssh/sshd_config\.d/' /etc/ssh/sshd_config; then
  printf '%s\n' "$SSHD_BLOCK" > "$SSHD_DROPIN"
  ok "sshd 드롭인: $SSHD_DROPIN"
else
  if ! grep -q "Match User $TUNNEL_USER" /etc/ssh/sshd_config; then
    printf '\n# SkyLens tunnel\n%s\n' "$SSHD_BLOCK" >> /etc/ssh/sshd_config
  fi
  ok "sshd_config 에 Match 블록 추가 (Include 미지원 버전)"
fi
if sshd -t; then
  systemctl reload ssh 2>/dev/null || systemctl reload sshd
  ok "sshd 설정 검증 후 reload"
else
  echo "sshd 설정 오류 — reload 하지 않았습니다. 위 메시지를 확인하세요." >&2; exit 1
fi
ACTUAL_SSH_PORT="$(sshd -T 2>/dev/null | awk '/^port /{print $2; exit}')"
[ -n "$ACTUAL_SSH_PORT" ] && SSH_PORT="$ACTUAL_SSH_PORT"

# ---------- 2. nginx ----------
step "nginx (:$HTTP_PORT → 127.0.0.1:$TUNNEL_PORT)"
sed "s/__HTTP_PORT__/$HTTP_PORT/g" nginx-skylens.conf > /etc/nginx/sites-available/skylens
ln -sf /etc/nginx/sites-available/skylens /etc/nginx/sites-enabled/skylens
# 우분투 기본 사이트가 같은 포트를 잡고 있으면 비활성화 (기본 파일일 때만)
if [ -L /etc/nginx/sites-enabled/default ] && \
   grep -q "Default server configuration" /etc/nginx/sites-available/default 2>/dev/null && \
   grep -qE "listen\s+$HTTP_PORT\b" /etc/nginx/sites-available/default; then
  rm -f /etc/nginx/sites-enabled/default
  warn "nginx 기본 사이트(default) 비활성화 — 포트 $HTTP_PORT 충돌"
fi
nginx -t
systemctl enable --now nginx >/dev/null 2>&1
systemctl reload nginx
ok "nginx 활성"

# ---------- 3. MediaMTX (SRT 수신) ----------
step "MediaMTX $MEDIAMTX_VERSION (SRT :$SRT_PORT)"
mkdir -p /opt/mediamtx
if [ ! -x /opt/mediamtx/mediamtx ] || [ "$(cat /opt/mediamtx/VERSION 2>/dev/null)" != "$MEDIAMTX_VERSION" ]; then
  TARBALL="mediamtx_${MEDIAMTX_VERSION}_linux_amd64.tar.gz"
  curl -fsSL -o "/tmp/$TARBALL" \
    "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/${TARBALL}"
  tar xzf "/tmp/$TARBALL" -C /opt/mediamtx mediamtx
  rm -f "/tmp/$TARBALL"
  echo "$MEDIAMTX_VERSION" > /opt/mediamtx/VERSION
  ok "바이너리 설치"
else
  ok "바이너리 있음"
fi
sed -e "s/__PUBLISH_PASS__/$PUBLISH_PASS/g" -e "s/__SRT_PORT__/$SRT_PORT/g" \
  mediamtx-bridge.yml > /opt/mediamtx/mediamtx.yml
chmod 0640 /opt/mediamtx/mediamtx.yml
id mediamtx >/dev/null 2>&1 || useradd -r -s /usr/sbin/nologin mediamtx
chown -R mediamtx:mediamtx /opt/mediamtx
cat > /etc/systemd/system/mediamtx.service <<'UNIT'
[Unit]
Description=MediaMTX (SkyLens bridge, SRT ingest)
After=network-online.target
Wants=network-online.target

[Service]
User=mediamtx
WorkingDirectory=/opt/mediamtx
ExecStart=/opt/mediamtx/mediamtx /opt/mediamtx/mediamtx.yml
Restart=always
RestartSec=3
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now mediamtx >/dev/null 2>&1
systemctl restart mediamtx
sleep 1
systemctl is-active --quiet mediamtx && ok "mediamtx 활성" || { journalctl -u mediamtx -n 20 --no-pager; exit 1; }

# ---------- 4. 방화벽 (ufw 가 켜져 있을 때만) ----------
step "방화벽"
if command -v ufw >/dev/null && ufw status 2>/dev/null | grep -q '^Status: active'; then
  ufw allow "$HTTP_PORT/tcp" >/dev/null
  ufw allow "$SRT_PORT/udp" >/dev/null
  ok "ufw: $HTTP_PORT/tcp, $SRT_PORT/udp 허용 (기존 규칙 유지)"
else
  ok "ufw 비활성 — 변경 없음"
fi

# ---------- 5. 요약 ----------
PUB_IP="$(curl -4 -s -m 5 ifconfig.me || echo '?')"
cat <<SUMMARY

============================================================
 브리지 설치 완료
============================================================
 공인 IP           : $PUB_IP
 SSH 포트          : $SSH_PORT        ← 대전 터널이 이 포트로 붙음
 터널 계정         : $TUNNEL_USER     (로컬 127.0.0.1:$TUNNEL_PORT 만 허용)
 브라우저 주소     : http://$PUB_IP:$HTTP_PORT/drone
 SRT 수신          : srt://$PUB_IP:$SRT_PORT?streamid=publish:drone:drone:<PASS>
 SRT publish 비번  : $PUBLISH_PASS$([ "$GENERATED_PASS" = 1 ] && printf '   (자동 생성 — bridge.env 에 적어두세요)')

 공유기 뒤라면 포트포워딩 필요:
   TCP $HTTP_PORT  → 이 서버      (브라우저)
   UDP $SRT_PORT   → 이 서버      (드론 게이트웨이)
   TCP $SSH_PORT   → 이 서버      (대전 터널)  ※ 이미 SSH 되고 있으면 되어 있는 것

 다음: 이 세 값을 정준모에게 전달  →  SSH 포트 / publish 비번 / 포트포워딩 여부
       ./check-bridge.sh 로 상태 확인
============================================================
SUMMARY
