#!/usr/bin/env bash
# SkyLens 브리지 설치 — 175.126.98.44 (Ubuntu 20.04+, amd64)
#
# 이 서버가 맡는 일:
#   1. nginx 가 브라우저 요청을 받는다          (:HTTP_PORT → / 는 client:8090, /live/ 는 대전 WHEP)
#   2. 대전 VM 으로 SSH 터널을 건다             (로컬 :20889 → 대전 :10889, autossh 로 상시 유지)
#   3. MediaMTX 가 드론 게이트웨이의 SRT 영상을 받아둔다 (:SRT_PORT/udp)
#
# 터널 방향: 이 서버 → 대전. 175 는 KOREN 등록 IP 라 대전 SSH(26022)에 들어갈 수 있다.
# 반대 방향(대전 → 175)은 175 에 SSH 포트를 열어야 해서 쓰지 않는다.
#
# 기존 서비스(WireGuard 등)는 건드리지 않는다. 추가만 한다. 여러 번 실행해도 안전하다.
#
# 사용:  sudo ./setup-bridge.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ "$(id -u)" -ne 0 ]; then
  echo "root 로 실행하세요:  sudo ./setup-bridge.sh" >&2; exit 1
fi

# ---------- 설정 ----------
HTTP_PORT=80
PUBLISH_PASS=""
SRT_PORT=10890
DAEJEON_HOST=116.89.187.181
DAEJEON_PORT=26022
DAEJEON_USER=ubuntu
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

TUNNEL_USER=skytunnel
TUNNEL_PORT=20889
TUNNEL_DIR=/etc/skylens
TUNNEL_KEY="$TUNNEL_DIR/tunnel_key"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }

# ---------- 0. 패키지 ----------
step "패키지 설치"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq nginx curl ca-certificates autossh openssh-client >/dev/null
ok "nginx, curl, autossh"

# ---------- 1. 대전으로 가는 터널 ----------
step "대전 터널 ($TUNNEL_USER: 로컬 :$TUNNEL_PORT → $DAEJEON_HOST:10889)"
id "$TUNNEL_USER" >/dev/null 2>&1 || useradd -r -M -s /usr/sbin/nologin "$TUNNEL_USER"
install -d -m 0750 -o "$TUNNEL_USER" -g "$TUNNEL_USER" "$TUNNEL_DIR"
if [ ! -f "$TUNNEL_KEY" ]; then
  ssh-keygen -q -t ed25519 -N '' -C "bridge-tunnel@skylens" -f "$TUNNEL_KEY"
  ok "터널 키 생성: $TUNNEL_KEY"
else
  ok "터널 키 있음"
fi
chown "$TUNNEL_USER:$TUNNEL_USER" "$TUNNEL_KEY" "$TUNNEL_KEY.pub"
chmod 0600 "$TUNNEL_KEY"
touch "$TUNNEL_DIR/known_hosts"; chown "$TUNNEL_USER:$TUNNEL_USER" "$TUNNEL_DIR/known_hosts"

cat > /etc/systemd/system/daejeon-tunnel.service <<UNIT
[Unit]
Description=SSH tunnel bridge -> Daejeon for WHEP signaling (SkyLens)
After=network-online.target
Wants=network-online.target

[Service]
User=$TUNNEL_USER
Environment=AUTOSSH_GATETIME=0
ExecStart=/usr/bin/autossh -M 0 -N \\
  -i $TUNNEL_KEY \\
  -o UserKnownHostsFile=$TUNNEL_DIR/known_hosts \\
  -o StrictHostKeyChecking=accept-new \\
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \\
  -o ExitOnForwardFailure=yes \\
  -o BatchMode=yes \\
  -L 127.0.0.1:$TUNNEL_PORT:127.0.0.1:10889 \\
  -p $DAEJEON_PORT $DAEJEON_USER@$DAEJEON_HOST
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now daejeon-tunnel >/dev/null 2>&1
systemctl restart daejeon-tunnel
ok "daejeon-tunnel 서비스 등록 (대전에 키가 등록될 때까지 재시도 상태로 대기)"

# ---------- 2. nginx ----------
step "nginx (:$HTTP_PORT)"
sed "s/__HTTP_PORT__/$HTTP_PORT/g" nginx-skylens.conf > /etc/nginx/sites-available/skylens
ln -sf /etc/nginx/sites-available/skylens /etc/nginx/sites-enabled/skylens
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
 브라우저 주소     : http://$PUB_IP:$HTTP_PORT/live/drone
 SRT 수신          : srt://$PUB_IP:$SRT_PORT?streamid=publish:drone:drone:<PASS>
 SRT publish 비번  : $PUBLISH_PASS$([ "$GENERATED_PASS" = 1 ] && printf '   (자동 생성 — bridge.env 에 적어두세요)')

 공유기 뒤라면 포트포워딩 (두 개):
   TCP $HTTP_PORT  → 이 서버      (브라우저)
   UDP $SRT_PORT   → 이 서버      (드론 게이트웨이)

 ★ 아래 공개키 한 줄을 정준모에게 보내세요. 대전에 등록되면 터널이 자동으로 붙습니다.
------------------------------------------------------------
$(cat "$TUNNEL_KEY.pub")
------------------------------------------------------------
 그다음 ./check-bridge.sh 로 상태 확인
============================================================
SUMMARY
