#!/bin/bash
# SkyLens bridge installer for macOS. The bridge host is a Raspberry Pi today,
# so this is the spare for moving the role onto a Mac; the Linux installer is
# setup-bridge.sh.
#
# What this host does:
#   1. nginx takes the browser requests  (:HTTP_PORT; / is the static board in
#      WEB_DIR, /viewer and /uplink go through tunnel :20882 to the Daejeon
#      proxy, /live/ through tunnel :20889 to the Daejeon WHEP endpoint)
#   2. autossh holds one SSH tunnel to the Daejeon VM, carrying both forwards
#      (local :20889 -> Daejeon :10889, local :20882 -> Daejeon proxy :8082)
#   3. MediaMTX takes the SRT feed from the drone gateway (:SRT_PORT/udp)
#
# One tunnel carries both forwards with ExitOnForwardFailure=yes, so if the
# Daejeon key lacks permitopen for 8082 the whole SSH connection drops and
# takes WHEP (20889) down with it. Hence the install order in the README.
#
# Same design as the Linux installer; packages come from Homebrew and services
# run as launchd LaunchDaemons under /opt/skylens, so they start without a
# login. Existing configuration is left alone; this only adds. Safe to run
# again.
#
# Needs Homebrew (https://brew.sh).
# Usage: sudo ./setup-bridge-macos.sh
set -euo pipefail
cd "$(dirname "$0")"; cd "$(pwd -P)"

[ "$(uname)" = Darwin ] || { echo "macOS 전용입니다. Ubuntu 는 setup-bridge.sh" >&2; exit 1; }
[ "$(id -u)" -eq 0 ]    || { echo "root 로 실행하세요:  sudo ./setup-bridge-macos.sh" >&2; exit 1; }
RUN_USER="${SUDO_USER:-}"
if [ -z "$RUN_USER" ] || [ "$RUN_USER" = root ]; then
  echo "일반 계정에서 sudo 로 실행하세요 (root 로그인 상태 불가)" >&2; exit 1
fi

# ---------- settings ----------
HTTP_PORT=80
PUBLISH_PASS=""
SRT_PORT=10890
HTTPS_PORT=443
TLS_NAMES=
REDIRECT_TO_HTTPS=yes
TOWER_PORT=8090
TOWER_TLS_PORT=8091
TOWER_ALLOW="127.0.0.1 10.9.8.0/24 10.9.9.0/24"
VIDEO_MODE=ingest
WHEP_SOURCE=local
WHEP_PORT=10889
WEBRTC_HOSTS=
WEB_DIR=/opt/skylens/web
DAEJEON_HOST=116.89.187.181
DAEJEON_PORT=26022
DAEJEON_USER=ubuntu
MEDIAMTX_VERSION=v1.21.0
if [ -f bridge.env ]; then
  # shellcheck disable=SC1091
  . ./bridge.env
fi
if [ -z "$PUBLISH_PASS" ]; then
  # With head last it exits first, tr dies of SIGPIPE, and set -e plus pipefail
  # stops the script at 141 without a word. Cut the input first, filter after.
  PUBLISH_PASS="$(LC_ALL=C head -c 512 /dev/urandom | LC_ALL=C tr -dc 'A-Za-z0-9' | cut -c1-20)"
  [ ${#PUBLISH_PASS} -eq 20 ] || { echo "비밀번호 생성 실패 (${#PUBLISH_PASS}자)" >&2; exit 1; }
  GENERATED_PASS=1
else
  GENERATED_PASS=0
fi
if [ "$HTTP_PORT" = 8080 ]; then
  echo "HTTP_PORT=8080 은 Homebrew nginx 기본 서버와 겹칩니다. 80 이나 다른 포트를 쓰세요." >&2; exit 1
fi

BASE=/opt/skylens
TUNNEL_PORT=20889
PROXY_TUNNEL_PORT=20882
RTSP_TUNNEL_PORT=20554
TUNNEL_KEY="$BASE/tunnel/tunnel_key"
LOGS="$BASE/logs"
TLS_DIR="$BASE/tls"
TLS_CERT="$TLS_DIR/board.crt"
TLS_KEY="$TLS_DIR/board.key"
TOWER_ACL="$BASE/tower-acl.conf"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
as_user() { sudo -u "$RUN_USER" -H "$@"; }

# ---------- 0. Homebrew packages ----------
step "Homebrew 패키지"
BREW="$(as_user bash -lc 'command -v brew' 2>/dev/null || true)"
[ -n "$BREW" ] || { echo "Homebrew 가 없습니다. https://brew.sh 안내대로 설치 후 다시 실행하세요." >&2; exit 1; }
PREFIX="$(as_user "$BREW" --prefix)"
as_user "$BREW" install -q nginx autossh >/dev/null 2>&1 || as_user "$BREW" install nginx autossh
ok "nginx, autossh  ($PREFIX)"

mkdir -p "$BASE/mediamtx" "$BASE/tunnel" "$LOGS"
chown "$RUN_USER" "$BASE/mediamtx" "$BASE/tunnel" "$LOGS"

# ---------- 1. the tunnel to Daejeon ----------
step "대전 터널 (로컬 :$TUNNEL_PORT → :10889, :$PROXY_TUNNEL_PORT → :8082, :$RTSP_TUNNEL_PORT → :8554)"
if [ ! -f "$TUNNEL_KEY" ]; then
  as_user ssh-keygen -q -t ed25519 -N '' -C 'bridge-tunnel@skylens' -f "$TUNNEL_KEY"
  ok "터널 키 생성: $TUNNEL_KEY"
else
  ok "터널 키 있음"
fi
chmod 0600 "$TUNNEL_KEY"; chown "$RUN_USER" "$TUNNEL_KEY" "$TUNNEL_KEY.pub"
touch "$BASE/tunnel/known_hosts"; chown "$RUN_USER" "$BASE/tunnel/known_hosts"

cat > /Library/LaunchDaemons/com.skylens.daejeon-tunnel.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.skylens.daejeon-tunnel</string>
  <key>UserName</key><string>$RUN_USER</string>
  <key>ProgramArguments</key><array>
    <string>$PREFIX/bin/autossh</string>
    <string>-M</string><string>0</string>
    <string>-N</string>
    <string>-i</string><string>$TUNNEL_KEY</string>
    <string>-o</string><string>UserKnownHostsFile=$BASE/tunnel/known_hosts</string>
    <string>-o</string><string>StrictHostKeyChecking=accept-new</string>
    <string>-o</string><string>ServerAliveInterval=15</string>
    <string>-o</string><string>ServerAliveCountMax=3</string>
    <string>-o</string><string>ExitOnForwardFailure=yes</string>
    <string>-o</string><string>BatchMode=yes</string>
    <string>-L</string><string>127.0.0.1:$TUNNEL_PORT:127.0.0.1:10889</string>
    <string>-L</string><string>127.0.0.1:$PROXY_TUNNEL_PORT:127.0.0.1:8082</string>
    <string>-L</string><string>127.0.0.1:$RTSP_TUNNEL_PORT:127.0.0.1:8554</string>
    <string>-p</string><string>$DAEJEON_PORT</string>
    <string>$DAEJEON_USER@$DAEJEON_HOST</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>AUTOSSH_GATETIME</key><string>0</string>
    <key>PATH</key><string>$PREFIX/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>StandardOutPath</key><string>$LOGS/daejeon-tunnel.log</string>
  <key>StandardErrorPath</key><string>$LOGS/daejeon-tunnel.log</string>
</dict></plist>
PLIST
chmod 0644 /Library/LaunchDaemons/com.skylens.daejeon-tunnel.plist
launchctl bootout system/com.skylens.daejeon-tunnel 2>/dev/null || true
launchctl bootstrap system /Library/LaunchDaemons/com.skylens.daejeon-tunnel.plist
ok "daejeon-tunnel 등록 (대전에 키가 등록될 때까지 재시도 상태로 대기)"

# ---------- 2. the static board files ----------
step "현황판 정적 파일 ($WEB_DIR)"
mkdir -p "$WEB_DIR"
chmod 0755 "$WEB_DIR"
if [ -f "$WEB_DIR/status.html" ]; then
  ok "현황판 있음 ($(find "$WEB_DIR" -type f | wc -l | tr -d ' ') 개 파일)"
else
  warn "현황판 파일 없음. 빌드 산출물(status.html 포함)을 $WEB_DIR 에 복사하세요"
  warn "  bash src/skylens_client/build-web.sh  →  sudo cp -R src/skylens_client/dist/. $WEB_DIR/"
fi

# ---------- 3. board certificate ----------
step "board certificate (TLS on :$HTTPS_PORT)"
# The browser only grants WebGPU to a secure context, and the board's renderer
# panics without it, so the board is served over TLS even on the LAN. A
# self-signed certificate makes the context secure (the viewer clicks through
# the warning once). Drop a trusted pair at the same paths to lose the warning;
# an existing certificate is never overwritten.
mkdir -p "$TLS_DIR"; chmod 0750 "$TLS_DIR"
if [ -s "$TLS_CERT" ] && [ -s "$TLS_KEY" ]; then
  ok "certificate exists: $TLS_CERT"
else
  if [ -n "$TLS_NAMES" ]; then
    SAN="$TLS_NAMES"
  else
    SAN="DNS:$(hostname),DNS:$(hostname).local,DNS:localhost,IP:127.0.0.1"
    for ip in $(ifconfig 2>/dev/null | awk '/inet /{print $2}'); do
      [ "$ip" = 127.0.0.1 ] && continue
      SAN="$SAN,IP:$ip"
    done
  fi
  openssl req -x509 -newkey rsa:2048 -sha256 -days 1825 -nodes \
    -keyout "$TLS_KEY" -out "$TLS_CERT" \
    -subj "/CN=skylens-board" -addext "subjectAltName=$SAN" >/dev/null 2>&1
  chmod 0640 "$TLS_KEY"; chmod 0644 "$TLS_CERT"; chown -R "$RUN_USER" "$TLS_DIR"
  ok "self-signed certificate written ($SAN)"
fi

# ---------- 4. nginx ----------
step "nginx (:$HTTP_PORT, TLS :$HTTPS_PORT)"
NGINX_ETC="$PREFIX/etc/nginx"
mkdir -p "$NGINX_ETC/servers"
# A redirect to the default TLS port must not carry :443 in the URL.
if [ "$HTTPS_PORT" = 443 ]; then HTTPS_SUFFIX=""; else HTTPS_SUFFIX=":$HTTPS_PORT"; fi
# Where /live/ sends the WHEP exchange: this host's MediaMTX, or the Daejeon one
# through the tunnel. Local is the default, because Daejeon's UDP media does not
# cross the VPN boundary.
case "$WHEP_SOURCE" in
  local)  WHEP_UPSTREAM="127.0.0.1:$WHEP_PORT" ;;
  tunnel) WHEP_UPSTREAM="127.0.0.1:$TUNNEL_PORT" ;;
  *) echo "WHEP_SOURCE 은 local 또는 tunnel" >&2; exit 1 ;;
esac
if [ "$REDIRECT_TO_HTTPS" = no ]; then
  HTTPS_REDIRECT="# redirect off (REDIRECT_TO_HTTPS=no)"
  TOWER_REDIRECT="# redirect off (REDIRECT_TO_HTTPS=no)"
else
  HTTPS_REDIRECT="if (\$scheme = http) { return 301 https://\$host$HTTPS_SUFFIX\$request_uri; }"
  TOWER_REDIRECT="if (\$skylens_tower_redirect) { return 301 https://\$host:$TOWER_TLS_PORT\$request_uri; }"
fi
# The tower is reachable from the operator LAN and the VPN only. nginx reads
# these lines in order, so the deny has to come last.
{ for net in $TOWER_ALLOW; do echo "allow $net;"; done; echo "deny all;"; } > "$TOWER_ACL"
chmod 0644 "$TOWER_ACL"
sed -e "s/__HTTP_PORT__/$HTTP_PORT/g" -e "s|__WEB_DIR__|$WEB_DIR|g" \
  -e "s/__HTTPS_PORT__/$HTTPS_PORT/g" -e "s|__HTTPS_HOSTSUFFIX__|$HTTPS_SUFFIX|g" \
  -e "s|__TLS_CERT__|$TLS_CERT|g" -e "s|__TLS_KEY__|$TLS_KEY|g" \
  -e "s|__HTTPS_REDIRECT__|$HTTPS_REDIRECT|g" \
  -e "s/__TOWER_PORT__/$TOWER_PORT/g" -e "s/__TOWER_TLS_PORT__/$TOWER_TLS_PORT/g" \
  -e "s|__TOWER_ACL__|$TOWER_ACL|g" -e "s|__TOWER_REDIRECT__|$TOWER_REDIRECT|g" \
  -e "s|__WHEP_UPSTREAM__|$WHEP_UPSTREAM|g" \
  nginx-skylens.conf > "$NGINX_ETC/servers/skylens.conf"
grep -qE '^\s*include\s+servers/\*;' "$NGINX_ETC/nginx.conf" \
  || warn "nginx.conf 에 'include servers/*;' 가 없습니다 — http { } 블록 안에 추가해야 합니다"
"$PREFIX/bin/nginx" -t -c "$NGINX_ETC/nginx.conf"

NGX_PID="$(pgrep -x nginx | head -1 || true)"
if [ -n "$NGX_PID" ]; then
  NGX_OWNER="$(ps -o user= -p "$NGX_PID" | tr -d ' ')"
  if [ "$NGX_OWNER" = root ]; then "$PREFIX/bin/nginx" -s reload; else as_user "$PREFIX/bin/nginx" -s reload; fi
  ok "이미 떠 있는 nginx($NGX_OWNER) 에 설정 반영(reload)"
  [ "$HTTP_PORT" -lt 1024 ] && [ "$NGX_OWNER" != root ] && warn "포트 $HTTP_PORT 은 root 가 아니면 못 엽니다. 기존 nginx 를 내리고 재실행하거나 HTTP_PORT 를 1024 이상으로"
else
  cat > /Library/LaunchDaemons/com.skylens.nginx.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.skylens.nginx</string>
  <key>ProgramArguments</key><array>
    <string>$PREFIX/bin/nginx</string>
    <string>-c</string><string>$NGINX_ETC/nginx.conf</string>
    <string>-g</string><string>daemon off;</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOGS/nginx.log</string>
  <key>StandardErrorPath</key><string>$LOGS/nginx.log</string>
</dict></plist>
PLIST
  chmod 0644 /Library/LaunchDaemons/com.skylens.nginx.plist
  launchctl bootout system/com.skylens.nginx 2>/dev/null || true
  launchctl bootstrap system /Library/LaunchDaemons/com.skylens.nginx.plist
  ok "nginx 를 LaunchDaemon(root) 으로 기동"
fi

# ---------- 5. MediaMTX (SRT ingest) ----------
step "MediaMTX $MEDIAMTX_VERSION (SRT :$SRT_PORT)"
case "$(uname -m)" in arm64) MARCH=arm64 ;; x86_64) MARCH=amd64 ;; *) echo "지원하지 않는 CPU" >&2; exit 1 ;; esac
if [ ! -x "$BASE/mediamtx/mediamtx" ] || [ "$(cat "$BASE/mediamtx/VERSION" 2>/dev/null)" != "$MEDIAMTX_VERSION" ]; then
  TARBALL="mediamtx_${MEDIAMTX_VERSION}_darwin_${MARCH}.tar.gz"
  curl -fsSL -o "/tmp/$TARBALL" "https://github.com/bluenviron/mediamtx/releases/download/${MEDIAMTX_VERSION}/${TARBALL}"
  tar xzf "/tmp/$TARBALL" -C "$BASE/mediamtx" mediamtx
  rm -f "/tmp/$TARBALL"
  echo "$MEDIAMTX_VERSION" > "$BASE/mediamtx/VERSION"
  xattr -d com.apple.quarantine "$BASE/mediamtx/mediamtx" 2>/dev/null || true
  ok "바이너리 설치 (darwin_$MARCH)"
else
  ok "바이너리 있음"
fi
case "$VIDEO_MODE" in
  ingest) MTX_CONF=mediamtx-bridge.yml ;;
  relay)  MTX_CONF=mediamtx-bridge-relay.yml ;;
  *) echo "VIDEO_MODE 은 ingest 또는 relay" >&2; exit 1 ;;
esac
if [ -n "$WEBRTC_HOSTS" ]; then
  HOSTS="$WEBRTC_HOSTS"
else
  HOSTS=""
  TICK="'"          # the YAML list wants each host quoted
  for ip in $(ifconfig 2>/dev/null | awk '/inet /{print $2}'); do
    [ "$ip" = 127.0.0.1 ] && continue
    HOSTS="${HOSTS:+$HOSTS,}$TICK$ip$TICK"
  done
fi
sed -e "s/__PUBLISH_PASS__/$PUBLISH_PASS/g" -e "s/__SRT_PORT__/$SRT_PORT/g" \
  -e "s/__WHEP_PORT__/$WHEP_PORT/g" -e "s/__RTSP_TUNNEL_PORT__/$RTSP_TUNNEL_PORT/g" \
  -e "s|__WEBRTC_HOSTS__|$HOSTS|g" \
  "$MTX_CONF" > "$BASE/mediamtx/mediamtx.yml"
ok "$MTX_CONF, WebRTC 후보 $HOSTS"
chmod 0640 "$BASE/mediamtx/mediamtx.yml"
chown -R "$RUN_USER" "$BASE/mediamtx"

cat > /Library/LaunchDaemons/com.skylens.mediamtx.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.skylens.mediamtx</string>
  <key>UserName</key><string>$RUN_USER</string>
  <key>WorkingDirectory</key><string>$BASE/mediamtx</string>
  <key>ProgramArguments</key><array>
    <string>$BASE/mediamtx/mediamtx</string>
    <string>$BASE/mediamtx/mediamtx.yml</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$LOGS/mediamtx.log</string>
  <key>StandardErrorPath</key><string>$LOGS/mediamtx.log</string>
</dict></plist>
PLIST
chmod 0644 /Library/LaunchDaemons/com.skylens.mediamtx.plist
launchctl bootout system/com.skylens.mediamtx 2>/dev/null || true
launchctl bootstrap system /Library/LaunchDaemons/com.skylens.mediamtx.plist
sleep 2
if launchctl print system/com.skylens.mediamtx 2>/dev/null | grep -q 'state = running'; then
  ok "mediamtx 실행 중"
else
  warn "mediamtx 상태 확인 실패 — 로그: $LOGS/mediamtx.log"; tail -n 15 "$LOGS/mediamtx.log" 2>/dev/null || true
fi

# ---------- 6. macOS firewall (only when it is already on) ----------
step "방화벽"
SFW=/usr/libexec/ApplicationFirewall/socketfilterfw
if "$SFW" --getglobalstate 2>/dev/null | grep -qi 'enabled'; then
  for b in "$BASE/mediamtx/mediamtx" "$PREFIX/bin/nginx"; do
    "$SFW" --add "$b" >/dev/null 2>&1 || true
    "$SFW" --unblockapp "$b" >/dev/null 2>&1 || true
  done
  ok "macOS 방화벽에 mediamtx, nginx 허용 추가"
else
  ok "macOS 방화벽 꺼짐 — 변경 없음"
fi

# ---------- 7. summary ----------
PUB_IP="$(curl -4 -s -m 5 ifconfig.me || echo '?')"
cat <<SUMMARY

============================================================
 브리지 설치 완료 (macOS)
============================================================
 공인 IP           : $PUB_IP
 현황판            : https://$(ipconfig getifaddr en0 2>/dev/null || hostname)$HTTPS_SUFFIX/   ($WEB_DIR/status.html)
                     WebGPU 가 보안 컨텍스트에서만 열려서 HTTPS 로 띄웁니다.
                     자체 서명 인증서라 브라우저 경고를 한 번 넘겨야 합니다.
 이 서버 생존      : http://$PUB_IP:$HTTP_PORT/health      (nginx 만 확인)
 터널 생존         : http://$PUB_IP:$HTTP_PORT/healthz     (대전까지 확인)
 라이브 영상       : http://$PUB_IP:$HTTP_PORT/live/drone1/whep  (drone1~3)
 관제탑            : https://$(ipconfig getifaddr en0 2>/dev/null || hostname):$TOWER_TLS_PORT/  (랜·VPN 전용)
                     허용 범위: $TOWER_ALLOW
 SRT 수신          : srt://$PUB_IP:$SRT_PORT?streamid=publish:drone1:drone:<PASS>
 SRT publish 비번  : $PUBLISH_PASS$([ "$GENERATED_PASS" = 1 ] && printf '   (자동 생성 — bridge.env 에 적어두세요)')

 공유기 뒤라면 포트포워딩 (두 개):
   TCP $HTTP_PORT  → 이 맥      (브라우저)
   UDP $SRT_PORT   → 이 맥      (드론 게이트웨이, 그리고 대전이 당겨감)

 ★ 아래 공개키 한 줄을 정준모에게 보내세요. 대전에서 permitopen 10889·8082 로
   등록되면 터널 두 포워드가 자동으로 붙습니다. 8082 가 빠지면 WHEP 까지 같이 죽습니다.
------------------------------------------------------------
$(cat "$TUNNEL_KEY.pub")
------------------------------------------------------------
 로그: $LOGS/     상태 확인: ./check-bridge-macos.sh
============================================================
SUMMARY
