#!/bin/bash
# SkyLens 브리지 설치 — macOS (맥미니 175.126.98.44)
#
# 이 서버가 맡는 일:
#   1. nginx 가 브라우저 요청을 받는다          (:HTTP_PORT → / 는 client:8090, /live/ 는 대전 WHEP)
#   2. 대전 VM 으로 SSH 터널을 건다             (로컬 :20889 → 대전 :10889, autossh 로 상시 유지)
#   3. MediaMTX 가 드론 게이트웨이의 SRT 영상을 받아둔다 (:SRT_PORT/udp)
#
# Ubuntu 판(setup-bridge.sh)과 같은 설계. 패키지는 Homebrew, 서비스는 launchd 로 올린다.
# 서비스는 /opt/skylens 아래에 두고, 로그인 없이도 돌도록 LaunchDaemon 으로 등록한다.
# 기존 설정은 건드리지 않고 추가만 한다. 여러 번 실행해도 안전하다.
#
# 준비: Homebrew 가 설치돼 있어야 한다 (https://brew.sh)
# 사용: sudo ./setup-bridge-macos.sh
set -euo pipefail
cd "$(dirname "$0")"; cd "$(pwd -P)"

[ "$(uname)" = Darwin ] || { echo "macOS 전용입니다. Ubuntu 는 setup-bridge.sh" >&2; exit 1; }
[ "$(id -u)" -eq 0 ]    || { echo "root 로 실행하세요:  sudo ./setup-bridge-macos.sh" >&2; exit 1; }
RUN_USER="${SUDO_USER:-}"
if [ -z "$RUN_USER" ] || [ "$RUN_USER" = root ]; then
  echo "일반 계정에서 sudo 로 실행하세요 (root 로그인 상태 불가)" >&2; exit 1
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
  PUBLISH_PASS="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 20)"
  GENERATED_PASS=1
else
  GENERATED_PASS=0
fi
if [ "$HTTP_PORT" = 8080 ]; then
  echo "HTTP_PORT=8080 은 Homebrew nginx 기본 서버와 겹칩니다. 80 이나 다른 포트를 쓰세요." >&2; exit 1
fi

BASE=/opt/skylens
TUNNEL_PORT=20889
TUNNEL_KEY="$BASE/tunnel/tunnel_key"
LOGS="$BASE/logs"

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }
as_user() { sudo -u "$RUN_USER" -H "$@"; }

# ---------- 0. Homebrew / 패키지 ----------
step "Homebrew 패키지"
BREW="$(as_user bash -lc 'command -v brew' 2>/dev/null || true)"
[ -n "$BREW" ] || { echo "Homebrew 가 없습니다. https://brew.sh 안내대로 설치 후 다시 실행하세요." >&2; exit 1; }
PREFIX="$(as_user "$BREW" --prefix)"
as_user "$BREW" install -q nginx autossh >/dev/null 2>&1 || as_user "$BREW" install nginx autossh
ok "nginx, autossh  ($PREFIX)"

mkdir -p "$BASE/mediamtx" "$BASE/tunnel" "$LOGS"
chown "$RUN_USER" "$BASE/mediamtx" "$BASE/tunnel" "$LOGS"

# ---------- 1. 대전으로 가는 터널 ----------
step "대전 터널 (로컬 :$TUNNEL_PORT → $DAEJEON_HOST:10889)"
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

# ---------- 2. nginx ----------
step "nginx (:$HTTP_PORT)"
NGINX_ETC="$PREFIX/etc/nginx"
mkdir -p "$NGINX_ETC/servers"
sed "s/__HTTP_PORT__/$HTTP_PORT/g" nginx-skylens.conf > "$NGINX_ETC/servers/skylens.conf"
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

# ---------- 3. MediaMTX (SRT 수신) ----------
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
sed -e "s/__PUBLISH_PASS__/$PUBLISH_PASS/g" -e "s/__SRT_PORT__/$SRT_PORT/g" \
  mediamtx-bridge.yml > "$BASE/mediamtx/mediamtx.yml"
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

# ---------- 4. macOS 방화벽 (켜져 있을 때만) ----------
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

# ---------- 5. 요약 ----------
PUB_IP="$(curl -4 -s -m 5 ifconfig.me || echo '?')"
cat <<SUMMARY

============================================================
 브리지 설치 완료 (macOS)
============================================================
 공인 IP           : $PUB_IP
 브라우저 주소     : http://$PUB_IP:$HTTP_PORT/live/drone
 SRT 수신          : srt://$PUB_IP:$SRT_PORT?streamid=publish:drone:drone:<PASS>
 SRT publish 비번  : $PUBLISH_PASS$([ "$GENERATED_PASS" = 1 ] && printf '   (자동 생성 — bridge.env 에 적어두세요)')

 공유기 뒤라면 포트포워딩 (두 개):
   TCP $HTTP_PORT  → 이 맥      (브라우저)
   UDP $SRT_PORT   → 이 맥      (드론 게이트웨이)

 ★ 아래 공개키 한 줄을 정준모에게 보내세요. 대전에 등록되면 터널이 자동으로 붙습니다.
------------------------------------------------------------
$(cat "$TUNNEL_KEY.pub")
------------------------------------------------------------
 로그: $LOGS/     상태 확인: ./check-bridge-macos.sh
============================================================
SUMMARY
