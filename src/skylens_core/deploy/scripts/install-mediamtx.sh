#!/usr/bin/env bash
# MediaMTX 설치 (Ubuntu, amd64). /opt/mediamtx 에 바이너리와 설정을 둔다.
# 사용: sudo ./install-mediamtx.sh <config.yml> [version]
#   version 생략 시 GitHub 최신 릴리스.
set -euo pipefail

CFG="$(readlink -f "${1:?config.yml 경로 필요}")"
VER="${2:-}"

if [ -z "$VER" ]; then
  VER=$(curl -fsSL https://api.github.com/repos/bluenviron/mediamtx/releases/latest \
        | grep -oP '"tag_name":\s*"\K[^"]+')
fi
echo "MediaMTX $VER"

TARBALL="mediamtx_${VER}_linux_amd64.tar.gz"
URL="https://github.com/bluenviron/mediamtx/releases/download/${VER}/${TARBALL}"

mkdir -p /opt/mediamtx
cd /opt/mediamtx
curl -fsSL -o "$TARBALL" "$URL"
tar xzf "$TARBALL" mediamtx
rm -f "$TARBALL"
chmod +x mediamtx
install -m 0644 "$CFG" /opt/mediamtx/mediamtx.yml
chown -R ubuntu:ubuntu /opt/mediamtx
echo "$VER" > /opt/mediamtx/VERSION

./mediamtx --version
