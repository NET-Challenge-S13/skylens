#!/usr/bin/env bash
# Install MediaMTX into /opt/mediamtx, binary and config. The build is picked
# from uname -m, so this also works on an arm64 host.
# Usage: sudo ./install-mediamtx.sh <config.yml> [version]
#   Without a version, the latest GitHub release is used.
set -euo pipefail

CFG="$(readlink -f "${1:?config.yml 경로 필요}")"
VER="${2:-}"

if [ -z "$VER" ]; then
  VER=$(curl -fsSL https://api.github.com/repos/bluenviron/mediamtx/releases/latest \
        | grep -oP '"tag_name":\s*"\K[^"]+')
fi
echo "MediaMTX $VER"

case "$(uname -m)" in
  x86_64)        MARCH=amd64 ;;
  aarch64|arm64) MARCH=arm64 ;;
  armv7l)        MARCH=armv7 ;;
  armv6l)        MARCH=armv6 ;;
  *) echo "unsupported CPU: $(uname -m)" >&2; exit 1 ;;
esac
TARBALL="mediamtx_${VER}_linux_${MARCH}.tar.gz"
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
