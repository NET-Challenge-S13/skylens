#!/usr/bin/env bash
# Register the bridge (175) tunnel public key on the Daejeon ubuntu account.
#
# The bridge uses this key to open two forwards and nothing else:
#   local 20889 -> Daejeon 10889 (WHEP)
#   local 20882 -> Daejeon 8082  (the proxy, board /viewer and uplink /uplink)
# The key carries command="/bin/false" plus restrict and two permitopen entries,
# so it cannot run a command, get a shell, forward another port, or reach the
# agent or X11.
# An already registered key whose options are stale (no 8082) gets its line redone.
# Order: run this first, then setup-bridge.sh on the bridge. The other way round,
# the refused permitopen drops the whole SSH connection and WHEP (10889) with it.
#
# Usage: ./register-bridge-key.sh 'ssh-ed25519 AAAA... bridge-tunnel@skylens'
#       ./register-bridge-key.sh < pubkey.txt
set -euo pipefail

if [ $# -ge 1 ]; then KEY="$*"; else KEY="$(cat)"; fi
KEY="$(printf '%s' "$KEY" | tr -d '\r' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

case "$KEY" in
  ssh-ed25519\ AAAA*|ssh-rsa\ AAAA*|ecdsa-sha2-*\ AAAA*) ;;
  *) echo "공개키 형식이 아닙니다: '$KEY'" >&2; exit 1 ;;
esac

AK="$HOME/.ssh/authorized_keys"
mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"; touch "$AK"; chmod 600 "$AK"
cp "$AK" "$AK.bak.$(date +%Y%m%d%H%M%S)"

KEYBODY="$(printf '%s' "$KEY" | awk '{print $2}')"

# restrict alone does not block command execution (checked), so command= makes the
# session useless. autossh runs with -N, so the forced command never runs anyway.
OPTS='command="/bin/false",restrict,port-forwarding,permitopen="127.0.0.1:10889",permitopen="127.0.0.1:8082"'

CUR="$(grep -F "$KEYBODY" "$AK" || true)"
if [ -n "$CUR" ]; then
  case "$CUR" in
    *"$OPTS"*)
      echo "이미 같은 권한으로 등록된 키입니다. 기존 줄:"
      grep -nF "$KEYBODY" "$AK"
      exit 0 ;;
  esac
  echo "같은 키가 옛 권한으로 등록되어 있어 권한 줄을 갱신합니다 (8082 추가)."
  TMP="$(mktemp)"
  grep -vF "$KEYBODY" "$AK" > "$TMP" || true
  cat "$TMP" > "$AK"
  rm -f "$TMP"
fi

printf '%s %s\n' "$OPTS" "$KEY" >> "$AK"
echo "등록 완료:"
tail -1 "$AK" | cut -c1-140
echo
echo "브리지에서 확인: ./check-bridge.sh  →  127.0.0.1:20889 · 127.0.0.1:20882 열림이 떠야 함"
