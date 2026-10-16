#!/usr/bin/env bash
# 브리지(175)의 터널 공개키를 대전 ubuntu 계정에 등록한다.
#
# 브리지는 이 키로 SSH 를 걸어 두 터널만 연다.
#   로컬 20889 → 대전 10889 (WHEP)
#   로컬 20882 → 대전 8082  (프록시, 현황판 /viewer · 업링크 /uplink)
# 키에 command="/bin/false" + restrict + permitopen 을 걸어 그 두 포트 외에는
# 아무것도 못 하게 한다 (명령 실행·셸·다른 포트 포워딩·에이전트·X11 전부 불가).
#
# 이미 등록된 키라도 권한 줄이 옛 값(8082 누락)이면 줄을 갱신한다.
# 순서: 이 스크립트를 먼저 돌리고, 그다음 브리지에서 setup-bridge.sh 를 돌린다.
# 거꾸로 하면 브리지 터널이 permitopen 거부로 끊기면서 10889 WHEP 까지 같이 죽는다.
#
# 사용: ./register-bridge-key.sh 'ssh-ed25519 AAAA... bridge-tunnel@skylens'
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

# restrict 만으로는 명령 실행이 막히지 않는다(검증됨). command= 로 세션을 무력화한다.
# autossh 는 -N(세션 없음)이라 forced command 가 실행되지 않고 포워딩만 된다.
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
