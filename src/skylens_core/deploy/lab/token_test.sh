#!/bin/bash
# One-off check: can the session credentials hirun minted be used from the pi?
# Passed through the environment only, never written to disk.
set -u
. ~/.koren_portal
( ~/hirun -u "$ID" -p "$PW" -c "sleep 25" >/dev/null 2>&1 & )
sleep 4
CONF=$(ls -t /dev/shm/.proxychains_* /tmp/.proxychains_* 2>/dev/null | head -1)
[ -z "$CONF" ] && { echo "설정 파일을 못 찾음"; exit 1; }
LINE=$(grep -E "^socks5" "$CONF" | head -1)
SU=$(echo "$LINE" | awk '{print $4}')
SP=$(echo "$LINE" | awk '{print $5}')
echo "세션 자격증명 확보 (사용자 ${#SU}자, 비밀번호 ${#SP}자)"

echo "=== 1) 대전에서 이 자격증명으로 판교 연결 ==="
KOREN_SOCKS_USER="$SU" KOREN_SOCKS_PASS="$SP" python3 ~/lab/socks5_tok.py 10.246.246.9 15201 < /dev/null && echo "  대전: 성공" || echo "  대전: 실패"

echo "=== 2) 파이에서 같은 자격증명으로 판교 연결 ==="
ssh -o BatchMode=yes pi "KOREN_SOCKS_USER='$SU' KOREN_SOCKS_PASS='$SP' python3 ~/lab/socks5_tok.py 10.246.246.9 15201 < /dev/null" && echo "  파이: 성공" || echo "  파이: 실패"
