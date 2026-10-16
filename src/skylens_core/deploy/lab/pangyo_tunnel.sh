#!/bin/bash
# 대전에서 판교로 가는 상시 터널. hirun 으로 프록시를 통과한 뒤 포트 하나를 로컬에 묶는다.
# 이게 떠 있으면 대전에서 'ssh pangyo' 가 바로 된다. 매번 프록시 인증을 다시 하지 않으므로
# 측정값이 인증 비용에 오염되지 않는다.
set -u
. ~/.koren_portal
LOCAL_PORT=${LOCAL_PORT:-19022}
exec ~/hirun -u "$ID" -p "$PW" -c \
  "ssh -p 11000 -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
       -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
       -L ${LOCAL_PORT}:127.0.0.1:11000 ${PANGYO_USER:-$ID}@10.246.246.9"
