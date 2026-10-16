#!/bin/bash
# The standing tunnel from Daejeon to Pangyo: pass the proxy with hirun, then bind
# one local port. With this up, `ssh pangyo` works straight from Daejeon, and no
# measurement is polluted by repeating the proxy authentication.
set -u
. ~/.koren_portal
LOCAL_PORT=${LOCAL_PORT:-19022}
exec ~/hirun -u "$ID" -p "$PW" -c \
  "ssh -p 11000 -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
       -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
       -L ${LOCAL_PORT}:127.0.0.1:11000 ${PANGYO_USER:-$ID}@10.246.246.9"
