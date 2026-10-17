#!/usr/bin/env bash
# Deployment agent for the bridge host (the Raspberry Pi). A timer runs it
# every minute.
#
# Two things arrive here, and they arrive differently:
#
#   the status board   a built web bundle (client-dist.tgz) from the rolling
#                      release, unpacked into the directory nginx serves
#   the host config    nginx, MediaMTX and the tunnel, which are files in the
#                      repository, so a git checkout is the deployment
#
# The installer is only re-run when something under src/skylens_gateway/ really
# changed between the old and the new commit. Running it on every unrelated
# push would restart MediaMTX, and that drops the drone video for a few seconds
# for no reason.
set -euo pipefail

REPO="${SKYLENS_REPO:-NET-Challenge-S13/skylens}"
TAG="${SKYLENS_RELEASE_TAG:-deploy-develop}"
REPO_DIR="${SKYLENS_REPO_DIR:-$HOME/skylens}"
WEB="${SKYLENS_WEB_DIR:-/opt/skylens/web}"
HOLD="${SKYLENS_HOLD:-/etc/skylens/hold}"
BRANCH="${SKYLENS_BRANCH:-develop}"
BASE="https://github.com/$REPO/releases/download/$TAG"

STATE="${SKYLENS_STATE:-$HOME/.skylens-agent-state}"
WORK="${SKYLENS_WORK:-$HOME/.cache/skylens-agent}"
mkdir -p "$WORK"
touch "$STATE"

# To stderr, because a helper's stdout is its return value. systemd keeps both.
log() { printf '%s %s\n' "$(date -Is)" "$*" >&2; }

if [ -e "$HOLD" ]; then
  exit 0   # held on purpose: a demo is running, or someone is working on the host
fi

exec 9>"$WORK/lock"
flock -n 9 || exit 0

installed() { awk -v f="$1" '$1 == f { print $2 }' "$STATE"; }
remember() {
  local f="$1" sum="$2" tmp
  tmp="$(mktemp)"
  awk -v f="$f" '$1 != f' "$STATE" > "$tmp"
  printf '%s %s\n' "$f" "$sum" >> "$tmp"
  mv "$tmp" "$STATE"
}

# ---- the status board web bundle -------------------------------------------
sum="$(curl -fsSL -m 20 "$BASE/manifest-screens.json" 2>/dev/null |
  python3 -I -c 'import json,sys; print(json.load(sys.stdin)["files"].get("client-dist.tgz",""))' 2>/dev/null || true)"

if [ -n "$sum" ] && [ "$sum" != "$(installed client-dist.tgz)" ]; then
  out="$WORK/client-dist.tgz"
  if curl -fsSL -m 600 -o "$out" "$BASE/client-dist.tgz" &&
     [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sum" ]; then
    rm -rf "$WORK/web.new"
    mkdir -p "$WORK/web.new"
    tar xzf "$out" -C "$WORK/web.new"
    if [ -f "$WORK/web.new/status.html" ]; then
      # Swap directories rather than copying into the live one, so nginx never
      # serves a page whose wasm is not there yet.
      sudo -n rm -rf "$WEB.previous"
      if [ -d "$WEB" ]; then
        sudo -n mv "$WEB" "$WEB.previous"
      fi
      sudo -n mkdir -p "$(dirname "$WEB")"
      sudo -n mv "$WORK/web.new" "$WEB"
      sudo -n chown -R root:root "$WEB"
      log "status board updated"
      remember client-dist.tgz "$sum"
    else
      log "client-dist.tgz has no status.html; left the board alone"
    fi
    rm -f "$out"
  else
    log "status board download failed or did not verify"
    rm -f "$out"
  fi
fi

# ---- the host config -------------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch -q origin "$BRANCH" || log "fetch failed"
  old="$(git -C "$REPO_DIR" rev-parse HEAD)"
  new="$(git -C "$REPO_DIR" rev-parse "origin/$BRANCH")"
  if [ "$old" != "$new" ]; then
    if [ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]; then
      # Someone is editing on the host. Their work is worth more than this
      # update, so stop here and say so on every tick until it is sorted out.
      log "$REPO_DIR has uncommitted changes; not moving to ${new:0:12}"
    else
      git -C "$REPO_DIR" reset -q --hard "$new"
      log "config now ${new:0:12}"
      # bridge.env and the tunnel key are untracked, so the checkout leaves
      # them where they are.
      if [ -n "$(git -C "$REPO_DIR" diff --name-only "$old" "$new" -- src/skylens_gateway/ | grep -v '^src/skylens_gateway/agent/' || true)" ]; then
        log "bridge files changed: running the installer"
        if (cd "$REPO_DIR/src/skylens_gateway" && sudo -n ./setup-bridge.sh >"$WORK/setup.log" 2>&1); then
          log "installer ok"
        else
          log "installer failed; see $WORK/setup.log"
        fi
      fi
    fi
  fi
fi
