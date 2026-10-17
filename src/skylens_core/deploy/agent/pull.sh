#!/usr/bin/env bash
# Deployment agent for the Daejeon host. A timer runs it every minute.
#
# KOREN blocks inbound, so nothing can push a build here. This asks instead:
# it reads the manifests of the rolling release, compares the sha256 of each
# file against what is installed, and fetches only what differs. Most minutes
# it makes one HTTPS request and exits.
#
# What it installs:
#   skylens-core      the core binary, into releases/core/<stamp>-<commit>/,
#                     then the bin/skylens-core symlink, then a restart
#   control-dist.tgz  the control tower web build, into web/dist (static files,
#                     so no restart)
#
# Safety, in the order it matters:
#   - a hold file stops everything, which is how a demo is protected
#   - a download is verified against the manifest before it is installed
#   - the swap is a symlink, so the previous release stays on disk
#   - the core has to answer /health after the restart, or the symlink goes
#     back and the service is restarted again on the version that worked
set -euo pipefail

REPO="${SKYLENS_REPO:-NET-Challenge-S13/skylens}"
TAG="${SKYLENS_RELEASE_TAG:-deploy-develop}"
ROOT="${SKYLENS_ROOT:-$HOME/skylens_online}"
REPO_DIR="${SKYLENS_REPO_DIR:-$HOME/skylens}"
BRANCH="${SKYLENS_BRANCH:-develop}"
HOLD="${SKYLENS_HOLD:-/etc/skylens/hold}"
HEALTH="${SKYLENS_HEALTH:-http://127.0.0.1:8080/health}"
SERVICE="${SKYLENS_SERVICE:-skylens-core}"
KEEP="${SKYLENS_KEEP_RELEASES:-5}"
BASE="https://github.com/$REPO/releases/download/$TAG"

STATE="$ROOT/agent-state"   # file name -> sha256 of what is installed
WORK="$ROOT/run/agent"
mkdir -p "$ROOT/releases/core" "$ROOT/web" "$WORK" "$(dirname "$STATE")"
touch "$STATE"

# To stderr, because a helper's stdout is its return value. systemd keeps both.
log() { printf '%s %s\n' "$(date -Is)" "$*" >&2; }

if [ -e "$HOLD" ]; then
  exit 0   # held on purpose: a demo is running, or someone is working on the host
fi

# Only one agent at a time. A long download must not meet the next tick.
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

# want <manifest> <file> prints the wanted sha256, empty when the manifest does
# not carry that file (its build did not run, so nothing changed for it).
want() {
  curl -fsSL -m 20 "$BASE/$1" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["files"].get(sys.argv[1], ""))' "$2" 2>/dev/null || true
}

fetch() { # fetch <file> <sha256> -> path, or nothing when it does not verify
  local f="$1" sum="$2" out="$WORK/$1"
  curl -fsSL -m 600 -o "$out" "$BASE/$f" || { log "download failed: $f"; return 1; }
  if [ "$(sha256sum "$out" | cut -d' ' -f1)" != "$sum" ]; then
    log "checksum mismatch: $f"
    rm -f "$out"
    return 1
  fi
  printf '%s' "$out"
}

commit_of() {
  curl -fsSL -m 20 "$BASE/$1" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["commit"][:12])' 2>/dev/null || echo unknown
}

# ---- the checkout ----------------------------------------------------------
# This is how the agent updates itself: systemd points at the script inside the
# clone, so the next tick runs whatever was just checked out. The host files
# under deploy/ are only brought in, never applied: applying them means
# restarting MediaMTX or the tunnels, which is a decision for a person at a
# keyboard, not for a timer.
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch -q origin "$BRANCH" || log "fetch failed"
  old_head="$(git -C "$REPO_DIR" rev-parse HEAD)"
  new_head="$(git -C "$REPO_DIR" rev-parse "origin/$BRANCH")"
  if [ "$old_head" != "$new_head" ]; then
    if [ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]; then
      log "$REPO_DIR has uncommitted changes; not moving to ${new_head:0:12}"
    else
      git -C "$REPO_DIR" reset -q --hard "$new_head"
      log "checkout now ${new_head:0:12}"
      changed="$(git -C "$REPO_DIR" diff --name-only "$old_head" "$new_head" -- src/skylens_core/deploy/ |
        grep -v '^src/skylens_core/deploy/agent/' || true)"
      if [ -n "$changed" ]; then
        log "host files changed and were NOT applied; apply by hand: $(tr '
' ' ' <<<"$changed")"
      fi
    fi
  fi
fi

# ---- the core binary -------------------------------------------------------
sum="$(want manifest-core.json skylens-core)"
if [ -n "$sum" ] && [ "$sum" != "$(installed skylens-core)" ]; then
  if bin="$(fetch skylens-core "$sum")"; then
    commit="$(commit_of manifest-core.json)"
    dir="$ROOT/releases/core/$(date -u +%Y%m%dT%H%M%SZ)-$commit"
    mkdir -p "$dir"
    install -m 0755 "$bin" "$dir/skylens-core"

    previous="$(readlink -f "$ROOT/bin/skylens-core" || true)"
    mkdir -p "$ROOT/bin"
    ln -sfn "$dir/skylens-core" "$ROOT/bin/skylens-core"
    sudo -n systemctl restart "$SERVICE"

    ok=0
    for _ in $(seq 1 30); do
      sleep 1
      curl -fsS -m 2 "$HEALTH" >/dev/null 2>&1 && { ok=1; break; }
    done

    if [ "$ok" = 1 ]; then
      log "core now $commit"
      remember skylens-core "$sum"
    elif [ -n "$previous" ] && [ -x "$previous" ]; then
      log "core $commit did not answer /health; rolling back to $(basename "$(dirname "$previous")")"
      ln -sfn "$previous" "$ROOT/bin/skylens-core"
      sudo -n systemctl restart "$SERVICE"
    else
      log "core $commit did not answer /health and there is nothing to roll back to"
    fi
    rm -f "$bin"
  fi
fi

# ---- the control tower web build -------------------------------------------
sum="$(want manifest-screens.json control-dist.tgz)"
if [ -n "$sum" ] && [ "$sum" != "$(installed control-dist.tgz)" ]; then
  if tgz="$(fetch control-dist.tgz "$sum")"; then
    # Unpack beside the live directory and swap, so the core never serves a
    # half-written tree.
    rm -rf "$WORK/dist.new"
    mkdir -p "$WORK/dist.new"
    tar xzf "$tgz" -C "$WORK/dist.new"
    rm -rf "$ROOT/web/dist.old"
    if [ -d "$ROOT/web/dist" ]; then
      mv "$ROOT/web/dist" "$ROOT/web/dist.old"
    fi
    mv "$WORK/dist.new" "$ROOT/web/dist"
    rm -rf "$ROOT/web/dist.old"
    log "control tower now $(commit_of manifest-screens.json)"
    remember control-dist.tgz "$sum"
    rm -f "$tgz"
  fi
fi

# ---- keep the last few releases --------------------------------------------
live="$(readlink -f "$ROOT/bin/skylens-core" 2>/dev/null || true)"
ls -1dt "$ROOT/releases/core"/*/ 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
  if [ -n "$live" ] && [ "$live" = "$(readlink -f "$old/skylens-core" 2>/dev/null || true)" ]; then
    continue
  fi
  rm -rf "$old"
done
