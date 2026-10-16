#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Cut a release: scripts/release.sh patch|minor|major [--dry-run]
#
# Checks (fmt, clippy -D warnings, tests, public API compatibility), bumps the workspace version and the inter-crate dependency
# versions, refreshes Cargo.lock, moves the "## Unreleased" entries of CHANGELOG.md / CHANGELOG.ko.md under the new version,
# commits "Release X.Y.Z" and pushes. The push to main starts the Release workflow, which publishes to crates.io, builds the
# binaries and creates the tag and the GitHub Release.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { echo "usage: scripts/release.sh patch|minor|major [--dry-run]" >&2; exit 2; }
[ $# -ge 1 ] || usage
bump=$1; dry=0
case "$bump" in patch|minor|major) ;; *) usage ;; esac
if [ $# -ge 2 ]; then [ "$2" = "--dry-run" ] && dry=1 || usage; fi
[ $# -le 2 ] || usage

die() { echo "error: $*" >&2; exit 1; }

# --- preconditions
[ "$(git rev-parse --abbrev-ref HEAD)" = main ] || die "release from the main branch"
[ -z "$(git status --porcelain --untracked-files=no)" ] || die "working tree has uncommitted changes"
git fetch -q origin main
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || die "main differs from origin/main (pull or push first)"

old=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)
IFS=. read -r maj min pat <<< "$old"
case "$bump" in
  patch) new="$maj.$min.$((pat + 1))" ;;
  minor) new="$maj.$((min + 1)).0" ;;
  major) new="$((maj + 1)).0.0" ;;
esac
echo "release: $old -> $new ($bump)$([ $dry = 1 ] && echo ' [dry run]')"

# --- changelogs: both need entries under "## Unreleased"
has_entries() {
  python3 -I - "$1" <<'PY'
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"^## Unreleased[ \t]*\n(.*?)(?=^## |\Z)", t, re.S | re.M)
sys.exit(0 if m and m.group(1).strip() else 1)
PY
}
for f in CHANGELOG.md CHANGELOG.ko.md; do
  has_entries "$f" || die "$f: no entries under '## Unreleased' (add them above the newest version heading)"
done

# --- checks
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --release --workspace --locked

# --- bump versions
perl -i -pe 's/^(version = ")\Q'"$old"'\E(")/${1}'"$new"'${2}/ if $. && /^version = /; s/^(cumulus3d-[a-z0-9-]+ = \{ path = "[^"]+", version = ")\Q'"$old"'\E(")/${1}'"$new"'${2}/' Cargo.toml
[ "$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)" = "$new" ] || die "version bump in Cargo.toml failed"
cargo update -w -q

# --- public API compatibility with the last published version, judged on the bumped version (0.x: breaking changes need 'minor')
revert() { git checkout -q -- Cargo.toml Cargo.lock; }
if ! command -v cargo-semver-checks > /dev/null 2>&1; then
  echo "cargo-semver-checks not found; installing"
  cargo install cargo-semver-checks --locked || echo "warning: cannot install cargo-semver-checks" >&2
fi
if command -v cargo-semver-checks > /dev/null 2>&1; then
  if curl -sf -m 20 -A "cumulus3d-release" https://crates.io/api/v1/crates/cumulus3d-core > /dev/null; then
    if ! cargo semver-checks --workspace; then
      revert
      die "public API changes need more than a '$bump' bump (breaking changes in 0.x need 'minor')"
    fi
  else
    echo "warning: crates.io unreachable or crate not published; semver check skipped" >&2
  fi
else
  echo "warning: semver check skipped (cargo-semver-checks unavailable)" >&2
fi

if [ $dry = 1 ]; then
  revert
  echo "dry run: would bump $old -> $new, update Cargo.toml/Cargo.lock/changelogs, commit 'Release $new' and push"
  exit 0
fi

# --- changelogs: "## Unreleased" becomes "## [new] — date"; a fresh empty "## Unreleased" is kept on top
today=$(date +%Y-%m-%d)
for f in CHANGELOG.md CHANGELOG.ko.md; do
  python3 -I - "$f" "$new" "$today" <<'PY'
import sys
f, new, today = sys.argv[1:]
t = open(f, encoding="utf-8").read()
t = t.replace("## Unreleased", "## Unreleased\n\n## [%s] — %s" % (new, today), 1)
open(f, "w", encoding="utf-8").write(t)
PY
done

cargo build --workspace --locked -q   # Cargo.lock consistent with the new version
git add Cargo.toml Cargo.lock CHANGELOG.md CHANGELOG.ko.md
git commit -q -m "Release $new"

# --- push (the post-commit hook may already be pushing in the background; retry until origin/main has the commit)
for i in 1 2 3 4 5; do
  git push -q origin HEAD 2> /dev/null || true
  git fetch -q origin main
  [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] && break
  sleep 3
done
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || die "push failed; run 'git push origin main' and the release workflow starts"
echo "released $new: pushed $(git rev-parse --short HEAD). The Release workflow publishes the crates and creates tag v$new:"
echo "  gh run watch \$(gh run list --workflow Release --limit 1 --json databaseId --jq '.[0].databaseId')"
