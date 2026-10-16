#!/usr/bin/env bash
# Re-vendor third_party/cumulus3d at a new upstream release, carrying the SkyLens
# patches forward with a per-file 3-way merge.
#
#   scripts/cumulus3d-sync.sh resolve [--version X.Y.Z | --tag vX.Y.Z]   decide the target
#   scripts/cumulus3d-sync.sh apply   [--version X.Y.Z | --tag vX.Y.Z]   re-vendor + update pins
#   scripts/cumulus3d-sync.sh check                                      build/test checks only
#   scripts/cumulus3d-sync.sh run     [--version ...] [--no-checks]      apply, then check
#
# Without --version/--tag the newest stable cumulus3d-cli on crates.io is used.
# The script only edits the working tree and writes a report to $SYNC_OUT
# (default ./.cumulus3d-sync). It never commits or pushes, so it is safe to run locally.
# Outputs are also written to $GITHUB_OUTPUT when set.
#
# Exit codes: 0 ok (or nothing to do), 2 merge conflicts remain, 3 checks failed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/third_party/cumulus3d"
UPSTREAM_FILE="$VENDOR/UPSTREAM"
OUT="${SYNC_OUT:-$ROOT/.cumulus3d-sync}"
UA="skylens-cumulus3d-sync (https://github.com/NET-Challenge-S13/skylens)"

die() { echo "error: $*" >&2; exit 1; }
say() { echo "[cumulus3d-sync] $*" >&2; }
gh_out() { if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "$1=$2" >> "$GITHUB_OUTPUT"; fi; echo "$1=$2" >> "$OUT/result.env"; }

# --- UPSTREAM file -----------------------------------------------------------
up_get() { sed -n "s/^$1=//p" "$UPSTREAM_FILE" | head -n1; }
REPO="$(up_get repository)"; OLD_REV="$(up_get rev)"; OLD_VERSION="$(up_get version)"
[ -n "$REPO" ] && [ -n "$OLD_REV" ] || die "cannot read $UPSTREAM_FILE"

# --- arguments ---------------------------------------------------------------
CMD="${1:-}"; [ -n "$CMD" ] && shift || true
REQ_VERSION=""; REQ_TAG=""; RUN_CHECKS=1
while [ $# -gt 0 ]; do
  case "$1" in
    --version) REQ_VERSION="${2#v}"; shift 2 ;;
    --tag) REQ_TAG="$2"; shift 2 ;;
    --no-checks) RUN_CHECKS=0; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done
mkdir -p "$OUT"

# --- resolve target ----------------------------------------------------------
resolve_target() {
  if [ -n "$REQ_TAG" ] && [ -z "$REQ_VERSION" ]; then REQ_VERSION="${REQ_TAG#v}"; fi
  if [ -z "$REQ_VERSION" ]; then
    REQ_VERSION="$(curl -fsS -A "$UA" https://crates.io/api/v1/crates/cumulus3d-cli | jq -r '.crate.max_stable_version')"
    [ -n "$REQ_VERSION" ] && [ "$REQ_VERSION" != null ] || die "crates.io lookup failed"
  fi
  NEW_VERSION="$REQ_VERSION"; NEW_TAG="${REQ_TAG:-v$NEW_VERSION}"
  # Prefer the peeled commit of an annotated tag.
  NEW_REV="$(git ls-remote "$REPO" "refs/tags/$NEW_TAG^{}" | awk '{print $1}')"
  [ -n "$NEW_REV" ] || NEW_REV="$(git ls-remote "$REPO" "refs/tags/$NEW_TAG" | awk '{print $1}')"
  [ -n "$NEW_REV" ] || die "tag $NEW_TAG not found in $REPO"
  if [ "$NEW_REV" = "$OLD_REV" ]; then NEEDS_UPDATE=false
  elif [ -z "$REQ_TAG" ] && [ "$(printf '%s\n%s\n' "$OLD_VERSION" "$NEW_VERSION" | sort -V | tail -n1)" = "$OLD_VERSION" ] \
       && [ "$OLD_VERSION" != "$NEW_VERSION" ]; then NEEDS_UPDATE=false  # target is older than what we use
  else NEEDS_UPDATE=true; fi
  : > "$OUT/result.env"
  gh_out old_version "$OLD_VERSION"; gh_out old_rev "$OLD_REV"
  gh_out new_version "$NEW_VERSION"; gh_out new_tag "$NEW_TAG"; gh_out new_rev "$NEW_REV"
  gh_out needs_update "$NEEDS_UPDATE"
  say "current $OLD_VERSION (${OLD_REV:0:9}) -> target $NEW_VERSION ($NEW_TAG, ${NEW_REV:0:9}); needs_update=$NEEDS_UPDATE"
}

# --- re-vendor with per-file 3-way merge -------------------------------------
fetch_tree() { # <rev> <dest>: upstream tree at rev
  mkdir -p "$2"
  git -C "$TMP/repo" fetch -q --depth=1 "$REPO" "$1"
  git -C "$TMP/repo" archive "$1" | tar -x -C "$2"
}

same() { # both missing counts as same
  if [ -e "$1" ] && [ -e "$2" ]; then cmp -s "$1" "$2"; elif [ ! -e "$1" ] && [ ! -e "$2" ]; then return 0; else return 1; fi
}

apply_sync() {
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  git init -q "$TMP/repo"
  say "fetching base ${OLD_REV:0:9} and target $NEW_TAG"
  fetch_tree "$OLD_REV" "$TMP/base"; fetch_tree "$NEW_REV" "$TMP/target"
  # Never treat build output or a stray lockfile of the vendored workspace as source.
  rm -rf "$VENDOR/target" "$VENDOR/Cargo.lock"
  mkdir -p "$TMP/merged"; cp -Rp "$VENDOR/." "$TMP/merged/"
  rm -f "$TMP/merged/UPSTREAM"

  diff -ruN --exclude=UPSTREAM --exclude=PATCHES.md "$TMP/base" "$VENDOR" > "$OUT/skylens-patch.diff" || true
  : > "$OUT/merge-status.tsv"
  (cd "$TMP/base" && find . -type f; cd "$TMP/target" && find . -type f; cd "$VENDOR" && find . -type f) \
    | sed 's|^\./||' | sort -u > "$TMP/paths"

  local p b t v n
  while IFS= read -r p; do
    # UPSTREAM is ours; Cargo.lock is deliberately not vendored (the root workspace locks).
    [ "$p" = UPSTREAM ] || [ "$p" = Cargo.lock ] && continue
    b="$TMP/base/$p"; t="$TMP/target/$p"; v="$VENDOR/$p"; m="$TMP/merged/$p"
    if same "$v" "$b"; then                       # SkyLens did not touch it
      if same "$t" "$b"; then continue; fi
      if [ -e "$t" ]; then mkdir -p "$(dirname "$m")"; cp -p "$t" "$m"
        if [ -e "$b" ]; then echo -e "UPSTREAM_UPDATED\t$p" >> "$OUT/merge-status.tsv"; else echo -e "UPSTREAM_ADDED\t$p" >> "$OUT/merge-status.tsv"; fi
      else rm -f "$m"; echo -e "UPSTREAM_DELETED\t$p" >> "$OUT/merge-status.tsv"; fi
      continue
    fi
    # SkyLens changed it from here on.
    if same "$t" "$b"; then                       # upstream did not
      if [ -e "$v" ]; then
        if [ -e "$b" ]; then echo -e "SKYLENS_KEPT\t$p" >> "$OUT/merge-status.tsv"; else echo -e "SKYLENS_ADDED\t$p" >> "$OUT/merge-status.tsv"; fi
      else echo -e "SKYLENS_DELETED\t$p" >> "$OUT/merge-status.tsv"; fi
      continue
    fi
    # Both changed it.
    if same "$v" "$t"; then                       # upstream now has the same content
      echo -e "SAME_AS_UPSTREAM\t$p" >> "$OUT/merge-status.tsv"; continue
    fi
    if [ ! -e "$v" ]; then                        # SkyLens deleted, upstream modified
      echo -e "CONFLICT_DELETE_VS_MODIFY\t$p (SkyLens deleted it, upstream changed it; kept deleted, review)" >> "$OUT/merge-status.tsv"; continue
    fi
    if [ ! -e "$t" ]; then                        # SkyLens modified, upstream deleted
      echo -e "CONFLICT_MODIFY_VS_DELETE\t$p (upstream deleted it, SkyLens version kept)" >> "$OUT/merge-status.tsv"; continue
    fi
    if ! grep -Iq . "$v" || ! grep -Iq . "$t"; then  # binary
      cp -p "$t" "$m"; echo -e "CONFLICT_BINARY\t$p (upstream copy taken)" >> "$OUT/merge-status.tsv"; continue
    fi
    local base_file="$b"; [ -e "$b" ] || base_file=/dev/null
    set +e
    git merge-file -p --diff3 -L "skylens (vendored)" -L "base ${OLD_REV:0:9}" -L "upstream $NEW_TAG" \
      "$v" "$base_file" "$t" > "$TMP/mf.out"
    n=$?
    set -e
    mkdir -p "$(dirname "$m")"; cat "$TMP/mf.out" > "$m"
    if [ "$n" -eq 0 ]; then echo -e "MERGED_CLEAN\t$p" >> "$OUT/merge-status.tsv"
    else echo -e "CONFLICT\t$p ($n hunk(s))" >> "$OUT/merge-status.tsv"; fi
  done < "$TMP/paths"

  # Install the merged tree.
  rsync -a --delete "$TMP/merged/" "$VENDOR/"
  update_pins
  write_upstream

  CONFLICTS="$(grep -c '^CONFLICT' "$OUT/merge-status.tsv" || true)"
  grep '^CONFLICT' "$OUT/merge-status.tsv" | cut -f2 > "$OUT/conflicts.txt" || true
  write_merge_report
  gh_out conflicts "$CONFLICTS"
  say "merge done: $CONFLICTS conflicting file(s); report in $OUT/sync-report.md"
}

update_pins() {
  # Both root Cargo.toml and src/skylens_model/Cargo.toml pin the git rev.
  local f
  for f in $(grep -rl --include=Cargo.toml "$OLD_REV" "$ROOT" --exclude-dir=third_party --exclude-dir=target --exclude-dir=.git --exclude-dir=.cumulus3d-sync || true); do
    perl -pi -e "s/\Q$OLD_REV\E/$NEW_REV/g" "$f"; say "updated rev pin in ${f#"$ROOT"/}"
  done
  # PATCHES.md names the base revision in its first paragraph.
  OLD_REV="$OLD_REV" NEW_REV="$NEW_REV" DESC="upstream $NEW_TAG" \
    perl -pi -e 's/\Q$ENV{OLD_REV}\E`\([^)]*\)/$ENV{NEW_REV}`($ENV{DESC})/' "$VENDOR/PATCHES.md"
}

write_upstream() {
  cat > "$UPSTREAM_FILE" <<EOT
# Machine-readable record of the upstream revision third_party/cumulus3d is based on.
# Read and rewritten by scripts/cumulus3d-sync.sh. Keep KEY=VALUE, no spaces around "=".
repository=$REPO
# Commit the vendored tree was copied from (the base of the SkyLens patches in PATCHES.md).
rev=$NEW_REV
# Workspace version at that commit.
version=$NEW_VERSION
# Release tag, empty when rev is not a release.
tag=$NEW_TAG
EOT
}

write_merge_report() {
  {
    echo "| 상태 | 파일 |"; echo "|---|---|"
    local label
    for s in CONFLICT_DELETE_VS_MODIFY CONFLICT_MODIFY_VS_DELETE CONFLICT_BINARY CONFLICT MERGED_CLEAN SAME_AS_UPSTREAM SKYLENS_KEPT SKYLENS_ADDED SKYLENS_DELETED UPSTREAM_UPDATED UPSTREAM_ADDED UPSTREAM_DELETED; do
      case $s in
        CONFLICT*) label="충돌 ($s)" ;;
        MERGED_CLEAN) label="양쪽 수정, 자동 병합 성공" ;;
        SAME_AS_UPSTREAM) label="upstream에 같은 내용이 이미 있음 (패치 삭제 후보)" ;;
        SKYLENS_KEPT) label="SkyLens 패치만 있음, 유지" ;;
        SKYLENS_ADDED) label="SkyLens가 추가한 파일, 유지" ;;
        SKYLENS_DELETED) label="SkyLens가 삭제한 파일, 계속 삭제" ;;
        UPSTREAM_UPDATED) label="upstream만 수정, 새 버전 사용" ;;
        UPSTREAM_ADDED) label="upstream이 추가" ;;
        UPSTREAM_DELETED) label="upstream이 삭제" ;;
      esac
      local n; n="$(grep -c "^$s	" "$OUT/merge-status.tsv" || true)"
      [ "$n" -gt 0 ] || continue
      if [[ $s == CONFLICT* || $s == MERGED_CLEAN || $s == SAME_AS_UPSTREAM || $s == SKYLENS_KEPT || $s == SKYLENS_ADDED || $s == SKYLENS_DELETED ]]; then
        grep "^$s	" "$OUT/merge-status.tsv" | cut -f2 | while IFS= read -r f; do echo "| $label | \`$f\` |"; done
      else
        echo "| $label | ${n}개 파일 (전체 목록은 \`merge-status.tsv\`) |"
      fi
    done
  } > "$OUT/sync-report.md"
}

# --- checks ------------------------------------------------------------------
# Name|directory|command. The vendored workspace is its own cargo workspace (excluded
# from the root one). skylens-model is the only SkyLens crate that depends on cumulus3d.
CHECKS=(
  "cumulus3d workspace check|third_party/cumulus3d|cargo check --workspace --all-targets"
  "cumulus3d unit tests (no CUDA device)|third_party/cumulus3d|cargo test --workspace --exclude cumulus3d-cuda --lib"
  "skylens-model check|.|cargo check -p skylens-model --all-targets"
  "skylens-model unit tests|.|cargo test -p skylens-model --lib"
)

run_checks() {
  mkdir -p "$OUT/checks"; : > "$OUT/check-results.tsv"
  local failed=0 i=0 spec name dir cmd start rc log
  for spec in "${CHECKS[@]}"; do
    IFS='|' read -r name dir cmd <<< "$spec"; i=$((i+1)); log="$OUT/checks/$i.log"
    if [ "${SKIP_CHECKS:-}" != "" ] && [[ ",$SKIP_CHECKS," == *",$i,"* ]]; then
      echo -e "$i\t$name\tskipped\t0" >> "$OUT/check-results.tsv"; continue; fi
    say "check $i: $name"
    start=$SECONDS; set +e; (cd "$ROOT/$dir" && eval "$cmd") > "$log" 2>&1; rc=$?; set -e
    if [ $rc -eq 0 ]; then echo -e "$i\t$name\tpass\t$((SECONDS-start))" >> "$OUT/check-results.tsv"
    else echo -e "$i\t$name\tFAIL\t$((SECONDS-start))" >> "$OUT/check-results.tsv"; failed=1; fi
  done
  rm -f "$VENDOR/Cargo.lock"  # the vendored workspace is not locked in this repo
  {
    echo "| 검사 | 결과 | 시간(초) |"; echo "|---|---|---|"
    while IFS=$'\t' read -r i name st secs; do echo "| $name | $st | $secs |"; done < "$OUT/check-results.tsv"
  } > "$OUT/check-report.md"
  : > "$OUT/check-failures.log"
  while IFS=$'\t' read -r i name st secs; do
    [ "$st" = FAIL ] && { echo "=== $name ==="; tail -n 80 "$OUT/checks/$i.log"; } >> "$OUT/check-failures.log"
  done < "$OUT/check-results.tsv"
  gh_out check_status "$([ $failed -eq 0 ] && echo pass || echo fail)"
  return $failed
}

# --- main --------------------------------------------------------------------
case "$CMD" in
  resolve) resolve_target ;;
  apply|run)
    resolve_target
    if [ "$NEEDS_UPDATE" != true ]; then say "already up to date"; exit 0; fi
    apply_sync
    ( cd "$ROOT" && cargo update -p cumulus3d-cli >/dev/null 2>&1 || cargo metadata --format-version 1 >/dev/null 2>&1 ) || true
    if [ "$CONFLICTS" -gt 0 ]; then gh_out check_status skipped-conflicts; exit 2; fi
    if [ "$CMD" = run ] && [ "$RUN_CHECKS" = 1 ]; then run_checks || exit 3; fi
    ;;
  check) : > "$OUT/result.env"; run_checks || exit 3 ;;
  *) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
