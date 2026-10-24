#!/usr/bin/env bash
# Publishes build outputs to the rolling release the hosts pull from, and writes
# the manifest that tells them what changed.
#
# Usage: publish-artifact.sh <manifest name> <file> [file...]
#
# One release holds the current build of develop. Assets are replaced in place,
# so the download URL never changes and a host can ask for the same path every
# minute:
#   https://github.com/<repo>/releases/download/deploy-develop/<file>
#
# Each workflow writes its own manifest (manifest-core.json, manifest-screens.json).
# One shared manifest would be a lost update the moment two workflows publish at
# the same time, and they do: a commit that touches the proto starts both.
#
# The manifest carries a sha256 per file, which is how a host decides whether to
# download at all. A build that did not run leaves its manifest untouched, so its
# files stay at the version that is already on the hosts.
set -euo pipefail

MANIFEST="${1:?manifest name}"
shift
[ $# -gt 0 ] || { echo "no files to publish" >&2; exit 1; }

TAG="${RELEASE_TAG:-deploy-develop}"

if ! gh release view "$TAG" >/dev/null 2>&1; then
  gh release create "$TAG" \
    --target develop \
    --prerelease \
    --title "develop (rolling)" \
    --notes "The current build of develop. The Daejeon and bridge hosts pull from here; see docs/DEPLOYMENT.md. Assets are replaced on every build, so do not link to a file here expecting it to stay put."
fi

{
  printf '{\n'
  printf '  "commit": "%s",\n' "${GITHUB_SHA:-unknown}"
  printf '  "built_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "run": "%s",\n' "${GITHUB_RUN_ID:-local}"
  printf '  "files": {\n'
  first=1
  for f in "$@"; do
    # The separator is printed as a format string, never passed as an argument:
    # printf expands escapes in the format only, so a ",\n" argument lands in
    # the file as those two characters and the JSON stops parsing.
    if [ "$first" = 0 ]; then printf ',\n'; fi
    printf '    "%s": "%s"' "$(basename "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
    first=0
  done
  printf '\n  }\n}\n'
} > "$MANIFEST"

# A manifest that does not parse stops every host from deploying, and does it
# quietly. Fail the build here instead.
python3 -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$MANIFEST"

cat "$MANIFEST"
gh release upload "$TAG" "$@" "$MANIFEST" --clobber
