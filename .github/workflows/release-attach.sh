#!/usr/bin/env bash
# Attaches build outputs to the release of the tag being built, creating the
# release the first time.
#
# Usage: release-attach.sh <file> [file...]
#
# Every platform job calls this, so the first one to finish creates the release
# and the others add to it. gh release create fails when the release is already
# there, which is why the create is guarded rather than retried.
set -euo pipefail

[ $# -gt 0 ] || { echo "nothing to attach" >&2; exit 1; }

tag="${GITHUB_REF_NAME:?no tag}"

if ! gh release view "$tag" >/dev/null 2>&1; then
  gh release create "$tag" --title "$tag" --notes "Desktop and Android builds of the two screens, from the commit this tag points at.

The servers do not use these: Daejeon and the bridge take the web builds from ci-screens. The status board is also a web page, served by the bridge, if you would rather not install anything.

- skylens-client: the status board (points, mesh, detections, live video)
- skylens-control: the control tower (map, route planning, telemetry)
- skylens-board-android.apk: the status board on Android, phones and emulators in one file

Each screen takes the core it should connect to: \`--core <host:port>\`.

None of this is signed for distribution. macOS and Linux binaries are unsigned, so macOS asks you to allow the app in System Settings (or run \`xattr -d com.apple.quarantine <file>\`), and the APK carries the Android debug key, so install it with \`adb install -r\` or allow unknown sources. The APK has not been checked on a real device yet." || true
fi

gh release upload "$tag" "$@" --clobber
