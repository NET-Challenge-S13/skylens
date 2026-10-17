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
- SkyLens-macos-arm64.dmg: both screens as macOS apps (Apple Silicon)
- skylens-board-android.apk: the status board on Android, phones and emulators in one file

The status board connects to the bridge (wss://175.126.98.44:8443) first and to a core on the same machine (127.0.0.1:8080) when the bridge does not answer. The control tower connects to 127.0.0.1:8080. Either takes another core with \`--core <host:port>\`.

None of this is signed for distribution. The macOS apps carry an ad-hoc signature only, so the first start is right-click > Open (or \`xattr -dr com.apple.quarantine <app>\`); the Linux binaries are unsigned; the APK is signed with the project's own key, not a store key, so allow unknown sources (or \`adb install -r\`); later releases install over it. The APK has not been checked on a real device yet." || true
fi

gh release upload "$tag" "$@" --clobber
