#!/usr/bin/env bash
# Wraps the two macOS screen binaries in out/ into .app bundles, signs them
# ad-hoc and puts both in one dmg, replacing the bare binaries:
#
#   bash .github/workflows/macos-dmg.sh out macos-arm64
#
# A bare executable opened from Finder starts in Terminal and has no icon; an
# .app does not. The signature is ad-hoc ("-", no certificate): Apple Silicon
# will not run unsigned code at all, and Gatekeeper still asks once (right
# click > Open). A dmg keeps the mode bits and the signatures as they are.
set -euo pipefail

out="$1"
suffix="$2"
version="$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)"
stage="$(mktemp -d)"

# binary | bundle name | name shown in Finder and the Dock | bundle id | icon (1024 px png)
apps=(
  "skylens-client|SkyLens Status Board|SkyLens 현황판|kr.skylens.board|res/icon/status/skylens-status-1024.png"
  "skylens-control|SkyLens Control Tower|SkyLens 관제탑|kr.skylens.control|res/icon/control/skylens-control-1024.png"
)

for entry in "${apps[@]}"; do
  IFS='|' read -r bin name display id icon <<<"$entry"
  app="$stage/$name.app"
  mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
  cp "$out/$bin-$suffix" "$app/Contents/MacOS/$bin"
  chmod +x "$app/Contents/MacOS/$bin"

  # .icns from the 1024 px png: every size iconutil expects, 1x and 2x.
  set_dir="$(mktemp -d)/AppIcon.iconset"
  mkdir -p "$set_dir"
  for s in 16 32 128 256 512; do
    sips -z "$s" "$s" "$icon" --out "$set_dir/icon_${s}x${s}.png" >/dev/null
    sips -z "$((s * 2))" "$((s * 2))" "$icon" --out "$set_dir/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$set_dir" -o "$app/Contents/Resources/AppIcon.icns"

  cat >"$app/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleExecutable</key><string>$bin</string>
  <key>CFBundleIdentifier</key><string>$id</string>
  <key>CFBundleName</key><string>$name</string>
  <key>CFBundleDisplayName</key><string>$display</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
  plutil -lint "$app/Contents/Info.plist"

  codesign --force --deep --sign - --timestamp=none "$app"
  codesign --verify --deep --strict --verbose=2 "$app"
  rm "$out/$bin-$suffix"
done

# Drag-to-install: the two apps next to a link to /Applications.
ln -s /Applications "$stage/Applications"
hdiutil create -volname "SkyLens" -srcfolder "$stage" -ov -format UDZO "$out/SkyLens-$suffix.dmg"
ls -l "$out"
