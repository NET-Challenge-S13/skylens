#!/usr/bin/env bash
# Builds the web bundle of skylens_client into src/skylens_client/dist/.
#
#   dist/index.html     status board (default screen)
#   dist/status.html    status board   (the gateway serves this one)
#   dist/control.html   control tower  (the core serves this one)
#   dist/pkg/           wasm + JS glue from wasm-bindgen
#   dist/fonts/         Korean font registered by the wasm app
#
# Needs: rustup target wasm32-unknown-unknown, wasm-bindgen-cli at the version in
# Cargo.lock, curl (first run only, for the font). wasm-opt is used if present.
#
# Environment:
#   PROFILE=release|dev          cargo profile (default release)
#   SKYLENS_FONT_FILE=path.ttf   use this font instead of downloading Noto Sans KR
#   CARGO_TARGET_DIR             honoured like cargo does
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$HERE/dist"
PROFILE="${PROFILE:-release}"
TARGET=wasm32-unknown-unknown
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
FONT_CACHE="$HERE/web/fonts/NotoSansKR.ttf"
FONT_URL="${SKYLENS_FONT_URL:-https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR%5Bwght%5D.ttf}"

if [ "$PROFILE" = "dev" ]; then PROFILE_DIR=debug; else PROFILE_DIR="$PROFILE"; fi

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "wasm-bindgen-cli not found." >&2
  echo "  cargo install wasm-bindgen-cli --version <wasm-bindgen version in Cargo.lock> --locked" >&2
  exit 1
fi

# wasm-bindgen-cli must match the wasm-bindgen crate exactly.
LOCK_VER="$(awk '/^name = "wasm-bindgen"$/ { getline; gsub(/version = |"/, ""); print; exit }' "$ROOT/Cargo.lock" || true)"
CLI_VER="$(wasm-bindgen --version | awk '{ print $2 }')"
if [ -n "$LOCK_VER" ] && [ "$LOCK_VER" != "$CLI_VER" ]; then
  echo "wasm-bindgen-cli $CLI_VER does not match wasm-bindgen $LOCK_VER in Cargo.lock." >&2
  echo "  cargo install wasm-bindgen-cli --version $LOCK_VER --locked" >&2
  exit 1
fi

echo "==> cargo build ($PROFILE, $TARGET)"
cargo build --manifest-path "$ROOT/Cargo.toml" -p skylens-client --lib --target "$TARGET" --profile "$PROFILE"

WASM="$TARGET_DIR/$TARGET/$PROFILE_DIR/skylens_client.wasm"
if [ ! -f "$WASM" ]; then
  echo "build output not found: $WASM" >&2
  exit 1
fi

echo "==> wasm-bindgen -> $DIST/pkg"
rm -rf "$DIST"
mkdir -p "$DIST/pkg" "$DIST/fonts"
wasm-bindgen --target web --no-typescript --out-dir "$DIST/pkg" "$WASM"

if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" != "dev" ]; then
  echo "==> wasm-opt -Oz"
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$DIST/pkg/skylens_client_bg.wasm" -o "$DIST/pkg/skylens_client_bg.wasm"
fi

echo "==> pages"
cp "$HERE/web/index.html" "$DIST/index.html"
cp "$HERE/../../res/icon/skylens-64.png" "$DIST/favicon.png"
cp "$HERE/web/index.html" "$DIST/status.html"
sed 's/<meta name="skylens-screen" content="status" \/>/<meta name="skylens-screen" content="control" \/>/' \
  "$HERE/web/index.html" > "$DIST/control.html"
grep -q 'content="control"' "$DIST/control.html" || { echo "control.html: screen meta not set" >&2; exit 1; }

echo "==> font"
if [ -n "${SKYLENS_FONT_FILE:-}" ]; then
  cp "$SKYLENS_FONT_FILE" "$DIST/fonts/NotoSansKR.ttf"
else
  if [ ! -s "$FONT_CACHE" ]; then
    mkdir -p "$(dirname "$FONT_CACHE")"
    echo "    downloading Noto Sans KR (OFL) once into web/fonts/"
    curl -fL --retry 2 -o "$FONT_CACHE" "$FONT_URL"
  fi
  cp "$FONT_CACHE" "$DIST/fonts/NotoSansKR.ttf"
fi

echo "done: $DIST"
echo "  serve it over HTTP (wasm cannot load from file://), e.g. python -m http.server -d \"$DIST\" 8000"
echo "  then open /control.html?core=ws://<core>:8080/viewer or /status.html"
