#!/usr/bin/env bash
# Builds the web bundle of the status board into src/skylens_client/dist/.
#
#   dist/status.html    the board (the gateway serves this one: nginx `index status.html`)
#   dist/index.html     the same page
#   dist/pkg/           wasm + JS glue from wasm-bindgen
#   dist/favicon.png
#
# Fonts are inside the wasm (skylens-viewer bundles them), so nothing else is fetched.
#
# Needs: rustup target wasm32-unknown-unknown, wasm-bindgen-cli at the version in Cargo.lock.
# wasm-opt is used if present.
#
# Environment:
#   PROFILE=release|dev          cargo profile (default release)
#   CARGO_TARGET_DIR             honoured like cargo does
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DIST="$HERE/dist"
PROFILE="${PROFILE:-release}"
TARGET=wasm32-unknown-unknown
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"

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

WASM="$TARGET_DIR/$TARGET/$PROFILE_DIR/status_screen.wasm"
if [ ! -f "$WASM" ]; then
  echo "build output not found: $WASM" >&2
  exit 1
fi

echo "==> wasm-bindgen -> $DIST/pkg"
rm -rf "$DIST"
mkdir -p "$DIST/pkg"
wasm-bindgen --target web --no-typescript --out-dir "$DIST/pkg" "$WASM"

if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" != "dev" ]; then
  # reference-types is not optional: the bundle has an externref table, and without
  # the feature wasm-opt renumbers the tables and leaves __wbindgen_externrefs
  # pointing at the funcref one. The screen then dies on its first line, in
  # Table.grow, with nothing wrong in the build log or the file size.
  # .github/workflows/check-wasm-exports.py is what catches it.
  echo "==> wasm-opt -Oz"
  wasm-opt -Oz --enable-reference-types --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$DIST/pkg/status_screen_bg.wasm" -o "$DIST/pkg/status_screen_bg.wasm"
fi

echo "==> pages"
cp "$HERE/web/index.html" "$DIST/status.html"
cp "$HERE/web/index.html" "$DIST/index.html"
cp "$ROOT/res/icon/status/skylens-status-64.png" "$DIST/favicon.png"

echo "done: $DIST"
echo "  serve it over HTTP (wasm cannot load from file://), e.g. python -m http.server -d \"$DIST\" 8000"
echo "  then open /status.html?core=ws://<core or gateway>:8080/viewer"
