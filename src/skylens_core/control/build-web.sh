#!/usr/bin/env bash
# Builds the web bundle of the control tower into src/skylens_core/control/dist/,
# which the Go core serves at / (SKYLENS_CORE_WEB_DIR).
#
#   dist/index.html     the tower page (one full-window canvas)
#   dist/favicon.png
#   dist/pkg/           wasm + JS glue from wasm-bindgen
#
# Fonts are inside the wasm (skylens-viewer bundles them), so nothing else is copied.
#
# Needs: rustup target wasm32-unknown-unknown, wasm-bindgen-cli at the version in
# Cargo.lock. wasm-opt is used if present.
#
# Environment:
#   PROFILE=release|dev          cargo profile (default release)
#   CARGO_TARGET_DIR             honoured like cargo does
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
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
cargo build --manifest-path "$ROOT/Cargo.toml" -p skylens-control --lib --target "$TARGET" --profile "$PROFILE"

WASM="$TARGET_DIR/$TARGET/$PROFILE_DIR/skylens_control.wasm"
if [ ! -f "$WASM" ]; then
  echo "build output not found: $WASM" >&2
  exit 1
fi

echo "==> wasm-bindgen -> $DIST/pkg"
rm -rf "$DIST"
mkdir -p "$DIST/pkg"
wasm-bindgen --target web --no-typescript --out-dir "$DIST/pkg" "$WASM"

if command -v wasm-opt >/dev/null 2>&1 && [ "$PROFILE" != "dev" ]; then
  echo "==> wasm-opt -Oz"
  wasm-opt -Oz --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$DIST/pkg/skylens_control_bg.wasm" -o "$DIST/pkg/skylens_control_bg.wasm"
fi

echo "==> page"
cp "$HERE/web/index.html" "$DIST/index.html"
cp "$ROOT/res/icon/skylens-64.png" "$DIST/favicon.png"

echo "done: $DIST"
echo "  the core serves it: (cd src/skylens_core && go run .) then open http://<core>:8080/"
echo "  elsewhere: serve dist over HTTP and add ?core=ws://<core>:8080/control"
