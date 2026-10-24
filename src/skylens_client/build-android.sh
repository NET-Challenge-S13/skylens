#!/usr/bin/env bash
# Builds the Android APK of the status board with cargo-apk, and optionally installs and starts it.
#
#   bash src/skylens_client/build-android.sh [all|device|emulator] [--install] [--run]
#
#   all        aarch64 (phones, tablets) + x86_64 (emulator) in one APK (default)
#   device     aarch64-linux-android only
#   emulator   x86_64-linux-android only
#   --install  adb install -r the APK on the connected device or emulator
#   --run      also start it (implies --install) and follow its log (adb logcat -s skylens)
#
# Output: target/<profile>/apk/skylens-board.apk (package kr.skylens.board).
#
# Needs:
#   cargo install cargo-apk
#   rustup target add aarch64-linux-android x86_64-linux-android
#   Android SDK with platforms;android-35, build-tools and an NDK (r25 or newer)
#   a JDK (keytool and apksigner run on it), e.g. Android Studio's or any JetBrains IDE's jbr
#
# Environment (found automatically when unset):
#   ANDROID_HOME       SDK root          default %LOCALAPPDATA%\Android\Sdk or ~/Android/Sdk
#   ANDROID_NDK_ROOT   NDK root          default the newest $ANDROID_HOME/ndk/<version>
#   JAVA_HOME          JDK root          default Android Studio's jbr, else a JetBrains IDE's jbr
#   PROFILE            release|dev       default release (a dev build is very slow on a phone)
#   CARGO_APK_RELEASE_KEYSTORE, CARGO_APK_RELEASE_KEYSTORE_PASSWORD
#                      release signing key; default the SDK debug keystore (~/.android/debug.keystore,
#                      created here if missing). Fine for installing with adb, not for a store.
#   CARGO_TARGET_DIR   honoured like cargo does
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PROFILE="${PROFILE:-release}"
TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
PACKAGE=kr.skylens.board

WHICH=all
INSTALL=0
RUN=0
for arg in "$@"; do
  case "$arg" in
    all|device|emulator) WHICH="$arg" ;;
    --install) INSTALL=1 ;;
    --run) INSTALL=1; RUN=1 ;;
    -h|--help) sed -n '2,29p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $arg (see --help)" >&2; exit 2 ;;
  esac
done

# Windows (Git Bash): tools started from here are Windows programs and want Windows paths.
winpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s\n' "$1"; fi
}
unixpath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s\n' "$1"; fi
}

# --- SDK and NDK ------------------------------------------------------------------------------
if [ -z "${ANDROID_HOME:-}" ]; then
  for c in "${ANDROID_SDK_ROOT:-}" "${LOCALAPPDATA:+$(unixpath "$LOCALAPPDATA")/Android/Sdk}" "$HOME/Android/Sdk" "$HOME/Library/Android/sdk"; do
    if [ -n "$c" ] && [ -d "$c/platforms" ]; then ANDROID_HOME="$c"; break; fi
  done
fi
if [ -z "${ANDROID_HOME:-}" ]; then
  echo "Android SDK not found; set ANDROID_HOME." >&2
  exit 1
fi
SDK="$(unixpath "$ANDROID_HOME")"
if [ ! -d "$SDK/platforms/android-35" ]; then
  echo "platforms;android-35 is missing in $SDK (target_sdk_version in Cargo.toml)." >&2
  echo "  sdkmanager \"platforms;android-35\"" >&2
  exit 1
fi

if [ -z "${ANDROID_NDK_ROOT:-}" ]; then
  NEWEST="$(ls -1 "$SDK/ndk" 2>/dev/null | sort -V | tail -n 1 || true)"
  if [ -n "$NEWEST" ]; then ANDROID_NDK_ROOT="$SDK/ndk/$NEWEST"; fi
fi
if [ -z "${ANDROID_NDK_ROOT:-}" ] || [ ! -d "$(unixpath "$ANDROID_NDK_ROOT")" ]; then
  echo "Android NDK not found; install one (sdkmanager \"ndk;28.0.12433566\") or set ANDROID_NDK_ROOT." >&2
  exit 1
fi

# --- JDK --------------------------------------------------------------------------------------
has_keytool() { [ -x "$1/bin/keytool" ] || [ -x "$1/bin/keytool.exe" ]; }
if [ -z "${JAVA_HOME:-}" ] || ! has_keytool "$(unixpath "$JAVA_HOME")"; then
  JAVA_HOME=""
  CANDIDATES=(
    "/c/Program Files/Android/Android Studio/jbr"
    "${LOCALAPPDATA:+$(unixpath "$LOCALAPPDATA")/Programs/Android Studio/jbr}"
    "/opt/android-studio/jbr"
    "/Applications/Android Studio.app/Contents/jbr/Contents/Home"
  )
  if [ -n "${LOCALAPPDATA:-}" ]; then
    for d in "$(unixpath "$LOCALAPPDATA")"/Programs/*/jbr; do CANDIDATES+=("$d"); done
  fi
  for c in "${CANDIDATES[@]}"; do
    if [ -n "$c" ] && has_keytool "$c"; then JAVA_HOME="$c"; break; fi
  done
fi
if [ -z "$JAVA_HOME" ]; then
  if command -v keytool >/dev/null 2>&1; then
    JAVA_HOME="$(cd "$(dirname "$(command -v keytool)")/.." && pwd)"
  else
    echo "No JDK found (keytool, apksigner need one); set JAVA_HOME, e.g. to Android Studio's jbr." >&2
    exit 1
  fi
fi
JAVA_UNIX="$(unixpath "$JAVA_HOME")"
export PATH="$JAVA_UNIX/bin:$PATH"

export ANDROID_HOME="$(winpath "$SDK")"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export ANDROID_NDK_ROOT="$(winpath "$(unixpath "$ANDROID_NDK_ROOT")")"
export JAVA_HOME="$(winpath "$JAVA_UNIX")"

# --- tools ------------------------------------------------------------------------------------
if ! cargo apk --help >/dev/null 2>&1; then
  echo "cargo-apk not found." >&2
  echo "  cargo install cargo-apk" >&2
  exit 1
fi

case "$WHICH" in
  device) TARGETS=(aarch64-linux-android) ;;
  emulator) TARGETS=(x86_64-linux-android) ;;
  all) TARGETS=(aarch64-linux-android x86_64-linux-android) ;;
esac
INSTALLED_TARGETS="$(rustup target list --installed 2>/dev/null || true)"
for t in "${TARGETS[@]}"; do
  if ! grep -qx "$t" <<<"$INSTALLED_TARGETS"; then
    echo "Rust target $t is not installed." >&2
    echo "  rustup target add $t" >&2
    exit 1
  fi
done

# --- signing ----------------------------------------------------------------------------------
# cargo-apk signs dev builds with the debug keystore by itself and wants a key for other profiles.
if [ "$PROFILE" = "dev" ]; then
  PROFILE_DIR=debug
  PROFILE_ARGS=()
else
  PROFILE_DIR="$PROFILE"
  if [ "$PROFILE" = "release" ]; then PROFILE_ARGS=(--release); else PROFILE_ARGS=(--profile "$PROFILE"); fi
  VAR_PROFILE="$(printf '%s' "$PROFILE" | tr '[:lower:]-' '[:upper:]_')"
  KEY_VAR="CARGO_APK_${VAR_PROFILE}_KEYSTORE"
  PASS_VAR="CARGO_APK_${VAR_PROFILE}_KEYSTORE_PASSWORD"
  if [ -z "${!KEY_VAR:-}" ]; then
    DEBUG_KEYSTORE="$HOME/.android/debug.keystore"
    if [ ! -f "$DEBUG_KEYSTORE" ]; then
      echo "==> creating the debug keystore $DEBUG_KEYSTORE"
      mkdir -p "$HOME/.android"
      keytool -genkeypair -v -keystore "$(winpath "$DEBUG_KEYSTORE")" -storepass android \
        -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 -validity 10000 \
        -dname "CN=Android Debug,O=Android,C=US"
    fi
    export "$KEY_VAR=$(winpath "$DEBUG_KEYSTORE")"
    export "$PASS_VAR=android"
    echo "==> signing with the debug keystore (set $KEY_VAR for a real key)"
  fi
fi

# --- build ------------------------------------------------------------------------------------
echo "==> cargo apk build ($PROFILE, ${TARGETS[*]})"
echo "    ANDROID_HOME=$ANDROID_HOME"
echo "    ANDROID_NDK_ROOT=$ANDROID_NDK_ROOT"
echo "    JAVA_HOME=$JAVA_HOME"
TARGET_ARGS=()
if [ "$WHICH" != "all" ]; then
  # Without --target cargo-apk builds every build_targets entry of Cargo.toml into one APK.
  TARGET_ARGS=(--target "${TARGETS[0]}")
fi
(cd "$ROOT" && cargo apk build -p skylens-client --lib "${PROFILE_ARGS[@]}" "${TARGET_ARGS[@]}")

APK="$TARGET_DIR/$PROFILE_DIR/apk/skylens-board.apk"
if [ ! -f "$APK" ]; then
  echo "build output not found: $APK" >&2
  exit 1
fi
echo "done: $APK"

# --- install and run --------------------------------------------------------------------------
ADB="$SDK/platform-tools/adb"
if [ ! -x "$ADB" ] && [ -x "$ADB.exe" ]; then ADB="$ADB.exe"; fi
if [ ! -x "$ADB" ]; then ADB=adb; fi

if [ "$INSTALL" = 1 ]; then
  echo "==> adb install"
  "$ADB" install -r "$(winpath "$APK")"
fi
if [ "$RUN" = 1 ]; then
  echo "==> starting $PACKAGE (Ctrl+C stops the log, not the app)"
  "$ADB" logcat -c || true
  "$ADB" shell am start -n "$PACKAGE/android.app.NativeActivity"
  "$ADB" logcat -s skylens RustStdoutStderr AndroidRuntime
fi

if [ "$INSTALL" = 0 ]; then
  echo "  install: \"$ADB\" install -r \"$APK\""
  echo "  core:    adb reverse tcp:8080 tcp:8080   (the default ws://127.0.0.1:8080/viewer then reaches this PC)"
  echo "           or adb shell setprop debug.skylens.core ws://<core or gateway>:8080/viewer"
fi
