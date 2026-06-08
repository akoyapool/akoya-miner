#!/usr/bin/env bash
# Build a HiveOS package for akoya-miner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_VERSION=$(tr -d '[:space:]' < "$REPO_ROOT/version.txt" 2>/dev/null || echo "0.0.0")
VERSION="${1:-$DEFAULT_VERSION}"
PACKAGE_SUFFIX="${2:-}"
DIST_DIR="$REPO_ROOT/dist"
if [[ -n "$PACKAGE_SUFFIX" ]]; then
    PKG_NAME="akoya-miner-${VERSION}-${PACKAGE_SUFFIX}"
else
    PKG_NAME="akoya-miner-${VERSION}"
fi
PKG_DIR="$DIST_DIR/$PKG_NAME"
PACKAGE_FLAVOR="${AKOYA_PACKAGE_FLAVOR:-${PACKAGE_SUFFIX:-modern}}"
REQUIRED_CUDA_MAJOR="${AKOYA_REQUIRED_CUDA_MAJOR:-12}"
REQUIRED_CUDA_MINOR="${AKOYA_REQUIRED_CUDA_MINOR:-8}"
REQUIRED_MIN_SM="${AKOYA_REQUIRED_MIN_SM:-80}"

DOTNET="${DOTNET:-dotnet}"
if ! command -v "$DOTNET" >/dev/null 2>&1; then
    DOTNET="$HOME/.dotnet/dotnet"
fi

rm -rf "$PKG_DIR" "$DIST_DIR/$PKG_NAME.tar.gz" "$DIST_DIR/$PKG_NAME.tar.gz.sha256"
mkdir -p "$PKG_DIR/miner/lib" "$PKG_DIR/scripts"

echo "=== Akoya Miner HiveOS package ==="
echo "Version: $VERSION"
echo "Output:  $DIST_DIR/$PKG_NAME.tar.gz"

if [[ -n "${PORTABLE_ARTIFACTS_DIR:-}" ]]; then
    SRC_BIN="$PORTABLE_ARTIFACTS_DIR/akoya-miner/akoya-miner"
    [[ -f "$SRC_BIN" ]] || { echo "x no binary at $SRC_BIN"; exit 1; }
    cp "$SRC_BIN" "$PKG_DIR/miner/AkoyaMiner"
else
    "$DOTNET" publish "$REPO_ROOT/src/Akoya.Miner/Akoya.Miner.csproj" \
        -c Release -r linux-x64 --self-contained \
        -o "$PKG_DIR/miner" \
        -p:PublishAot=true \
        -p:StripSymbols=true \
        --nologo -v q
    mv "$PKG_DIR/miner/akoya-miner" "$PKG_DIR/miner/AkoyaMiner"
fi
rm -f "$PKG_DIR"/miner/*.pdb "$PKG_DIR"/miner/*.dbg

if [[ -z "${PORTABLE_ARTIFACTS_DIR:-}" ]]; then
    GEMM_H100="${PEARL_GEMM_H100_LIB:-/tmp/libpearl_gemm_capi_h100.so}"
    GEMM_VOLTA="${PEARL_GEMM_VOLTA_LIB:-/tmp/libpearl_gemm_capi_volta.so}"
    GEMM_TURING="${PEARL_GEMM_TURING_LIB:-/tmp/libpearl_gemm_capi_turing.so}"
    GEMM_PORTABLE="${PEARL_GEMM_PORTABLE_LIB:-$REPO_ROOT/native/pearl-gemm/csrc/capi/build/libpearl_gemm_capi.so}"
    GEMM_AMPERE="${PEARL_GEMM_AMPERE_LIB:-/tmp/libpearl_gemm_capi_ampere.so}"
    GEMM_ADA="${PEARL_GEMM_ADA_LIB:-/tmp/libpearl_gemm_capi_ada.so}"
    GEMM_BLACKWELL="${PEARL_GEMM_BLACKWELL_LIB:-/tmp/libpearl_gemm_capi_blackwell.so}"
    GEMM_B200="${PEARL_GEMM_B200_LIB:-/tmp/libpearl_gemm_capi_b200.so}"
    MINING_LIB="${PEARL_MINING_LIB:-$REPO_ROOT/native/target/release/libpearl_mining_capi.so}"
fi

copy_lib() {
    local src="$1"
    local dest="$2"
    [[ -f "$src" ]] || { echo "x missing native lib: $src"; exit 1; }
    cp "$src" "$PKG_DIR/miner/lib/$dest"
}

if [[ -n "${PORTABLE_ARTIFACTS_DIR:-}" ]]; then
    shopt -s nullglob
    gemm_libs=("$PORTABLE_ARTIFACTS_DIR"/lib/libpearl_gemm_capi_*.so)
    shopt -u nullglob
    [[ "${#gemm_libs[@]}" -gt 0 ]] || { echo "x no GEMM libs in $PORTABLE_ARTIFACTS_DIR/lib"; exit 1; }
    for src in "${gemm_libs[@]}"; do
        copy_lib "$src" "$(basename "$src")"
    done
    copy_lib "$PORTABLE_ARTIFACTS_DIR/lib/libpearl_mining_capi.so" "libpearl_mining_capi.so"
else
    copy_lib "$GEMM_H100" "libpearl_gemm_capi_h100.so"
    copy_lib "$GEMM_VOLTA" "libpearl_gemm_capi_volta.so"
    copy_lib "$GEMM_TURING" "libpearl_gemm_capi_turing.so"
    copy_lib "$GEMM_PORTABLE" "libpearl_gemm_capi_portable.so"
    copy_lib "$GEMM_AMPERE" "libpearl_gemm_capi_ampere.so"
    copy_lib "$GEMM_ADA" "libpearl_gemm_capi_ada.so"
    copy_lib "$GEMM_BLACKWELL" "libpearl_gemm_capi_blackwell.so"
    copy_lib "$GEMM_B200" "libpearl_gemm_capi_b200.so"
    copy_lib "$MINING_LIB" "libpearl_mining_capi.so"
fi
strip --strip-all "$PKG_DIR/miner/lib/"*.so 2>/dev/null || true

cp "$REPO_ROOT/hiveos/h-manifest.conf" "$PKG_DIR/"
cp "$REPO_ROOT/hiveos/h-config.sh" "$PKG_DIR/"
cp "$REPO_ROOT/hiveos/h-run.sh" "$PKG_DIR/"
cp "$REPO_ROOT/hiveos/h-stats.sh" "$PKG_DIR/"
cp "$REPO_ROOT/hiveos/h-readme.md" "$PKG_DIR/"
cp "$REPO_ROOT/hiveos/scripts/check-driver.sh" "$PKG_DIR/scripts/"

chmod +x "$PKG_DIR/h-config.sh" "$PKG_DIR/h-run.sh" "$PKG_DIR/h-stats.sh" \
         "$PKG_DIR/scripts/check-driver.sh" "$PKG_DIR/miner/AkoyaMiner"

sed -i "s/CUSTOM_VERSION=\".*\"/CUSTOM_VERSION=\"$VERSION\"/" "$PKG_DIR/h-manifest.conf"
cat >> "$PKG_DIR/h-manifest.conf" <<EOF
AKOYA_PACKAGE_FLAVOR="$PACKAGE_FLAVOR"
AKOYA_REQUIRED_CUDA_MAJOR="$REQUIRED_CUDA_MAJOR"
AKOYA_REQUIRED_CUDA_MINOR="$REQUIRED_CUDA_MINOR"
AKOYA_REQUIRED_MIN_SM="$REQUIRED_MIN_SM"
EOF
echo "$VERSION" > "$PKG_DIR/miner/VERSION"

bash "$SCRIPT_DIR/leak-scan.sh" "$PKG_DIR"

cd "$DIST_DIR"
tar -czf "$PKG_NAME.tar.gz" "$PKG_NAME/"
sha256sum "$PKG_NAME.tar.gz" > "$PKG_NAME.tar.gz.sha256"

echo "Package: $DIST_DIR/$PKG_NAME.tar.gz"
