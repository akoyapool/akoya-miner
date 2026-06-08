#!/usr/bin/env bash
# Build a standalone akoya-miner tarball.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_VERSION=$(tr -d '[:space:]' < "$REPO_ROOT/version.txt" 2>/dev/null || echo "0.0.0")
VERSION="${1:-$DEFAULT_VERSION}"
ARCH="${2:-portable}"
DIST_DIR="$REPO_ROOT/dist"
PKG_NAME="akoya-miner-${VERSION}-${ARCH}"
PKG_DIR="$DIST_DIR/$PKG_NAME"

DOTNET="${DOTNET:-dotnet}"
if ! command -v "$DOTNET" >/dev/null 2>&1; then
    DOTNET="$HOME/.dotnet/dotnet"
fi

rm -rf "$PKG_DIR" "$DIST_DIR/$PKG_NAME.tar.gz" "$DIST_DIR/$PKG_NAME.tar.gz.sha256"
mkdir -p "$PKG_DIR/lib"

echo "=== Akoya Miner standalone package ==="
echo "Version: $VERSION"
echo "Output:  $DIST_DIR/$PKG_NAME.tar.gz"

if [[ -n "${PORTABLE_ARTIFACTS_DIR:-}" ]]; then
    SRC_BIN="$PORTABLE_ARTIFACTS_DIR/akoya-miner/akoya-miner"
    [[ -f "$SRC_BIN" ]] || { echo "x no binary at $SRC_BIN"; exit 1; }
    cp "$SRC_BIN" "$PKG_DIR/akoya-miner.bin"
else
    "$DOTNET" publish "$REPO_ROOT/src/Akoya.Miner/Akoya.Miner.csproj" \
        -c Release -r linux-x64 --self-contained \
        -o "$PKG_DIR/_pub" \
        -p:PublishAot=true \
        -p:StripSymbols=true \
        --nologo -v q
    mv "$PKG_DIR/_pub/akoya-miner" "$PKG_DIR/akoya-miner.bin"
    rm -rf "$PKG_DIR/_pub"
fi
chmod +x "$PKG_DIR/akoya-miner.bin"

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
    cp "$src" "$PKG_DIR/lib/$dest"
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
strip --strip-all "$PKG_DIR/lib/"*.so 2>/dev/null || true

cat > "$PKG_DIR/akoya-miner" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
LIB="$DIR/lib"
TARGET="$LIB/libpearl_gemm_capi.so"

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]') || true
MAJOR="${CC%%.*}"
MINOR="${CC#*.}"

variant_exists() { [[ -f "$LIB/libpearl_gemm_capi_$1.so" ]]; }
select_variant() {
    local forced="${AKOYA_GEMM_VARIANT:-auto}"
    case "$forced" in
        auto|"") ;;
        h100|volta|turing|portable|ampere|ada|blackwell|b200)
            if ! variant_exists "$forced"; then
                echo "AKOYA_GEMM_VARIANT=$forced requested, but libpearl_gemm_capi_${forced}.so is missing" >&2
                exit 64
            fi
            echo "$forced"; return ;;
        *)
            echo "invalid AKOYA_GEMM_VARIANT=$forced (expected auto|volta|turing|portable|ampere|ada|blackwell|b200|h100)" >&2
            exit 64 ;;
    esac

    if [[ "$MAJOR" == "7" && "$MINOR" == "0" ]] && variant_exists volta; then echo volta
    elif [[ "$MAJOR" == "7" && "$MINOR" == "5" ]] && variant_exists turing; then echo turing
    elif [[ "$MAJOR" == "10" ]] && variant_exists b200; then echo b200
    elif [[ "$MAJOR" == "12" ]] && variant_exists blackwell; then echo blackwell
    elif [[ "$MAJOR" == "9" ]] && variant_exists h100; then echo h100
    elif [[ "$MAJOR" == "8" && "$MINOR" == "9" ]] && variant_exists ada; then echo ada
    elif [[ "$MAJOR" == "8" ]] && variant_exists ampere; then echo ampere
    else echo portable
    fi
}

VARIANT="$(select_variant)"
if [[ "$VARIANT" == "b200" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_b200.so" "$TARGET"
elif [[ "$VARIANT" == "volta" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_volta.so" "$TARGET"
elif [[ "$VARIANT" == "turing" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_turing.so" "$TARGET"
elif [[ "$VARIANT" == "blackwell" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_blackwell.so" "$TARGET"
elif [[ "$VARIANT" == "h100" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_h100.so" "$TARGET"
elif [[ "$VARIANT" == "ada" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_ada.so" "$TARGET"
elif [[ "$VARIANT" == "ampere" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_ampere.so" "$TARGET"
else
    ln -sf "$LIB/libpearl_gemm_capi_portable.so" "$TARGET"
fi

export LD_LIBRARY_PATH="${LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[[ -d "$LIB/cuda" ]] && export LD_LIBRARY_PATH="$LIB/cuda:$LD_LIBRARY_PATH"
[[ -d /usr/lib/wsl/lib ]] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib/wsl/lib"
export AKOYA_PEARL_GEMM_LIB="${AKOYA_PEARL_GEMM_LIB:-$TARGET}"
export AKOYA_PEARL_MINING_LIB="${AKOYA_PEARL_MINING_LIB:-$LIB/libpearl_mining_capi.so}"
exec "$DIR/akoya-miner.bin" "$@"
LAUNCHER
chmod +x "$PKG_DIR/akoya-miner"

echo "$VERSION" > "$PKG_DIR/VERSION"
cat > "$PKG_DIR/README.md" <<README
# Akoya Miner ${VERSION}

Standalone pool miner for Pearl.

Required:

    AKOYA_POOL_WALLET=PRL1... ./akoya-miner mine-blocks

Optional:

    AKOYA_POOL_HOST=pool-v2.akoyapool.com
    AKOYA_POOL_PORT=443
    AKOYA_POOL_TLS=1
    AKOYA_POOL_WORKER=my-rig
    AKOYA_GPU_INDICES=all
README

bash "$SCRIPT_DIR/leak-scan.sh" "$PKG_DIR"

cd "$DIST_DIR"
tar -czf "$PKG_NAME.tar.gz" "$PKG_NAME/"
sha256sum "$PKG_NAME.tar.gz" > "$PKG_NAME.tar.gz.sha256"

echo "Package: $DIST_DIR/$PKG_NAME.tar.gz"
