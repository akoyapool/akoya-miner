#!/usr/bin/env bash
#
# h-run.sh — HiveOS launcher for akoya-miner.
#
# 1. Pre-flight driver check (package-specific CUDA + SM floor)
# 2. Auto-select best GEMM kernel based on GPU compute cap
# 3. Source h-config.sh to populate AKOYA_* env vars from the flight sheet
# 4. exec the AOT binary with `mine-blocks`

set -u

cd "$(dirname "$0")" || { echo "[h-run] cannot cd to script dir"; exit 1; }
# shellcheck source=h-manifest.conf disable=SC1091
. h-manifest.conf
export AKOYA_REQUIRED_CUDA_MAJOR="${AKOYA_REQUIRED_CUDA_MAJOR:-12}"
export AKOYA_REQUIRED_CUDA_MINOR="${AKOYA_REQUIRED_CUDA_MINOR:-8}"
export AKOYA_REQUIRED_MIN_SM="${AKOYA_REQUIRED_MIN_SM:-80}"

# --- Pre-flight driver check ---
if ! ./scripts/check-driver.sh; then
    echo "Driver compatibility check failed. See h-readme.md for requirements."
    exit 5
fi

# --- GPU auto-detect -> symlink the right GEMM kernel ---
# Always re-evaluate: a stale symlink from a previous install (or from v1's
# detect-gpu.sh which only knew about h100/portable) must be refreshed each
# launch so Ada/Blackwell rigs pick up their arch-specific kernel.
LIB="$(pwd)/miner/lib"
TARGET="$LIB/libpearl_gemm_capi.so"
RUNTIME_LIB_PATH="$LIB"
if [[ -d "$LIB/cuda" ]]; then
    RUNTIME_LIB_PATH="$LIB/cuda:$RUNTIME_LIB_PATH"
fi
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]') || true
MAJOR="${CC%%.*}"
MINOR="${CC#*.}"

variant_exists() { [[ -f "$LIB/libpearl_gemm_capi_$1.so" ]]; }
select_gemm_variant() {
    local forced="${AKOYA_GEMM_VARIANT:-auto}"
    case "$forced" in
        auto|"") ;;
        h100|volta|turing|portable|ampere|ada|blackwell|b200)
            if ! variant_exists "$forced"; then
                echo "[h-run] ERROR: AKOYA_GEMM_VARIANT=$forced requested, but libpearl_gemm_capi_${forced}.so is missing" >&2
                exit 64
            fi
            echo "$forced"; return ;;
        *)
            echo "[h-run] ERROR: invalid AKOYA_GEMM_VARIANT=$forced (expected auto|volta|turing|portable|ampere|ada|blackwell|b200|h100)" >&2
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

VARIANT="$(select_gemm_variant)" || exit $?
if [[ "$VARIANT" == "b200" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_b200.so" "$TARGET"
    echo "[h-run] GPU SM $CC -> B200/B300 kernels"
elif [[ "$VARIANT" == "volta" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_volta.so" "$TARGET"
    echo "[h-run] GPU SM $CC -> Volta kernels"
elif [[ "$VARIANT" == "turing" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_turing.so" "$TARGET"
    echo "[h-run] GPU SM $CC -> Turing kernels"
elif [[ "$VARIANT" == "blackwell" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_blackwell.so" "$TARGET"
    echo "[h-run] GPU SM $CC -> Blackwell kernels"
elif [[ "$VARIANT" == "h100" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_h100.so" "$TARGET"
    echo "[h-run] GPU SM $CC -> H100/H200 WGMMA kernels"
elif [[ "$VARIANT" == "ada" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_ada.so" "$TARGET"
    echo "[h-run] GPU SM $CC -> Ada kernels"
elif [[ "$VARIANT" == "ampere" ]]; then
    ln -sf "$LIB/libpearl_gemm_capi_ampere.so" "$TARGET"
    echo "[h-run] GPU SM $CC -> Ampere kernels"
else
    ln -sf "$LIB/libpearl_gemm_capi_portable.so" "$TARGET"
    echo "[h-run] GPU SM ${CC:-unknown} -> portable kernels"
fi

# CUDA runtime libs (libcudart.so.12). The HiveOS image
# bundles the NVIDIA driver AND the CUDA toolkit; if these libs are missing
# the rig has a broken/non-standard image and the user needs to fix it.
# We fail fast with a clear message rather than silently pulling ~480 MB on
# every rig that happens to be misconfigured.
have_cuda_libs() {
    LD_LIBRARY_PATH="$RUNTIME_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$TARGET" 2>/dev/null \
        | grep -E 'libcudart\.so\.12' \
        | grep -q 'not found' && return 1
    return 0
}
if ! have_cuda_libs; then
    cat >&2 <<'EOM'
[h-run] ERROR: CUDA 12 runtime libraries are missing on this rig.
                libcudart.so.12 cannot be resolved.

  The standard HiveOS image ships these as part of the NVIDIA driver+toolkit
  bundle. To restore them:

    1. Reinstall / upgrade the HiveOS NVIDIA driver:
         nvidia-driver-update --list
         nvidia-driver-update <latest_550+>

    2. Or install the CUDA 12 toolkit packages:
         apt-get update && apt-get install -y cuda-runtime-12-9

    3. Verify with:
         ldconfig -p | grep -E 'libcudart\.so\.12'

  If you are running a custom image and cannot install the toolkit, you can
  pre-stage the libs at  ./miner/lib/cuda/  (libcudart.so.12).
EOM
    echo "  This package requires driver/toolkit CUDA ${AKOYA_REQUIRED_CUDA_MAJOR}.${AKOYA_REQUIRED_CUDA_MINOR}+." >&2
    echo "" >&2
    exit 6
fi
export LD_LIBRARY_PATH="$RUNTIME_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AKOYA_PEARL_GEMM_LIB="${AKOYA_PEARL_GEMM_LIB:-$TARGET}"
export AKOYA_PEARL_MINING_LIB="${AKOYA_PEARL_MINING_LIB:-$LIB/libpearl_mining_capi.so}"

# --- Load flight-sheet config (sourced so exports stick) ---
# shellcheck source=h-config.sh disable=SC1091
. ./h-config.sh

# --- Log dirs ---
mkdir -p /var/log/miner/akoya-miner /run/hive

echo "[$(date -u +%FT%TZ)] launching: ./miner/AkoyaMiner mine-blocks" \
    >> /var/log/miner/akoya-miner/h-run.log

# Exec replaces this shell process with the miner.
exec ./miner/AkoyaMiner mine-blocks 2>&1
