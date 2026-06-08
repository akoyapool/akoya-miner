#!/usr/bin/env bash
# Select the best GEMM library for the first visible NVIDIA GPU.
set -euo pipefail

LIB_DIR="/app/lib"
TARGET="$LIB_DIR/libpearl_gemm_capi.so"

variant_exists() {
    [[ -f "$LIB_DIR/libpearl_gemm_capi_$1.so" ]]
}

select_for_cc() {
    local cc="$1"
    local major="${cc%%.*}"
    local minor="${cc#*.}"

    if [[ "$major" -eq 7 ]] && [[ "$minor" -eq 0 ]] && variant_exists volta; then
        echo "volta"
    elif [[ "$major" -eq 7 ]] && [[ "$minor" -eq 5 ]] && variant_exists turing; then
        echo "turing"
    elif [[ "$major" -eq 10 ]] && variant_exists b200; then
        echo "b200"
    elif [[ "$major" -eq 12 ]] && variant_exists blackwell; then
        echo "blackwell"
    elif [[ "$major" -eq 9 ]] && variant_exists h100; then
        echo "h100"
    elif [[ "$major" -eq 8 ]] && [[ "$minor" -eq 9 ]] && variant_exists ada; then
        echo "ada"
    elif [[ "$major" -eq 8 ]] && variant_exists ampere; then
        echo "ampere"
    else
        echo "portable"
    fi
}

select_for_name() {
    local name="$1"

    case "$name" in
        *H100*|*H200*)
            if variant_exists h100; then echo "h100"; return; fi
            ;;
        *B200*|*B100*|*GB200*)
            if variant_exists b200; then echo "b200"; return; fi
            ;;
        *RTX*50[0-9][0-9]*)
            if variant_exists blackwell; then echo "blackwell"; return; fi
            ;;
        *RTX*40[0-9][0-9]*|*L40*|*L4*|*"6000 Ada"*)
            if variant_exists ada; then echo "ada"; return; fi
            ;;
        *RTX*30[0-9][0-9]*|*A100*|*A40*|*A6000*)
            if variant_exists ampere; then echo "ampere"; return; fi
            ;;
    esac

    echo "portable"
}

select_gemm_lib() {
    local forced="${AKOYA_GEMM_VARIANT:-auto}"
    case "$forced" in
        auto|"") ;;
        h100|volta|turing|portable|ampere|ada|blackwell|b200)
            if ! variant_exists "$forced"; then
                echo "[entrypoint] AKOYA_GEMM_VARIANT=$forced requested, but libpearl_gemm_capi_${forced}.so is missing" >&2
                exit 64
            fi
            echo "[entrypoint] AKOYA_GEMM_VARIANT=$forced" >&2
            echo "$forced"
            return
            ;;
        *)
            echo "[entrypoint] invalid AKOYA_GEMM_VARIANT=$forced (expected auto|volta|turing|portable|ampere|ada|blackwell|b200|h100)" >&2
            exit 64
            ;;
    esac

    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]') || true

    if [[ -z "$cc" ]]; then
        local name
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) || true
        if [[ -n "$name" ]]; then
            echo "[entrypoint] Detected GPU name: $name" >&2
            local variant
            variant="$(select_for_name "$name")"
            echo "[entrypoint] Selecting $variant kernels" >&2
            echo "$variant"
            return
        fi

        echo "[entrypoint] Could not detect GPU; using portable kernels" >&2
        echo "portable"
        return
    fi

    echo "[entrypoint] Detected GPU compute capability: $cc" >&2
    local variant
    variant="$(select_for_cc "$cc")"
    echo "[entrypoint] Selecting $variant kernels" >&2
    echo "$variant"
}

if [[ "${AKOYA_PEARL_GEMM_LIB:-$TARGET}" == "$TARGET" ]]; then
    variant=$(select_gemm_lib | tail -1)
    ln -sf "$LIB_DIR/libpearl_gemm_capi_${variant}.so" "$TARGET"
    echo "[entrypoint] AKOYA_PEARL_GEMM_LIB=$TARGET -> libpearl_gemm_capi_${variant}.so" >&2
fi

exec /app/akoya-miner "$@"
