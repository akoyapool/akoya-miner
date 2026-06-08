#!/usr/bin/env bash
# Build the universal Akoya Miner Docker image.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/build-docker.sh [tag] [--legacy-cuda122] [--all] [--blackwell-only] [--variants <list>] [--low-memory]

Examples:
  ./scripts/build-docker.sh akoya-miner:latest
  ./scripts/build-docker.sh akoya-miner:cuda122 --legacy-cuda122
  ./scripts/build-docker.sh akoya-miner:all --all
  ./scripts/build-docker.sh akoya-miner:blackwell --blackwell-only
  ./scripts/build-docker.sh akoya-miner:custom --variants blackwell,portable

Environment overrides:
  CUDA_VERSION, CUDA_UBUNTU, UBUNTU_CODENAME, DOTNET_INSTALL_MODE
  AKOYA_GEMM_VARIANTS
  PEARL_GEMM_JOBS, CARGO_BUILD_JOBS, DOTNET_MAX_CPU_COUNT, NVCC_THREADS
  DOCKER_BUILD_MEMORY, DOCKER_BUILD_MEMORY_SWAP, DOCKER_BUILD_CPUS,
  DOCKER_BUILD_CPUSET_CPUS
  PEARL_GEMM_BLACKWELL_LOAD_POLICY, PEARL_GEMM_BLACKWELL_MANUAL_IMMA,
  PEARL_GEMM_BLACKWELL_XOR_ACCUMS, PEARL_GEMM_BLACKWELL_BM,
  PEARL_GEMM_BLACKWELL_BN, PEARL_GEMM_BLACKWELL_STAGES,
  PEARL_GEMM_BLACKWELL_KBLOCK, PEARL_GEMM_BLACKWELL_SWIZZLE_BITS,
  PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS,
  PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS,
  PEARL_GEMM_BLACKWELL_MIN_BLOCKS
USAGE
    exit "${1:-1}"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TAG="${1:-akoya-miner:latest}"
case "$TAG" in -*) TAG="akoya-miner:latest" ;; *) shift || true ;; esac

LOW_MEMORY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --legacy-cuda122)
            export CUDA_VERSION="${CUDA_VERSION:-12.2.2}"
            export CUDA_UBUNTU="${CUDA_UBUNTU:-ubuntu22.04}"
            export UBUNTU_CODENAME="${UBUNTU_CODENAME:-jammy}"
            export DOTNET_INSTALL_MODE="${DOTNET_INSTALL_MODE:-script}"
            export AKOYA_GEMM_VARIANTS="${AKOYA_GEMM_VARIANTS:-legacy-cuda122}"
            ;;
        --all)
            export AKOYA_GEMM_VARIANTS="${AKOYA_GEMM_VARIANTS:-all}"
            ;;
        --blackwell-only)
            export AKOYA_GEMM_VARIANTS="blackwell"
            ;;
        --variants|--gemm-variants)
            shift
            if [[ $# -eq 0 || "$1" == -* ]]; then
                echo "--variants requires a value" >&2
                usage 1
            fi
            export AKOYA_GEMM_VARIANTS="$1"
            ;;
        --variants=*|--gemm-variants=*)
            export AKOYA_GEMM_VARIANTS="${1#*=}"
            ;;
        --low-memory)
            LOW_MEMORY=1
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage 1
            ;;
    esac
    shift
done

DEFAULT_JOBS="$(nproc 2>/dev/null || echo 1)"
BALANCED_JOBS="$DEFAULT_JOBS"
if [[ "$BALANCED_JOBS" -gt 2 ]]; then
    BALANCED_JOBS=2
fi

if [[ "$LOW_MEMORY" -eq 1 ]]; then
    export PEARL_GEMM_JOBS="${PEARL_GEMM_JOBS:-1}"
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
    export DOTNET_MAX_CPU_COUNT="${DOTNET_MAX_CPU_COUNT:-1}"
    export NVCC_THREADS="${NVCC_THREADS:-1}"
else
    export PEARL_GEMM_JOBS="${PEARL_GEMM_JOBS:-1}"
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$BALANCED_JOBS}"
    export DOTNET_MAX_CPU_COUNT="${DOTNET_MAX_CPU_COUNT:-$BALANCED_JOBS}"
    export NVCC_THREADS="${NVCC_THREADS:-2}"
fi

CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
CUDA_UBUNTU="${CUDA_UBUNTU:-ubuntu24.04}"
UBUNTU_CODENAME="${UBUNTU_CODENAME:-noble}"
DOTNET_INSTALL_MODE="${DOTNET_INSTALL_MODE:-apt}"
AKOYA_GEMM_VARIANTS="${AKOYA_GEMM_VARIANTS:-modern}"
PEARL_GEMM_BLACKWELL_LOAD_POLICY="${PEARL_GEMM_BLACKWELL_LOAD_POLICY:-tma}"
PEARL_GEMM_BLACKWELL_MANUAL_IMMA="${PEARL_GEMM_BLACKWELL_MANUAL_IMMA:-1}"
PEARL_GEMM_BLACKWELL_XOR_ACCUMS="${PEARL_GEMM_BLACKWELL_XOR_ACCUMS:-4}"
PEARL_GEMM_BLACKWELL_BM="${PEARL_GEMM_BLACKWELL_BM:-}"
PEARL_GEMM_BLACKWELL_BN="${PEARL_GEMM_BLACKWELL_BN:-}"
PEARL_GEMM_BLACKWELL_STAGES="${PEARL_GEMM_BLACKWELL_STAGES:-}"
PEARL_GEMM_BLACKWELL_KBLOCK="${PEARL_GEMM_BLACKWELL_KBLOCK:-}"
PEARL_GEMM_BLACKWELL_SWIZZLE_BITS="${PEARL_GEMM_BLACKWELL_SWIZZLE_BITS:-}"
PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS="${PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS:-}"
PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS="${PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS:-}"
PEARL_GEMM_BLACKWELL_MIN_BLOCKS="${PEARL_GEMM_BLACKWELL_MIN_BLOCKS:-}"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

DOCKER_BUILD_FLAGS=()
[[ -n "${DOCKER_BUILD_MEMORY:-}" ]] && DOCKER_BUILD_FLAGS+=(--memory "$DOCKER_BUILD_MEMORY")
[[ -n "${DOCKER_BUILD_MEMORY_SWAP:-}" ]] && DOCKER_BUILD_FLAGS+=(--memory-swap "$DOCKER_BUILD_MEMORY_SWAP")
[[ -n "${DOCKER_BUILD_CPUS:-}" ]] && DOCKER_BUILD_FLAGS+=(--cpus "$DOCKER_BUILD_CPUS")
[[ -n "${DOCKER_BUILD_CPUSET_CPUS:-}" ]] && DOCKER_BUILD_FLAGS+=(--cpuset-cpus "$DOCKER_BUILD_CPUSET_CPUS")

echo "Building $TAG"
echo "  cuda:    $CUDA_VERSION / $CUDA_UBUNTU"
echo "  gemm:    $AKOYA_GEMM_VARIANTS"
echo "  jobs:    gemm=$PEARL_GEMM_JOBS cargo=$CARGO_BUILD_JOBS dotnet=$DOTNET_MAX_CPU_COUNT nvcc_threads=$NVCC_THREADS"
echo "  bw:      load=$PEARL_GEMM_BLACKWELL_LOAD_POLICY manual_imma=$PEARL_GEMM_BLACKWELL_MANUAL_IMMA xor=$PEARL_GEMM_BLACKWELL_XOR_ACCUMS"
if [[ -n "$PEARL_GEMM_BLACKWELL_BM$PEARL_GEMM_BLACKWELL_BN$PEARL_GEMM_BLACKWELL_STAGES$PEARL_GEMM_BLACKWELL_KBLOCK$PEARL_GEMM_BLACKWELL_SWIZZLE_BITS$PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS$PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS$PEARL_GEMM_BLACKWELL_MIN_BLOCKS" ]]; then
    echo "  bw-extra: bm=$PEARL_GEMM_BLACKWELL_BM bn=$PEARL_GEMM_BLACKWELL_BN stages=$PEARL_GEMM_BLACKWELL_STAGES kblock=$PEARL_GEMM_BLACKWELL_KBLOCK swizzle=$PEARL_GEMM_BLACKWELL_SWIZZLE_BITS min_blocks=$PEARL_GEMM_BLACKWELL_MIN_BLOCKS"
fi
if [[ "${#DOCKER_BUILD_FLAGS[@]}" -gt 0 ]]; then
    echo "  docker:  ${DOCKER_BUILD_FLAGS[*]}"
fi

docker build \
    "${DOCKER_BUILD_FLAGS[@]}" \
    -f "$REPO_ROOT/Dockerfile" \
    -t "$TAG" \
    --build-arg AKOYA_GIT_SHA="$GIT_SHA" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg CUDA_UBUNTU="$CUDA_UBUNTU" \
    --build-arg UBUNTU_CODENAME="$UBUNTU_CODENAME" \
    --build-arg DOTNET_INSTALL_MODE="$DOTNET_INSTALL_MODE" \
    --build-arg AKOYA_GEMM_VARIANTS="$AKOYA_GEMM_VARIANTS" \
    --build-arg PEARL_GEMM_JOBS="$PEARL_GEMM_JOBS" \
    --build-arg CARGO_BUILD_JOBS="$CARGO_BUILD_JOBS" \
    --build-arg DOTNET_MAX_CPU_COUNT="$DOTNET_MAX_CPU_COUNT" \
    --build-arg NVCC_THREADS="$NVCC_THREADS" \
    --build-arg PEARL_GEMM_BLACKWELL_BM="$PEARL_GEMM_BLACKWELL_BM" \
    --build-arg PEARL_GEMM_BLACKWELL_BN="$PEARL_GEMM_BLACKWELL_BN" \
    --build-arg PEARL_GEMM_BLACKWELL_STAGES="$PEARL_GEMM_BLACKWELL_STAGES" \
    --build-arg PEARL_GEMM_BLACKWELL_KBLOCK="$PEARL_GEMM_BLACKWELL_KBLOCK" \
    --build-arg PEARL_GEMM_BLACKWELL_SWIZZLE_BITS="$PEARL_GEMM_BLACKWELL_SWIZZLE_BITS" \
    --build-arg PEARL_GEMM_BLACKWELL_LOAD_POLICY="$PEARL_GEMM_BLACKWELL_LOAD_POLICY" \
    --build-arg PEARL_GEMM_BLACKWELL_MANUAL_IMMA="$PEARL_GEMM_BLACKWELL_MANUAL_IMMA" \
    --build-arg PEARL_GEMM_BLACKWELL_XOR_ACCUMS="$PEARL_GEMM_BLACKWELL_XOR_ACCUMS" \
    --build-arg PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS="$PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS" \
    --build-arg PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS="$PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS" \
    --build-arg PEARL_GEMM_BLACKWELL_MIN_BLOCKS="$PEARL_GEMM_BLACKWELL_MIN_BLOCKS" \
    "$REPO_ROOT"
