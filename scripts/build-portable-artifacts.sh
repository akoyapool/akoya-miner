#!/usr/bin/env bash
# Build native package artifacts inside a controlled CUDA builder image.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/build-portable-artifacts.sh [--low-memory]

Builds native Linux artifacts into dist/portable-artifacts by default.

Environment:
  OUT_DIR, IMAGE_TAG, CUDA_VERSION, CUDA_UBUNTU, AKOYA_GEMM_VARIANTS
  PEARL_GEMM_JOBS, CARGO_BUILD_JOBS, DOTNET_MAX_CPU_COUNT, NVCC_THREADS
  DOCKER_BUILD_MEMORY, DOCKER_BUILD_MEMORY_SWAP, DOCKER_BUILD_CPUS,
  DOCKER_BUILD_CPUSET_CPUS
USAGE
    exit "${1:-1}"
}

LOW_MEMORY=0
for arg in "$@"; do
    case "$arg" in
        --low-memory) LOW_MEMORY=1 ;;
        -h|--help) usage 0 ;;
        *) echo "x unknown arg: $arg" >&2; usage 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUT_DIR="${OUT_DIR:-$REPO_ROOT/dist/portable-artifacts}"
DOCKERFILE="$SCRIPT_DIR/Dockerfile.builder-portable"
DEFAULT_JOBS="$(nproc 2>/dev/null || echo 1)"
# Local Docker Desktop commonly has 4 CPUs / 8 GB RAM. CUTLASS/nvcc is the
# memory-heavy phase, so keep GEMM builds serialized and only parallelize within
# a compile unless the caller opts into more.
BALANCED_JOBS="$DEFAULT_JOBS"
if [[ "$BALANCED_JOBS" -gt 2 ]]; then
    BALANCED_JOBS=2
fi

if [[ "$LOW_MEMORY" -eq 1 ]]; then
    PEARL_GEMM_JOBS="${PEARL_GEMM_JOBS:-1}"
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
    DOTNET_MAX_CPU_COUNT="${DOTNET_MAX_CPU_COUNT:-1}"
    NVCC_THREADS="${NVCC_THREADS:-1}"
else
    PEARL_GEMM_JOBS="${PEARL_GEMM_JOBS:-1}"
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$BALANCED_JOBS}"
    DOTNET_MAX_CPU_COUNT="${DOTNET_MAX_CPU_COUNT:-$BALANCED_JOBS}"
    NVCC_THREADS="${NVCC_THREADS:-1}"
fi
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
CUDA_UBUNTU="${CUDA_UBUNTU:-ubuntu22.04}"
AKOYA_GEMM_VARIANTS="${AKOYA_GEMM_VARIANTS:-modern}"
IMAGE_TAG="${IMAGE_TAG:-akoya-miner-builder-portable:${AKOYA_GEMM_VARIANTS}-${CUDA_VERSION}}"
PEARL_GEMM_AMPERE_BM="${PEARL_GEMM_AMPERE_BM:-128}"
PEARL_GEMM_AMPERE_BN="${PEARL_GEMM_AMPERE_BN:-256}"
PEARL_GEMM_AMPERE_KBLOCK="${PEARL_GEMM_AMPERE_KBLOCK:-64}"
PEARL_GEMM_AMPERE_STAGES="${PEARL_GEMM_AMPERE_STAGES:-3}"
PEARL_GEMM_AMPERE_SWIZZLE_BITS="${PEARL_GEMM_AMPERE_SWIZZLE_BITS:-2}"
PEARL_GEMM_AMPERE_MIN_BLOCKS="${PEARL_GEMM_AMPERE_MIN_BLOCKS:-1}"
PEARL_GEMM_ADA_BM="${PEARL_GEMM_ADA_BM:-128}"
PEARL_GEMM_ADA_BN="${PEARL_GEMM_ADA_BN:-256}"
PEARL_GEMM_ADA_KBLOCK="${PEARL_GEMM_ADA_KBLOCK:-64}"
PEARL_GEMM_ADA_STAGES="${PEARL_GEMM_ADA_STAGES:-3}"
PEARL_GEMM_ADA_SWIZZLE_BITS="${PEARL_GEMM_ADA_SWIZZLE_BITS:-2}"
PEARL_GEMM_ADA_MIN_BLOCKS="${PEARL_GEMM_ADA_MIN_BLOCKS:-1}"
PEARL_GEMM_BLACKWELL_BM="${PEARL_GEMM_BLACKWELL_BM:-}"
PEARL_GEMM_BLACKWELL_BN="${PEARL_GEMM_BLACKWELL_BN:-}"
PEARL_GEMM_BLACKWELL_STAGES="${PEARL_GEMM_BLACKWELL_STAGES:-}"
PEARL_GEMM_BLACKWELL_KBLOCK="${PEARL_GEMM_BLACKWELL_KBLOCK:-}"
PEARL_GEMM_BLACKWELL_SWIZZLE_BITS="${PEARL_GEMM_BLACKWELL_SWIZZLE_BITS:-}"
PEARL_GEMM_BLACKWELL_LOAD_POLICY="${PEARL_GEMM_BLACKWELL_LOAD_POLICY:-tma}"
PEARL_GEMM_BLACKWELL_MANUAL_IMMA="${PEARL_GEMM_BLACKWELL_MANUAL_IMMA:-1}"
PEARL_GEMM_BLACKWELL_XOR_ACCUMS="${PEARL_GEMM_BLACKWELL_XOR_ACCUMS:-4}"
PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS="${PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS:-}"
PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS="${PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS:-}"
PEARL_GEMM_BLACKWELL_MIN_BLOCKS="${PEARL_GEMM_BLACKWELL_MIN_BLOCKS:-}"

DOCKER_BUILD_FLAGS=()
[[ -n "${DOCKER_BUILD_MEMORY:-}" ]] && DOCKER_BUILD_FLAGS+=(--memory "$DOCKER_BUILD_MEMORY")
[[ -n "${DOCKER_BUILD_MEMORY_SWAP:-}" ]] && DOCKER_BUILD_FLAGS+=(--memory-swap "$DOCKER_BUILD_MEMORY_SWAP")
[[ -n "${DOCKER_BUILD_CPUS:-}" ]] && DOCKER_BUILD_FLAGS+=(--cpus "$DOCKER_BUILD_CPUS")
[[ -n "${DOCKER_BUILD_CPUSET_CPUS:-}" ]] && DOCKER_BUILD_FLAGS+=(--cpuset-cpus "$DOCKER_BUILD_CPUSET_CPUS")

command -v docker >/dev/null 2>&1 || { echo "x docker not on PATH"; exit 1; }
[[ -f "$DOCKERFILE" ]] || { echo "x missing $DOCKERFILE"; exit 1; }

GIT_SHA=$(cd "$REPO_ROOT" && git rev-parse HEAD)

echo "=== Building akoya-miner native artifacts ==="
echo "  git_sha: $GIT_SHA"
echo "  out:     $OUT_DIR"
echo "  cuda:    $CUDA_VERSION / $CUDA_UBUNTU"
echo "  gemm:    $AKOYA_GEMM_VARIANTS"
echo "  jobs:    gemm=$PEARL_GEMM_JOBS cargo=$CARGO_BUILD_JOBS dotnet=$DOTNET_MAX_CPU_COUNT nvcc_threads=$NVCC_THREADS"
echo "  ampere:  tile=${PEARL_GEMM_AMPERE_BM}x${PEARL_GEMM_AMPERE_BN}x${PEARL_GEMM_AMPERE_KBLOCK} stages=$PEARL_GEMM_AMPERE_STAGES swizzle=$PEARL_GEMM_AMPERE_SWIZZLE_BITS minBlocks=$PEARL_GEMM_AMPERE_MIN_BLOCKS"
echo "  ada:     tile=${PEARL_GEMM_ADA_BM}x${PEARL_GEMM_ADA_BN}x${PEARL_GEMM_ADA_KBLOCK} stages=$PEARL_GEMM_ADA_STAGES swizzle=$PEARL_GEMM_ADA_SWIZZLE_BITS minBlocks=$PEARL_GEMM_ADA_MIN_BLOCKS"
echo "  bw:      tile=${PEARL_GEMM_BLACKWELL_BM}x${PEARL_GEMM_BLACKWELL_BN} swizzle=$PEARL_GEMM_BLACKWELL_SWIZZLE_BITS load=$PEARL_GEMM_BLACKWELL_LOAD_POLICY manual_imma=$PEARL_GEMM_BLACKWELL_MANUAL_IMMA xor=$PEARL_GEMM_BLACKWELL_XOR_ACCUMS"

docker build \
    "${DOCKER_BUILD_FLAGS[@]}" \
    -f "$DOCKERFILE" \
    --target export \
    -t "$IMAGE_TAG" \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg CUDA_UBUNTU="$CUDA_UBUNTU" \
    --build-arg AKOYA_GIT_SHA="$GIT_SHA" \
    --build-arg AKOYA_GEMM_VARIANTS="$AKOYA_GEMM_VARIANTS" \
    --build-arg PEARL_GEMM_JOBS="$PEARL_GEMM_JOBS" \
    --build-arg CARGO_BUILD_JOBS="$CARGO_BUILD_JOBS" \
    --build-arg DOTNET_MAX_CPU_COUNT="$DOTNET_MAX_CPU_COUNT" \
    --build-arg NVCC_THREADS="$NVCC_THREADS" \
    --build-arg PEARL_GEMM_AMPERE_BM="$PEARL_GEMM_AMPERE_BM" \
    --build-arg PEARL_GEMM_AMPERE_BN="$PEARL_GEMM_AMPERE_BN" \
    --build-arg PEARL_GEMM_AMPERE_KBLOCK="$PEARL_GEMM_AMPERE_KBLOCK" \
    --build-arg PEARL_GEMM_AMPERE_STAGES="$PEARL_GEMM_AMPERE_STAGES" \
    --build-arg PEARL_GEMM_AMPERE_SWIZZLE_BITS="$PEARL_GEMM_AMPERE_SWIZZLE_BITS" \
    --build-arg PEARL_GEMM_AMPERE_MIN_BLOCKS="$PEARL_GEMM_AMPERE_MIN_BLOCKS" \
    --build-arg PEARL_GEMM_ADA_BM="$PEARL_GEMM_ADA_BM" \
    --build-arg PEARL_GEMM_ADA_BN="$PEARL_GEMM_ADA_BN" \
    --build-arg PEARL_GEMM_ADA_KBLOCK="$PEARL_GEMM_ADA_KBLOCK" \
    --build-arg PEARL_GEMM_ADA_STAGES="$PEARL_GEMM_ADA_STAGES" \
    --build-arg PEARL_GEMM_ADA_SWIZZLE_BITS="$PEARL_GEMM_ADA_SWIZZLE_BITS" \
    --build-arg PEARL_GEMM_ADA_MIN_BLOCKS="$PEARL_GEMM_ADA_MIN_BLOCKS" \
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

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
CID=$(docker create "$IMAGE_TAG")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
docker cp "$CID:/out/." "$OUT_DIR/"
docker rm "$CID" >/dev/null
trap - EXIT

want_variant() {
    case ",${AKOYA_GEMM_VARIANTS}," in *,all,*|*,"$1",*) return 0;; esac
    case "${AKOYA_GEMM_VARIANTS}:$1" in
        modern:h100|modern:portable|modern:ampere|modern:ada|modern:blackwell|modern:b200) return 0;;
        legacy-cuda122:volta|legacy-cuda122:turing|legacy-cuda122:portable|legacy-cuda122:ampere|legacy-cuda122:ada) return 0;;
        legacy:volta|legacy:turing|legacy:portable|legacy:ampere|legacy:ada) return 0;;
    esac
    return 1
}

required=(
    "$OUT_DIR/akoya-miner/akoya-miner"
    "$OUT_DIR/lib/libpearl_mining_capi.so"
)
for variant in h100 volta turing portable ampere ada blackwell b200; do
    if want_variant "$variant"; then
        required+=("$OUT_DIR/lib/libpearl_gemm_capi_${variant}.so")
    fi
done

for f in "${required[@]}"; do
    [[ -f "$f" ]] || { echo "x missing artifact: $f"; exit 1; }
done

echo ""
echo "GLIBC floor:"
HIGHEST="GLIBC_2.0"
for f in "$OUT_DIR"/lib/*.so "$OUT_DIR/akoya-miner/akoya-miner"; do
    floor=$(objdump -T "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)
    floor="${floor:-none}"
    printf '  %-44s %s\n' "$(basename "$f")" "$floor"
    if [[ "$floor" != "none" ]] && [[ "$(printf '%s\n%s' "$HIGHEST" "$floor" | sort -V | tail -1)" == "$floor" ]]; then
        HIGHEST="$floor"
    fi
done

EXPECTED_MAX="GLIBC_2.35"
if [[ "$(printf '%s\n%s' "$HIGHEST" "$EXPECTED_MAX" | sort -V | tail -1)" != "$EXPECTED_MAX" ]]; then
    echo "x GLIBC floor $HIGHEST exceeds $EXPECTED_MAX"
    exit 1
fi

bash "$SCRIPT_DIR/leak-scan.sh" "$OUT_DIR"
echo "Artifacts ready: $OUT_DIR"
