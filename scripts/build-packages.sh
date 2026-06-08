#!/usr/bin/env bash
# Build standalone "-portable" and HiveOS tarballs.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  ./scripts/build-packages.sh [version] [--low-memory] [--with-legacy-cuda122]

Outputs:
  dist/akoya-miner-<version>-portable.tar.gz
  dist/akoya-miner-<version>.tar.gz

With --with-legacy-cuda122, also outputs:
  dist/akoya-miner-<version>-cuda122-portable.tar.gz
  dist/akoya-miner-<version>-cuda122.tar.gz

Environment:
  DOCKER_BUILD_MEMORY, DOCKER_BUILD_CPUS, DOCKER_BUILD_MEMORY_SWAP,
  DOCKER_BUILD_CPUSET_CPUS
USAGE
    exit "${1:-1}"
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGET_VERSION=""
LOW_MEMORY=0
WITH_LEGACY_CUDA122=0
for arg in "$@"; do
    case "$arg" in
        --low-memory) LOW_MEMORY=1 ;;
        --with-legacy-cuda122|--legacy-cuda122) WITH_LEGACY_CUDA122=1 ;;
        -h|--help) usage 0 ;;
        -*)
            echo "x unknown arg: $arg" >&2
            usage 1
            ;;
        *)
            [[ -z "$TARGET_VERSION" ]] || { echo "x only one version argument is allowed" >&2; exit 1; }
            TARGET_VERSION="$arg"
            ;;
    esac
done

VERSION="${TARGET_VERSION:-$(tr -d '[:space:]' < "$REPO_ROOT/version.txt" 2>/dev/null || echo "0.0.0")}"
LOW_MEMORY_FLAG=()
if [[ "$LOW_MEMORY" -eq 1 ]]; then
    LOW_MEMORY_FLAG=(--low-memory)
fi

echo "=== Akoya Miner package build ==="
echo "Version: $VERSION"
[[ "$LOW_MEMORY" -eq 1 ]] && echo "Mode:    low-memory"

CUDA_VERSION="${CUDA_VERSION:-12.8.1}" \
CUDA_UBUNTU="${CUDA_UBUNTU:-ubuntu22.04}" \
AKOYA_GEMM_VARIANTS="${AKOYA_GEMM_VARIANTS:-modern}" \
OUT_DIR="$REPO_ROOT/dist/portable-artifacts" \
IMAGE_TAG="akoya-miner-builder-portable:modern-${CUDA_VERSION:-12.8.1}" \
    bash "$SCRIPT_DIR/build-portable-artifacts.sh" "${LOW_MEMORY_FLAG[@]}"

PORTABLE_ARTIFACTS_DIR="$REPO_ROOT/dist/portable-artifacts" \
    bash "$SCRIPT_DIR/package-standalone.sh" "$VERSION" portable

PORTABLE_ARTIFACTS_DIR="$REPO_ROOT/dist/portable-artifacts" \
AKOYA_PACKAGE_FLAVOR="modern" \
AKOYA_REQUIRED_CUDA_MAJOR="12" \
AKOYA_REQUIRED_CUDA_MINOR="8" \
AKOYA_REQUIRED_MIN_SM="80" \
    bash "$SCRIPT_DIR/package-hiveos.sh" "$VERSION"

if [[ "$WITH_LEGACY_CUDA122" -eq 1 ]]; then
    LEGACY_CUDA_VERSION="${LEGACY_CUDA_VERSION:-12.2.2}" \
    CUDA_VERSION="${LEGACY_CUDA_VERSION:-12.2.2}" \
    CUDA_UBUNTU="${LEGACY_CUDA_UBUNTU:-ubuntu22.04}" \
    AKOYA_GEMM_VARIANTS="${LEGACY_GEMM_VARIANTS:-legacy-cuda122}" \
    OUT_DIR="$REPO_ROOT/dist/portable-artifacts-cuda122" \
    IMAGE_TAG="akoya-miner-builder-portable:cuda122-${LEGACY_CUDA_VERSION:-12.2.2}" \
        bash "$SCRIPT_DIR/build-portable-artifacts.sh" "${LOW_MEMORY_FLAG[@]}"

    PORTABLE_ARTIFACTS_DIR="$REPO_ROOT/dist/portable-artifacts-cuda122" \
        bash "$SCRIPT_DIR/package-standalone.sh" "$VERSION" cuda122-portable

    PORTABLE_ARTIFACTS_DIR="$REPO_ROOT/dist/portable-artifacts-cuda122" \
    AKOYA_PACKAGE_FLAVOR="cuda122" \
    AKOYA_REQUIRED_CUDA_MAJOR="12" \
    AKOYA_REQUIRED_CUDA_MINOR="2" \
    AKOYA_REQUIRED_MIN_SM="70" \
        bash "$SCRIPT_DIR/package-hiveos.sh" "$VERSION" cuda122
fi

echo ""
echo "Packages ready under $REPO_ROOT/dist:"
find "$REPO_ROOT/dist" -maxdepth 1 -type f \( -name "akoya-miner-${VERSION}*.tar.gz" -o -name "akoya-miner-${VERSION}*.sha256" \) -print | sort
