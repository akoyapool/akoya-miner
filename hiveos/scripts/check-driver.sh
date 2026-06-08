#!/usr/bin/env bash

set -e

# Package defaults are modern CUDA 12.8 / sm_80+. The CUDA 12.2 legacy
# HiveOS tarball overrides these in h-manifest.conf before this script runs.
REQUIRED_CUDA_MAJOR="${AKOYA_REQUIRED_CUDA_MAJOR:-12}"
REQUIRED_CUDA_MINOR="${AKOYA_REQUIRED_CUDA_MINOR:-8}"
REQUIRED_MIN_SM="${AKOYA_REQUIRED_MIN_SM:-80}"

# Get driver-reported CUDA version
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver not installed?"
    exit 1
fi

cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')

if [[ -z "$cuda_version" ]]; then
    echo "ERROR: Could not parse CUDA version from nvidia-smi"
    exit 1
fi

cuda_major="${cuda_version%.*}"
cuda_minor="${cuda_version#*.}"

# Compare versions
if (( cuda_major < REQUIRED_CUDA_MAJOR )) || \
   (( cuda_major == REQUIRED_CUDA_MAJOR && cuda_minor < REQUIRED_CUDA_MINOR )); then
    echo "ERROR: CUDA driver version $cuda_version too old."
    echo "Required: CUDA $REQUIRED_CUDA_MAJOR.$REQUIRED_CUDA_MINOR or later."
    echo ""
    echo "On HiveOS:    nvidia-driver-update"
    echo "On Ubuntu:    sudo apt install nvidia-driver-570"
    echo ""
    echo "After upgrade, reboot and try again."
    exit 5
fi

# Check GPU compute capability against the package's compiled kernels.
sm_versions=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sort -u)
unsupported=""
for sm in $sm_versions; do
    sm_int=$(echo "$sm" | tr -d '.')
    if (( sm_int < REQUIRED_MIN_SM )); then
        unsupported="$unsupported $sm"
    fi
done

if [[ -n "$unsupported" ]]; then
    echo "ERROR: Detected GPU(s) with compute capability$unsupported (below required floor)."
    echo "This Akoya Miner package requires sm_${REQUIRED_MIN_SM} or later."
    echo "Install the CUDA 12.2 legacy package for V100/T4/RTX 20-series rigs."
    exit 5
fi

exit 0
