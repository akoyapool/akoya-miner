# syntax=docker/dockerfile:1.7
#
# Universal Akoya Miner image.
#
# The builder compiles one GEMM shared library per selected NVIDIA GPU profile
# and the runtime entrypoint selects the best library from the detected compute
# capability. By default this builds the "modern" image:
#   h100, portable, ampere, ada, blackwell, b200
#
# To include Volta/Turing in the same image:
#   docker build --build-arg AKOYA_GEMM_VARIANTS=all -t akoya-miner:all .
#
# To build the CUDA 12.2 legacy image used for older sm_70/sm_75 hosts:
#   docker build \
#     --build-arg CUDA_VERSION=12.2.2 \
#     --build-arg CUDA_UBUNTU=ubuntu22.04 \
#     --build-arg UBUNTU_CODENAME=jammy \
#     --build-arg DOTNET_INSTALL_MODE=script \
#     --build-arg AKOYA_GEMM_VARIANTS=legacy-cuda122 \
#     -t akoya-miner:cuda122 .

ARG CUDA_VERSION=12.8.1
ARG CUDA_UBUNTU=ubuntu24.04
ARG UBUNTU_CODENAME=noble
ARG DOTNET_VERSION=10.0
ARG DOTNET_INSTALL_MODE=apt
ARG CUTLASS_REF=25e252bdce504932d83f43f07c4b8cc7f9b8e2b6
ARG AKOYA_GEMM_VARIANTS=modern
ARG PEARL_GEMM_JOBS=1
ARG CARGO_BUILD_JOBS=1
ARG DOTNET_MAX_CPU_COUNT=1
ARG NVCC_THREADS=1
ARG PEARL_GEMM_AMPERE_BM=128
ARG PEARL_GEMM_AMPERE_BN=256
ARG PEARL_GEMM_AMPERE_KBLOCK=64
ARG PEARL_GEMM_AMPERE_STAGES=3
ARG PEARL_GEMM_AMPERE_SWIZZLE_BITS=2
ARG PEARL_GEMM_AMPERE_MIN_BLOCKS=1
ARG PEARL_GEMM_ADA_BM=128
ARG PEARL_GEMM_ADA_BN=256
ARG PEARL_GEMM_ADA_KBLOCK=64
ARG PEARL_GEMM_ADA_STAGES=3
ARG PEARL_GEMM_ADA_SWIZZLE_BITS=2
ARG PEARL_GEMM_ADA_MIN_BLOCKS=1
ARG PEARL_GEMM_BLACKWELL_BM=
ARG PEARL_GEMM_BLACKWELL_BN=
ARG PEARL_GEMM_BLACKWELL_STAGES=
ARG PEARL_GEMM_BLACKWELL_KBLOCK=
ARG PEARL_GEMM_BLACKWELL_SWIZZLE_BITS=
ARG PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma
ARG PEARL_GEMM_BLACKWELL_MANUAL_IMMA=1
ARG PEARL_GEMM_BLACKWELL_XOR_ACCUMS=4
ARG PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS=
ARG PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS=
ARG PEARL_GEMM_BLACKWELL_MIN_BLOCKS=

FROM nvidia/cuda:${CUDA_VERSION}-devel-${CUDA_UBUNTU} AS builder

ARG AKOYA_GIT_SHA=unknown
ARG CUTLASS_REF
ARG UBUNTU_CODENAME
ARG DOTNET_VERSION
ARG DOTNET_INSTALL_MODE
ARG AKOYA_GEMM_VARIANTS
ARG PEARL_GEMM_JOBS
ARG CARGO_BUILD_JOBS
ARG DOTNET_MAX_CPU_COUNT
ARG NVCC_THREADS
ARG PEARL_GEMM_AMPERE_BM
ARG PEARL_GEMM_AMPERE_BN
ARG PEARL_GEMM_AMPERE_KBLOCK
ARG PEARL_GEMM_AMPERE_STAGES
ARG PEARL_GEMM_AMPERE_SWIZZLE_BITS
ARG PEARL_GEMM_AMPERE_MIN_BLOCKS
ARG PEARL_GEMM_ADA_BM
ARG PEARL_GEMM_ADA_BN
ARG PEARL_GEMM_ADA_KBLOCK
ARG PEARL_GEMM_ADA_STAGES
ARG PEARL_GEMM_ADA_SWIZZLE_BITS
ARG PEARL_GEMM_ADA_MIN_BLOCKS
ARG PEARL_GEMM_BLACKWELL_BM
ARG PEARL_GEMM_BLACKWELL_BN
ARG PEARL_GEMM_BLACKWELL_STAGES
ARG PEARL_GEMM_BLACKWELL_KBLOCK
ARG PEARL_GEMM_BLACKWELL_SWIZZLE_BITS
ARG PEARL_GEMM_BLACKWELL_LOAD_POLICY
ARG PEARL_GEMM_BLACKWELL_MANUAL_IMMA
ARG PEARL_GEMM_BLACKWELL_XOR_ACCUMS
ARG PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS
ARG PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS
ARG PEARL_GEMM_BLACKWELL_MIN_BLOCKS

RUN --mount=type=cache,id=apt-cache-akoya-miner-builder,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-akoya-miner-builder,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl git make cmake pkg-config \
        clang lld libssl-dev zlib1g-dev python3 \
        binutils gpg && \
    if [ "${DOTNET_INSTALL_MODE}" = "script" ]; then \
        curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh && \
        bash /tmp/dotnet-install.sh --channel "${DOTNET_VERSION}" --install-dir /usr/share/dotnet && \
        ln -sf /usr/share/dotnet/dotnet /usr/bin/dotnet; \
    else \
        install -d /etc/apt/keyrings && \
        curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg && \
        echo "deb [signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/microsoft-ubuntu-${UBUNTU_CODENAME}-prod ${UBUNTU_CODENAME} main" > /etc/apt/sources.list.d/microsoft.list && \
        apt-get update && \
        apt-get install -y --no-install-recommends dotnet-sdk-${DOTNET_VERSION}; \
    fi

COPY --from=rust:latest /usr/local/cargo /usr/local/cargo
COPY --from=rust:latest /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:${PATH}" \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PEARL_GEMM_JOBS=${PEARL_GEMM_JOBS} \
    CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS} \
    DOTNET_MAX_CPU_COUNT=${DOTNET_MAX_CPU_COUNT} \
    NVCC_THREADS=${NVCC_THREADS} \
    DOTNET_NOLOGO=1 \
    DOTNET_CLI_TELEMETRY_OPTOUT=1

WORKDIR /src

# CUDA kernels. Keep this before Rust/.NET so the expensive layer caches well.
COPY native/pearl-gemm/csrc/ ./native/pearl-gemm/csrc/
RUN set -eux; \
    mkdir -p native/pearl-gemm/third_party; \
    git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git \
        native/pearl-gemm/third_party/cutlass; \
    cd native/pearl-gemm/third_party/cutlass; \
    git fetch --depth 1 origin "${CUTLASS_REF}"; \
    git checkout --detach "${CUTLASS_REF}"

RUN set -eux; \
    want_variant() { \
        case ",${AKOYA_GEMM_VARIANTS}," in *,all,*|*,"$1",*) return 0;; esac; \
        case "${AKOYA_GEMM_VARIANTS}:$1" in \
            modern:h100|modern:portable|modern:ampere|modern:ada|modern:blackwell|modern:b200) return 0;; \
            legacy-cuda122:volta|legacy-cuda122:turing|legacy-cuda122:portable|legacy-cuda122:ampere|legacy-cuda122:ada) return 0;; \
            legacy:volta|legacy:turing|legacy:portable|legacy:ampere|legacy:ada) return 0;; \
        esac; \
        return 1; \
    }; \
    build_variant() { \
        variant="$1"; shift; \
        build_dir="/tmp/pearl-gemm-build/${variant}"; \
        make -C native/pearl-gemm/csrc/capi -j"${PEARL_GEMM_JOBS}" \
            BUILD="${build_dir}" \
            PEARL_GEMM_ARCH="${variant}" \
            NVCC_THREADS="${NVCC_THREADS}" "$@"; \
        cp "${build_dir}/libpearl_gemm_capi.so" "/out/lib/libpearl_gemm_capi_${variant}.so"; \
        rm -rf "${build_dir}"; \
    }; \
    mkdir -p /out/lib; \
    built=0; \
    if want_variant h100; then build_variant h100; built=$((built + 1)); fi; \
    if want_variant volta; then build_variant volta; built=$((built + 1)); fi; \
    if want_variant turing; then build_variant turing; built=$((built + 1)); fi; \
    if want_variant portable; then build_variant portable; built=$((built + 1)); fi; \
    if want_variant ampere; then \
        build_variant ampere \
            PEARL_GEMM_AMPERE_BM="${PEARL_GEMM_AMPERE_BM}" \
            PEARL_GEMM_AMPERE_BN="${PEARL_GEMM_AMPERE_BN}" \
            PEARL_GEMM_AMPERE_KBLOCK="${PEARL_GEMM_AMPERE_KBLOCK}" \
            PEARL_GEMM_AMPERE_STAGES="${PEARL_GEMM_AMPERE_STAGES}" \
            PEARL_GEMM_AMPERE_SWIZZLE_BITS="${PEARL_GEMM_AMPERE_SWIZZLE_BITS}" \
            PEARL_GEMM_AMPERE_MIN_BLOCKS="${PEARL_GEMM_AMPERE_MIN_BLOCKS}"; \
        built=$((built + 1)); \
    fi; \
    if want_variant ada; then \
        build_variant ada \
            PEARL_GEMM_ADA_BM="${PEARL_GEMM_ADA_BM}" \
            PEARL_GEMM_ADA_BN="${PEARL_GEMM_ADA_BN}" \
            PEARL_GEMM_ADA_KBLOCK="${PEARL_GEMM_ADA_KBLOCK}" \
            PEARL_GEMM_ADA_STAGES="${PEARL_GEMM_ADA_STAGES}" \
            PEARL_GEMM_ADA_SWIZZLE_BITS="${PEARL_GEMM_ADA_SWIZZLE_BITS}" \
            PEARL_GEMM_ADA_MIN_BLOCKS="${PEARL_GEMM_ADA_MIN_BLOCKS}"; \
        built=$((built + 1)); \
    fi; \
    if want_variant blackwell; then \
        build_variant blackwell \
            PEARL_GEMM_BLACKWELL_BM="${PEARL_GEMM_BLACKWELL_BM}" \
            PEARL_GEMM_BLACKWELL_BN="${PEARL_GEMM_BLACKWELL_BN}" \
            PEARL_GEMM_BLACKWELL_STAGES="${PEARL_GEMM_BLACKWELL_STAGES}" \
            PEARL_GEMM_BLACKWELL_KBLOCK="${PEARL_GEMM_BLACKWELL_KBLOCK}" \
            PEARL_GEMM_BLACKWELL_SWIZZLE_BITS="${PEARL_GEMM_BLACKWELL_SWIZZLE_BITS}" \
            PEARL_GEMM_BLACKWELL_LOAD_POLICY="${PEARL_GEMM_BLACKWELL_LOAD_POLICY}" \
            PEARL_GEMM_BLACKWELL_MANUAL_IMMA="${PEARL_GEMM_BLACKWELL_MANUAL_IMMA}" \
            PEARL_GEMM_BLACKWELL_XOR_ACCUMS="${PEARL_GEMM_BLACKWELL_XOR_ACCUMS}" \
            PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS="${PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS}" \
            PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS="${PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS}" \
            PEARL_GEMM_BLACKWELL_MIN_BLOCKS="${PEARL_GEMM_BLACKWELL_MIN_BLOCKS}"; \
        built=$((built + 1)); \
    fi; \
    if want_variant b200; then build_variant b200; built=$((built + 1)); fi; \
    test "$built" -gt 0

# Rust mining C API.
COPY native/Cargo.toml native/Cargo.lock ./native/
COPY native/pearl-blake3/              ./native/pearl-blake3/
COPY native/pearl-mining-capi/         ./native/pearl-mining-capi/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/src/native/target \
    cd native && \
    RUSTFLAGS="--remap-path-prefix /src=akoya-miner --remap-path-prefix /usr/local/cargo=cargo --remap-path-prefix /usr/local/rustup=rustup" \
    cargo build --release --jobs "${CARGO_BUILD_JOBS}" && \
    cp target/release/libpearl_mining_capi.so /out/lib/libpearl_mining_capi.so

# .NET NativeAOT miner.
COPY proto/ ./proto/
COPY src/   ./src/
COPY version.txt Akoya.slnx ./
ENV AKOYA_GIT_SHA=${AKOYA_GIT_SHA}
RUN --mount=type=cache,target=/root/.nuget/packages \
    dotnet publish src/Akoya.Miner/Akoya.Miner.csproj \
        -c Release -r linux-x64 \
        --self-contained true \
        -maxcpucount:"${DOTNET_MAX_CPU_COUNT}" \
        -p:PublishAot=true \
        -p:BuildInParallel=false \
        -p:StripSymbols=true \
        -p:DeterministicSourcePaths=true \
        -o /out/akoya-miner

RUN mkdir -p /out/cuda && \
    for lib in libcudart.so.12; do \
        real="$(readlink -f "/usr/local/cuda/targets/x86_64-linux/lib/${lib}")"; \
        cp "${real}" /out/cuda/; \
        ln -s "$(basename "${real}")" "/out/cuda/${lib}"; \
    done && \
    strip --strip-all /out/lib/*.so /out/akoya-miner/akoya-miner && \
    set -e; \
    for f in /out/akoya-miner/akoya-miner /out/lib/*.so; do \
        if strings "$f" | grep -qiE '/home/[a-z]|/Users/|/root/\.|/mnt/[a-z]/[A-Z]'; then \
            echo "RELEASE BLOCKED: $(basename "$f") leaks build paths"; exit 1; \
        fi; \
    done; \
    echo "PII audit passed"

FROM nvidia/cuda:${CUDA_VERSION}-base-${CUDA_UBUNTU} AS final

RUN --mount=type=cache,id=apt-cache-akoya-miner-final,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=apt-lists-akoya-miner-final,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates tini bash procps && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /out/cuda/                   /app/lib/cuda/
COPY --from=builder /out/lib/                    /app/lib/
COPY --from=builder /out/akoya-miner/akoya-miner /app/akoya-miner
COPY docker-entrypoint.sh                         /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh /app/akoya-miner && \
    mkdir -p /var/lib/akoya-miner

ENV LD_LIBRARY_PATH=/app/lib:/app/lib/cuda \
    AKOYA_PEARL_GEMM_LIB=/app/lib/libpearl_gemm_capi.so \
    AKOYA_PEARL_MINING_LIB=/app/lib/libpearl_mining_capi.so \
    AKOYA_POOL_HOST=pool-v2.akoyapool.com \
    AKOYA_POOL_PORT=443 \
    AKOYA_POOL_TLS=1 \
    AKOYA_POOL_WORKER=docker \
    AKOYA_SESSION_FILE=/var/lib/akoya-miner/session.json \
    AKOYA_GPU_INDICES=all \
    AKOYA_METRICS_PORT=9100 \
    DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

VOLUME ["/var/lib/akoya-miner"]
EXPOSE 9100

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD /app/akoya-miner version > /dev/null 2>&1 || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]
CMD ["mine-blocks"]
