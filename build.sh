#!/usr/bin/env bash
#
# Build the Akoya reference miner end-to-end (native libraries + .NET) into a
# runnable layout. The native libs are staged next to the managed binary so the
# miner's P/Invoke calls resolve them automatically.
#
# Usage:
#   ./build.sh                          # NVIDIA CUDA, arch=h100, Release
#   PEARL_GEMM_ARCH=ada ./build.sh      # NVIDIA, RTX 40-series (Ada)
#   BACKEND=rocm ./build.sh             # AMD ROCm / HIP (CDNA3 / MI300)
#
# Runs on Linux (x64 and ARM64), including WSL2. Windows: build inside WSL2.
#
# Environment (all optional — sensible defaults):
#   BACKEND           cuda | rocm                  (default: cuda)
#   PEARL_GEMM_ARCH   h100|ampere|ada|blackwell|b200|volta|turing|portable
#                                                  (default: auto-detected from the GPU, else h100)
#   RID               .NET runtime identifier      (default: from uname -m → linux-x64 / linux-arm64)
#   CONFIG            .NET build configuration      (default: Release)
#   OUT               ready-to-run output folder    (default: ./out)
#
# Prerequisites are verified at startup (see preflight): .NET 10 SDK, Rust,
# git, make, clang+zlib1g-dev, python3, and the CUDA toolkit (nvcc) or ROCm
# (hipcc) with a matching GPU + driver.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${BACKEND:-cuda}"
PEARL_GEMM_ARCH="${PEARL_GEMM_ARCH:-}"   # empty ⇒ auto-detect from the GPU (CUDA), else h100
CONFIG="${CONFIG:-Release}"
# .NET runtime identifier for the AOT publish — default from the CPU arch
# (x86_64 → linux-x64, aarch64 → linux-arm64). Override with RID=…
case "$(uname -m 2>/dev/null)" in
  aarch64|arm64) _default_rid=linux-arm64 ;;
  *)             _default_rid=linux-x64 ;;
esac
RID="${RID:-$_default_rid}"
OUT="${OUT:-$ROOT/out}"          # ready-to-run output folder

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

SPIN=(⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏)
trap 'printf "\033[?25h" 2>/dev/null || true' EXIT   # always restore the cursor

# run_step "<label>" "<progress|empty>" <cmd...>
# Runs <cmd> with its output captured to a temp log and an animated spinner in
# its place. On success prints a green check (+ the live progress text); on
# failure prints the captured output and exits. <progress> is a shell snippet
# eval'd each tick for a short count — it may reference $log (the output file).
run_step() {
  local label="$1" progress="$2"; shift 2
  local tty=0; [ -t 1 ] && tty=1
  local log p=""; log="$(mktemp)"
  "$@" >"$log" 2>&1 &
  local pid=$!
  if [ "$tty" = 1 ]; then
    printf '\033[?25l'                                   # hide cursor
    local i=0 n=${#SPIN[@]}
    while kill -0 "$pid" 2>/dev/null; do
      p=""; if [ -n "$progress" ]; then p=" — $(eval "$progress" 2>/dev/null || true)"; fi
      printf '\r  \033[36m%s\033[0m %s%s\033[K' "${SPIN[i]}" "$label" "$p"
      i=$(( (i + 1) % n )); sleep 0.1
    done
    printf '\033[?25h'                                   # show cursor
  fi
  local rc=0; wait "$pid" || rc=$?
  p=""; if [ -n "$progress" ]; then p=" — $(eval "$progress" 2>/dev/null || true)"; fi
  if [ "$rc" -eq 0 ]; then
    if [ "$tty" = 1 ]; then printf '\r  \033[1;32m✓\033[0m %s%s\033[K\n' "$label" "$p"
    else                    printf '  ✓ %s%s\n' "$label" "$p"; fi
    rm -f "$log"
  else
    if [ "$tty" = 1 ]; then printf '\r  \033[1;31m✗\033[0m %s\033[K\n' "$label"
    else                    printf '  ✗ %s\n' "$label"; fi
    printf '\n\033[1;31m──── %s failed (exit %d) ────\033[0m\n' "$label" "$rc" >&2
    cat "$log" >&2; rm -f "$log"; exit "$rc"
  fi
}

# Major CUDA version of an nvcc binary (echoes a number, or 0 on failure).
nvcc_major() { "$1" --version 2>/dev/null | grep -oiE 'release [0-9]+' | grep -oE '[0-9]+' | head -1; }

# Pick an nvcc that supports -std=c++20 (CUDA >= 12). Honours $NVCC, then PATH,
# then common install dirs (so a conda-shadowed old nvcc on PATH is bypassed).
pick_nvcc() {
  if [ -n "${NVCC:-}" ]; then echo "$NVCC"; return; fi
  if command -v nvcc >/dev/null 2>&1 && [ "$(nvcc_major nvcc)" -ge 12 ] 2>/dev/null; then
    command -v nvcc; return
  fi
  local n v best="" bestv=0
  for n in /usr/local/cuda*/bin/nvcc /opt/cuda*/bin/nvcc; do
    [ -x "$n" ] || continue
    v="$(nvcc_major "$n")"
    if [ "${v:-0}" -ge 12 ] 2>/dev/null && [ "${v:-0}" -gt "$bestv" ] 2>/dev/null; then
      best="$n"; bestv="$v"
    fi
  done
  echo "$best"
}

# Map the installed NVIDIA GPU to a PEARL_GEMM_ARCH via its compute capability
# (with a name-based fallback). Echoes an arch name, or "" if undetermined.
detect_arch() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local cap
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')"
  case "$cap" in
    7.0)         echo volta ;;
    7.5)         echo turing ;;
    8.0|8.6|8.7) echo ampere ;;
    8.9)         echo ada ;;
    9.0)         echo h100 ;;
    10.*)        echo b200 ;;
    12.*)        echo blackwell ;;
    *)
      # Older driver without compute_cap query — fall back to the GPU name.
      local name; name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
      case "$name" in
        *H100*|*H200*)            echo h100 ;;
        *B200*|*B100*|*GB200*)    echo b200 ;;
        *RTX*50[0-9][0-9]*)       echo blackwell ;;
        *RTX*40[0-9][0-9]*|*L40*|*L4*|*"6000 Ada"*) echo ada ;;
        *RTX*30[0-9][0-9]*|*A100*|*A40*|*A6000*)    echo ampere ;;
        *)                        echo "" ;;
      esac ;;
  esac
}

# Verify every required tool is present before doing any work. Reports ALL
# missing tools at once (with where to get them) and exits, rather than failing
# halfway through.
preflight() {
  local -a miss=()
  if ! command -v dotnet >/dev/null 2>&1; then
    miss+=( "dotnet (.NET 10 SDK)        →  https://dotnet.microsoft.com/download" )
  elif ! dotnet --list-sdks 2>/dev/null | grep -q '^10\.'; then
    miss+=( ".NET 10 SDK (have: $(dotnet --version 2>/dev/null || echo none)) →  https://dotnet.microsoft.com/download" )
  fi
  command -v cargo  >/dev/null 2>&1 || miss+=( "cargo (Rust toolchain)      →  https://rustup.rs" )
  command -v git    >/dev/null 2>&1 || miss+=( "git                         →  https://git-scm.com  (or your package manager)" )
  command -v make   >/dev/null 2>&1 || miss+=( "make                        →  apt install build-essential" )
  command -v clang  >/dev/null 2>&1 || miss+=( "clang + zlib1g-dev (.NET AOT) →  apt install clang zlib1g-dev" )
  if [ "$BACKEND" = "rocm" ]; then
    command -v hipcc >/dev/null 2>&1 || miss+=( "hipcc (ROCm / HIP SDK)      →  https://rocm.docs.amd.com" )
  else
    command -v python3 >/dev/null 2>&1 || miss+=( "python3 (CUDA kernel codegen) →  apt install python3" )
    if [ -z "$(pick_nvcc)" ]; then
      miss+=( "nvcc (CUDA Toolkit >= 12)   →  https://developer.nvidia.com/cuda-downloads
       (-std=c++20 needs CUDA 12+; a conda 'cuda-nvcc' may shadow a newer one — install CUDA 12+ or set NVCC=/usr/local/cuda-12.x/bin/nvcc)" )
    fi
  fi
  if [ "${#miss[@]}" -gt 0 ]; then
    printf '\n\033[1;31mMissing prerequisites:\033[0m\n' >&2
    printf '  • %s\n' "${miss[@]}" >&2
    printf '\nInstall the tools above, then re-run ./build.sh\n\n' >&2
    exit 1
  fi
}

say "Checking prerequisites (BACKEND=$BACKEND)"
preflight

declare -a STAGE   # native libraries to place next to the miner

# ── 1. pearl-gemm — CUDA/ROCm proof-of-work GEMM kernels ─────────────────────
if [ "$BACKEND" = "rocm" ]; then
  ROCM_DIR="$ROOT/native/pearl-gemm/csrc/rocm/host"
  run_step "Building libpearl_gemm_capi.so (ROCm / HIP)" \
    'echo "$(find "$ROCM_DIR" -name "*.o" 2>/dev/null | wc -l) files compiled"' \
    make -C "$ROCM_DIR"
  STAGE+=( "$ROCM_DIR/libpearl_gemm_capi.so"
           "$ROCM_DIR/libcuda.so.1" )   # CUDA→HIP shim
else
  # Resolve the GPU architecture: explicit $PEARL_GEMM_ARCH wins; otherwise
  # auto-detect the installed card; otherwise fall back to h100.
  if [ -z "$PEARL_GEMM_ARCH" ]; then
    PEARL_GEMM_ARCH="$(detect_arch)"
    if [ -n "$PEARL_GEMM_ARCH" ]; then
      say "Auto-detected GPU → PEARL_GEMM_ARCH=$PEARL_GEMM_ARCH"
    else
      PEARL_GEMM_ARCH="h100"
      say "No supported GPU detected → defaulting PEARL_GEMM_ARCH=h100 (override with PEARL_GEMM_ARCH=…)"
    fi
  fi

  # CUTLASS (git submodule) must be checked out.
  if [ ! -e "$ROOT/native/pearl-gemm/third_party/cutlass/include/cutlass/cutlass.h" ]; then
    say "Fetching CUTLASS submodule"
    git -C "$ROOT" submodule update --init --depth 1 native/pearl-gemm/third_party/cutlass \
      || die "CUTLASS is missing and the submodule fetch failed. Clone with
  git clone --recurse-submodules <repo-url>
or run
  git submodule update --init native/pearl-gemm/third_party/cutlass"
  fi

  # Need a CUDA >= 12 nvcc for -std=c++20.
  NVCC_BIN="$(pick_nvcc)"
  [ -n "$NVCC_BIN" ] || die "no CUDA >= 12 nvcc found (it provides -std=c++20).
On this machine 'nvcc' is $(command -v nvcc >/dev/null 2>&1 && echo "CUDA $(nvcc_major nvcc) ($(command -v nvcc))" || echo 'not on PATH').
  • If conda is shadowing your toolkit, 'conda deactivate' or remove its cuda-nvcc.
  • Otherwise install CUDA 12+, or point at it:  NVCC=/usr/local/cuda-12.x/bin/nvcc ./build.sh"
  printf '  \033[2mnvcc: %s\033[0m\n' "$NVCC_BIN"
  # Build each arch into its OWN object dir (build/<arch>/). The Makefile caches
  # objects by source mtime only and is blind to the arch (-D…) flags, so a
  # shared dir would link stale objects from a previous arch ("undefined
  # reference to run_pearl_*"). Per-arch dirs avoid that AND keep same-arch
  # rebuilds incremental. `make BUILD=…` overrides the default build path.
  GEMM_BUILD="$ROOT/native/pearl-gemm/csrc/capi/build/$PEARL_GEMM_ARCH"
  # The Makefile decides freshness by source mtime alone — it cannot see the
  # arch/-D flags an object was compiled with. A build dir left over from an
  # earlier run (or shipped in the tree) can therefore relink stale objects
  # whose math no longer matches the host reference: the GPU finds "winning"
  # tiles the host then rejects (claimedHash > target). Wipe the dir for a
  # guaranteed-consistent build. Set AKOYA_INCREMENTAL=1 to keep objects for
  # fast same-arch dev rebuilds.
  [ "${AKOYA_INCREMENTAL:-0}" = "1" ] || rm -rf "$GEMM_BUILD"
  run_step "Building libpearl_gemm_capi.so (CUDA $(nvcc_major "$NVCC_BIN"), $PEARL_GEMM_ARCH)" \
    'echo "$(find "$GEMM_BUILD" -name "*.o" 2>/dev/null | wc -l) files compiled"' \
    make -C "$ROOT/native/pearl-gemm/csrc/capi" BUILD="$GEMM_BUILD" NVCC="$NVCC_BIN" PEARL_GEMM_ARCH="$PEARL_GEMM_ARCH"
  STAGE+=( "$GEMM_BUILD/libpearl_gemm_capi.so" )
fi

# ── 2. pearl-mining-capi — BLAKE3 keyed-merkle C ABI (Rust) ──────────────────
run_step "Building libpearl_mining_capi.so (Rust)" \
  'echo "$(grep -c "Compiling " "$log" 2>/dev/null || true) crates compiled"' \
  cargo build --release --manifest-path "$ROOT/native/Cargo.toml"
STAGE+=( "$ROOT/native/target/release/libpearl_mining_capi.so" )

# ── 3. .NET miner — Native AOT publish into ./out ───────────────────────────
rm -rf "$OUT"
run_step "Publishing akoya-miner (Native AOT, $RID) → ./out" "" \
  dotnet publish "$ROOT/src/Akoya.Miner/Akoya.Miner.csproj" \
    -c "$CONFIG" -r "$RID" --self-contained true -p:PublishAot=true \
    -p:DebugType=none -p:DebugSymbols=false -o "$OUT"

# Keep ./out clean: no managed PDBs, no Native AOT .dbg symbol files.
rm -f "$OUT"/*.pdb "$OUT"/*.dbg

# ── 4. Stage native libs into the ready-to-run folder ───────────────────────
for so in "${STAGE[@]}"; do
  [ -f "$so" ] || die "expected native library not found: $so"
  cp "$so" "$OUT/"
done
printf '  \033[1;32m✓\033[0m Staged %d native librar%s into ./out\n' "${#STAGE[@]}" "$([ "${#STAGE[@]}" -eq 1 ] && echo y || echo ies)"

BIN="$OUT/akoya-miner"
cat <<EOF

✅ Build complete — ready-to-run folder:
   $OUT
   $(ls -1 "$OUT" 2>/dev/null | sed 's/^/     /')

Run it:
   AKOYA_POOL_WALLET=prl1youraddresshere "$BIN"
EOF
