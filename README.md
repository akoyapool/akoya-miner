<p align="center">
  <img src="card.png" alt="Akoya Miner" />
</p>

# Akoya Miner

**The reference miner for the [Akoya Pool](https://akoyapool.com).**

This is the official open-source reference implementation of a GPU miner for the
Akoya Pool, which mines **Pearl (PRL)**. Use it directly to mine, or as a
reference for building your own miner that connects to the pool — the wire
protocol it speaks is defined in [`proto/v2/miner.proto`](proto/v2/miner.proto).

The proof of work is a low-rank-noised integer GEMM (matrix-multiply): each
candidate is a tile of `A · Bᵀ` that is hashed and checked against a difficulty
target. The heavy compute runs on the GPU (NVIDIA CUDA or AMD ROCm); the host
side handles the pool connection, BLAKE3 keyed-merkle commitments, and share
submission.

---

## Supported GPUs

`build.sh` auto-detects your card and picks the matching kernel — override with
`PEARL_GEMM_ARCH=…` if needed.

| Architecture | Example GPUs | Compute (SM) | `PEARL_GEMM_ARCH` |
|---|---|---|---|
| **Blackwell** | RTX 50-series (5090 / 5080 …) | sm_120 | `blackwell` |
| **Blackwell (DC)** | B200 | sm_100 | `b200` |
| **Hopper** | H100, H200 | sm_90 | `h100` |
| **Ada** | RTX 40-series, L4, L40S, RTX 6000 Ada | sm_89 | `ada` |
| **Ampere** | RTX 30-series, A100, A40, A6000 | sm_80 / sm_86 | `ampere` |
| **Turing** | RTX 20-series, T4 | sm_75 | `turing` |
| **Volta** | V100 | sm_70 | `volta` |
| **AMD CDNA3** | MI300X | gfx942 | `rocm` (`BACKEND=rocm`) |

Other NVIDIA cards (sm_70+) build with the `portable` fallback.

---

## Repository layout

```
akoya-miner/
├── build.sh                     # one-shot build: native + AOT miner → ./out
├── Akoya.slnx                   # .NET solution
├── proto/v2/miner.proto         # gRPC wire protocol (pool ⇄ miner)
├── src/                         # C# miner (host application)
│   ├── Akoya.Miner/             #   entry point, mining loop, pool session
│   ├── Akoya.Pool/              #   gRPC connection + session state machine
│   ├── Akoya.Crypto/            #   managed BLAKE3 / noise / jackpot
│   ├── Akoya.Cuda / .PearlGemm  #   P/Invoke into the native GEMM library
│   ├── Akoya.Mining / .MinerCore
│   └── Akoya.Proto              #   generated gRPC client stubs
└── native/
    ├── pearl-gemm/              # CUDA + ROCm proof-of-work GEMM kernels
    │   └── third_party/cutlass  #   NVIDIA CUTLASS (git submodule)
    ├── pearl-blake3/            # BLAKE3 keyed-merkle (Rust)
    └── pearl-mining-capi/       # C ABI over pearl-blake3 (Rust → libpearl_mining_capi.so)
```

At runtime the miner loads two native libraries via P/Invoke:
`libpearl_gemm_capi.so` (the GPU kernels) and `libpearl_mining_capi.so` (the
host-side BLAKE3 merkle). `build.sh` compiles the miner with **Native AOT** and
assembles a self-contained, ready-to-run **`./out`** folder containing the
native binary and both libraries — no .NET runtime required to run it.

---

## Prerequisites

`build.sh` (Linux) and `build.ps1` (Windows) check for all of these at startup
and list any that are missing (with where to get them) before doing any work.
On **Windows**, `clang`/`zlib`/`make` are replaced by **Visual Studio** (the
"Desktop development with C++" workload provides the AOT linker, CMake, and
Ninja) — see [Windows (native)](#windows-native--no-wsl) below.

- **.NET 10 SDK** (plus **`clang`** + **`zlib1g-dev`** — required by Native AOT on Linux)
- **Rust** toolchain (`cargo`)
- **git** (CUTLASS is a submodule) and **`make`**
- A C++/CUDA toolchain for your GPU:
  - **NVIDIA:** CUDA Toolkit **12.4+** (`nvcc`; needed for `-std=c++20`, `12.8` for sm90+),
    `python3`, the driver, and a GPU of compute capability **sm_70+** (sm_80+
    recommended). Per-architecture builds exist for Hopper (H100), Ampere, Ada
    (RTX 40xx), Blackwell (RTX 50xx), B200, and a `portable` fallback.
  - **AMD:** ROCm / HIP (`hipcc`) and a **CDNA3** GPU (e.g. MI300).

---

## Platform setup

The build runs on **Linux (x64 and ARM64)** via `./build.sh`, and **natively on
Windows** via `.\build.ps1` (no WSL required). Pick your platform, install the
toolchain, then run the matching build script.

**Common toolchain** — identical on every Linux/WSL target:

```bash
# .NET 10 SDK (installs to ~/.dotnet)
curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 10.0
export PATH="$HOME/.dotnet:$PATH"            # add this line to ~/.bashrc to persist

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# build tools + Native-AOT deps + codegen
sudo apt update && sudo apt install -y build-essential clang zlib1g-dev git python3
```

What differs per platform is the **GPU driver + CUDA/ROCm toolkit**:

### Windows (native — no WSL)
`build.ps1` builds a pure-Windows miner: `pearl_gemm_capi.dll` + `pearl_mining_capi.dll`
+ a Native-AOT `akoya-miner.exe`. Install these, then run `.\build.ps1` from a
normal PowerShell (it locates Visual Studio's tools itself — no "Developer
PowerShell" needed):

1. **Visual Studio** (Community is fine) with the **"Desktop development with
   C++"** workload — provides `cl.exe`, the AOT linker, and bundled CMake +
   Ninja. [visualstudio.microsoft.com](https://visualstudio.microsoft.com)
2. **NVIDIA Windows driver** (Game Ready / Studio) + the **CUDA Toolkit 12.4+**
   (`nvcc`) for Windows — [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   → Windows → x86_64.
3. **.NET 10 SDK** ([dotnet.microsoft.com](https://dotnet.microsoft.com/download)),
   the **Rust** toolchain ([rustup.rs](https://rustup.rs)), **git**, and
   **python** (CUDA kernel codegen).
4. `.\build.ps1` — auto-detects your GPU architecture (override with
   `-Arch ada` etc.) and writes the ready-to-run `.\out` folder.

> CUDA's `nvcc` officially supports MSVC from VS 2019–2022. On a newer Visual
> Studio (e.g. 2026) `build.ps1` automatically passes
> `-allow-unsupported-compiler`; install the VS 2022 build tools if you hit a
> host-compiler incompatibility.

Prefer WSL2? That path still works — follow **WSL2** below instead.

### WSL2 (Ubuntu)
1. Install the common toolchain above.
2. Install the **CUDA Toolkit, “WSL-Ubuntu” variant** (provides `nvcc`; the driver
   comes from Windows) — from
   [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   → Linux → x86_64 → **WSL-Ubuntu**.
3. `./build.sh` — the miner auto-detects WSL's `libcuda.so.1`.

### Linux (x64)
1. Install the common toolchain above.
2. **NVIDIA:** install the driver + **CUDA Toolkit 12+** for your distro from
   [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   (Linux → x86_64). **AMD:** install **ROCm**
   ([rocm.docs.amd.com](https://rocm.docs.amd.com)) and build with `BACKEND=rocm ./build.sh`.
3. `./build.sh`.

### Linux (ARM64 / aarch64)
Same as Linux x64 — `build.sh` auto-selects `RID=linux-arm64`. Use the **ARM**
CUDA packages:

- **Grace-Hopper (GH200) / SBSA servers:** the **arm64-sbsa** CUDA Toolkit from
  the CUDA downloads page. Auto-detects as `h100` (sm_90).
- **Jetson (Orin):** use the **JetPack** CUDA that ships with L4T. `nvidia-smi` is
  not present on Jetson, so arch auto-detection won't work — set it explicitly:
  `PEARL_GEMM_ARCH=ampere ./build.sh` (Orin is sm_87).

---

## Quick start

```bash
git clone --recurse-submodules https://github.com/akoyapool/akoya-miner.git akoya-miner
cd akoya-miner

# NVIDIA — build.sh auto-detects your GPU architecture (override via PEARL_GEMM_ARCH).
./build.sh

# …or AMD:
BACKEND=rocm ./build.sh
```

On **native Windows** (PowerShell), use `build.ps1` instead — same result, a
self-contained `.\out`:

```powershell
git clone --recurse-submodules https://github.com/akoyapool/akoya-miner.git akoya-miner
cd akoya-miner
.\build.ps1                      # auto-detects your GPU (override with -Arch ada)
```

`build.sh` / `build.ps1` fetch CUTLASS (CUDA only), build the native libraries,
compile the miner with **Native AOT**, and assemble a self-contained **`./out`**
folder. Then just run the native binary — no .NET runtime needed:

```bash
AKOYA_POOL_WALLET=prl1youraddresshere \
AKOYA_POOL_WORKER=rig01 \
./out/akoya-miner
```

```powershell
# native Windows
$env:AKOYA_POOL_WALLET = 'prl1youraddresshere'
$env:AKOYA_POOL_WORKER = 'rig01'
.\out\akoya-miner.exe
```

That's the only required setting — `AKOYA_POOL_WALLET`. The miner defaults to
the production pool at `pool-v2.akoyapool.com:443` (TLS). The `./out` folder is
self-contained — copy it to any matching machine and run it as-is.

### Verify the build

Before mining, run the built-in **self-test** — it checks config, loads both
native libraries, reaches the pool, and verifies the session store, then exits
`0` on success (no GPU mining required). Run it on any platform right after
building:

```bash
AKOYA_POOL_WALLET=prl1youraddresshere ./out/akoya-miner selftest
echo "exit code: $?"        # 0 = ready to mine
./out/akoya-miner version   # prints miner version + git sha
```

If `selftest` fails on a native library, it means the `.so` wasn't found or
couldn't load (check that `libpearl_gemm_capi.so` / `libpearl_mining_capi.so` are
in `./out`, or that the CUDA/ROCm runtime is installed).

### Build options

| Variable | Values | Default |
|---|---|---|
| `BACKEND` | `cuda`, `rocm` | `cuda` |
| `PEARL_GEMM_ARCH` | `h100`, `ampere`, `ada`, `blackwell`, `b200`, `volta`, `turing`, `portable` | auto-detect (else `h100`) |
| `PEARL_GEMM_BLACKWELL_LOAD_POLICY` | `cp_async`, `tma` | `cp_async` |
| `PEARL_GEMM_BLACKWELL_MANUAL_IMMA` | `0`, `1` | unset (`0`) |
| `PEARL_GEMM_BLACKWELL_XOR_ACCUMS` | `4`, `8`, `16` | unset (`4`) |
| `CONFIG` | `Release`, `Debug` | `Release` |
| `RID` | .NET runtime identifier for the AOT publish | `linux-x64` |
| `OUT` | ready-to-run output folder | `./out` |

### GPU architecture (auto-detected)

If you don't set `PEARL_GEMM_ARCH`, `build.sh` **auto-detects your GPU** (via
`nvidia-smi` compute capability, with a name-based fallback) and selects the
matching architecture — only falling back to `h100` if no supported card is
found. Set `PEARL_GEMM_ARCH` explicitly to override the detection:

| Your GPU | `PEARL_GEMM_ARCH` |
|---|---|
| **RTX 40-series** (4090 / 4080 / 4070 …), L4, L40S, RTX 6000 Ada | **`ada`** |
| **RTX 50-series** (5090 / 5080 …) | **`blackwell`** |
| RTX 30-series (3090 / 3080 …), A100, A6000 | `ampere` |
| H100 / H200 (Hopper) | `h100` (default) |
| B200 (datacenter Blackwell) | `b200` |
| Older / unsure (sm_70+) | `portable` |

```bash
# RTX 40-series (Ada) — the most common case:
PEARL_GEMM_ARCH=ada ./build.sh

# RTX 50-series (Blackwell):
PEARL_GEMM_ARCH=blackwell ./build.sh
```

### RTX 50-series / SM120 production profile

The Blackwell RTX 50-series path now uses a dedicated SM120 transcript GEMM
implementation at `native/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu`
instead of the shared Ampere/Ada consumer transcript kernel.

For Linux `build.sh` / Makefile builds, the tested RTX 5060 Ti production
profile is:

```bash
PEARL_GEMM_ARCH=blackwell \
PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma \
PEARL_GEMM_BLACKWELL_MANUAL_IMMA=1 \
PEARL_GEMM_BLACKWELL_XOR_ACCUMS=4 \
./build.sh
```

Validation notes from the 2026-06-06 RTX 5060 Ti (SM120) migration:

- Original pre-tuning baseline from the tuning log used
  `AKOYA_MINE_M=4096`, `AKOYA_MINE_N=131072`, `AKOYA_MINE_K=4096`,
  `AKOYA_MINE_NOISE_RANK=128`, and `AKOYA_GPU_INDICES=all`.

  | Profile | Window | GPUs | Per-GPU avg TMADs/s | Total avg TMADs/s |
  |---|---:|---:|---:|---:|
  | CUDA 13.1 default Blackwell/cp_async | 5 minutes | 2x RTX 5060 Ti | 71.060 | 142.121 |

- Ubuntu 26.04 used CUDA 13.3 (`/usr/local/cuda-13.3/bin/nvcc`) for the clean
  production build. CUDA 13.1 hit a system-header `rsqrt` / `rsqrtf` conflict on
  that host.
- The tuned profile registered successfully against the production pool and
  benchmarked at about `84.23 TMADs/s`, with live stats around
  `85.3-85.6 TMADs/s`; submitted shares were accepted.
- The build output is not bundled with `libcudart.so.13`, so keep a compatible
  CUDA runtime installed on the target host or package it next to
  `libpearl_gemm_capi.so`.

---

## Building manually

`build.sh` runs these three steps; you can run them individually.

```bash
# 1. pearl-gemm — the CUDA proof-of-work kernels  →  libpearl_gemm_capi.so
git submodule update --init --depth 1 native/pearl-gemm/third_party/cutlass
make -C native/pearl-gemm/csrc/capi PEARL_GEMM_ARCH=ada   # set to match your GPU (see table above)
#    (AMD instead:  make -C native/pearl-gemm/csrc/rocm/host)

# 2. pearl-mining-capi — BLAKE3 merkle C ABI      →  libpearl_mining_capi.so
cargo build --release --manifest-path native/Cargo.toml

# 3. the .NET miner — Native AOT, self-contained, into ./out
dotnet publish src/Akoya.Miner/Akoya.Miner.csproj -c Release -r linux-x64 \
  --self-contained true -p:PublishAot=true -o out
```

Then copy `libpearl_gemm_capi.so` and `libpearl_mining_capi.so` into `out/` next
to the `akoya-miner` binary (or point the miner at them explicitly with
`AKOYA_PEARL_GEMM_LIB` / `AKOYA_PEARL_MINING_LIB`).

---

## Running & configuration

The miner is configured entirely through environment variables. The essentials:

| Variable | Default | Meaning |
|---|---|---|
| `AKOYA_POOL_WALLET` | — (**required**) | Your Pearl payout address (`prl1…`). |
| `AKOYA_POOL_WORKER` | machine name | Worker label. |
| `AKOYA_POOL_HOST` | `pool-v2.akoyapool.com` | Pool host. |
| `AKOYA_POOL_PORT` | `443` (TLS) | Pool port. |
| `AKOYA_POOL_TLS` | `true` | Use TLS. |
| `AKOYA_GPU_INDICES` | `all` | `all` or comma-separated device indices (e.g. `0,1`). |
| `AKOYA_LOG_LEVEL` | `Information` | Log verbosity. |
| `AKOYA_PEARL_GEMM_LIB` / `AKOYA_PEARL_MINING_LIB` | unset | Absolute path to a native lib (overrides app-directory lookup). |

`build.sh` produces a self-contained Native AOT binary, so `./out/akoya-miner`
runs without a .NET runtime installed. The two native `.so` libraries sit beside
it in `./out` and are resolved automatically.

---

## Third-party components

`native/pearl-gemm/third_party/cutlass` is [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass),
included as a git submodule under its own (BSD-3-Clause) license.
