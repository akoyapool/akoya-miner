<#
.SYNOPSIS
  Build the Akoya reference miner end-to-end on NATIVE Windows (no WSL).

.DESCRIPTION
  Windows counterpart to build.sh. Builds the three native pieces and stages a
  ready-to-run .\out folder:
    1. pearl_gemm_capi.dll   — CUDA proof-of-work kernels  (CMake + nvcc + MSVC)
    2. pearl_mining_capi.dll — BLAKE3 keyed-merkle C ABI    (Rust / cargo)
    3. akoya-miner.exe       — the .NET host, Native AOT, self-contained

  Requires a Visual Studio install with the "Desktop development with C++"
  workload (provides cl.exe + the AOT linker), the CUDA Toolkit (nvcc), the
  Rust toolchain, the .NET 10 SDK, and python (CUDA kernel codegen). All are
  verified up front; missing ones are reported together.

.EXAMPLE
  .\build.ps1                       # auto-detect GPU arch, Release
  .\build.ps1 -Arch ampere          # force RTX 30-series / A100
  .\build.ps1 -Arch ada             # RTX 40-series
#>
[CmdletBinding()]
param(
  [ValidateSet('h100','volta','turing','portable','ampere','ada','blackwell','b200')]
  [string]$Arch = $env:PEARL_GEMM_ARCH,           # empty ⇒ auto-detect
  [ValidateSet('Release','Debug')]
  [string]$Config = $(if ($env:CONFIG) { $env:CONFIG } else { 'Release' }),
  [string]$Rid = 'win-x64',
  [string]$Out = (Join-Path $PSScriptRoot 'out'),
  # nvcc only supports MSVC from VS 2019–2022. Newer VS (e.g. 2026) needs this
  # override. 'auto' passes it only when the detected toolset is newer.
  [ValidateSet('auto','on','off')]
  [string]$AllowUnsupportedCompiler = 'auto'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

function Say  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Die  ($m) { Write-Host "`nERROR: $m" -ForegroundColor Red; exit 1 }
function Step ($m) { Write-Host "  - $m" -ForegroundColor DarkCyan }

# ── Locate a Visual Studio install with the C++ toolset ──────────────────────
function Find-VsInstall {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path $vswhere)) { return $null }
  $path = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath 2>$null | Select-Object -First 1
  if (-not $path) { return $null }
  [pscustomobject]@{
    Path     = $path
    VcVars   = Join-Path $path 'VC\Auxiliary\Build\vcvars64.bat'
    Installer= Split-Path $vswhere
  }
}

# Import vcvars64.bat's environment into the current PowerShell session so nvcc
# finds cl.exe and the .NET AOT linker finds link.exe. Filters out cmd's hidden
# "=X:" per-drive vars and the stale CXX/CC that break CMake auto-detection.
function Import-VcVars ($vcvars, $installerDir) {
  cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([A-Za-z_][A-Za-z0-9_()]*)=(.*)$') {
      Set-Item -Path "Env:\$($matches[1])" -Value $matches[2]
    }
  }
  Remove-Item Env:\CXX, Env:\CC -ErrorAction SilentlyContinue
  if ($installerDir) { $env:PATH = "$installerDir;$env:PATH" }  # vswhere for AOT linker
}

# Map the installed NVIDIA GPU's compute capability → PEARL_GEMM_ARCH.
function Detect-Arch {
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $smi) { return '' }
  $cap = (& $smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>$null |
          Select-Object -First 1).Trim()
  switch -Regex ($cap) {
    '^7\.0$'        { 'volta' ;    break }
    '^7\.5$'        { 'turing';    break }
    '^8\.(0|6|7)$'  { 'ampere';    break }
    '^8\.9$'        { 'ada' ;      break }
    '^9\.0$'        { 'h100';      break }
    '^10\.'         { 'b200';      break }
    '^12\.'         { 'blackwell'; break }
    default         { '' }
  }
}

# Resolve cmake/ninja: prefer PATH, else the copies bundled with Visual Studio.
function Resolve-Tool ($name, $vsRelPaths, $vsPath) {
  $c = Get-Command $name -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  foreach ($rel in $vsRelPaths) {
    $p = Join-Path $vsPath $rel
    if (Test-Path $p) { return $p }
  }
  return $null
}

# ── Preflight ────────────────────────────────────────────────────────────────
Say "Checking prerequisites"
$miss = @()

$vs = Find-VsInstall
if (-not $vs) { $miss += 'Visual Studio with "Desktop development with C++"  ->  https://visualstudio.microsoft.com (VC.Tools.x86.x64)' }

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
  $miss += 'dotnet (.NET 10 SDK)  ->  https://dotnet.microsoft.com/download'
} elseif (-not (dotnet --list-sdks 2>$null | Select-String '^10\.')) {
  $miss += ".NET 10 SDK (have: $(dotnet --version 2>$null))  ->  https://dotnet.microsoft.com/download"
}
if (-not (Get-Command cargo  -ErrorAction SilentlyContinue)) { $miss += 'cargo (Rust toolchain)  ->  https://rustup.rs' }
if (-not (Get-Command nvcc   -ErrorAction SilentlyContinue)) { $miss += 'nvcc (CUDA Toolkit >= 12)  ->  https://developer.nvidia.com/cuda-downloads' }
if (-not (Get-Command python -ErrorAction SilentlyContinue) -and
    -not (Get-Command py     -ErrorAction SilentlyContinue)) { $miss += 'python (CUDA kernel codegen)  ->  https://python.org' }
if (-not (Get-Command git    -ErrorAction SilentlyContinue)) { $miss += 'git (CUTLASS submodule)  ->  https://git-scm.com' }

$cmake = $null; $ninja = $null
if ($vs) {
  $cmake = Resolve-Tool 'cmake' @('Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe') $vs.Path
  $ninja = Resolve-Tool 'ninja' @('Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe')      $vs.Path
  if (-not $cmake) { $miss += 'cmake  ->  install the VS "C++ CMake tools" component, or https://cmake.org' }
  if (-not $ninja) { $miss += 'ninja  ->  install the VS "C++ CMake tools" component' }
}

if ($miss.Count -gt 0) {
  Write-Host "`nMissing prerequisites:" -ForegroundColor Red
  $miss | ForEach-Object { Write-Host "  - $_" }
  Die 'Install the tools above, then re-run .\build.ps1'
}

Import-VcVars $vs.VcVars $vs.Installer
$env:PATH = "$(Split-Path $ninja);$env:PATH"
Step "Visual Studio: $($vs.Path)"
Step "cmake: $cmake"
Step "nvcc:  $((Get-Command nvcc).Source)"

# ── Resolve GPU architecture ─────────────────────────────────────────────────
if (-not $Arch) {
  $Arch = Detect-Arch
  if ($Arch) { Say "Auto-detected GPU -> PEARL_GEMM_ARCH=$Arch" }
  else { $Arch = 'h100'; Say "No supported GPU detected -> defaulting to h100 (override with -Arch)" }
}

# Decide the nvcc unsupported-compiler override from the MSVC toolset version.
$cudaFlags = @()
$useAllow = $AllowUnsupportedCompiler
if ($useAllow -eq 'auto') {
  $clVer = (& cl 2>&1 | Select-String -Pattern 'Version (\d+)\.(\d+)' | Select-Object -First 1)
  $msvcMajorMinor = if ($clVer) { [version]("$($clVer.Matches[0].Groups[1].Value).$($clVer.Matches[0].Groups[2].Value)") } else { [version]'0.0' }
  # CUDA 13 supports MSVC up to 19.4x (VS 2022). Newer needs the override.
  $useAllow = if ($msvcMajorMinor -gt [version]'19.44') { 'on' } else { 'off' }
}
if ($useAllow -eq 'on') {
  Step "MSVC newer than nvcc's supported range -> passing -allow-unsupported-compiler"
  $cudaFlags += '-allow-unsupported-compiler'
}

# ── CUTLASS submodule (CUDA only) ────────────────────────────────────────────
$cutlassHdr = Join-Path $root 'native\pearl-gemm\third_party\cutlass\include\cutlass\cutlass.h'
if (-not (Test-Path $cutlassHdr)) {
  Say "Fetching CUTLASS submodule"
  git -C $root submodule update --init --depth 1 native/pearl-gemm/third_party/cutlass
  if ($LASTEXITCODE -ne 0) { Die "CUTLASS submodule fetch failed. Clone with --recurse-submodules." }
}

# ── 1. pearl-gemm -> pearl_gemm_capi.dll (CMake) ─────────────────────────────
Say "Building pearl_gemm_capi.dll (CUDA, $Arch)"
$gemmSrc   = Join-Path $root 'native\pearl-gemm\csrc\capi'
$gemmBuild = Join-Path $gemmSrc "build-win\$Arch"
$cfgArgs = @('-S', $gemmSrc, '-B', $gemmBuild, '-G', 'Ninja',
             "-DPEARL_GEMM_ARCH=$Arch", "-DCMAKE_BUILD_TYPE=$Config")
if ($cudaFlags.Count -gt 0) { $cfgArgs += "-DCMAKE_CUDA_FLAGS=$($cudaFlags -join ' ')" }
& $cmake @cfgArgs        ; if ($LASTEXITCODE -ne 0) { Die 'CMake configure failed' }
& $cmake --build $gemmBuild ; if ($LASTEXITCODE -ne 0) { Die 'CMake build failed' }
$gemmDll = Join-Path $gemmBuild 'pearl_gemm_capi.dll'
if (-not (Test-Path $gemmDll)) { Die "expected $gemmDll not produced" }

# ── 2. pearl-mining-capi -> pearl_mining_capi.dll (Rust) ─────────────────────
Say "Building pearl_mining_capi.dll (Rust)"
cargo build --release --manifest-path (Join-Path $root 'native\Cargo.toml')
if ($LASTEXITCODE -ne 0) { Die 'cargo build failed' }
$miningDll = Join-Path $root 'native\target\release\pearl_mining_capi.dll'
if (-not (Test-Path $miningDll)) { Die "expected $miningDll not produced" }

# ── 3. .NET host -> akoya-miner.exe (Native AOT) ─────────────────────────────
Say "Publishing akoya-miner.exe (Native AOT, $Rid)"
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
dotnet publish (Join-Path $root 'src\Akoya.Miner\Akoya.Miner.csproj') `
  -c $Config -r $Rid --self-contained true -p:PublishAot=true `
  -p:DebugType=none -p:DebugSymbols=false -o $Out
if ($LASTEXITCODE -ne 0) { Die 'dotnet publish failed' }
Get-ChildItem $Out -Filter *.pdb | Remove-Item -Force -ErrorAction SilentlyContinue

# ── 4. Stage native DLLs next to the binary ──────────────────────────────────
Copy-Item $gemmDll   $Out -Force
Copy-Item $miningDll $Out -Force

Write-Host "`nBuild complete - ready-to-run folder:" -ForegroundColor Green
Write-Host "   $Out"
Get-ChildItem $Out | ForEach-Object { Write-Host "     $($_.Name)" }
Write-Host "`nRun it:" -ForegroundColor Green
Write-Host "   `$env:AKOYA_POOL_WALLET='prl1youraddresshere'; & '$Out\akoya-miner.exe'"
