# Akoya Miner for HiveOS

Akoya Miner mines Pearl (PRL) on Akoya Pool and reports hashrate,
temperature, fan, uptime, and share counts to the HiveOS dashboard.

## Flight Sheet

- Miner: Custom
- Coin: Custom (`pearl`)
- Wallet: your Pearl address (`prl1...`)
- Pool: `pool-v2.akoyapool.com:443`
- Template: `%WAL%.%WORKER_NAME%`

The package defaults to TLS on port 443. No extra TLS setting is required for
the public Akoya Pool endpoint.

## Custom Miner Config

Use the HiveOS "Custom miner config" box for optional `AKOYA_*` overrides.
Put one setting per line, or separate settings with semicolons.

```text
AKOYA_GPU_INDICES=0,1
AKOYA_LOG_LEVEL=Information
AKOYA_BENCH_DURATION_SEC=10
```

For non-standard endpoints:

```text
AKOYA_POOL_TLS=1
```

`AKOYA_POOL_USE_TLS` is still accepted for older flight sheets, but
`AKOYA_POOL_TLS` is the current setting name.

## Requirements

- Modern package: NVIDIA GPU with compute capability sm_80 or newer and a driver exposing CUDA 12.8 or newer.
- Legacy CUDA 12.2 package: NVIDIA GPU with compute capability sm_70 or newer and a driver exposing CUDA 12.2 or newer.
- 8 GB or more VRAM per mining GPU.
- HiveOS image with `jq`, `bc`, `nvidia-smi`, and CUDA 12 runtime libraries.

## Dashboard Stats

The miner writes `/run/hive/akoya-miner-stats.json`. HiveOS reads it through
`h-stats.sh` and graphs total TMADs/sec as the miner hashrate, with accepted
and rejected share counts from the pool.

## Logs

- Launcher log: `/var/log/miner/akoya-miner/h-run.log`
- Config log: `/var/log/miner/akoya-miner/h-config.log`
- Miner output: HiveOS miner log view

## Driver Fix

If startup reports missing CUDA 12 libraries or an old driver, run:

```bash
nvidia-driver-update --list
nvidia-driver-update <latest_550+>
reboot
```

For V100, T4, and RTX 20-series rigs on older drivers, install the CUDA 12.2
package instead of the modern package.

Pool fee: 5%.
