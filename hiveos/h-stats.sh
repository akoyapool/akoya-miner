#!/usr/bin/env bash

script_dir="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=h-manifest.conf disable=SC1091
. "$script_dir/h-manifest.conf" 2>/dev/null || true

stats_file="${AKOYA_HIVEOS_STATS_PATH:-/run/hive/akoya-miner-stats.json}"
miner_version="${CUSTOM_VERSION:-unknown}"

# If miner hasn't written stats yet (just started), emit zeros
if [[ ! -f "$stats_file" ]]; then
    stats=$(jq -cn --arg ver "$miner_version" '{"hs":[0],"hs_units":"hs","total_hs":0,"temp":[0],"fan":[0],"uptime":0,"ver":$ver,"ar":[0,0],"algo":"pearl"}' 2>/dev/null \
        || printf '{"hs":[0],"hs_units":"hs","total_hs":0,"temp":[0],"fan":[0],"uptime":0,"ver":"%s","ar":[0,0],"algo":"pearl"}' "$miner_version")
    khs=0
    echo "$khs"
    echo "$stats"
    exit 0
fi

# Read miner stats
raw=$(cat "$stats_file")

# Parse with jq, transform to HiveOS expected structure
stats=$(echo "$raw" | jq -c '
{
    hs:        [.gpus[].tmads_per_sec | floor],
    hs_units:  "hs",
    total_hs:  (.total_tmads_per_sec | floor),
    temp:      [.gpus[].temp_c | floor],
    fan:       [.gpus[].fan_pct | floor],
    uptime:    .uptime_seconds,
    ver:       .version,
    ar:        [.shares.accepted, .shares.rejected],
    algo:      "pearl",
    bus_numbers: [.gpus[].pci_bus_id // 0]
}')

# total_hs in HiveOS is what gets graphed — we report TMADs/sec
total_hs=$(echo "$raw" | jq -r '.total_tmads_per_sec | floor')

# khs output (HiveOS legacy line)
khs=$(echo "scale=3; $total_hs / 1000" | bc -l 2>/dev/null || echo "0")

echo "$khs"
echo "$stats"
