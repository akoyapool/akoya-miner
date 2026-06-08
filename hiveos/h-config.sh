#!/usr/bin/env bash
#
# h-config.sh — Translate HiveOS flight-sheet variables into the AKOYA_* env
# vars consumed by the miner binary. It has NO config file; everything is
# driven by environment.
#
# Sourced (not executed) by h-run.sh so the exports propagate.
#
# HiveOS supplies:
#   $CUSTOM_TEMPLATE         "%WAL%.%WORKER_NAME%"  → already expanded to wallet.worker
#   $CUSTOM_URL              "host:port[,host:port,...]"
#   $CUSTOM_USER_CONFIG      Extra "KEY=value" lines from the flight sheet
#   $WORKER_NAME             Rig name
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIVE_CUSTOM_TEMPLATE="${CUSTOM_TEMPLATE:-}"
HIVE_CUSTOM_URL="${CUSTOM_URL:-}"
HIVE_CUSTOM_USER_CONFIG="${CUSTOM_USER_CONFIG:-}"

# shellcheck source=h-manifest.conf disable=SC1091
. "$SCRIPT_DIR/h-manifest.conf"

[[ -n "$HIVE_CUSTOM_TEMPLATE" ]] && CUSTOM_TEMPLATE="$HIVE_CUSTOM_TEMPLATE"
[[ -n "$HIVE_CUSTOM_URL" ]] && CUSTOM_URL="$HIVE_CUSTOM_URL"
[[ -n "$HIVE_CUSTOM_USER_CONFIG" ]] && CUSTOM_USER_CONFIG="$HIVE_CUSTOM_USER_CONFIG"

# --- Wallet + worker ---
# CUSTOM_TEMPLATE comes through as "wallet.worker" (HiveOS already substituted).
template="${CUSTOM_TEMPLATE}"
wallet="${template%%.*}"
worker="${template#*.}"
[[ "$worker" == "$template" ]] && worker="${WORKER_NAME:-default}"

export AKOYA_POOL_WALLET="$wallet"
export AKOYA_POOL_WORKER="$worker"

# --- Pool endpoint: take the first comma-separated entry, strip any scheme ---
pool_url="${CUSTOM_URL%%,*}"
pool_url="${pool_url#stratum+tcp://}"
pool_url="${pool_url#tcp://}"
pool_url="${pool_url#stratum2+tcp://}"
pool_url="${pool_url#grpc://}"
pool_url="${pool_url#grpcs://}"

pool_host="${pool_url%%:*}"
pool_port="${pool_url##*:}"
[[ "$pool_port" == "$pool_url" ]] && pool_port=443

export AKOYA_POOL_HOST="$pool_host"
export AKOYA_POOL_PORT="$pool_port"

# --- Extra arbitrary AKOYA_* overrides from the flight sheet ---
# Format in HiveOS "Custom miner config" box:
#   AKOYA_GPU_INDICES=0,1
#   AKOYA_MINE_M=8192
#   AKOYA_LOG_LEVEL=Debug
#   AKOYA_POOL_TLS=1
# Lines may be newline- or semicolon-separated.
if [[ -n "${CUSTOM_USER_CONFIG:-}" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^AKOYA_[A-Z0-9_]+= ]] || continue
        # 'line' is in KEY=value form; export accepts that as an assignment.
        # shellcheck disable=SC2163
        export "$line"
    done < <(echo "$CUSTOM_USER_CONFIG" | tr ';' '\n')
fi

# --- TLS compatibility + defaults ---
# AKOYA_POOL_TLS is the current miner env var. AKOYA_POOL_USE_TLS remains
# accepted for older flight sheets and package snippets.
if [[ -z "${AKOYA_POOL_TLS:-}" && -n "${AKOYA_POOL_USE_TLS:-}" ]]; then
    export AKOYA_POOL_TLS="$AKOYA_POOL_USE_TLS"
fi
if [[ -z "${AKOYA_POOL_TLS:-}" ]]; then
    if [[ "${AKOYA_POOL_PORT:-443}" == "443" ]]; then
        export AKOYA_POOL_TLS=1
    else
        export AKOYA_POOL_TLS=0
    fi
fi
export AKOYA_POOL_USE_TLS="$AKOYA_POOL_TLS"

# --- Observability defaults ---
export AKOYA_HIVEOS_STATS_PATH="${AKOYA_HIVEOS_STATS_PATH:-/run/hive/akoya-miner-stats.json}"
export AKOYA_LOG_LEVEL="${AKOYA_LOG_LEVEL:-Information}"
export AKOYA_METRICS_PORT="${AKOYA_METRICS_PORT:-9100}"

# --- Session persistence (identity_key survives restarts → same miner_id) ---
SESSION_DIR="$SCRIPT_DIR/session"
mkdir -p "$SESSION_DIR"
export AKOYA_SESSION_FILE="${AKOYA_SESSION_FILE:-$SESSION_DIR/session.json}"

mkdir -p /var/log/miner/akoya-miner
{
    echo "[$(date -u +%FT%TZ)] h-config:"
    echo "  AKOYA_POOL_HOST=$AKOYA_POOL_HOST"
    echo "  AKOYA_POOL_PORT=$AKOYA_POOL_PORT"
    echo "  AKOYA_POOL_TLS=$AKOYA_POOL_TLS"
    echo "  AKOYA_POOL_WORKER=$AKOYA_POOL_WORKER"
    echo "  AKOYA_SESSION_FILE=$AKOYA_SESSION_FILE"
    echo "  AKOYA_GPU_INDICES=${AKOYA_GPU_INDICES:-all}"
} >> /var/log/miner/akoya-miner/h-config.log
