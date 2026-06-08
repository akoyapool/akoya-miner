#!/usr/bin/env bash
# Privacy/secret leak gate for Akoya release artifacts.
#
# Scans ELF binaries and executables under one or more directories for build
# paths, personal identifiers, internal infra, secrets, and baked-in wallets.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 DIR [DIR...]" >&2
    exit 2
fi

PATTERNS=(
    "build-path-home::/home/[a-z][a-zA-Z0-9_.-]+"
    "build-path-users::/Users/[a-zA-Z0-9_.-]+"
    "build-path-root-dotfiles::/root/\\.[a-zA-Z0-9_.-]+"
    "build-path-wsl-drive::/mnt/[a-z]/[A-Z][a-zA-Z0-9_.-]+"
    "build-path-cargo-registry::\\.cargo/(registry|git)/"
    "build-path-rustup::\\.rustup/toolchains/"
    "build-path-dotnet-sdk::\\.dotnet/(sdk|tools)/"
    "build-path-nuget::\\.nuget/packages/"
    "build-path-tmp-build::/tmp/[a-zA-Z0-9_.-]+/(build|target|obj|bin)/"
    "personal-email::[a-zA-Z0-9._%+-]+@(gmail|outlook|hotmail|yahoo|protonmail|proton|icloud|live|me|aol|fastmail|tutanota|googlemail)\\.[a-z]{2,}"
    "tailscale-host::[a-z0-9-]+\\.[a-z0-9-]+\\.ts\\.net"
    "internal-tld::[a-z0-9-]+\\.(internal|lan|home|corp)([^a-zA-Z0-9]|$)"
    "mdns-host::(^|[^a-zA-Z0-9.-])[a-z0-9]([a-z0-9-]{2,})\\.local([^a-zA-Z0-9]|$)"
    "cloud-metadata-ip::169\\.254\\.169\\.254"
    "rfc1918-192::(^|[^0-9])192\\.168\\.[0-9]{1,3}\\.[0-9]{1,3}"
    "rfc1918-172::(^|[^0-9])172\\.(1[6-9]|2[0-9]|3[01])\\.[0-9]{1,3}\\.[0-9]{1,3}"
    "cgnat-100::(^|[^0-9])100\\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\\.[0-9]{1,3}\\.[0-9]{1,3}"
    "pem-marker::-----BEGIN [A-Z ]+-----"
    "aws-access-key::AKIA[0-9A-Z]{16}"
    "github-token::gh[pousr]_[A-Za-z0-9]{20,}"
    "openai-key::sk-[A-Za-z0-9]{20,}"
    "operator-wallet::prl1q[02-9ac-hj-np-z]{20,}"
)

ALLOWLIST=(
    "akoyapool.com"
    "akoyapool.org"
    "127.0.0.1"
    "0.0.0.0"
)

scan_file() {
    local f="$1"
    local hits=0
    local strs
    strs=$(strings -a "$f" 2>/dev/null) || { echo 0; return 0; }
    for entry in "${PATTERNS[@]}"; do
        local label="${entry%%::*}"
        local pat="${entry#*::}"
        local matches
        matches=$(printf '%s\n' "$strs" | grep -aE "$pat" 2>/dev/null || true)
        [[ -z "$matches" ]] && continue
        local filtered="$matches"
        for ok in "${ALLOWLIST[@]}"; do
            filtered=$(printf '%s\n' "$filtered" | grep -avF "$ok" || true)
        done
        if [[ -n "$filtered" ]]; then
            echo "  x [$label] in $(basename "$f"):" >&2
            printf '%s\n' "$filtered" | head -3 | sed 's/^/      /' >&2
            hits=$((hits + 1))
        fi
    done
    echo "$hits"
}

TOTAL_HITS=0
SCANNED=0
for root in "$@"; do
    [[ -d "$root" ]] || { echo "  ! skip (not a dir): $root"; continue; }
    while IFS= read -r -d '' f; do
        SCANNED=$((SCANNED + 1))
        h=$(scan_file "$f")
        TOTAL_HITS=$((TOTAL_HITS + h))
    done < <(find "$root" -type f \( -name '*.so' -o -name '*.so.*' -o -executable \) ! -name '*.json' ! -name '*.txt' ! -name '*.md' ! -name '*.sh' -print0)
done

echo ""
if [[ $TOTAL_HITS -gt 0 ]]; then
    echo "  x leak-scan: $TOTAL_HITS pattern hit(s) across $SCANNED file(s) - RELEASE BLOCKED"
    echo ""
    echo "  Common fixes:"
    echo "    Rust:  strip=true/debug=false and remap paths with RUSTFLAGS"
    echo "    CUDA:  avoid -lineinfo and use debug-prefix-map where possible"
    echo "    .NET:  publish with StripSymbols and no debug symbols"
    echo "    All:   strip --strip-all <binary> after build"
    exit 1
fi

echo "  ok leak-scan: $SCANNED file(s) clean"
exit 0
