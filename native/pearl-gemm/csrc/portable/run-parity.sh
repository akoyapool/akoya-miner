#!/usr/bin/env bash
# Entrypoint for pearl-gemm-kernel-parity images.
# Usage: docker run ... pearl-gemm-kernel-parity:tag [tiny|production|all]
set -euo pipefail

echo "============================================================"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total \
           --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi: not available"
echo "------------------------------------------------------------"
ls -lh /opt/goldens/
echo "------------------------------------------------------------"
PY=/app/.venv/bin/python
"$PY" -c "import pearl_gemm; print(f'pearl_gemm build_arch={getattr(pearl_gemm, \"__build_arch__\", \"(unknown)\")}')" 2>/dev/null || true
echo "============================================================"
echo

mode="${1:-all}"
case "$mode" in
    tiny)        files=(/opt/goldens/tiny.pt) ;;
    production)  files=(/opt/goldens/production.pt) ;;
    all)         files=(/opt/goldens/tiny.pt /opt/goldens/production.pt) ;;
    *)           echo "Usage: $0 [tiny|production|all]"; exit 2 ;;
esac

exec "$PY" /opt/verify_goldens.py "${files[@]}"
