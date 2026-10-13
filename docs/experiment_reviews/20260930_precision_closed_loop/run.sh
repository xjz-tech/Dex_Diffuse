#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
echo "BATCH START $(date -Is)"
/home/carus/miniforge3/envs/dp/bin/python "$EXPERIMENT_ROOT/run_precision_closed_loop.py" --workers 2
echo "SIMULATIONS COMPLETE $(date -Is)"
/home/carus/miniforge3/envs/decv2/bin/python "$EXPERIMENT_ROOT/analyze_precision_closed_loop.py"
echo "REPORT COMPLETE $(date -Is)"
