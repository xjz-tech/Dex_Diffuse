#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${ROOT}/../20260924_object_state_data"
export PYTHONPATH="${SOURCE}:${PYTHONPATH:-}"
PYTHON=/home/carus/miniforge3/envs/decv2/bin/python

echo "BATCH START $(date -Is)"
"$PYTHON" "$SOURCE/run_random10_cross_model.py"
"$PYTHON" "$SOURCE/summarize_random10_cross_model.py" > "$ROOT/report_generation.log" 2>&1
echo "REPORT COMPLETE $(date -Is)"
