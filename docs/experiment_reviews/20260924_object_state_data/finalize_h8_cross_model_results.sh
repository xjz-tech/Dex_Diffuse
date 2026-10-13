#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOGS="${ROOT}/.."
for _ in $(seq 1 1440); do
    if rg -q 'BATCH DONE 144' "$LOGS/20260929_h8_object_state_seed45_46.log" && \
       rg -q 'ONLINE AUDIT COMPLETE 144' "$LOGS/20260929_h8_object_state_seed45_46_audit.log"; then break; fi
    sleep 30
done
rg -q 'BATCH DONE 144' "$LOGS/20260929_h8_object_state_seed45_46.log"
rg -q 'ONLINE AUDIT COMPLETE 144' "$LOGS/20260929_h8_object_state_seed45_46_audit.log"
if rg -q '^FAILED ' "$LOGS/20260929_h8_object_state_seed45_46.log"; then
    echo 'A rollout failed; refusing to assemble an incomplete comparison.' >&2
    exit 1
fi
/home/carus/miniforge3/envs/decv2/bin/python \
    "$ROOT/assemble_h8_cross_model_results.py" \
    > "$LOGS/20260929_h8_object_state_assemble.log" 2>&1
/home/carus/miniforge3/envs/decv2/bin/python \
    "$ROOT/report_h8_cross_model_comparison.py" \
    > "$LOGS/20260929_h8_object_state_report.log" 2>&1
echo 'FINAL REPORT COMPLETE'
