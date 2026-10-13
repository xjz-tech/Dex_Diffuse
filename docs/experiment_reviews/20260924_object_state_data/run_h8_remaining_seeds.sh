#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT}/../20260929_h8_object_state_data"
FIRST_LOG="${ROOT}/../20260929_h8_object_state_seed44.log"
for _ in $(seq 1 1440); do
    if rg -q 'BATCH DONE 48' "$FIRST_LOG"; then break; fi
    sleep 30
done
rg -q 'BATCH DONE 48' "$FIRST_LOG"
if rg -q '^FAILED ' "$FIRST_LOG"; then
    echo 'Seed44 contains failed jobs; inspect before extending the sweep.' >&2
    exit 1
fi
for _ in $(seq 1 1440); do
    if [[ -f "${ROOT}/../20260929_h8_prior3000/comparison.json" ]]; then break; fi
    sleep 30
done
[[ -f "${ROOT}/../20260929_h8_prior3000/comparison.json" ]]
/home/carus/miniforge3/envs/dp/bin/python \
    "$ROOT/run_h8_cross_model_comparison.py" \
    --seeds 45 46 --models h8 1b 10b --workers 4 \
    > "$ROOT/../20260929_h8_object_state_seed45_46.log" 2>&1
if rg -q '^FAILED ' "$ROOT/../20260929_h8_object_state_seed45_46.log"; then
    echo 'Later seeds contain failed jobs; inspect before analysis.' >&2
    exit 1
fi
rg -q 'BATCH DONE 144' "$ROOT/../20260929_h8_object_state_seed45_46.log"
echo 'ALL ROLLOUTS COMPLETE'
