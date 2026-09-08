#!/usr/bin/env bash
# Grid: DDIM inference steps x executed action length, para failure criteria.
# Predict 9 throughout. Rank by mean hold-before-failure length.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INFERENCE_STEPS_LIST="${INFERENCE_STEPS_LIST:-2,4,8,16}"
EXECUTION_STEPS_LIST="${EXECUTION_STEPS_LIST:-2,3,4,5,6,7,8}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/hold_runs/${STAMP}_xjz_grid}"
mkdir -p "${RUN_DIR}"

die() {
    echo "xjz_grid: $*" >&2
    exit 2
}

IFS=',' read -r -a INFERENCE_STEPS_ARR <<< "${INFERENCE_STEPS_LIST}"
IFS=',' read -r -a EXECUTION_STEPS_ARR <<< "${EXECUTION_STEPS_LIST}"
[[ "${#INFERENCE_STEPS_ARR[@]}" -ge 1 ]] || die "INFERENCE_STEPS_LIST must not be empty"
[[ "${#EXECUTION_STEPS_ARR[@]}" -ge 1 ]] || die "EXECUTION_STEPS_LIST must not be empty"
for steps in "${INFERENCE_STEPS_ARR[@]}"; do
    [[ "${steps}" =~ ^[1-9][0-9]*$ ]] || die "invalid DDIM steps: ${steps}"
done
for steps in "${EXECUTION_STEPS_ARR[@]}"; do
    [[ "${steps}" =~ ^[1-9]$ ]] || die "invalid execution steps: ${steps}"
done

echo "[xjz_grid] run_dir=${RUN_DIR}"
echo "[xjz_grid] pred=9 ddim=${INFERENCE_STEPS_LIST} exec=${EXECUTION_STEPS_LIST}"

SUMMARY_LOGS=()
for ddim_steps in "${INFERENCE_STEPS_ARR[@]}"; do
    for exec_steps in "${EXECUTION_STEPS_ARR[@]}"; do
        name="ddim${ddim_steps}_exec${exec_steps}"
        echo "[xjz_grid] start ${name}"
        RUN_DIR="${RUN_DIR}" \
        INFERENCE_STEPS="${ddim_steps}" \
        EXECUTION_STEPS="${exec_steps}" \
        RUN_NAME="${name}" \
        EPISODE_LOG="${RUN_DIR}/${name}.jsonl" \
            bash "${SCRIPT_DIR}/xjz_test.sh"
        echo "[xjz_grid] finished ${name}"
        SUMMARY_LOGS+=("${RUN_DIR}/${name}.jsonl")
    done
done

MODEL_PYTHON="${MODEL_PYTHON:-/home/carus/miniforge3/envs/dp/bin/python}"
echo "[xjz_grid] summaries"
"${MODEL_PYTHON}" "${SCRIPT_DIR}/episode_stats.py" "${SUMMARY_LOGS[@]}"
echo "[xjz_grid] ranking by mean failure length"
"${MODEL_PYTHON}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for path in sorted(run_dir.glob("ddim*_exec*.jsonl")):
    lengths = []
    with path.open() as handle:
        for line in handle:
            rec = json.loads(line)
            if rec.get("reason") == "failure":
                lengths.append(int(rec["length"]))
    if not lengths:
        continue
    ordered = sorted(lengths)
    n = len(ordered)
    mean = sum(ordered) / float(n)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    name = path.stem
    ddim = int(name.split("_")[0].replace("ddim", ""))
    exec_steps = int(name.split("_")[1].replace("exec", ""))
    rows.append((mean, median, ordered[-1], n, ddim, exec_steps, name))

rows.sort(reverse=True)
print(f"{'rank':>4} {'ddim':>5} {'exec':>5} {'mean':>8} {'median':>8} {'max':>7} {'n':>5}")
for rank, (mean, median, max_len, n, ddim, exec_steps, name) in enumerate(rows, 1):
    print(f"{rank:4d} {ddim:5d} {exec_steps:5d} {mean:8.1f} {median:8.1f} {max_len:7d} {n:5d}")
if rows:
    best = rows[0]
    print(
        "best: DDIM %d / exec %d  mean=%.1f median=%.1f"
        % (best[4], best[5], best[0], best[1])
    )
PY
echo "[xjz_grid] logs: ${RUN_DIR}"
