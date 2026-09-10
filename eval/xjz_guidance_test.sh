#!/usr/bin/env bash
# Grid: guide prefix {2,4,6,8} x execute {2,4,6,8} x scale {50,100,150}.
# Skip cells where execution exceeds the guided prefix (fill with /).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PRIOR_CKPT_PATH="${PRIOR_CKPT_PATH:-/home/carus/data_usb/obs_4-66.ckpt}"
GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH:-${DEX_ROOT}/runs/sim_hand_10k_seed42/checkpoints/latest.ckpt}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-50,100,150}"
SAMPLER="${SAMPLER:-ddim}"
INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS:-4}"
FIXED_NOISE="${FIXED_NOISE:-1}"
GUIDANCE_STEPS_LIST="${GUIDANCE_STEPS_LIST:-2,4,6,8}"
EXECUTION_STEPS_LIST="${EXECUTION_STEPS_LIST:-2,4,6,8}"
MODEL_SERVER="${MODEL_SERVER:-${SCRIPT_DIR}/xjz_eval_strong_prior.py}"
XJZ_TEST_SCRIPT="${XJZ_TEST_SCRIPT:-${SCRIPT_DIR}/xjz_test.sh}"
MODEL_PYTHON="${MODEL_PYTHON:-/home/carus/miniforge3/envs/dp/bin/python}"

die() {
    echo "xjz_guidance_test: $*" >&2
    exit 2
}

[[ -f "${PRIOR_CKPT_PATH}" ]] || die "strong prior checkpoint not found: ${PRIOR_CKPT_PATH}"
[[ -f "${GUIDE_CKPT_PATH}" ]] || die "weak guide checkpoint not found: ${GUIDE_CKPT_PATH}"
[[ -f "${MODEL_SERVER}" ]] || die "guided model server not found: ${MODEL_SERVER}"
[[ -f "${XJZ_TEST_SCRIPT}" ]] || die "xjz test script not found: ${XJZ_TEST_SCRIPT}"

IFS=',' read -r -a SCALE_ARR <<< "${GUIDANCE_SCALES}"
IFS=',' read -r -a GUIDE_ARR <<< "${GUIDANCE_STEPS_LIST}"
IFS=',' read -r -a EXEC_ARR <<< "${EXECUTION_STEPS_LIST}"
[[ "${#SCALE_ARR[@]}" -ge 1 ]] || die "GUIDANCE_SCALES must not be empty"
[[ "${#GUIDE_ARR[@]}" -ge 1 ]] || die "GUIDANCE_STEPS_LIST must not be empty"
[[ "${#EXEC_ARR[@]}" -ge 1 ]] || die "EXECUTION_STEPS_LIST must not be empty"
for steps in "${GUIDE_ARR[@]}" "${EXEC_ARR[@]}"; do
    [[ "${steps}" =~ ^[1-9]$ ]] || die "guide/exec steps must be integers in 1..9, got ${steps}"
done
for scale in "${SCALE_ARR[@]}"; do
    [[ "${scale}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid guidance scale: ${scale}"
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/hold_runs/${STAMP}_strong1b_guide_exec_grid}"
mkdir -p "${RUN_DIR}"

echo "[xjz-guidance-grid] run_dir=${RUN_DIR}"
echo "[xjz-guidance-grid] pred=9 ddim=${INFERENCE_STEPS} guide_ddim=${GUIDE_INFERENCE_STEPS} scales=${GUIDANCE_SCALES}"
echo "[xjz-guidance-grid] guide=${GUIDANCE_STEPS_LIST} exec=${EXECUTION_STEPS_LIST} (skip exec>guide)"

SUMMARY_LOGS=()
for scale in "${SCALE_ARR[@]}"; do
    for guide_steps in "${GUIDE_ARR[@]}"; do
        for exec_steps in "${EXEC_ARR[@]}"; do
            name="scale${scale}_guide${guide_steps}_exec${exec_steps}"
            if (( exec_steps > guide_steps )); then
                echo "[xjz-guidance-grid] skip ${name} (exec ${exec_steps} > guide ${guide_steps})"
                continue
            fi
            echo "[xjz-guidance-grid] start ${name}"
            CKPT_PATH="${PRIOR_CKPT_PATH}" \
            GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH}" \
            MODEL_SERVER="${MODEL_SERVER}" \
            GUIDANCE_SCALE="${scale}" \
            GUIDANCE_STEPS="${guide_steps}" \
            GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS}" \
            FIXED_NOISE="${FIXED_NOISE}" \
            SAMPLER="${SAMPLER}" \
            INFERENCE_STEPS="${INFERENCE_STEPS}" \
            EXECUTION_STEPS="${exec_steps}" \
            RUN_DIR="${RUN_DIR}" \
            RUN_NAME="${name}" \
            EPISODE_LOG="${RUN_DIR}/${name}.jsonl" \
                bash "${XJZ_TEST_SCRIPT}"
            echo "[xjz-guidance-grid] finished ${name}"
            SUMMARY_LOGS+=("${RUN_DIR}/${name}.jsonl")
        done
    done
done

if [[ "${#SUMMARY_LOGS[@]}" -gt 0 && -x "${MODEL_PYTHON}" ]]; then
    echo "[xjz-guidance-grid] summaries"
    "${MODEL_PYTHON}" "${SCRIPT_DIR}/episode_stats.py" "${SUMMARY_LOGS[@]}"
    echo "[xjz-guidance-grid] 4x4 tables per scale (mean / median / max hold; / if exec>guide)"
    "${MODEL_PYTHON}" - "${RUN_DIR}" "${GUIDANCE_SCALES}" "${GUIDANCE_STEPS_LIST}" "${EXECUTION_STEPS_LIST}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
scale_vals = [item for item in sys.argv[2].split(",") if item]
guide_vals = [int(item) for item in sys.argv[3].split(",") if item]
exec_vals = [int(item) for item in sys.argv[4].split(",") if item]
stats = {}
for path in sorted(run_dir.glob("scale*_guide*_exec*.jsonl")):
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
    parts = path.stem.split("_")
    scale = parts[0].replace("scale", "")
    guide = int(parts[1].replace("guide", ""))
    exec_steps = int(parts[2].replace("exec", ""))
    stats[(scale, guide, exec_steps)] = (mean, median, ordered[-1], n)

def fmt(value, kind):
    if value is None:
        return f"{'/':>8}"
    mean, median, max_len, _n = value
    if kind == "mean":
        return f"{mean:8.1f}"
    if kind == "median":
        return f"{median:8.1f}"
    return f"{max_len:8d}"

for scale in scale_vals:
    print(f"===== scale {scale} =====")
    for kind, title in (("mean", "mean hold"), ("median", "median hold"), ("max", "max hold")):
        print(title)
        header = f"{'guide\\exec':>10}" + "".join(f"{exec_steps:>8}" for exec_steps in exec_vals)
        print(header)
        for guide in guide_vals:
            row = f"{guide:10d}"
            for exec_steps in exec_vals:
                if exec_steps > guide:
                    row += f"{'/':>8}"
                else:
                    row += fmt(stats.get((scale, guide, exec_steps)), kind)
            print(row)
        print()
PY
fi
echo "[xjz-guidance-grid] logs: ${RUN_DIR}"
