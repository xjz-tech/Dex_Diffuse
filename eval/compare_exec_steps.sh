#!/usr/bin/env bash
# Sweep executed action-chunk length while the policy still predicts 9 steps.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CKPT_PATH="${CKPT_PATH:-/home/carus/data_usb/obs_4-66.ckpt}"
NUM_ENV="${NUM_ENV:-1024}"
MAX_FAILURE_EPISODES="${MAX_FAILURE_EPISODES:-3000}"
DATA_INDICES="${DATA_INDICES:-000-149}"
HEADLESS="${HEADLESS:-1}"
RECORDING="${RECORDING:-0}"
PRINT_EVERY="${PRINT_EVERY:-100}"
SEED="${SEED:-42}"
SAMPLER="${SAMPLER:-ddim}"
INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
EXECUTION_STEPS_LIST="${EXECUTION_STEPS_LIST:-1,2,3,4,5,6,7,8,9}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/hold_runs/${STAMP}_obs66_${SAMPLER}${INFERENCE_STEPS}_exec}"
mkdir -p "${RUN_DIR}"

die() {
    echo "compare_exec_steps: $*" >&2
    exit 2
}

[[ -f "${CKPT_PATH}" ]] || die "checkpoint not found: ${CKPT_PATH}"
[[ "${NUM_ENV}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENV must be a positive integer"
[[ "${MAX_FAILURE_EPISODES}" =~ ^[1-9][0-9]*$ ]] || die "MAX_FAILURE_EPISODES must be a positive integer"
[[ "${SAMPLER}" == "ddpm" || "${SAMPLER}" == "ddim" ]] || die "SAMPLER must be ddpm or ddim"
[[ "${INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "INFERENCE_STEPS must be a positive integer"

IFS=',' read -r -a EXECUTION_STEPS_ARR <<< "${EXECUTION_STEPS_LIST}"
[[ "${#EXECUTION_STEPS_ARR[@]}" -ge 1 ]] || die "EXECUTION_STEPS_LIST must not be empty"
for steps in "${EXECUTION_STEPS_ARR[@]}"; do
    [[ "${steps}" =~ ^[1-9]$ ]] || die "invalid execution steps: ${steps}"
done

run_one() {
    local exec_steps="$1"
    local name="obs66_${SAMPLER}${INFERENCE_STEPS}_exec${exec_steps}"
    echo "[compare_exec_steps] start ${name} ckpt=${CKPT_PATH}"
    NUM_ENV="${NUM_ENV}" \
    MAX_FAILURE_EPISODES="${MAX_FAILURE_EPISODES}" \
    DATA_INDICES="${DATA_INDICES}" \
    HEADLESS="${HEADLESS}" \
    RECORDING="${RECORDING}" \
    PRINT_EVERY="${PRINT_EVERY}" \
    SEED="${SEED}" \
    SAMPLER="${SAMPLER}" \
    INFERENCE_STEPS="${INFERENCE_STEPS}" \
    EXECUTION_STEPS="${exec_steps}" \
    CKPT_PATH="${CKPT_PATH}" \
    RUN_NAME="${name}" \
    EPISODE_LOG="${RUN_DIR}/${name}.jsonl" \
        bash "${SCRIPT_DIR}/eval.sh"
    echo "[compare_exec_steps] finished ${name}"
}

echo "[compare_exec_steps] run_dir=${RUN_DIR}"
echo "[compare_exec_steps] num_env=${NUM_ENV} n_failure=${MAX_FAILURE_EPISODES} sampler=${SAMPLER} inference_steps=${INFERENCE_STEPS} pred=9 exec=${EXECUTION_STEPS_LIST} trajectories=${DATA_INDICES}"

SUMMARY_LOGS=()
for steps in "${EXECUTION_STEPS_ARR[@]}"; do
    run_one "${steps}"
    SUMMARY_LOGS+=("${RUN_DIR}/obs66_${SAMPLER}${INFERENCE_STEPS}_exec${steps}.jsonl")
done

MODEL_PYTHON="${MODEL_PYTHON:-/home/carus/miniforge3/envs/dp/bin/python}"
echo "[compare_exec_steps] summaries"
"${MODEL_PYTHON}" "${SCRIPT_DIR}/episode_stats.py" "${SUMMARY_LOGS[@]}"
echo "[compare_exec_steps] logs: ${RUN_DIR}"
