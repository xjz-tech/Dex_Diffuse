#!/usr/bin/env bash
# Compare 22-D vs 66-D Sim-Hand checkpoints on mean steps-before-failure.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CKPT_OBS22="${CKPT_OBS22:-${DEX_ROOT}/data/outputs/2026.08.28/10.55.14_train_diffusion_unet_sim_hand_sim_hand_lowdim/checkpoints/step_01700000.ckpt}"
CKPT_OBS66="${CKPT_OBS66:-/home/carus/data_usb/obs_4-66.ckpt}"

# Isaac Gym + 1D DP barely uses VRAM. 1024 envs is the current comparison default.
NUM_ENV="${NUM_ENV:-512}"
MAX_FAILURE_EPISODES="${MAX_FAILURE_EPISODES:-3000}"
DATA_INDICES="${DATA_INDICES:-000-149}"
HEADLESS="${HEADLESS:-1}"
RECORDING="${RECORDING:-0}"
PRINT_EVERY="${PRINT_EVERY:-100}"
SEED="${SEED:-42}"
SAMPLER="${SAMPLER:-ddim}"
INFERENCE_STEPS_LIST="${INFERENCE_STEPS_LIST:-8,16}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/hold_runs/${STAMP}}"
mkdir -p "${RUN_DIR}"

die() {
    echo "compare_hold: $*" >&2
    exit 2
}

[[ -f "${CKPT_OBS22}" ]] || die "22-D checkpoint not found: ${CKPT_OBS22}"
[[ -f "${CKPT_OBS66}" ]] || die "66-D checkpoint not found: ${CKPT_OBS66}"
[[ "${NUM_ENV}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENV must be a positive integer"
[[ "${MAX_FAILURE_EPISODES}" =~ ^[1-9][0-9]*$ ]] || die "MAX_FAILURE_EPISODES must be a positive integer"
[[ "${SAMPLER}" == "ddpm" || "${SAMPLER}" == "ddim" ]] || die "SAMPLER must be ddpm or ddim"

IFS=',' read -r -a INFERENCE_STEPS_ARR <<< "${INFERENCE_STEPS_LIST}"
[[ "${#INFERENCE_STEPS_ARR[@]}" -ge 1 ]] || die "INFERENCE_STEPS_LIST must not be empty"
for steps in "${INFERENCE_STEPS_ARR[@]}"; do
    [[ "${steps}" =~ ^[1-9][0-9]*$ ]] || die "invalid inference steps: ${steps}"
done

run_one() {
    local name="$1"
    local ckpt="$2"
    local steps="$3"
    echo "[compare_hold] start ${name} sampler=${SAMPLER} steps=${steps} ckpt=${ckpt}"
    NUM_ENV="${NUM_ENV}" \
    MAX_FAILURE_EPISODES="${MAX_FAILURE_EPISODES}" \
    DATA_INDICES="${DATA_INDICES}" \
    HEADLESS="${HEADLESS}" \
    RECORDING="${RECORDING}" \
    PRINT_EVERY="${PRINT_EVERY}" \
    SEED="${SEED}" \
    SAMPLER="${SAMPLER}" \
    INFERENCE_STEPS="${steps}" \
    CKPT_PATH="${ckpt}" \
    RUN_NAME="${name}" \
    EPISODE_LOG="${RUN_DIR}/${name}.jsonl" \
        bash "${SCRIPT_DIR}/eval.sh"
    echo "[compare_hold] finished ${name}"
}

echo "[compare_hold] run_dir=${RUN_DIR}"
echo "[compare_hold] num_env=${NUM_ENV} n_failure=${MAX_FAILURE_EPISODES} sampler=${SAMPLER} steps=${INFERENCE_STEPS_LIST} trajectories=${DATA_INDICES}"

SUMMARY_LOGS=()
for steps in "${INFERENCE_STEPS_ARR[@]}"; do
    name22="obs22_step01700000_${SAMPLER}${steps}"
    name66="obs66_${SAMPLER}${steps}"
    run_one "${name22}" "${CKPT_OBS22}" "${steps}"
    run_one "${name66}" "${CKPT_OBS66}" "${steps}"
    SUMMARY_LOGS+=("${RUN_DIR}/${name22}.jsonl" "${RUN_DIR}/${name66}.jsonl")
done

MODEL_PYTHON="${MODEL_PYTHON:-/home/carus/miniforge3/envs/dp/bin/python}"
echo "[compare_hold] summaries"
"${MODEL_PYTHON}" "${SCRIPT_DIR}/episode_stats.py" "${SUMMARY_LOGS[@]}"
echo "[compare_hold] logs: ${RUN_DIR}"
