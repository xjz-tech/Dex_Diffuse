#!/usr/bin/env bash
# Paired XJZ evaluation: strong 1B prior guided by the weak 10k policy.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PRIOR_CKPT_PATH="${PRIOR_CKPT_PATH:-/home/carus/data_usb/obs_4-66.ckpt}"
GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH:-${DEX_ROOT}/runs/sim_hand_10k_seed42/checkpoints/latest.ckpt}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0,25}"
GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS:-4}"
GUIDANCE_STEPS="${GUIDANCE_STEPS:-2}"
FIXED_NOISE="${FIXED_NOISE:-1}"
GUIDE_SEED="${GUIDE_SEED:-}"
MODEL_SERVER="${MODEL_SERVER:-${SCRIPT_DIR}/xjz_eval_strong_prior.py}"
XJZ_TEST_SCRIPT="${XJZ_TEST_SCRIPT:-${SCRIPT_DIR}/xjz_test.sh}"
FIRST_EPISODE_ONLY="${FIRST_EPISODE_ONLY:-1}"
CENSOR_UNFINISHED_AT_CAP="${CENSOR_UNFINISHED_AT_CAP:-1}"

die() {
    echo "xjz_eval_strong_prior: $*" >&2
    exit 2
}

[[ -f "${PRIOR_CKPT_PATH}" ]] || die "strong prior checkpoint not found: ${PRIOR_CKPT_PATH}"
[[ -f "${GUIDE_CKPT_PATH}" ]] || die "weak guide checkpoint not found: ${GUIDE_CKPT_PATH}"
[[ -f "${MODEL_SERVER}" ]] || die "strong-prior model server not found: ${MODEL_SERVER}"
[[ -f "${XJZ_TEST_SCRIPT}" ]] || die "xjz test script not found: ${XJZ_TEST_SCRIPT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/hold_runs/${STAMP}_strong1b_guided_by_weak10k}"
mkdir -p "${RUN_DIR}"

IFS=',' read -r -a SCALE_VALUES <<< "${GUIDANCE_SCALES}"
RUN_TAG="${RUN_TAG:-strong1b_weak10k_guide}"
for scale in "${SCALE_VALUES[@]}"; do
    [[ "${scale}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "invalid guidance scale: ${scale}"
    name="${RUN_TAG}_scale${scale}"
    echo "[xjz-strong-prior] start ${name}"
    CKPT_PATH="${PRIOR_CKPT_PATH}" \
    GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH}" \
    MODEL_SERVER="${MODEL_SERVER}" \
    GUIDANCE_SCALE="${scale}" \
    GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS}" \
    GUIDANCE_STEPS="${GUIDANCE_STEPS}" \
    FIXED_NOISE="${FIXED_NOISE}" \
    GUIDE_SEED="${GUIDE_SEED}" \
    FIRST_EPISODE_ONLY="${FIRST_EPISODE_ONLY}" \
    CENSOR_UNFINISHED_AT_CAP="${CENSOR_UNFINISHED_AT_CAP}" \
    RUN_DIR="${RUN_DIR}" \
    RUN_NAME="${name}" \
    EPISODE_LOG="${RUN_DIR}/${name}.jsonl" \
        bash "${XJZ_TEST_SCRIPT}"
    echo "[xjz-strong-prior] finished ${name}"
done

echo "[xjz-strong-prior] logs: ${RUN_DIR}"
