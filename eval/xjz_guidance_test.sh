#!/usr/bin/env bash
# Paired XJZ evaluation: weak 10k prior guided by the 1B obs4x66 policy.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

WEAK_CKPT_PATH="${WEAK_CKPT_PATH:-${DEX_ROOT}/runs/sim_hand_10k/checkpoints/latest.ckpt}"
GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH:-/home/carus/data_usb/obs_4-66.ckpt}"
GUIDANCE_SCALES="${GUIDANCE_SCALES:-0,100}"
GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS:-8}"
FIXED_NOISE="${FIXED_NOISE:-1}"
MODEL_SERVER="${MODEL_SERVER:-${SCRIPT_DIR}/guided_model_server.py}"
XJZ_TEST_SCRIPT="${XJZ_TEST_SCRIPT:-${SCRIPT_DIR}/xjz_test.sh}"

[[ -f "${WEAK_CKPT_PATH}" ]] || {
    echo "weak checkpoint not found: ${WEAK_CKPT_PATH}" >&2
    exit 2
}
[[ -f "${GUIDE_CKPT_PATH}" ]] || {
    echo "strong guide checkpoint not found: ${GUIDE_CKPT_PATH}" >&2
    exit 2
}
[[ -f "${MODEL_SERVER}" ]] || {
    echo "guided model server not found: ${MODEL_SERVER}" >&2
    exit 2
}
[[ -f "${XJZ_TEST_SCRIPT}" ]] || {
    echo "xjz test script not found: ${XJZ_TEST_SCRIPT}" >&2
    exit 2
}

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/hold_runs/${STAMP}_weak10k_guided_by_1b}"
mkdir -p "${RUN_DIR}"

IFS=',' read -r -a SCALE_VALUES <<< "${GUIDANCE_SCALES}"
for scale in "${SCALE_VALUES[@]}"; do
    [[ "${scale}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
        echo "invalid guidance scale: ${scale}" >&2
        exit 2
    }
    name="weak10k_guide1b_scale${scale}"
    echo "[xjz-guidance] start ${name}"
    CKPT_PATH="${WEAK_CKPT_PATH}" \
    GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH}" \
    MODEL_SERVER="${MODEL_SERVER}" \
    GUIDANCE_SCALE="${scale}" \
    GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS}" \
    FIXED_NOISE="${FIXED_NOISE}" \
    RUN_DIR="${RUN_DIR}" \
    RUN_NAME="${name}" \
    EPISODE_LOG="${RUN_DIR}/${name}.jsonl" \
        bash "${XJZ_TEST_SCRIPT}"
done

echo "[xjz-guidance] logs: ${RUN_DIR}"
