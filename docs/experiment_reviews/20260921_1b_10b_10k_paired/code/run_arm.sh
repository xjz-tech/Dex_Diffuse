#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 MODEL SEED NUM_ENVS MAX_STEPS OUTPUT_DIR" >&2
    exit 2
fi

MODEL="$1"
SEED_VALUE="$2"
NUM_ENVS_VALUE="$3"
MAX_STEPS_VALUE="$4"
OUTPUT_DIR="$(realpath -m "$5")"
PROJECT_ROOT="/home/carus/Program/Dexterous_Manipulation/Dex_diffuse"

case "${MODEL}" in
    1b) CHECKPOINT="/home/carus/data_usb/obs_4-66.ckpt" ;;
    10b) CHECKPOINT="/home/carus/data_usb/10B_obs_4-66.ckpt" ;;
    *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_DIR}"
unset GUIDE_CKPT_PATH

NUM_ENV="${NUM_ENVS_VALUE}" \
MAX_FAILURE_EPISODES="${NUM_ENVS_VALUE}" \
MAX_STEPS="${MAX_STEPS_VALUE}" \
DATA_INDICES="000-149" \
HEADLESS=1 \
RECORDING=0 \
PRINT_EVERY=1000 \
SEED="${SEED_VALUE}" \
SAMPLER=ddim \
INFERENCE_STEPS=4 \
EXECUTION_STEPS=2 \
FIRST_EPISODE_ONLY=1 \
CENSOR_UNFINISHED_AT_CAP=1 \
FAILURE_OBJ_POS_THRES_M=0.05 \
FAILURE_TIP_POS_THRES_M=0.1 \
FAILURE_OBJ_ROT_THRES_DEG=180 \
INVALID_OBJ_POS_THRES_M=0.15 \
FAILURE_TOLERANCE_SCALE=10000 \
FIXED_TOLERANCE_STEPS=20000 \
TRAJ_STEPS_LIMIT=12000 \
RESET_ON_REACH_GOAL=0 \
CROSS_TRAJECTORY_GOAL_PROB=0.3 \
MODEL_SERVER="${PROJECT_ROOT}/eval/model_server.py" \
MODEL_PYTHON="/home/carus/miniforge3/envs/dp/bin/python" \
CKPT_PATH="${CHECKPOINT}" \
RUN_DIR="${OUTPUT_DIR}" \
RUN_NAME="ordinary_${MODEL}_seed${SEED_VALUE}" \
EPISODE_LOG="${OUTPUT_DIR}/episodes.jsonl" \
INITIAL_STATE_DUMP="${OUTPUT_DIR}/initial_state.npz" \
    bash "${PROJECT_ROOT}/eval/xjz_test.sh"
