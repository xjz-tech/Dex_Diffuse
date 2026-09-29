#!/usr/bin/env bash
# Real visual DP -> reference-initialized SDEdit -> existing Franka/SharpA loop.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# A new script should first perform synthetic model inference with no hardware.
# For a live run, set CHECK_ONLY=0 LIVE=1 after reviewing both checkpoints.
export CHECK_ONLY="${CHECK_ONLY:-1}"
export LIVE="${LIVE:-0}"
export MAX_CHUNKS="${MAX_CHUNKS:-1}"
export CONTROLLER_CALLS_PER_DP="${CONTROLLER_CALLS_PER_DP:-1}"
export CONTROLLER_ACTION_CHUNK_SIZE="${CONTROLLER_ACTION_CHUNK_SIZE:-2}"
export DDIM_INFERENCE_STEPS="${DDIM_INFERENCE_STEPS:-4}"
export EDIT_NOISE_RATIO="${EDIT_NOISE_RATIO:-0.15}"
export EDIT_INTERPOLATE_LARGE_ACTIONS="${EDIT_INTERPOLATE_LARGE_ACTIONS:-0}"
export EDIT_INTERPOLATION_THRESHOLD_RAD="${EDIT_INTERPOLATION_THRESHOLD_RAD:-0.12}"
export FIXED_NOISE="${FIXED_NOISE:-1}"
export GUIDANCE_SCALE=0
export EDIT_MODE=1
export INFERENCE_SCRIPT="${SCRIPT_DIR}/inference_dp_edit.py"
export CONTROLLER_CKPT_PATH="${CONTROLLER_CKPT_PATH:-/media/frankagvl/U393/Dex_Diffuse/runs/obs_4-66.ckpt}"

exec bash "${SCRIPT_DIR}/eval_dp_controller_obs66.sh" "$@"
