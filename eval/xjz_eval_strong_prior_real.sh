#!/usr/bin/env bash
# Strong prior + weak trajectory guidance on the real SharpA hand.
# CHECK_ONLY=1 runs synthetic inference without connecting to hardware.
set -euo pipefail

# TENSORRT=1 TRT_PRECISION=fp32 FUSED_DDIM=1 GUIDANCE_SCALE=25 bash eval/xjz_eval_strong_prior_real.sh
# Append --record to save TXT and structured JSONL records (disabled by default).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/real/run_logging.sh"

export CKPT_PATH="${PRIOR_CKPT_PATH:-${DEX_ROOT}/runs/obs_4-66.ckpt}"
export GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH:-${DEX_ROOT}/runs/epoch_0200.ckpt}"
# One scale per hardware run; use GUIDANCE_SCALE=0 for the prior baseline.
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-25}"
# Match the obs66 real launcher and the 4+4 guidance grid configuration.
export INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
export GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS:-4}"
export GUIDANCE_STEPS="${GUIDANCE_STEPS:-2}"
export FIXED_NOISE="${FIXED_NOISE:-1}"
export GUIDE_SEED="${GUIDE_SEED:-${SEED:-42}}"
export ACTION_CHUNK_STEPS="${ACTION_CHUNK_STEPS:-2}"
export SAMPLER="${SAMPLER:-ddim}"
# Optional batch-1 TensorRT FP16 for both UNets; TRT also enables fused DDIM.
# FUSED_DDIM=1 alone uses analytic guidance with the existing PyTorch UNets.
export TENSORRT="${TENSORRT:-0}"
export FUSED_DDIM="${FUSED_DDIM:-1}"

# Robot init is executed by eval_para_obs66_real.sh using real/robot_init.py
# --skip-franka. Startup: initialize SharpA -> wait for Enter -> load both
# models -> infer. INIT_POSE selects ZERO or the recorded ROTATE pose.
export LIVE="${LIVE:-1}"
export CHECK_ONLY="${CHECK_ONLY:-0}"
export PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
export HAND_HOST="${HAND_HOST:-localhost}"
export HAND_PORT="${HAND_PORT:-5570}"
export HAND_TIMEOUT_MS="${HAND_TIMEOUT_MS:-2000}"
export INIT_POSE="${INIT_POSE:-ROTATE}"
export MOVE_TO_INITIAL_POSE="${MOVE_TO_INITIAL_POSE:-1}"
export INITIAL_POSE_STEPS="${INITIAL_POSE_STEPS:-50}"
export INITIAL_POSE_SECONDS="${INITIAL_POSE_SECONDS:-2.0}"
export INITIAL_POSE_TOLERANCE="${INITIAL_POSE_TOLERANCE:-0.1}"
export INITIAL_POSE_TIMEOUT_SECONDS="${INITIAL_POSE_TIMEOUT_SECONDS:-3.0}"
export INITIAL_POSE_POLL_SECONDS="${INITIAL_POSE_POLL_SECONDS:-0.1}"
export WAIT_FOR_ENTER="${WAIT_FOR_ENTER:-1}"

if [[ -n "${GUIDANCE_SCALES:-}" ]]; then
    echo "xjz_eval_strong_prior_real: use a single GUIDANCE_SCALE for each hardware run" >&2
    exit 2
fi

# Validate guidance settings before the shared launcher can initialize the hand.
[[ "${GUIDANCE_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid GUIDANCE_SCALE" >&2; exit 2; }
[[ "${GUIDANCE_STEPS}" =~ ^[1-9]$ ]] || { echo "GUIDANCE_STEPS must be in 1..9" >&2; exit 2; }
[[ "${GUIDE_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "GUIDE_INFERENCE_STEPS must be positive" >&2; exit 2; }
[[ "${FIXED_NOISE}" =~ ^[01]$ ]] || { echo "FIXED_NOISE must be 0 or 1" >&2; exit 2; }
[[ "${GUIDE_SEED}" =~ ^[0-9]+$ ]] || { echo "GUIDE_SEED must be non-negative integer" >&2; exit 2; }

echo "[real-guidance] prior=${CKPT_PATH} guide=${GUIDE_CKPT_PATH} scale=${GUIDANCE_SCALE}"
exec bash "${SCRIPT_DIR}/eval_para_obs66_real.sh" "$@"
