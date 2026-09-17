#!/usr/bin/env bash
# Real SharpA inference for the 66-D observation / 22-D action controller.
set -euo pipefail

# TENSORRT=1 TRT_PRECISION=fp32 FUSED_DDIM=1 bash eval/eval_para_obs66_real.sh
# Append --record to save TXT and structured JSONL records (disabled by default).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REAL_DIR="${SCRIPT_DIR}/real"
source "${REAL_DIR}/run_logging.sh"

MODEL_PYTHON="${MODEL_PYTHON:-/home/frankagvl/anaconda3/envs/dexIL/bin/python}"
CKPT_PATH="${CKPT_PATH:-${DEX_ROOT}/runs/obs_4-66.ckpt}"
# CKPT_PATH="${CKPT_PATH:-${DEX_ROOT}/runs/8_2_mixed.ckpt}"
DEVICE="${DEVICE:-cuda:0}"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
SEED="${SEED:-50}"
SAMPLER="${SAMPLER:-ddim}"
TENSORRT="${TENSORRT:-0}"
# FP32 reduces action drift; TRT_PRECISION=fp16 selects the faster engine.
TRT_PRECISION="${TRT_PRECISION:-fp32}"
FUSED_DDIM="${FUSED_DDIM:-${TENSORRT}}"
INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
OBSERVATION_MODE="${OBSERVATION_MODE:-qpos-target-residual}"
ACTION_CHUNK_STEPS="${ACTION_CHUNK_STEPS:-2}"
CONTROL_HZ="${CONTROL_HZ:-30}"
MAX_STEPS="${MAX_STEPS:-0}"

# LIVE=1 sends actions. CHECK_ONLY=1 never connects to hardware.
LIVE="${LIVE:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MODEL_WARMUP="${MODEL_WARMUP:-1}"

HAND_HOST="${HAND_HOST:-localhost}"
HAND_PORT="${HAND_PORT:-5570}"
HAND_TIMEOUT_MS="${HAND_TIMEOUT_MS:-2000}"
HAND_INTERPOLATE="${HAND_INTERPOLATE:-0}"
# Hold the position reached before inference; keep the policy target in obs66.
HOLD_DURING_INFERENCE="${HOLD_DURING_INFERENCE:-1}"
# Select the SharpA pose defined in robot_init.py: ZERO or ROTATE.
INIT_POSE="${INIT_POSE:-ROTATE}"
MOVE_TO_INITIAL_POSE="${MOVE_TO_INITIAL_POSE:-1}"
INITIAL_POSE_STEPS="${INITIAL_POSE_STEPS:-50}"
INITIAL_POSE_SECONDS="${INITIAL_POSE_SECONDS:-2.0}"
INITIAL_POSE_TOLERANCE="${INITIAL_POSE_TOLERANCE:-0.1}"
INITIAL_POSE_TIMEOUT_SECONDS="${INITIAL_POSE_TIMEOUT_SECONDS:-3.0}"
INITIAL_POSE_POLL_SECONDS="${INITIAL_POSE_POLL_SECONDS:-0.1}"
WAIT_FOR_ENTER="${WAIT_FOR_ENTER:-1}"

# Software limits are disabled by default for this launcher; each can be restored.
MAX_HAND_STEP="${MAX_HAND_STEP:-0.09}"
DISABLE_STEP_CLAMP="${DISABLE_STEP_CLAMP:-1}"
DISABLE_JOINT_LIMITS="${DISABLE_JOINT_LIMITS:-1}"
JOINT_LIMIT_MARGIN="${JOINT_LIMIT_MARGIN:-0.0}"
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-0}"
START_DELAY="${START_DELAY:-0.0}"
HOLD_CURRENT_ON_EXIT="${HOLD_CURRENT_ON_EXIT:-1}"
LOG_ACTION_STEPS="${LOG_ACTION_STEPS:-1}"

die() {
    echo "eval_para_obs66_real: $*" >&2
    exit 2
}

require_bool() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[01]$ ]] || die "${name} must be 0 or 1, got ${value}"
}

[[ -x "${MODEL_PYTHON}" ]] || die "Python is not executable: ${MODEL_PYTHON}"
[[ -d "${REAL_DIR}" ]] || die "real runner directory not found: ${REAL_DIR}"
[[ -f "${REAL_DIR}/inference_real.py" ]] || die "missing ${REAL_DIR}/inference_real.py"
[[ -f "${CKPT_PATH}" ]] || die "checkpoint not found: ${CKPT_PATH}"
[[ "${SAMPLER}" == "ddpm" || "${SAMPLER}" == "ddim" ]] || die "SAMPLER must be ddpm or ddim"
[[ "${OBSERVATION_MODE}" == "qpos-target-residual" ]] || die "this obs66 launcher requires OBSERVATION_MODE=qpos-target-residual"
[[ "${INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "INFERENCE_STEPS must be a positive integer"
[[ "${ACTION_CHUNK_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "ACTION_CHUNK_STEPS must be a positive integer"
(( ACTION_CHUNK_STEPS <= 5 )) || die "ACTION_CHUNK_STEPS cannot exceed the checkpoint output length 5"
[[ "${MAX_STEPS}" =~ ^(0|[1-9][0-9]*)$ ]] || die "MAX_STEPS must be a non-negative integer"
[[ "${INIT_POSE}" == "ZERO" || "${INIT_POSE}" == "ROTATE" ]] || die "INIT_POSE must be ZERO or ROTATE"
require_bool LIVE "${LIVE}"
require_bool CHECK_ONLY "${CHECK_ONLY}"
require_bool PREFLIGHT_ONLY "${PREFLIGHT_ONLY}"
require_bool MODEL_WARMUP "${MODEL_WARMUP}"
require_bool TENSORRT "${TENSORRT}"
require_bool FUSED_DDIM "${FUSED_DDIM}"
if [[ "${TENSORRT}" == "1" || "${FUSED_DDIM}" == "1" ]]; then
    [[ "${SAMPLER}" == "ddim" ]] || die "TensorRT/fused inference requires SAMPLER=ddim"
fi
if [[ "${TENSORRT}" == "1" ]]; then
    [[ "${TRT_PRECISION}" == "fp16" || "${TRT_PRECISION}" == "fp32" ]] || die "TRT_PRECISION must be fp16 or fp32"
    [[ "${DEVICE}" == cuda* ]] || die "TensorRT requires DEVICE=cuda:N"
fi
require_bool HAND_INTERPOLATE "${HAND_INTERPOLATE}"
require_bool HOLD_DURING_INFERENCE "${HOLD_DURING_INFERENCE}"
require_bool MOVE_TO_INITIAL_POSE "${MOVE_TO_INITIAL_POSE}"
require_bool WAIT_FOR_ENTER "${WAIT_FOR_ENTER}"
require_bool DISABLE_STEP_CLAMP "${DISABLE_STEP_CLAMP}"
require_bool DISABLE_JOINT_LIMITS "${DISABLE_JOINT_LIMITS}"
require_bool HOLD_CURRENT_ON_EXIT "${HOLD_CURRENT_ON_EXIT}"
require_bool LOG_ACTION_STEPS "${LOG_ACTION_STEPS}"
if [[ "${CHECK_ONLY}" == "1" && "${PREFLIGHT_ONLY}" == "1" ]]; then
    die "CHECK_ONLY and PREFLIGHT_ONLY are mutually exclusive"
fi
if [[ "${CHECK_ONLY}" == "0" && "${PREFLIGHT_ONLY}" == "0" && "${LIVE}" == "1" && "${MOVE_TO_INITIAL_POSE}" == "1" ]]; then
    [[ -f "${REAL_DIR}/robot_init.py" ]] || die "missing ${REAL_DIR}/robot_init.py"
fi

PYTHON_PREFIX="$(cd -- "$(dirname -- "${MODEL_PYTHON}")/.." && pwd)"
COMMON_ENV=(
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    PYTHONUNBUFFERED=1
    PYTHONDONTWRITEBYTECODE=1
    PYTHONNOUSERSITE=1
    PYTHONPATH="${REAL_DIR}:${PYTHONPATH:-}"
    LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
)

# Check the optional runtime before robot_init.py can move the hand.
if [[ "${TENSORRT}" == "1" && "${PREFLIGHT_ONLY}" == "0" ]]; then
    env "${COMMON_ENV[@]}" "${MODEL_PYTHON}" -c 'import torch, torch_tensorrt; assert torch.cuda.is_available(), "TensorRT requires CUDA"' \
        || die "TensorRT runtime unavailable; select MODEL_PYTHON with torch_tensorrt installed, or use TENSORRT=0 FUSED_DDIM=1"
fi

INFERENCE_ARGS=(
    "${REAL_DIR}/inference_real.py"
    --checkpoint "${CKPT_PATH}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --sampler "${SAMPLER}"
    --inference-steps "${INFERENCE_STEPS}"
    --observation-mode "${OBSERVATION_MODE}"
    --action-chunk-steps "${ACTION_CHUNK_STEPS}"
    --hz "${CONTROL_HZ}"
    --max-steps "${MAX_STEPS}"
    --hand-host "${HAND_HOST}"
    --hand-port "${HAND_PORT}"
    --hand-timeout-ms "${HAND_TIMEOUT_MS}"
    --no-move-initial-pose
    --initial-pose-steps "${INITIAL_POSE_STEPS}"
    --initial-pose-seconds "${INITIAL_POSE_SECONDS}"
    --initial-pose-tolerance "${INITIAL_POSE_TOLERANCE}"
    --initial-pose-timeout-seconds "${INITIAL_POSE_TIMEOUT_SECONDS}"
    --initial-pose-poll-seconds "${INITIAL_POSE_POLL_SECONDS}"
    --max-hand-step "${MAX_HAND_STEP}"
    --joint-limit-margin "${JOINT_LIMIT_MARGIN}"
    --max-tracking-error "${MAX_TRACKING_ERROR}"
    --start-delay "${START_DELAY}"
)
if [[ "${MODEL_WARMUP}" == "0" ]]; then
    INFERENCE_ARGS+=(--no-warmup)
fi
if [[ "${TENSORRT}" == "1" ]]; then
    INFERENCE_ARGS+=(--tensorrt --trt-precision "${TRT_PRECISION}")
elif [[ "${FUSED_DDIM}" == "1" ]]; then
    INFERENCE_ARGS+=(--fused-ddim)
fi
if [[ -n "${GUIDE_CKPT_PATH:-}" ]]; then
    [[ -f "${GUIDE_CKPT_PATH}" ]] || die "guide checkpoint not found: ${GUIDE_CKPT_PATH}"
    [[ "${SAMPLER}" == "ddim" ]] || die "guided inference requires SAMPLER=ddim"
    INFERENCE_ARGS+=(
        --guide-checkpoint "${GUIDE_CKPT_PATH}"
        --guidance-scale "${GUIDANCE_SCALE:-25}"
        --guidance-steps "${GUIDANCE_STEPS:-2}"
        --guide-inference-steps "${GUIDE_INFERENCE_STEPS:-4}"
        --guide-seed "${GUIDE_SEED:-${SEED}}"
        --fixed-noise "${FIXED_NOISE:-1}"
    )
fi
if [[ "${HAND_INTERPOLATE}" == "1" ]]; then
    INFERENCE_ARGS+=(--hand-interpolate)
fi
if [[ "${HOLD_DURING_INFERENCE}" == "1" ]]; then
    INFERENCE_ARGS+=(--hold-during-inference)
fi
if [[ "${WAIT_FOR_ENTER}" == "0" ]]; then
    INFERENCE_ARGS+=(--no-wait-for-enter)
fi
if [[ "${DISABLE_STEP_CLAMP}" == "1" ]]; then
    INFERENCE_ARGS+=(--disable-step-clamp)
fi
if [[ "${DISABLE_JOINT_LIMITS}" == "1" ]]; then
    INFERENCE_ARGS+=(--disable-joint-limits)
fi
if [[ "${HOLD_CURRENT_ON_EXIT}" == "0" ]]; then
    INFERENCE_ARGS+=(--no-hold-current-on-exit)
fi
if [[ "${LOG_ACTION_STEPS}" == "1" ]]; then
    INFERENCE_ARGS+=(--log-action-steps)
fi

echo "[real] python: ${MODEL_PYTHON}"
echo "[real] checkpoint: ${CKPT_PATH}"
echo "[real] policy: obs=66 action=22 sampler=${SAMPLER} inference_steps=${INFERENCE_STEPS} chunk=${ACTION_CHUNK_STEPS}"
echo "[real] acceleration: tensorrt=${TENSORRT} precision=${TRT_PRECISION} fused_ddim=$(( TENSORRT || FUSED_DDIM ))"

if [[ "${CHECK_ONLY}" == "1" ]]; then
    echo "[real] CHECK_ONLY: model load + synthetic inference; hardware will not be connected"
    exec env "${COMMON_ENV[@]}" "${MODEL_PYTHON}" -u "${INFERENCE_ARGS[@]}" --check-only
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    [[ -f "${REAL_DIR}/robot_init.py" ]] || die "missing ${REAL_DIR}/robot_init.py"
    echo "[real] PREFLIGHT_ONLY: checking SharpA communication; no motion command will be sent"
    exec env "${COMMON_ENV[@]}" "${MODEL_PYTHON}" -u "${REAL_DIR}/robot_init.py" \
        --skip-franka --init-pose "${INIT_POSE}" \
        --hand-host "${HAND_HOST}" \
        --hand-port "${HAND_PORT}" \
        --hand-timeout-ms "${HAND_TIMEOUT_MS}" \
        --check-only
fi

if [[ "${LIVE}" == "1" ]]; then
    INFERENCE_ARGS+=(--live --reset-before-load)
    if [[ "${MOVE_TO_INITIAL_POSE}" == "1" ]]; then
        echo "[real] startup: robot_init.py (SharpA only) -> wait for Enter (if enabled) -> load model -> infer"
        env "${COMMON_ENV[@]}" "${MODEL_PYTHON}" -u "${REAL_DIR}/robot_init.py" \
            --skip-franka --init-pose "${INIT_POSE}" \
            --hand-host "${HAND_HOST}" --hand-port "${HAND_PORT}" \
            --hand-timeout-ms "${HAND_TIMEOUT_MS}" \
            --hand-steps "${INITIAL_POSE_STEPS}" --hand-seconds "${INITIAL_POSE_SECONDS}" \
            --hand-tolerance "${INITIAL_POSE_TOLERANCE}" \
            --hand-timeout-seconds "${INITIAL_POSE_TIMEOUT_SECONDS}" \
            --poll-seconds "${INITIAL_POSE_POLL_SECONDS}"
    else
        echo "[real] startup: SharpA initial-pose motion disabled"
    fi
    echo "[real] LIVE: wait_for_enter=${WAIT_FOR_ENTER}; Ctrl-C holds measured hand qpos"
else
    echo "[real] DRY RUN: reading SharpA state and printing policy targets; no action commands"
fi
echo "[real] schedule: infer -> execute ${ACTION_CHUNK_STEPS} targets at ${CONTROL_HZ} Hz -> infer"
echo "[real] hold during inference: ${HOLD_DURING_INFERENCE} (live only)"
exec env "${COMMON_ENV[@]}" "${MODEL_PYTHON}" -u "${INFERENCE_ARGS[@]}"
