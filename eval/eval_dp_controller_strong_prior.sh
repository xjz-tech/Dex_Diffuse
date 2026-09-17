#!/usr/bin/env bash
# Real image DP + weak two-action reference -> strong Sim-Hand DDIM.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

die() {
    echo "eval_dp_controller: $*" >&2
    exit 2
}

# Bulb-task DP, strong obs66 prior, and weak observation-conditioned guide.
# All three checkpoint paths can be overridden through environment variables.
DP_CKPT_PATH="${DP_CKPT_PATH:-/media/frankagvl/U393/Dex_Diffuse/runs/0906_manip_bulb/epoch=0400-train_loss=0.0057.ckpt}"
# DP_CKPT_PATH="${DP_CKPT_PATH:-/media/frankagvl/U393/Dex_Diffuse/runs/0905_rotate_bulb_20/epoch=0300-train_loss=0.0145.ckpt}"
CONTROLLER_CKPT_PATH="${CONTROLLER_CKPT_PATH:-${PRIOR_CKPT_PATH:-${DEX_ROOT}/runs/obs_4-66.ckpt}}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-${SCRIPT_DIR}/inference_dp_controller.py}"

MODEL_PYTHON="${MODEL_PYTHON:-}"
if [[ -z "${MODEL_PYTHON}" ]]; then
    for candidate in \
        /home/bighand/miniconda3/envs/dp/bin/python \
        /home/wty/miniconda3/envs/rdp/bin/python \
        /home/frankagvl/anaconda3/envs/dp_w/bin/python; do
        if [[ -x "${candidate}" ]]; then
            MODEL_PYTHON="${candidate}"
            break
        fi
    done
fi

DEVICE="${DEVICE:-cuda:0}"
DP_INFERENCE_STEPS="${DP_INFERENCE_STEPS:-16}"
DDIM_INFERENCE_STEPS="${DDIM_INFERENCE_STEPS:-8}"
# Actions executed per call; the checkpoint determines the output length limit.
# EXECUTION_STEPS is a
# legacy fallback; CONTROLLER_ACTION_CHUNK_SIZE takes precedence when set.
CONTROLLER_ACTION_CHUNK_SIZE="${CONTROLLER_ACTION_CHUNK_SIZE:-${EXECUTION_STEPS:-2}}"
# DP guidance retains its own horizon and weight; weak guidance uses two actions.
GUIDANCE_STEPS="${DP_GUIDANCE_STEPS:-${GUIDANCE_STEPS:-9}}"
GUIDE_CKPT_PATH="${GUIDE_CKPT_PATH:-${DEX_ROOT}/runs/epoch_0200.ckpt}"
WEAK_GUIDANCE_STEPS="${WEAK_GUIDANCE_STEPS:-2}"
WEAK_GUIDANCE_SCALE="${WEAK_GUIDANCE_SCALE:-25}"
GUIDE_INFERENCE_STEPS="${GUIDE_INFERENCE_STEPS:-4}"
GUIDE_SEED="${GUIDE_SEED:-${SEED:-42}}"
INIT_POSE="${INIT_POSE:-ZERO}"
CONTROLLER_CALLS_PER_DP="${CONTROLLER_CALLS_PER_DP:-2}"
GUIDANCE_SCALE="${DP_GUIDANCE_SCALE:-${GUIDANCE_SCALE:-100}}"
ETA="${ETA:-0.0}"
FIXED_NOISE="${FIXED_NOISE:-1}"
SEED="${SEED:-42}"
ALLOW_SALVAGE="${ALLOW_SALVAGE:-1}"
# MAX_CHUNKS counts executed controller chunks, not Real DP proposals (0=unlimited).
MAX_CHUNKS="${MAX_CHUNKS:-0}"
HZ="${HZ:-30}"

# Run the full hardware evaluation by default. Override with CHECK_ONLY=1 and
# LIVE=0 when only checkpoint validation and synthetic inference are needed.
CHECK_ONLY="${CHECK_ONLY:-0}"
LIVE="${LIVE:-1}"

DINOV2_REPO_OR_DIR="${DINOV2_REPO_OR_DIR:-${DEX_ROOT}/assets/dinov2_assets/facebookresearch_dinov2_main}"
DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-${DEX_ROOT}/assets/dinov2_assets/dinov2_vits14_pretrain.pth}"
DINOV2_SOURCE="${DINOV2_SOURCE:-local}"

FRANKA_HOST="${FRANKA_HOST:-172.16.0.10}"
FRANKA_PORT="${FRANKA_PORT:-9090}"
FRANKA_URDF="${FRANKA_URDF:-${DEX_ROOT}/maniptrans_envs/assets/fr3_arm/fr3.urdf}"
HAND_HOST="${HAND_HOST:-localhost}"
HAND_PORT="${HAND_PORT:-5570}"
FRONT_SERIAL="${FRONT_SERIAL:-}"
WRIST_SERIAL="${WRIST_SERIAL:-}"
SHOW_CAMERA_INPUT="${SHOW_CAMERA_INPUT:-1}"
STOP_ON_CLOSE="${STOP_ON_CLOSE:-1}"
ROBOT_INIT_SCRIPT="${ROBOT_INIT_SCRIPT:-${SCRIPT_DIR}/real/robot_init.py}"

[[ $# -eq 0 ]] || die "this launcher takes no positional arguments; use environment variables"
[[ -n "${DP_CKPT_PATH}" ]] || die "set DP_CKPT_PATH to the real-task Diffusion Policy checkpoint"
[[ -n "${MODEL_PYTHON}" && -x "${MODEL_PYTHON}" ]] || die "set MODEL_PYTHON to the Diffusion Policy Python executable"
[[ -f "${INFERENCE_SCRIPT}" ]] || die "inference entrypoint not found: ${INFERENCE_SCRIPT}"
[[ -f "${DP_CKPT_PATH}" ]] || die "real DP checkpoint not found: ${DP_CKPT_PATH}"
[[ -f "${CONTROLLER_CKPT_PATH}" ]] || die "DDIM controller checkpoint not found: ${CONTROLLER_CKPT_PATH}"
[[ -f "${DINOV2_REPO_OR_DIR}/hubconf.py" ]] || die "DINOv2 repository not found: ${DINOV2_REPO_OR_DIR}"
[[ -f "${DINOV2_WEIGHTS}" ]] || die "DINOv2 weights not found: ${DINOV2_WEIGHTS}"
[[ -f "${GUIDE_CKPT_PATH}" ]] || die "weak guide checkpoint not found: ${GUIDE_CKPT_PATH}"
[[ "${INIT_POSE}" == "ZERO" || "${INIT_POSE}" == "ROTATE" ]] || die "INIT_POSE must be ZERO or ROTATE"
[[ "${WEAK_GUIDANCE_STEPS}" =~ ^[1-9]$ ]] || die "WEAK_GUIDANCE_STEPS must be in 1..9"
[[ "${WEAK_GUIDANCE_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "WEAK_GUIDANCE_SCALE must be non-negative"
[[ "${GUIDE_INFERENCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "GUIDE_INFERENCE_STEPS must be positive"
[[ "${GUIDE_SEED}" =~ ^[0-9]+$ ]] || die "GUIDE_SEED must be non-negative integer"
[[ "${CHECK_ONLY}" =~ ^[01]$ ]] || die "CHECK_ONLY must be 0 or 1"
[[ "${LIVE}" =~ ^[01]$ ]] || die "LIVE must be 0 or 1"
[[ "${FIXED_NOISE}" =~ ^[01]$ ]] || die "FIXED_NOISE must be 0 or 1"
[[ "${ALLOW_SALVAGE}" =~ ^[01]$ ]] || die "ALLOW_SALVAGE must be 0 or 1"
[[ "${SHOW_CAMERA_INPUT}" =~ ^[01]$ ]] || die "SHOW_CAMERA_INPUT must be 0 or 1"
[[ "${STOP_ON_CLOSE}" =~ ^[01]$ ]] || die "STOP_ON_CLOSE must be 0 or 1"
[[ "${CONTROLLER_ACTION_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "CONTROLLER_ACTION_CHUNK_SIZE must be a positive integer"
[[ "${GUIDANCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "GUIDANCE_STEPS must be a positive integer"
[[ "${CONTROLLER_CALLS_PER_DP}" =~ ^[1-9][0-9]*$ ]] || die "CONTROLLER_CALLS_PER_DP must be a positive integer"
if [[ "${CHECK_ONLY}" == "0" && "${LIVE}" != "1" ]]; then
    die "hardware evaluation requires LIVE=1 (or leave CHECK_ONLY=1 for a safe model check)"
fi

DP_PREFIX="$(cd -- "$(dirname -- "${MODEL_PYTHON}")/.." && pwd)"
NVJITLINK_LIB="${DP_PREFIX}/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
RUNTIME_LD_LIBRARY_PATH="${DP_PREFIX}/lib"
if [[ -d "${NVJITLINK_LIB}" ]]; then
    RUNTIME_LD_LIBRARY_PATH="${NVJITLINK_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    RUNTIME_LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
fi
env LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
    "${MODEL_PYTHON}" -c 'import diffusers, dill, hydra, torch' >/dev/null 2>&1 || \
    die "MODEL_PYTHON is missing a required package (torch, diffusers, hydra, or dill)"
if [[ "${CHECK_ONLY}" == "0" ]]; then
    env LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
        "${MODEL_PYTHON}" -c 'import cv2, pyrealsense2' >/dev/null 2>&1 || \
        die "hardware evaluation requires cv2 and pyrealsense2 in MODEL_PYTHON"
fi

ARGS=(
    "${INFERENCE_SCRIPT}"
    --dp-checkpoint "${DP_CKPT_PATH}"
    --controller-checkpoint "${CONTROLLER_CKPT_PATH}"
    --weak-guide-checkpoint "${GUIDE_CKPT_PATH}"
    --weak-guide-scale "${WEAK_GUIDANCE_SCALE}"
    --weak-guide-steps "${WEAK_GUIDANCE_STEPS}"
    --weak-guide-inference-steps "${GUIDE_INFERENCE_STEPS}"
    --weak-guide-seed "${GUIDE_SEED}"
    --device "${DEVICE}"
    --dp-inference-steps "${DP_INFERENCE_STEPS}"
    --ddim-inference-steps "${DDIM_INFERENCE_STEPS}"
    --controller-action-chunk-size "${CONTROLLER_ACTION_CHUNK_SIZE}"
    --guidance-steps "${GUIDANCE_STEPS}"
    --controller-calls-per-dp "${CONTROLLER_CALLS_PER_DP}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --eta "${ETA}"
    --fixed-noise "${FIXED_NOISE}"
    --seed "${SEED}"
    --max-chunks "${MAX_CHUNKS}"
    --hz "${HZ}"
)
if [[ "${ALLOW_SALVAGE}" == "0" ]]; then
    ARGS+=(--no-salvage)
fi
if [[ "${CHECK_ONLY}" == "1" ]]; then
    ARGS+=(--check-only)
else
    ARGS+=(
        --live
        --franka-host "${FRANKA_HOST}"
        --franka-port "${FRANKA_PORT}"
        --franka-control-mode joints
        --franka-urdf "${FRANKA_URDF}"
        --hand-host "${HAND_HOST}"
        --hand-port "${HAND_PORT}"
    )
    if [[ -n "${FRONT_SERIAL}" ]]; then
        ARGS+=(--front-serial "${FRONT_SERIAL}")
    fi
    if [[ -n "${WRIST_SERIAL}" ]]; then
        ARGS+=(--wrist-serial "${WRIST_SERIAL}")
    fi
    if [[ "${SHOW_CAMERA_INPUT}" == "1" ]]; then
        ARGS+=(--show-camera-input)
    fi
    if [[ "${STOP_ON_CLOSE}" == "1" ]]; then
        ARGS+=(--stop-on-close)
    fi
fi

echo "[pipeline] Real DP -> ${CONTROLLER_CALLS_PER_DP} x (strong DDIM: DP ${GUIDANCE_STEPS}-step + weak ${WEAK_GUIDANCE_STEPS}-step guidance -> execute ${CONTROLLER_ACTION_CHUNK_SIZE} -> observe) -> replan DP"
echo "[weak-guide] ${GUIDE_CKPT_PATH} scale=${WEAK_GUIDANCE_SCALE} actions=${WEAK_GUIDANCE_STEPS} ddim_steps=${GUIDE_INFERENCE_STEPS}"
echo "[dp] ${DP_CKPT_PATH} (steps=${DP_INFERENCE_STEPS})"
echo "[controller] ${CONTROLLER_CKPT_PATH} (DDIM steps=${DDIM_INFERENCE_STEPS})"
echo "[guidance] scale=${GUIDANCE_SCALE} eta=${ETA} ddim_steps=${DDIM_INFERENCE_STEPS} guide_steps=${GUIDANCE_STEPS} exec=${CONTROLLER_ACTION_CHUNK_SIZE} fixed_noise=${FIXED_NOISE}"

if [[ "${CHECK_ONLY}" == "0" ]]; then
    [[ -f "${FRANKA_URDF}" ]] || die "Franka URDF not found: ${FRANKA_URDF}"
    # Reset first; the inference entrypoint loads all three models afterward.
    "${MODEL_PYTHON}" - "${FRANKA_HOST}" "${FRANKA_PORT}" "${HAND_HOST}" "${HAND_PORT}" <<'PY'
import socket
import sys

failures = []
for label, host, port in (
    ("Franka", sys.argv[1], int(sys.argv[2])),
    ("SharpA", sys.argv[3], int(sys.argv[4])),
):
    try:
        with socket.create_connection((host, port), timeout=2.0):
            print(f"[preflight] {label} ready at {host}:{port}")
    except OSError as exc:
        failures.append(f"{label} {host}:{port}: {exc}")
if failures:
    raise SystemExit("hardware preflight failed; no command sent:\n  " + "\n  ".join(failures))
PY
    [[ -f "${ROBOT_INIT_SCRIPT}" ]] || die "ROBOT_INIT_SCRIPT not found: ${ROBOT_INIT_SCRIPT}"
    echo "[init] SharpA target: HAND_READY_JOINTS in ${ROBOT_INIT_SCRIPT}"
    env \
        PYTHONDONTWRITEBYTECODE=1 \
        LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
        "${MODEL_PYTHON}" "${ROBOT_INIT_SCRIPT}" \
        --init-pose "${INIT_POSE}" \
        --franka-host "${FRANKA_HOST}" \
        --franka-port "${FRANKA_PORT}" \
        --hand-host "${HAND_HOST}" \
        --hand-port "${HAND_PORT}"
fi

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${SCRIPT_DIR}:${DEX_ROOT}:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
    DINOV2_REPO_OR_DIR="${DINOV2_REPO_OR_DIR}" \
    DINOV2_WEIGHTS="${DINOV2_WEIGHTS}" \
    DINOV2_SOURCE="${DINOV2_SOURCE}" \
    "${MODEL_PYTHON}" -u "${ARGS[@]}"
