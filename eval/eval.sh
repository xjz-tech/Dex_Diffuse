#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONTROLLER_ROOT="${CONTROLLER_ROOT:-/home/carus/Program/dex-controller}"
DATA_ROOT="${DATA_ROOT:-${CONTROLLER_ROOT}/data}"
ASSETS_ROOT="${ASSETS_ROOT:-${CONTROLLER_ROOT}/maniptrans_envs/assets}"
SIM_DATASET="${SIM_DATASET:-/home/carus/Data/exp_data}"
SIM_CONFIG="${SIM_CONFIG:-${SIM_DATASET}/hydra_config.yaml}"
CKPT_PATH="${CKPT_PATH:-${DEX_ROOT}/runs/step_01700000.ckpt}"

# Isaac Gym is CPython 3.8-only here, while the trained DP stack is Python 3.10.
# The two processes communicate through a private UNIX socket.
if [[ -z "${SIM_PYTHON:-}" ]]; then
    for candidate in \
        /home/carus/miniforge3/envs/decv2/bin/python \
        /home/wty/miniconda3/envs/dec_sapg/bin/python; do
        if [[ -x "${candidate}" ]]; then
            SIM_PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ -z "${MODEL_PYTHON:-}" ]]; then
    for candidate in \
        /home/carus/miniforge3/envs/dp/bin/python \
        /home/wty/miniconda3/envs/rdp/bin/python; do
        if [[ -x "${candidate}" ]]; then
            MODEL_PYTHON="${candidate}"
            break
        fi
    done
fi
SIM_PYTHON="${SIM_PYTHON:-}"
MODEL_PYTHON="${MODEL_PYTHON:-}"

if [[ -z "${ISAACGYM_PYTHON:-}" ]]; then
    if [[ -d "${CONTROLLER_ROOT}/third_party/isaacgym/python" ]]; then
        ISAACGYM_PYTHON="${CONTROLLER_ROOT}/third_party/isaacgym/python"
    else
        ISAACGYM_PYTHON="/home/carus/opt/isaacgym/python"
    fi
fi

NOKOV3_DATA_DIR="${NOKOV3_DATA_DIR:-${DATA_ROOT}/NOKOV-v3}"
NOKOV3_RETARGET_DIR="${NOKOV3_RETARGET_DIR:-${DATA_ROOT}/retargeting/NOKOV-v3}"
SHARPA_ASSET_DIR="${SHARPA_ASSET_DIR:-${ASSETS_ROOT}/sharpa_hand}"

# for data_idx in {000..149}; do
#     DATA_INDICES+=("${data_idx}")
# done
DATA_INDICES="${DATA_INDICES:-000,001,002,003,004,005,006,007}"
# NUM_ENV is the public setting. Keep NUM_ENVS as a compatibility fallback for
# older commands, but do not let it override an explicitly supplied NUM_ENV.
NUM_ENV="${NUM_ENV:-${NUM_ENVS:-8}}"
RECORD_ENV="${RECORD_ENV:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
PRINT_EVERY="${PRINT_EVERY:-25}"
HEADLESS="${HEADLESS:-0}"
RANDOMIZE_DEMO_ON_FAILURE="${RANDOMIZE_DEMO_ON_FAILURE:-1}"

# Episode reset conditions (all pose errors are relative to the current target).
# Position/tip/rotation failures accumulate bad frames; regular targets allow
# FAILURE_TOLERANCE_SCALE * abs(skipSteps), while cross targets use the fixed
# tolerance. INVALID_OBJ_POS_THRES_M bypasses that tolerance and resets at once.
# TRAJ_STEPS_LIMIT is a full-trajectory success reset. RESET_ON_REACH_GOAL=1
# additionally resets after every stable target reach; 0 only advances target.
FAILURE_OBJ_POS_THRES_M="${FAILURE_OBJ_POS_THRES_M:-0.012}"
FAILURE_TIP_POS_THRES_M="${FAILURE_TIP_POS_THRES_M:-0.036}"
FAILURE_OBJ_ROT_THRES_DEG="${FAILURE_OBJ_ROT_THRES_DEG:-180.0}"
INVALID_OBJ_POS_THRES_M="${INVALID_OBJ_POS_THRES_M:-0.15}"
FAILURE_TOLERANCE_SCALE="${FAILURE_TOLERANCE_SCALE:-1.0}"
FIXED_TOLERANCE_STEPS="${FIXED_TOLERANCE_STEPS:-200}"
TRAJ_STEPS_LIMIT="${TRAJ_STEPS_LIMIT:-12000}"
RESET_ON_REACH_GOAL="${RESET_ON_REACH_GOAL:-0}"

RECORDING="${RECORDING:-0}"
RECORD_DIR="${RECORD_DIR:-${SCRIPT_DIR}/record}"
RECORD_WIDTH="${RECORD_WIDTH:-1280}"
RECORD_HEIGHT="${RECORD_HEIGHT:-720}"
RECORD_FPS="${RECORD_FPS:-30}"
RECORD_CAMERA_POSITION="${RECORD_CAMERA_POSITION:--0.10,0.55,0.10}"
RECORD_CAMERA_TARGET="${RECORD_CAMERA_TARGET:--0.10,0.00,-0.14}"
RECORD_CAMERA_FOV="${RECORD_CAMERA_FOV:-60.0}"
RECORD_AXIS_LENGTH="${RECORD_AXIS_LENGTH:-0.20}"
RECORD_AXIS_THICKNESS="${RECORD_AXIS_THICKNESS:-0.008}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:0}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
GRAPHICS_DEVICE_ID="${GRAPHICS_DEVICE_ID:-0}"
SEED="${SEED:-42}"
SAMPLER="${SAMPLER:-ddpm}"
INFERENCE_STEPS="${INFERENCE_STEPS:-}"
ALLOW_SALVAGE="${ALLOW_SALVAGE:-1}"
MODEL_WARMUP="${MODEL_WARMUP:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
MAX_FAILURE_EPISODES="${MAX_FAILURE_EPISODES:-0}"
EPISODE_LOG="${EPISODE_LOG:-}"
RUN_NAME="${RUN_NAME:-}"

die() {
    echo "eval: $*" >&2
    exit 2
}

[[ -x "${SIM_PYTHON}" ]] || die "Isaac Gym Python not executable: ${SIM_PYTHON}"
[[ -x "${MODEL_PYTHON}" ]] || die "Diffusion Policy Python not executable: ${MODEL_PYTHON}"
[[ -d "${ISAACGYM_PYTHON}" ]] || die "Isaac Gym python package not found: ${ISAACGYM_PYTHON}"
[[ -d "${CONTROLLER_ROOT}" ]] || die "dex-controller root not found: ${CONTROLLER_ROOT}"
[[ -f "${SIM_CONFIG}" ]] || die "simulation config not found: ${SIM_CONFIG}"
[[ -f "${CKPT_PATH}" ]] || die "checkpoint not found: ${CKPT_PATH}"
[[ -d "${NOKOV3_DATA_DIR}/data/bulb2" ]] || die "bulb2 source data not found under ${NOKOV3_DATA_DIR}"
[[ -d "${NOKOV3_RETARGET_DIR}/mano2sharpa_rh/bulb2" ]] || die "bulb2 retarget data not found under ${NOKOV3_RETARGET_DIR}"
[[ -f "${SHARPA_ASSET_DIR}/v3right_sharpa_wave-forhammer5.urdf" ]] || die "SharpA URDF not found under ${SHARPA_ASSET_DIR}"
[[ "${STARTUP_TIMEOUT}" =~ ^[0-9]+$ ]] || die "STARTUP_TIMEOUT must be an integer"
[[ "${FIXED_TOLERANCE_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "FIXED_TOLERANCE_STEPS must be a positive integer"
[[ "${TRAJ_STEPS_LIMIT}" =~ ^[1-9][0-9]*$ ]] || die "TRAJ_STEPS_LIMIT must be a positive integer"
[[ "${RESET_ON_REACH_GOAL}" =~ ^[01]$ ]] || die "RESET_ON_REACH_GOAL must be 0 or 1"
[[ "${MAX_FAILURE_EPISODES}" =~ ^[0-9]+$ ]] || die "MAX_FAILURE_EPISODES must be a non-negative integer"
[[ "${SAMPLER}" == "ddpm" || "${SAMPLER}" == "ddim" ]] || die "SAMPLER must be ddpm or ddim"
[[ "${NUM_ENV}" =~ ^[1-9][0-9]*$ ]] || die "NUM_ENV must be a positive integer"
[[ "${RECORD_ENV}" =~ ^(0|[1-9][0-9]*)$ ]] || die "RECORD_ENV must be a non-negative integer"
(( RECORD_ENV < NUM_ENV )) || die "RECORD_ENV must be in [0, NUM_ENV), got RECORD_ENV=${RECORD_ENV} NUM_ENV=${NUM_ENV}"

RUNTIME_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dex-diffuse-eval.XXXXXX")"
SOCKET_PATH="${RUNTIME_DIR}/policy.sock"
MODEL_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${MODEL_PID}" ]] && kill -0 "${MODEL_PID}" 2>/dev/null; then
        kill "${MODEL_PID}" 2>/dev/null || true
        wait "${MODEL_PID}" 2>/dev/null || true
    fi
    rm -f -- "${SOCKET_PATH}"
    rmdir -- "${RUNTIME_DIR}" 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

MODEL_ARGS=(
    "${SCRIPT_DIR}/model_server.py"
    --checkpoint "${CKPT_PATH}"
    --socket "${SOCKET_PATH}"
    --device "${MODEL_DEVICE}"
    --seed "${SEED}"
    --sampler "${SAMPLER}"
)
if [[ -n "${INFERENCE_STEPS}" ]]; then
    MODEL_ARGS+=(--inference-steps "${INFERENCE_STEPS}")
fi
if [[ -n "${EXECUTION_STEPS:-}" ]]; then
    [[ "${EXECUTION_STEPS}" =~ ^[1-9]$ ]] || die "EXECUTION_STEPS must be an integer in 1..9"
    MODEL_ARGS+=(--n-action-steps "${EXECUTION_STEPS}")
fi
if [[ "${ALLOW_SALVAGE}" == "0" ]]; then
    MODEL_ARGS+=(--no-salvage)
fi
if [[ "${MODEL_WARMUP}" == "0" ]]; then
    MODEL_ARGS+=(--no-warmup)
fi

echo "[eval] checkpoint: ${CKPT_PATH}"
echo "[eval] sampler: ${SAMPLER} inference_steps=${INFERENCE_STEPS:-checkpoint} execution_steps=${EXECUTION_STEPS:-checkpoint}"
echo "[eval] GPU visibility: ${CUDA_VISIBLE_DEVICES}"
echo "[eval] starting Diffusion Policy process (${MODEL_PYTHON})"
env \
    PATH="$(dirname "${MODEL_PYTHON}"):${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${SCRIPT_DIR}:${DEX_ROOT}:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$(dirname "$(dirname "${MODEL_PYTHON}")")/lib:${LD_LIBRARY_PATH:-}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "${MODEL_PYTHON}" -u "${MODEL_ARGS[@]}" &
MODEL_PID=$!

READY=0
for ((attempt = 0; attempt < STARTUP_TIMEOUT * 4; attempt++)); do
    if [[ -S "${SOCKET_PATH}" ]]; then
        READY=1
        break
    fi
    if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
        wait "${MODEL_PID}" || true
        die "Diffusion Policy process exited before becoming ready"
    fi
    sleep 0.25
done
[[ "${READY}" == "1" ]] || die "Diffusion Policy startup timed out after ${STARTUP_TIMEOUT}s"

SIM_ARGS=(
    "${SCRIPT_DIR}/sim_eval.py"
    --controller-root "${CONTROLLER_ROOT}"
    --sim-config "${SIM_CONFIG}"
    --socket "${SOCKET_PATH}"
    --num-envs "${NUM_ENV}"
    --record-env "${RECORD_ENV}"
    --data-indices "${DATA_INDICES}"
    --nokov3-data-dir "${NOKOV3_DATA_DIR}"
    --nokov3-retarget-dir "${NOKOV3_RETARGET_DIR}"
    --sharpa-asset-dir "${SHARPA_ASSET_DIR}"
    --sim-device "${SIM_DEVICE}"
    --rl-device "${RL_DEVICE}"
    --graphics-device-id "${GRAPHICS_DEVICE_ID}"
    --seed "${SEED}"
    --max-steps "${MAX_STEPS}"
    --print-every "${PRINT_EVERY}"
    --request-timeout "${REQUEST_TIMEOUT}"
    --failure-obj-pos-thres-m "${FAILURE_OBJ_POS_THRES_M}"
    --failure-tip-pos-thres-m "${FAILURE_TIP_POS_THRES_M}"
    --failure-obj-rot-thres-deg "${FAILURE_OBJ_ROT_THRES_DEG}"
    --invalid-obj-pos-thres-m "${INVALID_OBJ_POS_THRES_M}"
    --failure-tolerance-scale "${FAILURE_TOLERANCE_SCALE}"
    --fixed-tolerance-steps "${FIXED_TOLERANCE_STEPS}"
    --traj-steps-limit "${TRAJ_STEPS_LIMIT}"
    --reset-on-reach-goal "${RESET_ON_REACH_GOAL}"
    --max-failure-episodes "${MAX_FAILURE_EPISODES}"
)
if [[ -n "${EPISODE_LOG}" ]]; then
    SIM_ARGS+=(--episode-log "${EPISODE_LOG}")
fi
if [[ -n "${RUN_NAME}" ]]; then
    SIM_ARGS+=(--run-name "${RUN_NAME}")
fi
if [[ -n "${CROSS_TRAJECTORY_GOAL_PROB:-}" ]]; then
    SIM_ARGS+=(--cross-trajectory-goal-prob "${CROSS_TRAJECTORY_GOAL_PROB}")
fi
if [[ "${HEADLESS}" != "0" ]]; then
    SIM_ARGS+=(--headless)
fi
if [[ "${RANDOMIZE_DEMO_ON_FAILURE}" == "0" ]]; then
    SIM_ARGS+=(--no-randomize-demo-on-failure)
fi
if [[ "${RECORDING}" != "0" ]]; then
    SIM_ARGS+=(
        --recording
        --record-dir "${RECORD_DIR}"
        --record-width "${RECORD_WIDTH}"
        --record-height "${RECORD_HEIGHT}"
        --record-fps "${RECORD_FPS}"
        "--record-camera-position=${RECORD_CAMERA_POSITION}"
        "--record-camera-target=${RECORD_CAMERA_TARGET}"
        --record-camera-fov "${RECORD_CAMERA_FOV}"
        --record-axis-length "${RECORD_AXIS_LENGTH}"
        --record-axis-thickness "${RECORD_AXIS_THICKNESS}"
    )
fi

echo "[eval] starting Isaac Gym process (${SIM_PYTHON})"
echo "[eval] trajectories=${DATA_INDICES} num_envs=${NUM_ENV} record_env=${RECORD_ENV} headless=${HEADLESS}"
echo "[eval] reset failure_obj_pos_m=${FAILURE_OBJ_POS_THRES_M} failure_tip_pos_m=${FAILURE_TIP_POS_THRES_M} failure_obj_rot_deg=${FAILURE_OBJ_ROT_THRES_DEG} invalid_obj_pos_m=${INVALID_OBJ_POS_THRES_M} tolerance_scale=${FAILURE_TOLERANCE_SCALE} fixed_tolerance_steps=${FIXED_TOLERANCE_STEPS} traj_steps_limit=${TRAJ_STEPS_LIMIT} reset_on_reach_goal=${RESET_ON_REACH_GOAL}"
SIM_STATUS=0
if env \
    PATH="$(dirname "${SIM_PYTHON}"):${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${ISAACGYM_PYTHON}:${CONTROLLER_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
    LD_LIBRARY_PATH="$(dirname "$(dirname "${SIM_PYTHON}")")/lib:${LD_LIBRARY_PATH:-}" \
    "${SIM_PYTHON}" -u "${SIM_ARGS[@]}"; then
    SIM_STATUS=0
else
    SIM_STATUS=$?
fi

if [[ "${SIM_STATUS}" == "130" ]]; then
    echo "[eval] stopped by Ctrl-C"
fi
exit "${SIM_STATUS}"
