#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONTROLLER_ROOT="${CONTROLLER_ROOT:-/mnt/work/dexIL/dex-controller}"
SIM_DATASET="${SIM_DATASET:-${CONTROLLER_ROOT}/data/sim_data/bulb2/20260827175545}"
SIM_CONFIG="${SIM_CONFIG:-${SIM_DATASET}/hydra_config.yaml}"
CKPT_PATH="${CKPT_PATH:-${DEX_ROOT}/runs/step_01700000.ckpt}"

# Isaac Gym is CPython 3.8-only here, while the trained DP stack is Python 3.10.
# The two processes communicate through a private UNIX socket.
SIM_PYTHON="${SIM_PYTHON:-/home/wty/miniconda3/envs/dec_sapg/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-/home/wty/miniconda3/envs/rdp/bin/python}"

NOKOV3_DATA_DIR="${NOKOV3_DATA_DIR:-/mnt/work/dex-controller_sapg/data/NOKOV-v3}"
NOKOV3_RETARGET_DIR="${NOKOV3_RETARGET_DIR:-/mnt/work/dex-controller_sapg/data/retargeting/NOKOV-v3}"
DATA_INDICES="${DATA_INDICES:-000,001}"
# NUM_ENV is the public setting. Keep NUM_ENVS as a compatibility fallback for
# older commands, but do not let it override an explicitly supplied NUM_ENV.
NUM_ENV="${NUM_ENV:-${NUM_ENVS:-6}}"
RECORD_ENV="${RECORD_ENV:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
PRINT_EVERY="${PRINT_EVERY:-25}"
HEADLESS="${HEADLESS:-0}"
RANDOMIZE_DEMO_ON_FAILURE="${RANDOMIZE_DEMO_ON_FAILURE:-1}"
RECORDING="${RECORDING:-1}"
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
INFERENCE_STEPS="${INFERENCE_STEPS:-}"
ALLOW_SALVAGE="${ALLOW_SALVAGE:-1}"
MODEL_WARMUP="${MODEL_WARMUP:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-180}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"

die() {
    echo "eval: $*" >&2
    exit 2
}

[[ -x "${SIM_PYTHON}" ]] || die "Isaac Gym Python not executable: ${SIM_PYTHON}"
[[ -x "${MODEL_PYTHON}" ]] || die "Diffusion Policy Python not executable: ${MODEL_PYTHON}"
[[ -d "${CONTROLLER_ROOT}" ]] || die "dex-controller root not found: ${CONTROLLER_ROOT}"
[[ -f "${SIM_CONFIG}" ]] || die "simulation config not found: ${SIM_CONFIG}"
[[ -f "${CKPT_PATH}" ]] || die "checkpoint not found: ${CKPT_PATH}"
[[ -d "${NOKOV3_DATA_DIR}/data/bulb2" ]] || die "bulb2 source data not found under ${NOKOV3_DATA_DIR}"
[[ -d "${NOKOV3_RETARGET_DIR}/mano2sharpa_rh/bulb2" ]] || die "bulb2 retarget data not found under ${NOKOV3_RETARGET_DIR}"
[[ "${STARTUP_TIMEOUT}" =~ ^[0-9]+$ ]] || die "STARTUP_TIMEOUT must be an integer"
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
)
if [[ -n "${INFERENCE_STEPS}" ]]; then
    MODEL_ARGS+=(--inference-steps "${INFERENCE_STEPS}")
fi
if [[ "${ALLOW_SALVAGE}" == "0" ]]; then
    MODEL_ARGS+=(--no-salvage)
fi
if [[ "${MODEL_WARMUP}" == "0" ]]; then
    MODEL_ARGS+=(--no-warmup)
fi

echo "[eval] checkpoint: ${CKPT_PATH}"
echo "[eval] GPU visibility: ${CUDA_VISIBLE_DEVICES}"
echo "[eval] starting Diffusion Policy process (${MODEL_PYTHON})"
env \
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
    --sim-device "${SIM_DEVICE}"
    --rl-device "${RL_DEVICE}"
    --graphics-device-id "${GRAPHICS_DEVICE_ID}"
    --seed "${SEED}"
    --max-steps "${MAX_STEPS}"
    --print-every "${PRINT_EVERY}"
    --request-timeout "${REQUEST_TIMEOUT}"
)
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
SIM_STATUS=0
if env \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="${CONTROLLER_ROOT}/third_party/isaacgym/python:${CONTROLLER_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
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
