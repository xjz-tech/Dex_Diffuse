#!/usr/bin/env bash
# Diffusion Policy inference on the real Franka + SharpA setup.
# The EE action mode (absolute/relative) is read from the checkpoint config.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DP_CONDA_PREFIX="${DP_CONDA_PREFIX:-/home/frankagvl/anaconda3/envs/dp_w}"
PYTHON="${PYTHON:-${DP_CONDA_PREFIX}/bin/python}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-${SCRIPT_DIR}/inference_dp.py}"
ROBOT_INIT_SCRIPT="${ROBOT_INIT_SCRIPT:-/home/frankagvl/workspace/dex_setup/TacMP/scripts/robot_init.py}"

# Select the checkpoint here. Keep the directory and epoch separate; the loss
# suffix is discovered automatically from the checkpoint filename.
CHECKPOINT_DIR="/media/frankagvl/U393/diffusion_policy/data/outputs/2026.09.03/23.53.01_train_diffusion_unet_dino_image_bulb_image/checkpoints"
CKPT_EPOCH="200"

if [[ ! "${CKPT_EPOCH}" =~ ^[0-9]+$ ]]; then
  echo "CKPT_EPOCH must be a non-negative integer, got: ${CKPT_EPOCH}" >&2
  exit 2
fi
printf -v CKPT_EPOCH_PADDED '%04d' "$((10#${CKPT_EPOCH}))"

shopt -s nullglob
checkpoint_candidates=("${CHECKPOINT_DIR}/epoch=${CKPT_EPOCH_PADDED}-"*.ckpt)
shopt -u nullglob
if [[ ${#checkpoint_candidates[@]} -ne 1 ]]; then
  echo "Expected exactly one checkpoint for epoch ${CKPT_EPOCH_PADDED} in ${CHECKPOINT_DIR}; found ${#checkpoint_candidates[@]}" >&2
  exit 1
fi
CKPT_PATH="${checkpoint_candidates[0]}"
# Use the same local DINOv2 assets as the training launchers.
DINOV2_REPO_OR_DIR="${DINOV2_REPO_OR_DIR:-${SCRIPT_DIR}/assets/dinov2_assets/facebookresearch_dinov2_main}"
DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-${SCRIPT_DIR}/assets/dinov2_assets/dinov2_vits14_pretrain.pth}"
DINOV2_SOURCE="${DINOV2_SOURCE:-local}"
DEVICE="${DEVICE:-cuda:0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
ACTION_CHUNK_STEPS="${ACTION_CHUNK_STEPS:-8}"
FRANKA_URDF="${FRANKA_URDF:-/home/frankagvl/workspace/dex_setup/simtoolreal2teleop/assets/urdf/fr3_sharpa_description/fr3.urdf}"
FRANKA_HOST="${FRANKA_HOST:-172.16.0.10}"
FRANKA_PORT="${FRANKA_PORT:-9090}"
HAND_HOST="${HAND_HOST:-localhost}"
HAND_PORT="${HAND_PORT:-5570}"
# Leave serials empty by default and assign camera roles by model:
# D435/D455 -> front, D405 -> wrist. Set either variable to override.
FRONT_SERIAL="${FRONT_SERIAL:-}"
WRIST_SERIAL="${WRIST_SERIAL:-}"
CHECK_ONLY="${CHECK_ONLY:-0}"

# Use the project code and local DINOv2 assets, even when the checkpoint was
# produced on another machine and contains that machine's absolute paths.
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export DINOV2_REPO_OR_DIR DINOV2_WEIGHTS DINOV2_SOURCE
# Put dp_w first so Pillow/OpenCV use its matching libstdc++; its CUDA 12.1
# runtime also takes precedence over globally configured CUDA/ROS libraries.
export LD_LIBRARY_PATH="${DP_CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ $# -ne 0 ]]; then
  echo "Usage: $(basename "$0")" >&2
  exit 2
fi
[[ -x "${PYTHON}" ]] || { echo "Python not found: ${PYTHON}" >&2; exit 1; }
[[ -f "${INFERENCE_SCRIPT}" ]] || { echo "Inference entrypoint not found: ${INFERENCE_SCRIPT}" >&2; exit 1; }
[[ -f "${CKPT_PATH}" ]] || { echo "Checkpoint not found: ${CKPT_PATH}" >&2; exit 1; }
[[ -f "${DINOV2_REPO_OR_DIR}/hubconf.py" ]] || {
  echo "DINOv2 torch.hub repository not found: ${DINOV2_REPO_OR_DIR}" >&2
  exit 1
}
[[ -f "${DINOV2_WEIGHTS}" ]] || { echo "DINOv2 weights not found: ${DINOV2_WEIGHTS}" >&2; exit 1; }
[[ "${CHECK_ONLY}" == 0 || "${CHECK_ONLY}" == 1 ]] || {
  echo "CHECK_ONLY must be 0 or 1, got: ${CHECK_ONLY}" >&2
  exit 2
}

echo "[python] ${PYTHON}"
echo "[policy] ${CKPT_PATH}"
echo "[vision] DINOv2 ViT-S/14 from ${DINOV2_REPO_OR_DIR} (${DINOV2_SOURCE}); weights=${DINOV2_WEIGHTS}"
if [[ "${CHECK_ONLY}" == 1 ]]; then
  echo "[mode] CHECK ONLY: load checkpoint and run one synthetic inference; hardware is not connected"
  exec "${PYTHON}" "${INFERENCE_SCRIPT}" \
    --checkpoint "${CKPT_PATH}" \
    --device "${DEVICE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --action_chunk_steps "${ACTION_CHUNK_STEPS}" \
    --check_only
fi

[[ -f "${FRANKA_URDF}" ]] || { echo "Franka URDF not found: ${FRANKA_URDF}" >&2; exit 1; }
"${PYTHON}" - "${FRANKA_HOST}" "${FRANKA_PORT}" "${HAND_HOST}" "${HAND_PORT}" <<'PY'
import socket
import sys

endpoints = (
    ("Franka", sys.argv[1], int(sys.argv[2])),
    ("SharpA", sys.argv[3], int(sys.argv[4])),
)
failed = []
for label, host, port in endpoints:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"[preflight] {label} ready at {host}:{port}")
    except OSError as exc:
        failed.append(f"{label} {host}:{port}: {exc}")
if failed:
    raise SystemExit(
        "Hardware preflight failed; no robot command was sent:\n  "
        + "\n  ".join(failed)
    )
PY
[[ -f "${ROBOT_INIT_SCRIPT}" ]] || { echo "Robot init script not found: ${ROBOT_INIT_SCRIPT}" >&2; exit 1; }
"${PYTHON}" "${ROBOT_INIT_SCRIPT}"
echo "[hardware] Franka ${FRANKA_HOST}:${FRANKA_PORT} (joints server); SharpA ${HAND_HOST}:${HAND_PORT}"
echo "[cameras] front=${FRONT_SERIAL:-auto}; wrist=${WRIST_SERIAL:-auto}"
echo "[mode] LIVE, no tactile: serial ${ACTION_CHUNK_STEPS}-step chunks at 30 Hz"
echo "[schedule] infer -> execute all ${ACTION_CHUNK_STEPS} actions -> observe -> infer"

exec "${PYTHON}" "${INFERENCE_SCRIPT}" \
  --checkpoint "${CKPT_PATH}" \
  --device "${DEVICE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --action_chunk_steps "${ACTION_CHUNK_STEPS}" \
  --serial_chunks \
  --hz 30 \
  --live \
  --franka_host "${FRANKA_HOST}" \
  --franka_port "${FRANKA_PORT}" \
  --franka_control_mode joints \
  --franka_urdf "${FRANKA_URDF}" \
  --hand_host "${HAND_HOST}" \
  --hand_port "${HAND_PORT}" \
  --front_serial "${FRONT_SERIAL}" \
  --wrist_serial "${WRIST_SERIAL}" \
  --show_camera_input \
  --stop_on_close
