#!/usr/bin/env bash
# Diffusion Policy inference on the real Franka + SharpA setup.
# Action layout: relative EE pose (SE(3)) + absolute hand joints.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DP_CONDA_PREFIX="${DP_CONDA_PREFIX:-/home/frankagvl/anaconda3/envs/dp_w}"
PYTHON="${PYTHON:-${DP_CONDA_PREFIX}/bin/python}"
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-${SCRIPT_DIR}/inference_dp.py}"
ROBOT_INIT_SCRIPT="${ROBOT_INIT_SCRIPT:-${SCRIPT_DIR}/../../scripts/robot_init.py}"

CKPT_PATH="${CKPT_PATH:-/media/frankagvl/WD_wty/work/TacMP/third_party/diffusion_policy/data/outputs/2026.08.24/08.17.50_train_diffusion_unet_dino_image_bulb_image/checkpoints/epoch=0195-val_loss=0.2107.ckpt}"
CKPT_PATH="${CKPT_PATH:-/media/frankagvl/WD_wty/work/TacMP/third_party/diffusion_policy/data/outputs/2026.08.24/08.17.50_train_diffusion_unet_dino_image_bulb_image/checkpoints/epoch=0195-val_loss=0.2107.ckpt}"
# CKPT_PATH="${CKPT_PATH:-/media/frankagvl/WD_wty/work/TacMP/third_party/diffusion_policy/data/outputs/2026.08.24/08.30.59_train_diffusion_unet_dino_image_bulb_image/checkpoints/epoch=0100-val_loss=0.1289.ckpt}"
# CKPT_PATH="${CKPT_PATH:-/media/frankagvl/WD_wty/work/TacMP/third_party/diffusion_policy/data/outputs/2026.08.24/08.30.59_train_diffusion_unet_dino_image_bulb_image/checkpoints/epoch=0195-val_loss=0.2267.ckpt}"
DINOV2_REPO_OR_DIR="${DINOV2_REPO_OR_DIR:-/home/frankagvl/.cache/torch/hub/facebookresearch_dinov2_main}"
DINOV2_SOURCE="${DINOV2_SOURCE:-local}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-100}"
ACTION_CHUNK_STEPS="${ACTION_CHUNK_STEPS:-50}"
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

# This machine has CUDA/ROS library paths globally configured. Put the dp
# environment first so its Pillow/OpenCV dependencies use its newer libstdc++.
export LD_LIBRARY_PATH="${DP_CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DINOV2_REPO_OR_DIR DINOV2_SOURCE

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
[[ "${CHECK_ONLY}" == 0 || "${CHECK_ONLY}" == 1 ]] || {
  echo "CHECK_ONLY must be 0 or 1, got: ${CHECK_ONLY}" >&2
  exit 2
}

echo "[python] ${PYTHON}"
echo "[policy] ${CKPT_PATH}"
echo "[vision] DINOv2 ViT-S/14 from ${DINOV2_REPO_OR_DIR} (${DINOV2_SOURCE})"
if [[ "${CHECK_ONLY}" == 1 ]]; then
  echo "[mode] CHECK ONLY: load checkpoint and run one synthetic inference; hardware is not connected"
  exec "${PYTHON}" "${INFERENCE_SCRIPT}" \
    --checkpoint "${CKPT_PATH}" \
    --device cuda:0 \
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
  --device cuda:0 \
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
