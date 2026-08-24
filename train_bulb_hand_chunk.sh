#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-/home/wangtianyu/miniconda3/envs/dp/bin/python}"
DATASET_PATH="${DATASET_PATH:-/path/to/sim_bulb_hand_dp}"
NORMALIZER_ARGS=()
if [[ -n "${BULB_HAND_NORMALIZER:-}" ]]; then
  if [[ ! -f "$BULB_HAND_NORMALIZER" ]]; then
    echo "BULB_HAND_NORMALIZER file not found: $BULB_HAND_NORMALIZER" >&2
    exit 1
  fi
  NORMALIZER_ARGS+=(task.normalizer_path="$BULB_HAND_NORMALIZER")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec "$PYTHON" train.py \
    --config-name=train_diffusion_unet_bulb_hand_chunk_workspace \
    task=bulb_hand \
    task.dataset_path="$DATASET_PATH" \
    "${NORMALIZER_ARGS[@]}" \
    training.device=cuda:0 \
    logging.project=tacmp_diffusion_policy \
    logging.name=bulb_hand_chunk \
    "$@"
