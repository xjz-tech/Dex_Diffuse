#!/usr/bin/env bash
# Train Diffusion Policy with a frozen DINOv2 ViT-S/14 encoder on bulb_0705_dp.
# Action layout: relative EE pose (xyz + rotation-6D) + absolute hand joints.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/wangtianyu/miniconda3/envs/dp/bin/python}"
DATASET_PATH="${DATASET_PATH:-/home/wangtianyu/workspace/TacMP/data/bulb_tac_80_dp}"
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
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/data/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/data/matplotlib}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter is not executable: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$DATASET_PATH/replay_buffer.zarr" ]]; then
    echo "Dataset not found: $DATASET_PATH/replay_buffer.zarr" >&2
    echo "Run data2dp.py first." >&2
    exit 1
fi
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

exec "$PYTHON" train.py \
    --config-name=train_diffusion_unet_dino_image_workspace \
    task=bulb_image \
    task.dataset_path="$DATASET_PATH" \
    "${NORMALIZER_ARGS[@]}" \
    training.device=cuda:0 \
    training.num_epochs=200 \
    training.rollout_every=1000000 \
    dataloader.batch_size=256 \
    dataloader.num_workers=16 \
    val_dataloader.batch_size=256 \
    val_dataloader.num_workers=16 \
    logging.project=tacmp_diffusion_policy \
    logging.mode="$WANDB_MODE" \
    logging.name=bulb_tac_80_relative_ee_absolute_hand \
    checkpoint.topk.monitor_key=val_loss \
    checkpoint.topk.mode=min \
    "checkpoint.topk.format_str='epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt'" \
    "$@"
