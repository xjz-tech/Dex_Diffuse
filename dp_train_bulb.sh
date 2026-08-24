#!/usr/bin/env bash
# Train vanilla Diffusion Policy on bulb_0705_dp.
# Action layout: relative EE pose (xyz + rotation-6D) + absolute hand joints.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-/home/wangtianyu/miniconda3/envs/dp/bin/python}"
DATASET_PATH="${DATASET_PATH:-/home/wangtianyu/workspace/TacMP/data/bulb_tac_80_dp}"
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
    --config-name=train_diffusion_unet_image_workspace \
    task=bulb_image \
    task.dataset_path="$DATASET_PATH" \
    training.device=cuda:0 \
    training.num_epochs=1000 \
    training.rollout_every=1000000 \
    dataloader.batch_size=64 \
    dataloader.num_workers=8 \
    val_dataloader.batch_size=64 \
    val_dataloader.num_workers=8 \
    'policy.obs_encoder.resize_shape=[120,160]' \
    'policy.obs_encoder.crop_shape=[108,144]' \
    logging.project=tacmp_diffusion_policy \
    logging.mode="$WANDB_MODE" \
    logging.name=bulb_0717_relaee_abshand \
    checkpoint.topk.monitor_key=val_loss \
    checkpoint.topk.mode=min \
    "checkpoint.topk.format_str='epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt'" \
    "$@"
