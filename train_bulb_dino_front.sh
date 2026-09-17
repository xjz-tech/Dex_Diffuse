#!/usr/bin/env bash
# Train Diffusion Policy with front-camera images and a frozen DINOv2 ViT-S/14.
# RELATIVE=1: relative EE pose + absolute hand joints.
# RELATIVE=0: absolute EE pose + absolute hand joints.
# Usage: task_name=manipulate_bulb bash train_bulb_dino_front.sh
# Checkpoints are saved under $OUTPUT_DIR/checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
task_name="${task_name:-manipulate_bulb}"
OUTPUT_DIR="${SCRIPT_DIR}/runs/${task_name}"

PYTHON="${PYTHON:-/home/bighand/miniconda3/envs/dp/bin/python}"
DATASET_PATH="${DATASET_PATH:-/mnt/work/dexIL/Dex_Diffuse/data/real_data/realworld_bulb_sft_260909_dp}"
DINOV2_REPO_OR_DIR="$SCRIPT_DIR/assets/dinov2_assets/facebookresearch_dinov2_main"
DINOV2_WEIGHTS="$SCRIPT_DIR/assets/dinov2_assets/dinov2_vits14_pretrain.pth"
RELATIVE="${RELATIVE:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HORIZON="${HORIZON:-16}"
N_OBS_STEPS="${N_OBS_STEPS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/data/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/data/matplotlib}"
export DINOV2_SOURCE=local
export DINOV2_REPO_OR_DIR
export DINOV2_WEIGHTS
# The dp environment ships its matching CUDA 12.4 runtime. A system CUDA path
# here makes libcusparse load an incompatible libnvJitLink.
unset LD_LIBRARY_PATH

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter is not executable: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$DINOV2_REPO_OR_DIR/hubconf.py" ]]; then
    echo "Local DINOv2 source not found: $DINOV2_REPO_OR_DIR/hubconf.py" >&2
    exit 1
fi
if [[ ! -f "$DINOV2_WEIGHTS" ]]; then
    echo "Local DINOv2 weights not found: $DINOV2_WEIGHTS" >&2
    exit 1
fi
if [[ ! -d "$DATASET_PATH/replay_buffer.zarr" ]]; then
    echo "Dataset not found: $DATASET_PATH/replay_buffer.zarr" >&2
    echo "Run data2dp.py first." >&2
    exit 1
fi
case "$RELATIVE" in
    0)
        RELATIVE_BOOL=false
        ACTION_MODE=absolute_ee_absolute_hand
        ;;
    1)
        RELATIVE_BOOL=true
        ACTION_MODE=relative_ee_absolute_hand
        ;;
    *)
        echo "RELATIVE must be 0 or 1, got: $RELATIVE" >&2
        exit 1
        ;;
esac
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

exec "$PYTHON" train.py \
    --config-name=train_diffusion_unet_dino_image_workspace \
    task=bulb_front_image \
    horizon="$HORIZON" \
    n_obs_steps="$N_OBS_STEPS" \
    n_action_steps="$N_ACTION_STEPS" \
    task.dataset_path="$DATASET_PATH" \
    task.dataset.relative="$RELATIVE_BOOL" \
    task.dataset.val_ratio="$VALIDATION_RATIO" \
    optimizer.lr="$LEARNING_RATE" \
    optimizer.weight_decay="$WEIGHT_DECAY" \
    training.device=cuda:0 \
    training.num_epochs=500 \
    training.checkpoint_every=20 \
    training.rollout_every=50 \
    dataloader.batch_size=64 \
    dataloader.num_workers=16 \
    dataloader.persistent_workers=True \
    val_dataloader.batch_size=64 \
    val_dataloader.num_workers=16 \
    val_dataloader.persistent_workers=True \
    logging.project=tacmp_diffusion_policy \
    logging.mode="$WANDB_MODE" \
    logging.name="bulb_rotate_front_${ACTION_MODE}" \
    checkpoint.topk.monitor_key=train_loss \
    checkpoint.topk.mode=min \
    "checkpoint.topk.format_str='epoch={epoch:04d}-train_loss={train_loss:.4f}.ckpt'" \
    "hydra.run.dir='${OUTPUT_DIR}'" \
    "$@"
