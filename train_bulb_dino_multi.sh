#!/usr/bin/env bash
# Train the DINOv2 bulb policy with one DDP process per GPU.
# RELATIVE=1: relative EE pose + absolute hand joints.
# RELATIVE=0: absolute EE pose + absolute hand joints.
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash train_bulb_dino_multi.sh
# NUM_GPUS defaults to the number of visible GPUs. BATCH_SIZE is per GPU.
# OUTPUT_DIR selects an existing run for resume; RESUME_CKPT selects a file.
# Checkpoints are saved under $OUTPUT_DIR/checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
task_name="${task_name:-manipulate_bulb_hori50_his4_demo225}"
RUN_TIMESTAMP="$(date +%Y.%m.%d_%H.%M.%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/${RUN_TIMESTAMP}_${task_name}_ddp}"

PYTHON="${PYTHON:-/home/bighand/miniconda3/envs/dp/bin/python}"
DATASET_PATH="${DATASET_PATH:-/mnt/work/dexIL/Dex_Diffuse/data/real_data/bulb_tac_261001_dp}"
DINOV2_REPO_OR_DIR="${DINOV2_REPO_OR_DIR:-$SCRIPT_DIR/assets/dinov2_assets/facebookresearch_dinov2_main}"
DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-$SCRIPT_DIR/assets/dinov2_assets/dinov2_vits14_pretrain.pth}"
RELATIVE="${RELATIVE:-0}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
HORIZON="${HORIZON:-64}"
N_OBS_STEPS="${N_OBS_STEPS:-4}"
N_ACTION_STEPS="${N_ACTION_STEPS:-32}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
NUM_GPUS="${NUM_GPUS:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/runs/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/runs/matplotlib}"
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
if [[ -n "$NUM_GPUS" ]] && ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
    exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    if [[ -z "$NUM_GPUS" ]]; then
        NUM_GPUS="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
    fi
    if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "No CUDA GPUs detected. Set CUDA_VISIBLE_DEVICES and NUM_GPUS explicitly." >&2
        exit 1
    fi
    CUDA_VISIBLE_DEVICES=""
    for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
        CUDA_VISIBLE_DEVICES+="${CUDA_VISIBLE_DEVICES:+,}$gpu_index"
    done
fi
if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == "-1" ]]; then
    echo "CUDA_VISIBLE_DEVICES must expose at least one GPU." >&2
    exit 1
fi
IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS="${NUM_GPUS:-${#VISIBLE_GPUS[@]}}"
if ((NUM_GPUS > ${#VISIBLE_GPUS[@]})); then
    echo "NUM_GPUS=$NUM_GPUS exceeds the ${#VISIBLE_GPUS[@]} visible GPUs." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES
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

LAUNCH_ARGS=()
if ((NUM_GPUS > 1)); then
    LAUNCH_ARGS=(-m torch.distributed.run --standalone "--nproc_per_node=$NUM_GPUS")
fi
RESUME_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
    if [[ ! -f "$RESUME_CKPT" ]]; then
        echo "Checkpoint not found: $RESUME_CKPT" >&2
        exit 1
    fi
    RESUME_ARGS=("training.resume_path='$RESUME_CKPT'" training.resume=true)
fi

exec "$PYTHON" "${LAUNCH_ARGS[@]}" train.py \
    --config-name=train_diffusion_unet_dino_image_ddp_workspace \
    task=bulb_image \
    horizon="$HORIZON" \
    n_obs_steps="$N_OBS_STEPS" \
    n_action_steps="$N_ACTION_STEPS" \
    task.dataset_path="$DATASET_PATH" \
    task.dataset.relative="$RELATIVE_BOOL" \
    task.dataset.val_ratio="$VALIDATION_RATIO" \
    optimizer.lr="$LEARNING_RATE" \
    optimizer.weight_decay="$WEIGHT_DECAY" \
    training.device=cuda:0 \
    training.num_epochs=601 \
    training.checkpoint_every=100 \
    training.rollout_every=50 \
    dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$NUM_WORKERS" \
    dataloader.persistent_workers=False \
    +dataloader.prefetch_factor=1 \
    val_dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.num_workers="$VAL_NUM_WORKERS" \
    val_dataloader.persistent_workers=False \
    +val_dataloader.prefetch_factor=1 \
    logging.project=tacmp_diffusion_policy \
    logging.mode="$WANDB_MODE" \
    logging.name="bulb_rotate_${ACTION_MODE}_ddp" \
    checkpoint.topk.monitor_key=train_loss \
    checkpoint.topk.mode=min \
    "checkpoint.topk.format_str='epoch={epoch:04d}-train_loss={train_loss:.4f}.ckpt'" \
    "hydra.run.dir='${OUTPUT_DIR}'" \
    hydra/job_logging=disabled \
    hydra.output_subdir=null \
    "${RESUME_ARGS[@]}" \
    "$@"
