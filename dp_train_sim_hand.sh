#!/usr/bin/env bash
# Train the independent hand-only simulation Diffusion Policy.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data/sim_hand}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/data/wandb}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/data/matplotlib}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi
CACHE_READY="$DATASET_PATH/exp_data_mmap/READY"
if [[ ! -f "$CACHE_READY" ]]; then
    echo "Completed Sim-Hand mmap cache not found: $CACHE_READY" >&2
    echo "Build it first:" >&2
    echo "python -m diffusion_policy.scripts.prepare_sim_hand_mmap --dataset-path $DATASET_PATH" >&2
    exit 1
fi
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

exec "$PYTHON" train.py \
    --config-name=train_diffusion_unet_sim_hand_workspace \
    task.dataset_path="$DATASET_PATH" \
    logging.mode="$WANDB_MODE" \
    "$@"
