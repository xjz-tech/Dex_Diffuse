#!/usr/bin/env bash
# Train Sim-Hand DP with n_obs=4 / n_pred=21 / horizon=24.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
DATASET_PATH="${DATASET_PATH:-/home/carus/Data/exp_data}"
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
    echo "Build obs4/pred21/h24 cache first (or hardlink-update metadata):" >&2
    echo "python -m diffusion_policy.scripts.prepare_sim_hand_mmap \\" >&2
    echo "  --dataset-path $DATASET_PATH --horizon 24 --pad-before 3 --pad-after 20" >&2
    exit 1
fi
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

exec "$PYTHON" train.py \
    --config-name=train_diffusion_unet_sim_hand_obs4_pred21_workspace \
    task.dataset_path="$DATASET_PATH" \
    logging.mode="$WANDB_MODE" \
    "$@"
