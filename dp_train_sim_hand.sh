#!/usr/bin/env bash
# Train the independent hand-only simulation Diffusion Policy.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/data/sim_hand}"
NUM_GPUS="${NUM_GPUS:-1}"
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
    exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    CUDA_VISIBLE_DEVICES=""
    for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
        CUDA_VISIBLE_DEVICES+="${CUDA_VISIBLE_DEVICES:+,}$gpu_index"
    done
fi
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/data/wandb}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
# Prefer existing private credential stores; never echo the key.
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  WANDB_API_KEY="$(python - <<'PY'
import netrc
from pathlib import Path
auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("api.wandb.ai")
print((auth[2] if auth else ""), end="")
PY
)"
  export WANDB_API_KEY
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/data/matplotlib}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -d "$DATASET_PATH/replay_buffer.zarr" && ! -f "$DATASET_PATH/manifest.json" ]]; then
    echo "Dataset not found: expected $DATASET_PATH/replay_buffer.zarr or $DATASET_PATH/manifest.json" >&2
    exit 1
fi
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

LAUNCH_ARGS=()
HYDRA_ARGS=()
if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCH_ARGS=(
        -m torch.distributed.run
        --standalone
        "--nproc_per_node=$NUM_GPUS"
    )
    HYDRA_ARGS=(
        hydra/job_logging=disabled
        hydra.output_subdir=null
    )
fi

exec "$PYTHON" "${LAUNCH_ARGS[@]}" train.py \
    --config-name=train_diffusion_unet_sim_hand_workspace \
    task.dataset_path="$DATASET_PATH" \
    logging.mode="$WANDB_MODE" \
    "${HYDRA_ARGS[@]}" \
    "$@"
