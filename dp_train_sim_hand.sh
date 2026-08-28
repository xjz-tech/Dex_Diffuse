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

exec "$PYTHON" train.py \
    --config-name=train_diffusion_unet_sim_hand_workspace \
    task.dataset_path="$DATASET_PATH" \
    logging.mode="$WANDB_MODE" \
    "$@"
