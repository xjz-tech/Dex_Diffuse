#!/usr/bin/env bash
set -euo pipefail
REPORT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEX_ROOT="$(cd -- "${REPORT_DIR}/../.." && pwd)"
SCRIPT_DIR="${DEX_ROOT}/eval"
source "${SCRIPT_DIR}/real/run_logging.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/frankagvl/anaconda3/envs/dexIL/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${MODEL_PYTHON}" -c 'import torch, torch_tensorrt; assert torch.cuda.is_available()'
"${MODEL_PYTHON}" "${REPORT_DIR}/initialize_pose.py" \
    --skip-franka --init-pose ROTATE --hand-host localhost --hand-port 5570 \
    --hand-timeout-ms 2000 --hand-steps 50 --hand-seconds 2.0 \
    --hand-tolerance 0.1 --hand-timeout-seconds 3.0 --poll-seconds 0.1
exec bash "${SCRIPT_DIR}/eval_para_obs66_real.sh"
