#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DP_PREFIX="${DP_PREFIX:-${HOME}/miniforge3/envs/dp}"
NVJITLINK_LIB="${DP_PREFIX}/lib/python3.10/site-packages/nvidia/nvjitlink/lib"

export LD_LIBRARY_PATH="${NVJITLINK_LIB}:${DP_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}/eval:${ROOT}:${PYTHONPATH:-}"
export DINOV2_REPO_OR_DIR="${DINOV2_REPO_OR_DIR:-/home/carus/Program/diffusion_policy/dinov2_assets/facebookresearch_dinov2_main}"
export DINOV2_WEIGHTS="${DINOV2_WEIGHTS:-/home/carus/Program/diffusion_policy/dinov2_assets/dinov2_vits14_pretrain.pth}"
export DINOV2_SOURCE="${DINOV2_SOURCE:-local}"

cd "${ROOT}"
exec "${DP_PREFIX}/bin/python" -u exp/offline_guidance/offline_guidance_eval.py "$@"
