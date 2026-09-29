#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ENV="${DEMO_ENV:-/home/carus/miniforge3/envs/decv2}"
export PATH="${DEMO_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${DEMO_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
exec "${DEMO_ENV}/bin/python" -u "${SCRIPT_DIR}/view_demonstration.py" "$@"
