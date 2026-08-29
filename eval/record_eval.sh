#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RECORDING=1
export HEADLESS=0
export NUM_ENV="${NUM_ENV:-${NUM_ENVS:-1}}"
export RECORD_ENV="${RECORD_ENV:-0}"
export RECORD_DIR="${RECORD_DIR:-${SCRIPT_DIR}/record}"

exec "${SCRIPT_DIR}/eval.sh"
