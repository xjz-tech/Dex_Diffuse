#!/usr/bin/env bash
# Compatibility entrypoint; use xjz_eval_guidance.sh for new runs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/xjz_eval_guidance.sh" "$@"
