#!/usr/bin/env bash
# Standalone sim eval of the real-hand DP (epoch_1400) itself: no prior, no guide.
# Waits for run_real1400_grid.sh to finish (GPU is busy), then runs the same
# protocol as the grid: NUM_ENV=3000, MAX_STEPS=12000, first-episode-only,
# censor-at-cap, seed=42, DDIM 4 steps, pred 9 exec 2.
set -uo pipefail

DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_LOG="${DEX_ROOT}/data/real_hand_sft/grid1400.log"
CKPT="${DEX_ROOT}/data/outputs/2026.09.11/11.37.42_train_diffusion_unet_sim_hand_sim_hand_lowdim/checkpoints/epoch_1400.ckpt"
LOG="${DEX_ROOT}/data/real_hand_sft/alone1400.log"

echo "[alone1400] waiting for grid to finish $(date)" >> "${LOG}"
while ! grep -qa "ALL DONE" "${GRID_LOG}" 2>/dev/null; do
    sleep 120
done

[[ -f "${CKPT}" ]] || { echo "[alone1400] FATAL: no ckpt ${CKPT}" >> "${LOG}"; exit 1; }

RUN_DIR="${DEX_ROOT}/eval/hold_runs/$(date +%Y%m%d_%H%M%S)_real1400_alone"
echo "[alone1400] start ${RUN_DIR} $(date)" >> "${LOG}"
env -u LD_LIBRARY_PATH \
    NUM_ENV=3000 MAX_STEPS=12000 SEED=42 SAMPLER=ddim \
    INFERENCE_STEPS=4 EXECUTION_STEPS=2 \
    FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 \
    CKPT_PATH="${CKPT}" \
    RUN_NAME="realE1400_alone" \
    RUN_DIR="${RUN_DIR}" \
    EPISODE_LOG="${RUN_DIR}/realE1400_alone.jsonl" \
    bash "${DEX_ROOT}/eval/xjz_test.sh" >> "${LOG}" 2>&1
echo "[alone1400] done $(date)" >> "${LOG}"
