#!/usr/bin/env bash
# Seed-robustness pass: re-run the full real1400 experiment set with SEED=8.
#   prior in {1B, mixed} x scale in {0,25,50}, guide = real epoch_1400
#   + real epoch_1400 standalone (no prior, no guide)
# Everything else identical to run_real1400_grid.sh / run_real1400_alone.sh:
# NUM_ENV=3000, MAX_STEPS=12000, first-episode-only, censor-at-cap,
# DDIM 4 steps, pred 9 exec 2, guidance_steps=2.
# Waits for the seed-42 grid AND the seed-42 standalone run to finish first.
set -uo pipefail

DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_LOG="${DEX_ROOT}/data/real_hand_sft/grid1400.log"
ALONE_LOG="${DEX_ROOT}/data/real_hand_sft/alone1400.log"
CKPT_DIR="${DEX_ROOT}/data/outputs/2026.09.11/11.37.42_train_diffusion_unet_sim_hand_sim_hand_lowdim/checkpoints"
REAL_CKPT="${CKPT_DIR}/epoch_1400.ckpt"
PRIOR_1B="/home/carus/data_usb/obs_4-66.ckpt"
PRIOR_MIXED="/home/carus/data_usb/8_2_mixed.ckpt"
SEED=8
LOG="${DEX_ROOT}/data/real_hand_sft/grid1400_seed8.log"

echo "[g1400-s8] start $(date)" >> "${LOG}"

# --- wait for the seed-42 grid and the seed-42 standalone eval to finish ---
while ! grep -qa "ALL DONE" "${GRID_LOG}" 2>/dev/null; do
    sleep 120
done
echo "[g1400-s8] seed-42 grid done $(date)" >> "${LOG}"
while ! grep -qa "\[alone1400\] done" "${ALONE_LOG}" 2>/dev/null; do
    sleep 120
done
echo "[g1400-s8] seed-42 standalone done $(date)" >> "${LOG}"

[[ -f "${REAL_CKPT}" ]] || { echo "[g1400-s8] FATAL: no ckpt ${REAL_CKPT}" >> "${LOG}"; exit 1; }

run_one() {
    local prior="$1" prior_tag="$2" scale="$3"
    local run_dir="${DEX_ROOT}/eval/hold_runs/$(date +%Y%m%d_%H%M%S)_real1400_${prior_tag}_scale${scale}_seed8"
    echo "[g1400-s8] prior=${prior_tag} scale=${scale} seed=${SEED} -> ${run_dir} $(date)" >> "${LOG}"
    env -u LD_LIBRARY_PATH \
        NUM_ENV=3000 MAX_STEPS=12000 SEED=${SEED} SAMPLER=ddim \
        INFERENCE_STEPS=4 EXECUTION_STEPS=2 \
        FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 \
        PRIOR_CKPT_PATH="${prior}" \
        GUIDE_CKPT_PATH="${REAL_CKPT}" \
        GUIDANCE_SCALES="${scale}" GUIDE_INFERENCE_STEPS=4 GUIDANCE_STEPS=2 \
        RUN_TAG="realE1400guide_${prior_tag}_seed8" \
        RUN_DIR="${run_dir}" \
        bash "${DEX_ROOT}/eval/xjz_eval_strong_prior.sh" >> "${LOG}" 2>&1
    for f in "${run_dir}"/*.jsonl; do
        [[ -f "$f" ]] && /home/carus/miniforge3/envs/dp/bin/python "${DEX_ROOT}/eval/episode_stats.py" "$f" >> "${LOG}" 2>&1
    done
    echo "[g1400-s8] prior=${prior_tag} scale=${scale} done $(date)" >> "${LOG}"
}

for scale in 0 25 50; do
    run_one "${PRIOR_1B}" "prior1B" "${scale}"
done
for scale in 0 25 50; do
    run_one "${PRIOR_MIXED}" "priorMixed" "${scale}"
done

# --- real DP standalone, seed=8 ---
RUN_DIR="${DEX_ROOT}/eval/hold_runs/$(date +%Y%m%d_%H%M%S)_real1400_alone_seed8"
echo "[g1400-s8] real-alone seed=${SEED} -> ${RUN_DIR} $(date)" >> "${LOG}"
env -u LD_LIBRARY_PATH \
    NUM_ENV=3000 MAX_STEPS=12000 SEED=${SEED} SAMPLER=ddim \
    INFERENCE_STEPS=4 EXECUTION_STEPS=2 \
    FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 \
    CKPT_PATH="${REAL_CKPT}" \
    RUN_NAME="realE1400_alone_seed8" \
    RUN_DIR="${RUN_DIR}" \
    EPISODE_LOG="${RUN_DIR}/realE1400_alone_seed8.jsonl" \
    bash "${DEX_ROOT}/eval/xjz_test.sh" >> "${LOG}" 2>&1
echo "[g1400-s8] real-alone done $(date)" >> "${LOG}"

echo "[g1400-s8] ALL DONE $(date)" >> "${LOG}"
