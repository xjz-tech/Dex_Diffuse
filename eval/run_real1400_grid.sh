#!/usr/bin/env bash
# Wait for the fixed real-hand DP training to reach epoch 1400, stop it, then run:
#   prior in {1B, mixed} x scale in {0,25,50}, guide = new 1400-epoch real DP.
# Same protocol as before: NUM_ENV=3000, MAX_STEPS=12000, first-episode-only,
# censor-at-cap, seed=42, DDIM 4 steps, pred 9 exec 2, guidance_steps=2.
set -uo pipefail

DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_LOG="/tmp/train_v3.log"
CKPT_DIR="${DEX_ROOT}/data/outputs/2026.09.11/11.37.42_train_diffusion_unet_sim_hand_sim_hand_lowdim/checkpoints"
REAL_CKPT="${CKPT_DIR}/latest.ckpt"
PRIOR_1B="/home/carus/data_usb/obs_4-66.ckpt"
PRIOR_MIXED="/home/carus/data_usb/8_2_mixed.ckpt"
TARGET_EPOCH=1400
LOG="${DEX_ROOT}/data/real_hand_sft/grid1400.log"

echo "[g1400] start $(date)" >> "${LOG}"

# --- 1. wait for the epoch_1400 checkpoint file (checkpoint_every=5) ---
TARGET_CKPT="${CKPT_DIR}/epoch_1400.ckpt"
while true; do
    if [[ -f "${TARGET_CKPT}" ]]; then
        echo "[g1400] ${TARGET_CKPT} saved $(date)" >> "${LOG}"
        break
    fi
    if ! pgrep -f "task.dataset_path=.*real_hand_sft" >/dev/null; then
        echo "[g1400] WARNING: training gone before epoch 1400 $(date)" >> "${LOG}"
        break
    fi
    sleep 120
done

# --- 2. stop training (narrow match: only this dataset's run) ---
pkill -f "task.dataset_path=.*real_hand_sft" || true
sleep 10
REAL_CKPT="${TARGET_CKPT}"
[[ -f "${REAL_CKPT}" ]] || REAL_CKPT="${CKPT_DIR}/latest.ckpt"
[[ -f "${REAL_CKPT}" ]] || { echo "[g1400] FATAL: no ckpt" >> "${LOG}"; exit 1; }
echo "[g1400] guide ckpt: ${REAL_CKPT} ($(stat -c %y "${REAL_CKPT}"))" >> "${LOG}"

run_one() {
    local prior="$1" prior_tag="$2" scale="$3"
    local run_dir="${DEX_ROOT}/eval/hold_runs/$(date +%Y%m%d_%H%M%S)_real1400_${prior_tag}_scale${scale}"
    echo "[g1400] prior=${prior_tag} scale=${scale} -> ${run_dir} $(date)" >> "${LOG}"
    env -u LD_LIBRARY_PATH \
        NUM_ENV=3000 MAX_STEPS=12000 SEED=42 SAMPLER=ddim \
        INFERENCE_STEPS=4 EXECUTION_STEPS=2 \
        FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 \
        PRIOR_CKPT_PATH="${prior}" \
        GUIDE_CKPT_PATH="${REAL_CKPT}" \
        GUIDANCE_SCALES="${scale}" GUIDE_INFERENCE_STEPS=4 GUIDANCE_STEPS=2 \
        RUN_TAG="realE1400guide_${prior_tag}" \
        RUN_DIR="${run_dir}" \
        bash "${DEX_ROOT}/eval/xjz_eval_strong_prior.sh" >> "${LOG}" 2>&1
    for f in "${run_dir}"/*.jsonl; do
        [[ -f "$f" ]] && /home/carus/miniforge3/envs/dp/bin/python "${DEX_ROOT}/eval/episode_stats.py" "$f" >> "${LOG}" 2>&1
    done
    echo "[g1400] prior=${prior_tag} scale=${scale} done $(date)" >> "${LOG}"
}

for scale in 0 25 50; do
    run_one "${PRIOR_1B}" "prior1B" "${scale}"
done
for scale in 0 25 50; do
    run_one "${PRIOR_MIXED}" "priorMixed" "${scale}"
done

echo "[g1400] ALL DONE $(date)" >> "${LOG}"
