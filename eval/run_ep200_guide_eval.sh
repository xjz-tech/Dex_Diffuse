#!/usr/bin/env bash
# Wait for the running mixed-guide (scale 0,25) eval to finish, then run scale grids:
#   1) real-hand epoch_0200 (best val loss) as guide: scales 10,25,50,75
#   2) 8_2_mixed as guide: scales 10,50,75  (0 and 25 already running)
# scale=0 baseline is guide-independent: 0.104 / 75.9s.
set -uo pipefail

DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REAL_EP200="${DEX_ROOT}/data/outputs/2026.09.10/18.09.41_train_diffusion_unet_sim_hand_sim_hand_lowdim/checkpoints/epoch_0200.ckpt"
MIXED_CKPT="/home/carus/data_usb/8_2_mixed.ckpt"
LOG="${DEX_ROOT}/data/real_hand_sft/scale_grid_eval.log"

echo "[grid] waiting for current eval to finish $(date)" >> "${LOG}"
while pgrep -f "xjz_eval_strong_prior.py" >/dev/null || pgrep -f "sim_eval.py" >/dev/null; do
    sleep 60
done
echo "[grid] gpu free, starting $(date)" >> "${LOG}"

run_grid() {
    local ckpt="$1" tag="$2" scales="$3"
    [[ -f "${ckpt}" ]] || { echo "[grid] FATAL: no ckpt ${ckpt}" >> "${LOG}"; return 1; }
    local run_dir="${DEX_ROOT}/eval/hold_runs/$(date +%Y%m%d_%H%M%S)_${tag}"
    echo "[grid] ${tag} scales=${scales} -> ${run_dir} $(date)" >> "${LOG}"
    env -u LD_LIBRARY_PATH \
        NUM_ENV=3000 MAX_STEPS=12000 SEED=42 SAMPLER=ddim \
        INFERENCE_STEPS=4 EXECUTION_STEPS=2 \
        FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 \
        GUIDE_CKPT_PATH="${ckpt}" \
        GUIDANCE_SCALES="${scales}" GUIDE_INFERENCE_STEPS=4 GUIDANCE_STEPS=2 \
        RUN_DIR="${run_dir}" \
        bash "${DEX_ROOT}/eval/xjz_eval_strong_prior.sh" >> "${LOG}" 2>&1
    for f in "${run_dir}"/*.jsonl; do
        [[ -f "$f" ]] && /home/carus/miniforge3/envs/dp/bin/python "${DEX_ROOT}/eval/episode_stats.py" "$f" >> "${LOG}" 2>&1
    done
    echo "[grid] ${tag} done $(date)" >> "${LOG}"
}

run_grid "${REAL_EP200}" "real_ep200_guide_scale_grid" "10,25,50,75"
run_grid "${MIXED_CKPT}" "mixed_guide_scale_grid" "10,50,75"

echo "[grid] ALL DONE $(date)" >> "${LOG}"
