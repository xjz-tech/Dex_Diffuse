#!/usr/bin/env bash
# Orchestrate: wait for real-hand guide training (epoch>=700), stop it, then run
#   A) 1B prior guided by real-data policy, scale=0,25
#   B) real-data policy alone in sim (no guidance)
# Same protocol as the 10k-guide runs: NUM_ENV=3000, MAX_STEPS=12000,
# first-episode-only, censor-at-cap, seed=42, DDIM 4 steps, pred 9 exec 2.
set -uo pipefail

DEX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_LOG="${DEX_ROOT}/data/real_hand_sft/train_h12.log"
RUN_CKPT_DIR="${DEX_ROOT}/data/outputs/2026.09.10/18.09.41_train_diffusion_unet_sim_hand_sim_hand_lowdim/checkpoints"
REAL_CKPT="${RUN_CKPT_DIR}/latest.ckpt"
TARGET_EPOCH=700
LOG="${DEX_ROOT}/data/real_hand_sft/orchestrator.log"

echo "[orc] start $(date)" >> "${LOG}"

# --- 1. wait for epoch >= TARGET_EPOCH ---
while true; do
    ep=$(tail -1 "${TRAIN_LOG}" | tr '\r' '\n' | grep -oE 'epoch [0-9]+' | tail -1 | grep -oE '[0-9]+' || true)
    if [[ -n "${ep}" && "${ep}" -ge ${TARGET_EPOCH} ]]; then
        echo "[orc] epoch ${ep} reached $(date)" >> "${LOG}"
        break
    fi
    if ! pgrep -f "train.py --config-name=train_diffusion_unet_sim_hand_workspace" >/dev/null; then
        echo "[orc] WARNING: training process gone at epoch ${ep:-?} $(date)" >> "${LOG}"
        break
    fi
    sleep 60
done

# --- 2. stop training, let it flush latest.ckpt ---
pkill -f "train.py --config-name=train_diffusion_unet_sim_hand_workspace" || true
sleep 10
[[ -f "${REAL_CKPT}" ]] || { echo "[orc] FATAL: no checkpoint at ${REAL_CKPT}" >> "${LOG}"; exit 1; }
echo "[orc] using ckpt ${REAL_CKPT} ($(stat -c %y "${REAL_CKPT}"))" >> "${LOG}"

COMMON_ENV=(
    NUM_ENV=3000
    MAX_STEPS=12000
    SEED=42
    SAMPLER=ddim
    INFERENCE_STEPS=4
    EXECUTION_STEPS=2
    FIRST_EPISODE_ONLY=1
    CENSOR_UNFINISHED_AT_CAP=1
)

# --- 3. eval A: real-data guide + 1B prior ---
RUN_A="${DEX_ROOT}/eval/hold_runs/$(date +%Y%m%d_%H%M%S)_real_guide_scale0_25"
echo "[orc] eval A (guide) -> ${RUN_A} $(date)" >> "${LOG}"
env -u LD_LIBRARY_PATH "${COMMON_ENV[@]}" \
    GUIDE_CKPT_PATH="${REAL_CKPT}" \
    GUIDANCE_SCALES="0,25" \
    GUIDE_INFERENCE_STEPS=4 \
    GUIDANCE_STEPS=2 \
    RUN_DIR="${RUN_A}" \
    bash "${DEX_ROOT}/eval/xjz_eval_strong_prior.sh" >> "${LOG}" 2>&1
echo "[orc] eval A done $(date)" >> "${LOG}"

# --- 4. eval B: real-data policy alone ---
RUN_B="${DEX_ROOT}/eval/hold_runs/$(date +%Y%m%d_%H%M%S)_real_policy_alone"
mkdir -p "${RUN_B}"
echo "[orc] eval B (alone) -> ${RUN_B} $(date)" >> "${LOG}"
env -u LD_LIBRARY_PATH "${COMMON_ENV[@]}" \
    CKPT_PATH="${REAL_CKPT}" \
    RUN_DIR="${RUN_B}" \
    RUN_NAME="real_policy_alone" \
    EPISODE_LOG="${RUN_B}/real_policy_alone.jsonl" \
    bash "${DEX_ROOT}/eval/xjz_test.sh" >> "${LOG}" 2>&1
echo "[orc] eval B done $(date)" >> "${LOG}"

# --- 5. summary ---
echo "[orc] === summary $(date) ===" >> "${LOG}"
for f in "${RUN_A}"/*.jsonl "${RUN_B}"/*.jsonl; do
    [[ -f "$f" ]] && /home/carus/miniforge3/envs/dp/bin/python "${DEX_ROOT}/eval/episode_stats.py" "$f" >> "${LOG}" 2>&1
done
echo "[orc] ALL DONE $(date)" >> "${LOG}"
