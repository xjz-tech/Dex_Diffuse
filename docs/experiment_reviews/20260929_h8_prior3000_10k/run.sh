#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL="${ROOT}/../20260929_h8_prior3000/source/eval/xjz_test.sh"
exec 9>"${ROOT}/batch.lock"
flock -n 9 || { echo 'Another 10k batch is already running.' >&2; exit 1; }

for MODEL in h8 1b 10b; do
    case "$MODEL" in
        h8) CHECKPOINT=/home/carus/data_usb/aggresive_random_ckpt/h8.ckpt ;;
        1b) CHECKPOINT=/home/carus/data_usb/aggresive_random_ckpt/obs_4-66.ckpt ;;
        10b) CHECKPOINT=/home/carus/data_usb/aggresive_random_ckpt/10B_obs_4-66.ckpt ;;
    esac
    RUN="${ROOT}/seed42_${MODEL}"
    mkdir -p "$RUN"
    if [[ -f "$RUN/RUN_COMPLETE" ]]; then
        echo "SKIP completed ${MODEL}" | tee -a "${ROOT}/batch_status.log"
        continue
    fi
    echo "START ${MODEL} $(date -Is)" | tee -a "${ROOT}/batch_status.log"
    env CKPT_PATH="$CHECKPOINT" \
        NUM_ENV=3000 MAX_STEPS=10000 MAX_FAILURE_EPISODES=0 DATA_INDICES=000-149 \
        SEED=42 HEADLESS=1 RECORDING=0 PRINT_EVERY=100 \
        SAMPLER=ddim INFERENCE_STEPS=4 EXECUTION_STEPS=2 \
        GUIDE_CKPT_PATH= ALLOW_SALVAGE=0 \
        FAILURE_OBJ_POS_THRES_M=0.05 FAILURE_TIP_POS_THRES_M=0.1 \
        FAILURE_OBJ_ROT_THRES_DEG=180 INVALID_OBJ_POS_THRES_M=0.15 \
        FAILURE_TOLERANCE_SCALE=10000 FIXED_TOLERANCE_STEPS=20000 \
        TRAJ_STEPS_LIMIT=12000 RESET_ON_REACH_GOAL=0 CROSS_TRAJECTORY_GOAL_PROB=0.3 \
        FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 \
        ROTATION_RUN_DIR="$RUN" RUN_DIR="$RUN" RUN_NAME="prior_${MODEL}_joint_turn_seed42_10k" \
        EPISODE_LOG="$RUN/episodes.jsonl" INITIAL_STATE_DUMP="$RUN/initial_state.npz" \
        bash "$EVAL" > "$RUN/run.log" 2>&1
    /home/carus/miniforge3/envs/dp/bin/python "${ROOT}/analyze.py" --validate-model "$MODEL" \
        > "$RUN/validation.log" 2>&1
    touch "$RUN/RUN_COMPLETE"
    echo "DONE ${MODEL} $(date -Is)" | tee -a "${ROOT}/batch_status.log"
done

/home/carus/miniforge3/envs/dp/bin/python "${ROOT}/analyze.py" \
    > "${ROOT}/analysis.log" 2>&1
echo "BATCH COMPLETE $(date -Is)" | tee -a "${ROOT}/batch_status.log"
