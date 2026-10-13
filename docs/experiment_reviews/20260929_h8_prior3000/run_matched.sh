#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVAL="${ROOT}/source/eval/xjz_test.sh"
for MODEL in h8 1b; do
    case "$MODEL" in
      h8) CHECKPOINT=/home/carus/data_usb/aggresive_random_ckpt/h8.ckpt ;;
      1b) CHECKPOINT=/home/carus/data_usb/aggresive_random_ckpt/obs_4-66.ckpt ;;
    esac
    RUN="${ROOT}/seed42_${MODEL}"
    mkdir -p "$RUN"
    if [[ -f "$RUN/RUN_COMPLETE" ]]; then continue; fi
    env CKPT_PATH="$CHECKPOINT" \
      NUM_ENV=3000 MAX_STEPS=900 MAX_FAILURE_EPISODES=0 DATA_INDICES=000-149 \
      SEED=42 HEADLESS=1 RECORDING=0 PRINT_EVERY=100 \
      SAMPLER=ddim INFERENCE_STEPS=4 EXECUTION_STEPS=2 \
      GUIDE_CKPT_PATH= ALLOW_SALVAGE=0 \
      FAILURE_OBJ_POS_THRES_M=0.05 FAILURE_TIP_POS_THRES_M=0.1 \
      FAILURE_OBJ_ROT_THRES_DEG=180 INVALID_OBJ_POS_THRES_M=0.15 \
      FAILURE_TOLERANCE_SCALE=10000 FIXED_TOLERANCE_STEPS=20000 \
      TRAJ_STEPS_LIMIT=12000 RESET_ON_REACH_GOAL=0 CROSS_TRAJECTORY_GOAL_PROB=0.3 \
      FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 \
      ROTATION_RUN_DIR="$RUN" RUN_DIR="$RUN" RUN_NAME="prior_${MODEL}_joint_turn_seed42" \
      EPISODE_LOG="$RUN/episodes.jsonl" INITIAL_STATE_DUMP="$RUN/initial_state.npz" \
      bash "$EVAL" > "$RUN/run.log" 2>&1
done
/home/carus/miniforge3/envs/dp/bin/python "$ROOT/analyze_comparison.py" > "$ROOT/analyze.log" 2>&1
