#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse
DEST="$ROOT/docs/experiment_reviews/20260918_historical_3k_restore"
seed="$1"
export RESTORE_OUT="$DEST/seed${seed}_scale25"
mkdir -p "$RESTORE_OUT"
export NUM_ENV=3000 MAX_FAILURE_EPISODES=3000 MAX_STEPS=150
export SAMPLER=ddim INFERENCE_STEPS=4 EXECUTION_STEPS=2 GUIDE_INFERENCE_STEPS=4 GUIDANCE_STEPS=2 FIXED_NOISE=0
export FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1 DATA_INDICES=000-149 HEADLESS=1 RECORDING=0 PRINT_EVERY=50
export FAILURE_OBJ_POS_THRES_M=.05 FAILURE_TIP_POS_THRES_M=.1 FAILURE_OBJ_ROT_THRES_DEG=180 INVALID_OBJ_POS_THRES_M=.15 FAILURE_TOLERANCE_SCALE=10000 FIXED_TOLERANCE_STEPS=20000 TRAJ_STEPS_LIMIT=12000 RESET_ON_REACH_GOAL=0 CROSS_TRAJECTORY_GOAL_PROB=.3
export PRIOR_CKPT_PATH=/home/carus/data_usb/obs_4-66.ckpt GUIDE_CKPT_PATH="$ROOT/runs/sim_hand_10k_seed42/checkpoints/latest.ckpt"
export SEED="$seed" GUIDE_SEED="$((seed+100000))" GUIDANCE_SCALES=25
export MODEL_SERVER="$DEST/code/model.py" SIM_PYTHON="$DEST/code/sim_python" MODEL_PYTHON=/home/carus/miniforge3/envs/dp/bin/python
export RUN_DIR="$RESTORE_OUT" RUN_TAG=replay PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"
bash eval/xjz_eval_strong_prior.sh > "$RESTORE_OUT/console.log" 2>&1
