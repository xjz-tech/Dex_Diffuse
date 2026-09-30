#!/usr/bin/env bash
set -euo pipefail

DEX=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse
ROOT="$DEX/eval/hold_runs/20260917_100k_guides_10b_seed8_3k"
export PYTHONDONTWRITEBYTECODE=1
export NUM_ENV=3000 MAX_FAILURE_EPISODES=3000 MAX_STEPS=12000
export SAMPLER=ddim INFERENCE_STEPS=4 EXECUTION_STEPS=2
export GUIDE_INFERENCE_STEPS=4 GUIDANCE_STEPS=2 FIXED_NOISE=0
export FIRST_EPISODE_ONLY=1 CENSOR_UNFINISHED_AT_CAP=1
export DATA_INDICES=000-149 HEADLESS=1 RECORDING=0 PRINT_EVERY=100
export FAILURE_OBJ_POS_THRES_M=0.05 FAILURE_TIP_POS_THRES_M=0.1
export FAILURE_OBJ_ROT_THRES_DEG=180 INVALID_OBJ_POS_THRES_M=0.15
export FAILURE_TOLERANCE_SCALE=10000 FIXED_TOLERANCE_STEPS=20000
export TRAJ_STEPS_LIMIT=12000 RESET_ON_REACH_GOAL=0 CROSS_TRAJECTORY_GOAL_PROB=0.3
export SEED=8 GUIDE_SEED=100008
export PRIOR_CKPT_PATH=/home/carus/data_usb/10B_obs_4-66.ckpt
export GUIDE_CKPT_PATH="$DEX/runs/sim_hand_100k_ownnorm_seed42_equal_epochs/checkpoints/latest.ckpt"
export GUIDANCE_SCALES=25
export MODEL_SERVER="$DEX/eval/xjz_eval_strong_prior.py"
export XJZ_TEST_SCRIPT="$DEX/eval/xjz_test.sh"
export MODEL_PYTHON=/home/carus/miniforge3/envs/dp/bin/python
export RUN_DIR="$ROOT/seed8_10B_100k_scale25"
export RUN_TAG=seed8_10B_100k

cd "$DEX"
exec bash "$DEX/eval/xjz_eval_strong_prior.sh"
