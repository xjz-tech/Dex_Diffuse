#!/usr/bin/env bash
set -euo pipefail
VIDEO_ROOT=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/videos/20260914_10k_guides_1b_seed8_optimized
CODE_ROOT=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/parallel-guidance-1b-base
export PATH="/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/videos/20260914_10k_guides_1b_seed8/bin:$PATH"
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority
export CKPT_PATH=/home/carus/data_usb/obs_4-66.ckpt
export GUIDE_CKPT_PATH=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/sim_hand_10k_seed42/checkpoints/latest.ckpt
export MODEL_SERVER="$CODE_ROOT/eval/optimized_serial_server.py"
export SAMPLER=ddim INFERENCE_STEPS=4 GUIDE_INFERENCE_STEPS=4
export EXECUTION_STEPS=2 GUIDANCE_STEPS=2 GUIDANCE_SCALE=25
export FIXED_NOISE=0 SEED=8 GUIDE_SEED=100008 STARTUP_TIMEOUT=600
export RECORDING=1 HEADLESS=0 NUM_ENV=1 RECORD_ENV=0 MAX_STEPS=1800 PRINT_EVERY=100
export RECORD_DIR="$VIDEO_ROOT" EPISODE_LOG="$VIDEO_ROOT/episodes.jsonl"
export REQUEST_METRICS_PATH="$VIDEO_ROOT/requests.jsonl" RUN_NAME=video_optimized_original_10k_1b_guide2_scale25_seed8
export DATA_INDICES=000-149
export FAILURE_OBJ_POS_THRES_M=0.05 FAILURE_TIP_POS_THRES_M=0.1 FAILURE_OBJ_ROT_THRES_DEG=180.0
export INVALID_OBJ_POS_THRES_M=0.15 FAILURE_TOLERANCE_SCALE=10000.0 FIXED_TOLERANCE_STEPS=20000
export TRAJ_STEPS_LIMIT=12000 RESET_ON_REACH_GOAL=0 CROSS_TRAJECTORY_GOAL_PROB=0.3
export FIRST_EPISODE_ONLY=0 MAX_FAILURE_EPISODES=0 CENSOR_UNFINISHED_AT_CAP=0
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/eval/eval.sh"
