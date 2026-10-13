#!/usr/bin/env bash
set -euo pipefail
VIDEO_ROOT="${ARM_DIR:-/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260916_10k_obs466_30hz_clamp003_3k/hz${CONTROL_HZ:?}}"
CODE_ROOT=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260916_10k_obs466_30hz_clamp003_3k/code
export WAIT=1 CONTROL_HZ=30
export PATH="/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/videos/20260914_10k_guides_1b_seed8/bin:$PATH"
export DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority
export CKPT_PATH=/home/carus/data_usb/obs_4-66.ckpt
export GUIDE_CKPT_PATH=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/sim_hand_10k_seed42/checkpoints/latest.ckpt
export MODEL_SERVER="$CODE_ROOT/eval/optimized_serial_server.py"
export SAMPLER=ddim INFERENCE_STEPS=4 GUIDE_INFERENCE_STEPS=4
export EXECUTION_STEPS=2 GUIDANCE_STEPS=2 GUIDANCE_SCALE=25
export FIXED_NOISE=0 SEED=8 GUIDE_SEED=100008 STARTUP_TIMEOUT=600
export RECORDING=0 HEADLESS=1 NUM_ENV="${NUM_ENV:-3000}" RECORD_ENV=0 MAX_STEPS="${MAX_STEPS:-$((400 * CONTROL_HZ))}" PRINT_EVERY=100
export RECORD_DIR="$VIDEO_ROOT" EPISODE_LOG="$VIDEO_ROOT/episodes.jsonl"
export REQUEST_METRICS_PATH="$VIDEO_ROOT/requests.jsonl" RUN_NAME=trt_10k_obs466_hz30_clamp003_seed8_3k
export DATA_INDICES="${DATA_INDICES:-000-149}"
export FAILURE_OBJ_POS_THRES_M=0.05 FAILURE_TIP_POS_THRES_M=0.1 FAILURE_OBJ_ROT_THRES_DEG=180.0
export INVALID_OBJ_POS_THRES_M=0.15 FAILURE_TOLERANCE_SCALE=10000.0 FIXED_TOLERANCE_STEPS=20000
export TRAJ_STEPS_LIMIT=0 RESET_ON_REACH_GOAL=0 CROSS_TRAJECTORY_GOAL_PROB=0.3
export FIRST_EPISODE_ONLY=1 MAX_FAILURE_EPISODES="${NUM_ENV}" CENSOR_UNFINISHED_AT_CAP=1
mkdir -p "$VIDEO_ROOT"
cd "$CODE_ROOT"
exec bash "$CODE_ROOT/eval/eval.sh"
