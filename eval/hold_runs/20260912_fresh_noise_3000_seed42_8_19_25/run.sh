#!/usr/bin/env bash
# Four seeds, two priors, two guidance scales; each process draws fresh noise.
set -euo pipefail
DEX=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse
ROOT="$DEX/eval/hold_runs/20260912_fresh_noise_3000_seed42_8_19_25"
MODEL_PYTHON=/home/carus/miniforge3/envs/dp/bin/python
GUIDE="$DEX/runs/sim_hand_10k_seed42/checkpoints/latest.ckpt"
cd "$DEX"
exec 9>"$ROOT/queue.lock"
flock -n 9 || { echo 'Queue already running'; exit 2; }
exec >>"$ROOT/queue.log" 2>&1
trap 'code=$?; printf "%s queue-exit=%s\n" "$(date -Is)" "$code"' EXIT
printf '%s\n' "$$" > "$ROOT/supervisor.pid"
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
export GUIDE_CKPT_PATH="$GUIDE" MODEL_PYTHON
export MODEL_SERVER="$DEX/eval/xjz_eval_strong_prior.py"
export XJZ_TEST_SCRIPT="$DEX/eval/xjz_test.sh"
printf '%s queue-start; 16 runs; fresh prior and guide noise\n' "$(date -Is)"
sha256sum /home/carus/data_usb/obs_4-66.ckpt /home/carus/data_usb/8_2_mixed.ckpt "$GUIDE" > "$ROOT/checkpoints.sha256"
for seed in 42 8 19 25; do
  for scale in 0 25; do
    for prior in 1B mixed; do
      tag="seed${seed}_${prior}_fresh"
      run="$ROOT/seed${seed}_${prior}_scale${scale}"
      if [[ -f "$run/COMPLETE" ]]; then
        printf '%s skip-complete %s\n' "$(date -Is)" "$run"
        continue
      fi
      if [[ -d "$run" ]]; then
        printf '%s refusing-to-overwrite-incomplete %s\n' "$(date -Is)" "$run"
        exit 3
      fi
      # One extra 3000-env job needs roughly 8 GB; allow startup headroom.
      waiting=0
      while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
        if (( free_mb >= 10500 )); then break; fi
        if (( waiting == 0 )); then
          printf '%s waiting-for-gpu free_mb=%s\n' "$(date -Is)" "$free_mb"
          waiting=1
        fi
        sleep 30
      done
      mkdir -p "$run"
      ckpt=/home/carus/data_usb/obs_4-66.ckpt
      [[ "$prior" == mixed ]] && ckpt=/home/carus/data_usb/8_2_mixed.ckpt
      export SEED="$seed" GUIDE_SEED="$((seed + 100000))"
      export PRIOR_CKPT_PATH="$ckpt" GUIDANCE_SCALES="$scale" RUN_DIR="$run" RUN_TAG="$tag"
      printf '%s start seed=%s prior=%s scale=%s prior_seed=%s guide_seed=%s\n' "$(date -Is)" "$seed" "$prior" "$scale" "$SEED" "$GUIDE_SEED"
      env | LC_ALL=C sort | rg '^(NUM_ENV|MAX_FAILURE_EPISODES|MAX_STEPS|SAMPLER|INFERENCE_STEPS|EXECUTION_STEPS|GUIDE_INFERENCE_STEPS|GUIDANCE_STEPS|FIXED_NOISE|FIRST_EPISODE_ONLY|CENSOR_UNFINISHED_AT_CAP|DATA_INDICES|FAILURE_[A-Z_]+|INVALID_[A-Z_]+|FIXED_TOLERANCE_STEPS|TRAJ_STEPS_LIMIT|RESET_ON_REACH_GOAL|CROSS_TRAJECTORY_GOAL_PROB|SEED|GUIDE_SEED|PRIOR_CKPT_PATH|GUIDE_CKPT_PATH|GUIDANCE_SCALES|MODEL_SERVER)=' > "$run/parameters.txt"
      git rev-parse HEAD > "$run/git-head.txt"
      sha256sum eval/{xjz_eval_strong_prior.sh,xjz_test.sh,eval.sh,guided_pair_policy.py,inference_dp_controller.py,sim_eval.py,episode_stats.py,checkpoint_loader.py,model_server.py} diffusion_policy/guidance/guided_ddim.py > "$run/source.sha256"
      bash "$DEX/eval/xjz_eval_strong_prior.sh" > "$run/console.log" 2>&1
      "$MODEL_PYTHON" - "$run/${tag}_scale${scale}.jsonl" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, 'eval')
from episode_stats import summarize_episode_records, format_hold_summary
p = Path(sys.argv[1])
records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
assert len(records) == 3000, (p, len(records))
assert {r['env'] for r in records} == set(range(3000)), (p, 'env coverage')
assert all(r['episode'] == 0 for r in records), (p, 'non-first episode')
assert all(r['reason'] in ('failure', 'timeout', 'success') for r in records)
assert all(r['length'] == 12000 for r in records if r['reason'] == 'timeout')
print(format_hold_summary(str(p), summarize_episode_records(records)))
PY
      touch "$run/COMPLETE"
      printf '%s finished seed=%s prior=%s scale=%s\n' "$(date -Is)" "$seed" "$prior" "$scale"
    done
  done
done
printf '%s all-finished\n' "$(date -Is)"
