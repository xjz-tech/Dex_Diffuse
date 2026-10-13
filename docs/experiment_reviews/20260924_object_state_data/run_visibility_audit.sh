#!/usr/bin/env bash
set -euo pipefail
base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
root="$base/corrected_direct_vs_reference/all_full_episodes"
audit="$base/visibility_audit_20260926"
for episode in 0 2 28 51; do
  printf -v label '%02d' "$episode"
  out="$audit/episode_$label"
  mkdir -p "$out"
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
    PYTHONPATH=/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib \
    /home/carus/miniforge3/envs/decv2/bin/python "$base/compare_corrected_rollouts.py" \
    --mode direct --out "$out" --reference "$root/episode_$label/reference_full.npz" \
    --reference-id 0 --source-episode "$episode" --guidance-steps 2 \
    --guidance-scale 50 --execution-steps 2 --audit-recording > "$out/sim.log" 2>&1
  echo "complete ep$episode"
done
