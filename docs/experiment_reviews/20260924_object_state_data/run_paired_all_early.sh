#!/usr/bin/env bash
set -euo pipefail
base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
first=${1:-0}
last=${2:-78}
if (( first < 0 || last > 78 || first > last || first % 2 != 0 || last % 2 != 0 )); then
  echo 'usage: run_paired_all_early.sh [even_first_0_to_78] [even_last_0_to_78]' >&2
  exit 2
fi
for episode in $(seq "$first" 2 "$last"); do
  second=$((episode+1))
  name=$(printf 'pair_%02d_%02d' "$episode" "$second")
  echo "START $name $(date '+%F %T')"
  bash "$base/run_vectorized_all_full_episodes.sh" "$episode,$second" "$name" early
  echo "FINISH $name $(date '+%F %T')"
done
