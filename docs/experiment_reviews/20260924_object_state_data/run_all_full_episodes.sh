#!/usr/bin/env bash
set -euo pipefail
base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
root="$base/corrected_direct_vs_reference/all_full_episodes"
first=${1:-0}
last=${2:-79}
protocol=${3:-full}
if [[ "$protocol" != full && "$protocol" != early ]]; then
  echo 'protocol must be full or early' >&2
  exit 2
fi
run_sim() {
  local episode=$1 kind=$2 execute=$3 folder=$4 socket=${5:-}
  local reference="$root/episode_$(printf '%02d' "$episode")/reference_full.npz"
  mkdir -p "$folder"
  if [[ -f "$folder/summary.json" ]]; then echo "already complete: ep$episode $kind exec$execute"; return; fi
  local opts=(--mode "$kind" --out "$folder" --reference "$reference" --reference-id 0
    --source-episode "$episode" --no-video --front-only
    --guidance-steps 2 --guidance-scale 50 --execution-steps "$execute")
  if [[ "$protocol" == early ]]; then opts+=(--stop-on-native-failure); fi
  if [[ "$kind" == guided ]]; then opts+=(--socket "$socket" --reference-interpolation 1); fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
    PYTHONPATH=/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib \
    /home/carus/miniforge3/envs/decv2/bin/python "$base/compare_corrected_rollouts.py" "${opts[@]}" \
    > "$folder/sim.log" 2>&1
  echo "complete: ep$episode $kind exec$execute"
}
run_guided() {
  local episode=$1 execute=$2 folder="$root/episode_$(printf '%02d' "$1")/$protocol/guide2_exec${2}"
  local socket="/tmp/object_all_ep${episode}_exec${execute}_$$.sock"
  local server_pid=
  cleanup() {
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
    rm -f "$socket"
  }
  trap cleanup EXIT
  mkdir -p "$folder"
  if [[ -f "$folder/summary.json" ]]; then echo "already complete: ep$episode guided exec$execute"; trap - EXIT; return; fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/dp/lib \
    /home/carus/miniforge3/envs/dp/bin/python "$base/server.py" \
    --socket "$socket" --log "$folder/predictions.json" \
    --checkpoint /home/carus/data_usb/10B_obs_4-66.ckpt \
    --reference "$root/episode_$(printf '%02d' "$episode")/reference_full.npz" \
    --guidance-steps 2 --guidance-scale 50 --execution-steps "$execute" --reference-interpolation 1 \
    > "$folder/server.log" 2>&1 &
  server_pid=$!
  local ready=0
  for attempt in $(seq 1 200); do
    if [[ -S "$socket" ]]; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep .1
  done
  if [[ "$ready" != 1 ]]; then echo "server not ready: ep$episode exec$execute" >&2; exit 1; fi
  run_sim "$episode" guided "$execute" "$folder" "$socket"
  wait "$server_pid"
  server_pid=
  trap - EXIT
}
if (( first < 0 || last > 79 || first > last )); then
  echo 'usage: run_all_full_episodes.sh [first_episode_0_to_79] [last_episode_0_to_79]' >&2
  exit 2
fi
for episode in $(seq "$first" "$last"); do
  folder="$root/episode_$(printf '%02d' "$episode")/$protocol"
  echo "START ep$episode $(date '+%F %T')"
  run_sim "$episode" direct 2 "$folder/direct"
  run_guided "$episode" 2
  run_guided "$episode" 1
  echo "FINISH ep$episode $(date '+%F %T')"
done
