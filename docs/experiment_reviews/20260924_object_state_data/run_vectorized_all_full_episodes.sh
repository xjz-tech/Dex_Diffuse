#!/usr/bin/env bash
set -euo pipefail
base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
data="$base/corrected_direct_vs_reference/all_full_episodes"
episodes=${1:-all}
name=${2:-all80}
stop_mode=${3:-full}
if [[ "$stop_mode" != full && "$stop_mode" != early ]]; then
  echo 'stop mode must be full or early' >&2
  exit 2
fi
out="$data/$name"
mkdir -p "$out"
run_sim() {
  local mode=$1 execute=$2 folder=$3 socket=${4:-}
  mkdir -p "$folder"
  if [[ -f "$folder/summary.json" ]]; then echo "already complete: $mode exec$execute"; return; fi
  local opts=(--mode "$mode" --execution-steps "$execute" --episodes "$episodes" --out "$folder")
  if [[ "$stop_mode" == early ]]; then opts+=(--stop-when-all-failed); fi
  if [[ "$mode" == guided ]]; then opts+=(--socket "$socket"); fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
    PYTHONPATH=/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib \
    /home/carus/miniforge3/envs/decv2/bin/python "$base/sim_all_full_episodes.py" "${opts[@]}" \
    > "$folder/sim.log" 2>&1
  echo "complete: $mode exec$execute"
}
run_guided() {
  local execute=$1 folder="$out/guide2_exec${1}"
  local socket="/tmp/object_vectorized_${name}_exec${execute}_$$.sock"
  local server_pid=
  cleanup() {
    if [[ -n "${server_pid:-}" ]]; then kill "$server_pid" 2>/dev/null || true; fi
    if [[ -n "${socket:-}" ]]; then rm -f "$socket"; fi
  }
  trap cleanup EXIT
  mkdir -p "$folder"
  if [[ -f "$folder/summary.json" ]]; then echo "already complete: guided exec$execute"; trap - EXIT; return; fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/dp/lib \
    /home/carus/miniforge3/envs/dp/bin/python "$base/server.py" \
    --socket "$socket" --log "$folder/predictions.json" \
    --checkpoint /home/carus/data_usb/10B_obs_4-66.ckpt \
    --reference "$data/packed_reference.npz" \
    --guidance-steps 2 --guidance-scale 50 --execution-steps "$execute" --reference-interpolation 1 \
    > "$folder/server.log" 2>&1 &
  server_pid=$!
  local ready=0
  for attempt in $(seq 1 200); do
    if [[ -S "$socket" ]]; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep .1
  done
  if [[ "$ready" != 1 ]]; then echo "server not ready exec$execute" >&2; exit 1; fi
  run_sim guided "$execute" "$folder" "$socket"
  wait "$server_pid"
  server_pid=
  trap - EXIT
}
run_sim direct 2 "$out/direct"
run_guided 2
run_guided 1
echo "complete all $episodes"
