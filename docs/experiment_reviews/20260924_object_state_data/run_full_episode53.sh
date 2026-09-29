#!/usr/bin/env bash
set -euo pipefail
base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
out="$base/corrected_direct_vs_reference/full_episode53"
reference="$out/reference_full.npz"
mode=${1:-all}
run_sim() {
  local kind=$1 folder=$2 execute=$3 socket=${4:-}
  mkdir -p "$folder"
  if [[ -f "$folder/summary.json" ]]; then echo "already complete: $folder"; return; fi
  local opts=(--mode "$kind" --out "$folder" --reference "$reference" --reference-id 0
    --front-only --guidance-steps 2 --guidance-scale 50 --execution-steps "$execute")
  if [[ "$kind" == guided ]]; then opts+=(--socket "$socket" --reference-interpolation 1); fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
    PYTHONPATH=/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib \
    /home/carus/miniforge3/envs/decv2/bin/python "$base/compare_corrected_rollouts.py" "${opts[@]}" \
    > "$folder/sim.log" 2>&1
}
run_guided() {
  local execute=$1 folder="$out/insert1_scale50_guide2_exec${1}"
  local socket="/tmp/object53_full_exec${execute}_$$.sock"
  local server_pid=
  cleanup() {
    if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
    rm -f "$socket"
  }
  trap cleanup EXIT
  mkdir -p "$folder"
  if [[ -f "$folder/summary.json" ]]; then echo "already complete: $folder"; trap - EXIT; return; fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/dp/lib \
    /home/carus/miniforge3/envs/dp/bin/python "$base/server.py" \
    --socket "$socket" --log "$folder/predictions.json" \
    --checkpoint /home/carus/data_usb/10B_obs_4-66.ckpt \
    --reference "$reference" \
    --guidance-steps 2 --guidance-scale 50 --execution-steps "$execute" --reference-interpolation 1 \
    > "$folder/server.log" 2>&1 &
  server_pid=$!
  local ready=0
  for attempt in $(seq 1 200); do
    if [[ -S "$socket" ]]; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep .1
  done
  if [[ "$ready" != 1 ]]; then echo 'server not ready' >&2; exit 1; fi
  run_sim guided "$folder" "$execute" "$socket"
  wait "$server_pid"
  server_pid=
  trap - EXIT
  echo "complete: $folder"
}
case "$mode" in
  all)
    run_sim direct "$out/direct_original" 2
    run_guided 2
    run_guided 1
    ;;
  direct) run_sim direct "$out/direct_original" 2 ;;
  exec2) run_guided 2 ;;
  exec1) run_guided 1 ;;
  *) echo "usage: $0 [all|direct|exec2|exec1]" >&2; exit 2 ;;
esac
