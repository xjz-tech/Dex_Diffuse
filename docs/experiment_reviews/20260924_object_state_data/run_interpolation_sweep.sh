#!/usr/bin/env bash
set -euo pipefail

base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
start=${1:-1}
stop=${2:-5}
results="$base/corrected_direct_vs_reference/interpolation_${start}_to_${stop}"
checkpoint=/home/carus/data_usb/10B_obs_4-66.ckpt
front_flag=()
if (( start >= 6 )); then front_flag=(--front-only); fi
server_pid=
socket=
cleanup() {
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
  if [[ -n "$socket" ]]; then rm -f "$socket"; fi
}
trap cleanup EXIT
mkdir -p "$results"

for inserted in $(seq "$start" "$stop"); do
  for execute in 2 4; do
    out="$results/insert${inserted}_exec${execute}"
    mkdir -p "$out"
    if [[ -f "$out/summary.json" ]]; then
      echo "already complete: insert=$inserted exec=$execute"
      continue
    fi
    socket="/tmp/object53_interp_${inserted}_${execute}_$$.sock"
    echo "starting insert=$inserted exec=$execute"
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      LD_LIBRARY_PATH=/home/carus/miniforge3/envs/dp/lib \
      /home/carus/miniforge3/envs/dp/bin/python "$base/server.py" \
      --socket "$socket" --log "$out/predictions.json" \
      --checkpoint "$checkpoint" --reference "$base/reference/reference.npz" \
      --guidance-steps 9 --execution-steps "$execute" \
      --reference-interpolation "$inserted" > "$out/server.log" 2>&1 &
    server_pid=$!
    ready=0
    for attempt in $(seq 1 200); do
      if [[ -S "$socket" ]]; then ready=1; break; fi
      if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
      sleep .1
    done
    if [[ "$ready" != 1 ]]; then
      echo "server not ready: insert=$inserted exec=$execute" >&2
      exit 1
    fi
    env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
      PYTHONPATH=/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval \
      LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib \
      /home/carus/miniforge3/envs/decv2/bin/python "$base/compare_corrected_rollouts.py" \
      --mode guided --out "$out" --socket "$socket" \
      --guidance-steps 9 --execution-steps "$execute" \
      --reference-interpolation "$inserted" "${front_flag[@]}" > "$out/sim.log" 2>&1
    wait "$server_pid"
    server_pid=
    socket=
    echo "complete insert=$inserted exec=$execute"
  done
done
