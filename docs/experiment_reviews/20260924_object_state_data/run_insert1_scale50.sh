#!/usr/bin/env bash
set -euo pipefail
base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
inserted=${1:-1}
if [[ "$inserted" == 0 ]]; then
  root="$base/corrected_direct_vs_reference/nointerp_scale50"
elif [[ "$inserted" == 1 ]]; then
  root="$base/corrected_direct_vs_reference/interpolation_1_to_5"
else
  echo 'this runner accepts interpolation 0 or 1' >&2
  exit 2
fi
server_pid=
socket=
cleanup() {
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
  if [[ -n "$socket" ]]; then rm -f "$socket"; fi
}
trap cleanup EXIT
for pair in 2:2 9:2 9:4; do
  guide=${pair%%:*}
  execute=${pair##*:}
  if [[ "$inserted" == 0 ]]; then
    out="$root/guide${guide}_exec${execute}"
  else
    out="$root/insert1_guide${guide}_exec${execute}_scale50"
  fi
  mkdir -p "$out"
  if [[ -f "$out/summary.json" ]]; then echo "already complete: $pair"; continue; fi
  socket="/tmp/object53_interp${inserted}_scale50_${guide}_${execute}_$$.sock"
  echo "starting interpolation=$inserted guide=$guide exec=$execute scale=50"
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/dp/lib \
    /home/carus/miniforge3/envs/dp/bin/python "$base/server.py" \
    --socket "$socket" --log "$out/predictions.json" \
    --checkpoint /home/carus/data_usb/10B_obs_4-66.ckpt \
    --reference "$base/reference/reference.npz" \
    --guidance-steps "$guide" --guidance-scale 50 --execution-steps "$execute" \
    --reference-interpolation "$inserted" > "$out/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for attempt in $(seq 1 200); do
    if [[ -S "$socket" ]]; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep .1
  done
  if [[ "$ready" != 1 ]]; then echo "server not ready: $pair" >&2; exit 1; fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
    PYTHONPATH=/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib \
    /home/carus/miniforge3/envs/decv2/bin/python "$base/compare_corrected_rollouts.py" \
    --mode guided --out "$out" --socket "$socket" --front-only \
    --guidance-steps "$guide" --guidance-scale 50 --execution-steps "$execute" \
    --reference-interpolation "$inserted" > "$out/sim.log" 2>&1
  wait "$server_pid"
  server_pid=
  socket=
  echo "complete interpolation=$inserted guide=$guide exec=$execute scale=50"
done
