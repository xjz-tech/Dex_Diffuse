#!/usr/bin/env bash
set -euo pipefail
base=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_object_state_data
root="$base/corrected_direct_vs_reference/interpolation_1_to_5"
execute=${1:-2}
if [[ "$execute" != 2 && "$execute" != 4 ]]; then
  echo 'execution steps must be 2 or 4' >&2
  exit 2
fi
server_pid=
socket=
cleanup() {
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
  if [[ -n "$socket" ]]; then rm -f "$socket"; fi
}
trap cleanup EXIT
for scale in 25 50; do
  out="$root/insert1_guide4_exec${execute}_scale${scale}"
  mkdir -p "$out"
  if [[ -f "$out/summary.json" ]]; then echo "already complete scale=$scale"; continue; fi
  socket="/tmp/object53_insert1_guide4_exec${execute}_scale${scale}_$$.sock"
  echo "starting insert1 guide4 exec${execute} scale=$scale"
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/dp/lib \
    /home/carus/miniforge3/envs/dp/bin/python "$base/server.py" \
    --socket "$socket" --log "$out/predictions.json" \
    --checkpoint /home/carus/data_usb/10B_obs_4-66.ckpt \
    --reference "$base/reference/reference.npz" \
    --guidance-steps 4 --guidance-scale "$scale" --execution-steps "$execute" \
    --reference-interpolation 1 > "$out/server.log" 2>&1 &
  server_pid=$!
  ready=0
  for attempt in $(seq 1 200); do
    if [[ -S "$socket" ]]; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then break; fi
    sleep .1
  done
  if [[ "$ready" != 1 ]]; then echo "server not ready scale=$scale" >&2; exit 1; fi
  env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
    PYTHONPATH=/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval \
    LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib \
    /home/carus/miniforge3/envs/decv2/bin/python "$base/compare_corrected_rollouts.py" \
    --mode guided --out "$out" --socket "$socket" --front-only \
    --guidance-steps 4 --guidance-scale "$scale" --execution-steps "$execute" \
    --reference-interpolation 1 > "$out/sim.log" 2>&1
  wait "$server_pid"
  server_pid=
  socket=
  echo "complete insert1 guide4 exec${execute} scale=$scale"
done
