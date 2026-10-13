#!/usr/bin/env bash
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$1"
OUT="$R/evaluation/$NAME"
mkdir -p "$OUT"
test ! -e "$OUT/results.json"
if [[ "$NAME" == ordinary_10b ]]; then
  ARM=0
  export GUIDE_CHECKPOINT=/home/carus/data_usb/obs_4-66.ckpt
elif [[ "$NAME" == guide_1b ]]; then
  ARM=2
  export GUIDE_CHECKPOINT=/home/carus/data_usb/obs_4-66.ckpt
elif [[ "$NAME" == guide_old10k ]]; then
  ARM=2
  export GUIDE_CHECKPOINT=/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/runs/sim_hand_10k_seed42/checkpoints/latest.ckpt
else
  echo "unknown method $NAME" >&2
  exit 2
fi
D="$(mktemp -d /tmp/10b-guide.XXXXXXXX)"
export AUDIT_SOCKET="$D/p.sock"
export PYTHONDONTWRITEBYTECODE=1
/home/carus/miniforge3/envs/dp/bin/python -u "$R/code/server.py" "$OUT" > "$OUT/server.log" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; rm -f "$AUDIT_SOCKET"; rmdir "$D"' EXIT
for attempt in $(seq 1 240); do
 [[ -S "$AUDIT_SOCKET" ]] && break
 kill -0 "$PID" || { cat "$OUT/server.log"; exit 1; }
 sleep .5
done
env PYTHONPATH=/home/carus/opt/isaacgym/python LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH /home/carus/miniforge3/envs/decv2/bin/python -u "$R/code/eval.py" "$OUT" "$ARM" > "$OUT/sim.log" 2>&1
wait "$PID"
