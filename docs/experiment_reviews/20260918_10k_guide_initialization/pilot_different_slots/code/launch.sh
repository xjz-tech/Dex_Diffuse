#!/usr/bin/env bash
set -euo pipefail
OUT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
SOCKET_DIR="$(mktemp -d /tmp/1b-init.XXXXXXXX)"
export AUDIT_SOCKET="$SOCKET_DIR/p.sock"
/home/carus/miniforge3/envs/dp/bin/python -u "$OUT/code/server.py" "$OUT" >"$OUT/server.log" 2>&1 &
MODEL_PID=$!
trap 'kill "$MODEL_PID" 2>/dev/null || true; rm -f "$AUDIT_SOCKET"; rmdir "$SOCKET_DIR"' EXIT
for attempt in $(seq 1 240); do
  [[ -S "$AUDIT_SOCKET" ]] && break
  kill -0 "$MODEL_PID" || { cat "$OUT/server.log"; exit 1; }
  sleep .5
done
env PYTHONPATH=/home/carus/opt/isaacgym/python LD_LIBRARY_PATH=/home/carus/miniforge3/envs/decv2/lib:${LD_LIBRARY_PATH:-} PATH=/home/carus/miniforge3/envs/decv2/bin:$PATH \
 /home/carus/miniforge3/envs/decv2/bin/python -u "$OUT/code/run.py" "$OUT" >"$OUT/run.log" 2>&1
wait "$MODEL_PID"
