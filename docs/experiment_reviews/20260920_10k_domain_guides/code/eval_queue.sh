#!/usr/bin/env bash
set -euo pipefail
R="$(cd "$(dirname "$0")/.." && pwd)"
for method in ordinary_1b old10k light_low light_high heavy_low heavy_high balanced; do
  if [[ ! -f "$R/evaluation/$method/results.json" ]]; then
    echo "START $method $(date -Iseconds)"
    bash "$R/code/eval_one.sh" "$method"
  fi
  /home/carus/miniforge3/envs/dp/bin/python "$R/code/report_results.py" > "$R/report_progress.log" 2>&1
  echo "COMPLETE $method $(date -Iseconds)"
done
/home/carus/miniforge3/envs/dp/bin/python "$R/code/videos.py" > "$R/videos.log" 2>&1
echo "ALL COMPLETE $(date -Iseconds)"
