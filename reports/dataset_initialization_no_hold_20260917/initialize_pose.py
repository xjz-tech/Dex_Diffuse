"""Use an archived dataset pose in this process only; preserve robot_init.py."""
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'eval/real'))
import robot_init

pose = json.loads(Path(os.environ['EXPERIMENT_INITIAL_POSE_JSON']).read_text())
robot_init.HAND_ROTATE_JOINTS = np.asarray(pose['qpos_real_rad'], dtype=np.float64)
print('[experiment] Dataset initialization (ROTATE slot overridden only in this process):', flush=True)
print(json.dumps(pose, ensure_ascii=False), flush=True)
raise SystemExit(robot_init.main())
