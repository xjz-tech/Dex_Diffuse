#!/usr/bin/env python3
"""Real visual DP hand-window editing with the formal SDEdit controller."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_dp_controller import main as run_real_dp_controller


if __name__ == "__main__":
    raise SystemExit(run_real_dp_controller(method="edit"))
