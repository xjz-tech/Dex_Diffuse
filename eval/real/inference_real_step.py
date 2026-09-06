#!/usr/bin/env python3
"""Run the real SharpA controller with the recovered step_01700000 policy."""

from __future__ import annotations

import inference_real
from policy_loader_step import load_policy


# Leave the complete-checkpoint entry untouched. This dedicated entry replaces
# only its loader with the strict base-model salvage path above.
inference_real.load_policy = load_policy


if __name__ == "__main__":
    try:
        raise SystemExit(inference_real.main())
    except KeyboardInterrupt:
        print("\n[real] stopped by Ctrl-C", flush=True)
        raise SystemExit(130)
