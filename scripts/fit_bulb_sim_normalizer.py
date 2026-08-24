#!/usr/bin/env python3
"""Freeze hand-joint min/max from a sim Diffusion Policy zarr."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from diffusion_policy.common.bulb_action_normalizer import collect_hand_joint_stat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True, help="Path to replay_buffer.zarr")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()
    stat = collect_hand_joint_stat(os.path.expanduser(args.zarr), chunk_size=args.chunk_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **stat)
    print(f"wrote {args.output} min={stat['min'][:3]} max={stat['max'][:3]}")


if __name__ == "__main__":
    main()
