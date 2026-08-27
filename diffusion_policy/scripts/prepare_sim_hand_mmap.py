from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from diffusion_policy.dataset.sim_hand_mmap_converter import (
    build_sim_hand_mmap_cache,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert Sim-Hand HDF5 rollouts into a memory-mapped cache.",
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--pad-before", type=int, default=3)
    parser.add_argument("--pad-after", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=262144)
    args = parser.parse_args(list(argv) if argv is not None else None)

    cache_dir = build_sim_hand_mmap_cache(
        args.dataset_path,
        output_path=args.output_path,
        horizon=args.horizon,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        val_ratio=args.val_ratio,
        seed=args.seed,
        chunk_rows=args.chunk_rows,
    )
    print(f"Sim-Hand mmap cache ready: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
