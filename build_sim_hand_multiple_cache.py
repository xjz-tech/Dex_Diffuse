#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or inspect the Sim-Hand multiple disk cache."
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Report discovered sources without building caches.",
    )
    args = parser.parse_args(argv)

    from diffusion_policy.dataset.sim_hand_multiple_cache import (
        ensure_sim_hand_multiple_cache,
        preflight_sim_hand_multiple_cache,
    )

    preflight = preflight_sim_hand_multiple_cache(args.dataset_path, args.cache_path)
    payload = {
        "dataset_path": preflight.dataset_path,
        "cache_path": preflight.cache_path,
        "n_sources": len(preflight.sources),
        "declared_transitions": preflight.declared_transitions,
        "estimated_cache_bytes": preflight.estimated_cache_bytes,
        "available_bytes": preflight.available_bytes,
        "ready_sources": preflight.ready_sources,
        "missing_sources": preflight.missing_sources,
        "sources": [
            {
                "relative_path": source.relative_path,
                "manifest_path": source.manifest_path,
                "declared_transitions": source.declared_transitions,
                "ready": source.ready,
            }
            for source in preflight.sources
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.preflight:
        return 0
    ensure_sim_hand_multiple_cache(
        args.dataset_path,
        args.cache_path,
        rebuild=args.rebuild,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
