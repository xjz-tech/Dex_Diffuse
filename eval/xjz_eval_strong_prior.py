#!/usr/bin/env python3
"""XJZ evaluation server: sample a strong prior, guided by a weaker checkpoint."""

from __future__ import annotations

from pathlib import Path
import sys


EVAL_DIR = Path(__file__).resolve().parent
DEX_ROOT = EVAL_DIR.parent
for import_path in (EVAL_DIR, DEX_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from guided_pair_policy import parse_guided_server_args, run_guided_pair_server  # noqa: E402


def _parse_args():
    return parse_guided_server_args(
        __doc__,
        checkpoint_help="strong prior",
        guide_help="weak (or other-seed) guide checkpoint",
    )


def main() -> None:
    args = _parse_args()
    run_guided_pair_server(
        args,
        prior_label="strong-prior",
        guide_label="weak-guide",
    )


if __name__ == "__main__":
    main()
