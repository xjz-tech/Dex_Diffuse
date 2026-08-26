"""Hardware-free CLI for Sim-hand guided check and fake dry-run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

from diffusion_policy.guidance.runtime import (
    CheckReport,
    DryRunReport,
    load_guided_policies,
    run_check,
    run_dry_run,
)
from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidanceConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sim-hand guided inference check / fake dry-run (no hardware).",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("check", "dry-run"),
        help="check: one Real proposal + guide + zero oracle; dry-run: fake closed loop",
    )
    parser.add_argument(
        "--real-checkpoint",
        required=True,
        type=Path,
        help="Path to Real policy checkpoint",
    )
    parser.add_argument(
        "--sim-checkpoint",
        required=True,
        type=Path,
        help="Path to Sim-hand policy checkpoint",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Torch device, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--execution-steps",
        type=int,
        default=5,
        help="Hand reference / execution horizon per segment (default: 5)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale (>=0; zero allowed) (default: 1.0)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=12,
        help="DDIM inference steps (default: 12)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="DDIM eta (must be 0.0) (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for synthetic history / noise (default: 0)",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.execution_steps <= 0:
        raise ValueError(
            f"execution_steps must be positive, got {args.execution_steps}"
        )
    if args.guidance_scale < 0:
        raise ValueError(
            f"guidance_scale must be >= 0, got {args.guidance_scale}"
        )
    if args.num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {args.num_inference_steps}"
        )
    if args.eta != 0.0:
        raise ValueError(f"eta must be 0.0, got {args.eta}")

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available for device {args.device}")
        index = 0 if device.index is None else int(device.index)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable "
                f"(device_count={torch.cuda.device_count()})"
            )


def _print_check_fields(report: CheckReport) -> None:
    print(f"real_action_shape={report.real_action_shape}")
    print(f"segment_count={report.segment_count}")
    print(f"sim_horizon={report.sim_horizon}")
    print(f"sim_obs_steps={report.sim_obs_steps}")
    print(f"sim_pred_action_steps={report.sim_pred_action_steps}")
    print(f"guided_slice={report.guided_slice}")
    print(f"timesteps={report.timesteps}")
    print(f"max_x0_error={report.max_x0_error}")
    print(f"max_prev_error={report.max_prev_error}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        _validate_args(args)
        device = torch.device(args.device)
        guidance_config = SimHandGuidanceConfig(
            execution_steps=args.execution_steps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            eta=args.eta,
        )
        loaded = load_guided_policies(
            args.real_checkpoint,
            args.sim_checkpoint,
            device,
            guidance_config,
        )
        if args.mode == "check":
            report: CheckReport | DryRunReport = run_check(loaded, seed=args.seed)
            _print_check_fields(report)
        else:
            dry: DryRunReport = run_dry_run(loaded, seed=args.seed)
            _print_check_fields(dry.check)
            print(f"completed_segments={len(dry.guided.records)}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
