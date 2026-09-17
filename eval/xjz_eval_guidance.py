#!/usr/bin/env python3
"""XJZ evaluation server: guide a weak prior with a stronger checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch


EVAL_DIR = Path(__file__).resolve().parent
DEX_ROOT = EVAL_DIR.parent
for import_path in (EVAL_DIR, DEX_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from checkpoint_loader import (
    build_policy,
    configure_policy_sampler,
    load_checkpoint,
)
from inference_dp_controller import GuidedDDIMController
from model_server import _device_or_raise, _normalizer_summary, _serve, _warm_up


SPEC_KEYS = (
    "obs_dim",
    "action_dim",
    "n_obs_steps",
    "n_pred_action_steps",
    "horizon",
)


def validate_compatible_specs(weak: dict, strong: dict) -> None:
    for key in SPEC_KEYS:
        if int(weak[key]) != int(strong[key]):
            raise ValueError(
                f"weak and strong checkpoints disagree on {key}: "
                f"{weak[key]} != {strong[key]}"
            )


def extract_strong_reference(
    policy,
    observation: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    action_start: int,
    reference_steps: int,
) -> np.ndarray:
    tensor = torch.as_tensor(observation, device=device, dtype=torch.float32)
    with torch.inference_mode():
        result = policy.predict_action({"obs": tensor})
    if "action_pred" not in result:
        raise RuntimeError(
            "strong guide policy must return full action_pred, not only its "
            "execution prefix"
        )
    full_prediction = result["action_pred"]
    stop = int(action_start) + int(reference_steps)
    if full_prediction.ndim != 3 or full_prediction.shape[1] < stop:
        raise RuntimeError(
            "strong guide full action_pred is shorter than the guidance window"
        )
    reference = full_prediction[:, int(action_start):stop]
    if reference.shape[-1] != 22 or not torch.isfinite(reference).all():
        raise RuntimeError("strong guide returned an invalid hand trajectory")
    return reference.detach().to(device="cpu", dtype=torch.float32).numpy()


class StrongGuidedWeakPolicy:
    def __init__(self, strong_policy, controller: GuidedDDIMController, device):
        self.strong_policy = strong_policy
        self.controller = controller
        self.device = device
        self.n_obs_steps = int(controller.spec["n_obs_steps"])
        self.obs_dim = int(controller.spec["obs_dim"])
        self.action_dim = int(controller.spec["action_dim"])
        self.n_action_steps = int(controller.execution_steps)
        self.normalizer = controller.policy.normalizer
        self._calls = 0

    def predict_action(self, obs_dict):
        observation = obs_dict["obs"]
        reference = extract_strong_reference(
            self.strong_policy,
            observation,
            device=self.device,
            action_start=self.controller.action_start,
            reference_steps=self.controller.reference_steps,
        )
        action, stats = self.controller.predict(
            observation.detach().to(device="cpu", dtype=torch.float32).numpy(),
            reference,
        )
        self._calls += 1
        if self._calls == 1 or self._calls % 100 == 0:
            print(
                "[guidance] call=%d scale=%g mse_before=%.6f mse_after=%.6f"
                % (
                    self._calls,
                    self.controller.guidance_scale,
                    stats.mse_before,
                    stats.mse_after,
                ),
                flush=True,
            )
        return {
            "action": torch.from_numpy(action).to(
                device=self.device, dtype=torch.float32
            )
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="weak prior")
    parser.add_argument("--guide-checkpoint", required=True, type=Path)
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", choices=("ddim",), default="ddim")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--guide-inference-steps", type=int, default=8)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=100.0)
    parser.add_argument(
        "--guidance-steps",
        type=int,
        default=int(os.environ.get("GUIDANCE_STEPS", "9")),
        help="Number of leading predicted actions constrained by guidance (default: 9)",
    )
    parser.add_argument("--fixed-noise", type=int, choices=(0, 1), default=1)
    parser.add_argument("--no-salvage", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = _device_or_raise(args.device)
    allow_salvage = not args.no_salvage

    print(f"[weak-prior] loading {args.checkpoint}", flush=True)
    controller = GuidedDDIMController(
        args.checkpoint.expanduser().resolve(),
        device,
        inference_steps=args.inference_steps,
        execution_steps=args.n_action_steps,
        guidance_scale=args.guidance_scale,
        eta=0.0,
        fixed_noise=bool(args.fixed_noise),
        seed=args.seed,
        allow_salvage=allow_salvage,
    )
    controller.set_guidance_horizon(args.guidance_steps)
    print(f"[strong-guide] loading {args.guide_checkpoint}", flush=True)
    strong_loaded = load_checkpoint(
        args.guide_checkpoint.expanduser().resolve(),
        allow_salvage=allow_salvage,
    )
    strong_policy, strong_spec = build_policy(strong_loaded)
    validate_compatible_specs(controller.spec, strong_spec)
    configure_policy_sampler(
        strong_policy, "ddim", inference_steps=args.guide_inference_steps
    )
    strong_policy = strong_policy.to(device).eval()
    for parameter in strong_policy.parameters():
        parameter.requires_grad_(False)

    policy = StrongGuidedWeakPolicy(strong_policy, controller, device)
    summary = _normalizer_summary(policy)
    print(
        "[pipeline] weak=%s strong=%s scale=%g weak_steps=%d strong_steps=%d "
        "guidance_actions=%d"
        % (
            args.checkpoint,
            args.guide_checkpoint,
            args.guidance_scale,
            args.inference_steps,
            args.guide_inference_steps,
            args.guidance_steps,
        ),
        flush=True,
    )
    if not args.no_warmup:
        _warm_up(policy, device, args.seed, summary["obs"]["mean"])
    spec = dict(controller.spec)
    spec["n_action_steps"] = int(args.n_action_steps)
    _serve(
        args,
        policy,
        device,
        controller.checkpoint_info,
        spec,
        summary,
    )


if __name__ == "__main__":
    main()
