"""Shared prior+guide DDIM pairing used by the XJZ eval servers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

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


def validate_compatible_specs(prior: dict, guide: dict) -> None:
    for key in SPEC_KEYS:
        if int(prior[key]) != int(guide[key]):
            raise ValueError(
                f"prior and guide checkpoints disagree on {key}: "
                f"{prior[key]} != {guide[key]}"
            )


def bind_conditional_sample_seed(policy, *, seed: int, device, fixed_noise: bool = True):
    """Make a policy's DDIM start from a dedicated, optionally fixed noise seed."""
    original = policy.conditional_sample
    seed = int(seed)
    rolling_generator = torch.Generator(device=device)
    rolling_generator.manual_seed(seed)

    def conditional_sample(*args, generator=None, **kwargs):
        if fixed_noise:
            sample_gen = torch.Generator(device=device)
            sample_gen.manual_seed(seed)
        else:
            sample_gen = rolling_generator
        kwargs["generator"] = sample_gen
        return original(*args, **kwargs)

    policy.conditional_sample = conditional_sample
    return policy


def extract_guide_reference(
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
            "guide policy must return full action_pred, not only its execution prefix"
        )
    full_prediction = result["action_pred"]
    stop = int(action_start) + int(reference_steps)
    if full_prediction.ndim != 3 or full_prediction.shape[1] < stop:
        raise RuntimeError(
            "guide full action_pred is shorter than the guidance window"
        )
    reference = full_prediction[:, int(action_start):stop]
    if reference.shape[-1] != 22 or not torch.isfinite(reference).all():
        raise RuntimeError("guide returned an invalid hand trajectory")
    return reference.detach().to(device="cpu", dtype=torch.float32).numpy()


extract_strong_reference = extract_guide_reference


class GuidedPairPolicy:
    def __init__(self, guide_policy, controller: GuidedDDIMController, device):
        self.guide_policy = guide_policy
        self.strong_policy = guide_policy
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
        reference = extract_guide_reference(
            self.guide_policy,
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


StrongGuidedWeakPolicy = GuidedPairPolicy


def parse_guided_server_args(
    description: str,
    *,
    checkpoint_help: str,
    guide_help: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", required=True, type=Path, help=checkpoint_help)
    parser.add_argument("--guide-checkpoint", required=True, type=Path, help=guide_help)
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--guide-seed",
        type=int,
        default=None,
        help="DDIM noise seed for the guide policy (default: same as --seed)",
    )
    parser.add_argument("--sampler", choices=("ddim",), default="ddim")
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--guide-inference-steps", type=int, default=8)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--guidance-metric", choices=("joint", "fingertip"),
                        default=os.environ.get("GUIDANCE_METRIC", "joint"))
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
    args = parser.parse_args()
    if args.guide_seed is None:
        args.guide_seed = args.seed
    return args


def run_guided_pair_server(
    args: argparse.Namespace,
    *,
    prior_label: str,
    guide_label: str,
) -> None:
    device = _device_or_raise(args.device)
    allow_salvage = not args.no_salvage

    print(f"[{prior_label}] loading {args.checkpoint}", flush=True)
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
    if getattr(args, "guidance_metric", "joint") == "fingertip":
        from diffusion_policy.guidance.fingertip_fk import FingertipGuidanceLoss
        controller.guidance_loss_fn = FingertipGuidanceLoss(
            controller.policy.normalizer["action"], device=device)
        print("[guidance] metric=fingertip; wrist-frame Cartesian MSE in m^2", flush=True)
    print(f"[{guide_label}] loading {args.guide_checkpoint}", flush=True)
    guide_loaded = load_checkpoint(
        args.guide_checkpoint.expanduser().resolve(),
        allow_salvage=allow_salvage,
    )
    guide_policy, guide_spec = build_policy(guide_loaded)
    validate_compatible_specs(controller.spec, guide_spec)
    configure_policy_sampler(
        guide_policy, "ddim", inference_steps=args.guide_inference_steps
    )
    guide_policy = guide_policy.to(device).eval()
    for parameter in guide_policy.parameters():
        parameter.requires_grad_(False)
    bind_conditional_sample_seed(
        guide_policy,
        seed=args.guide_seed,
        device=device,
        fixed_noise=bool(args.fixed_noise),
    )

    policy = GuidedPairPolicy(guide_policy, controller, device)
    summary = _normalizer_summary(policy)
    print(
        "[pipeline] prior=%s guide=%s scale=%g prior_steps=%d guide_steps=%d "
        "guidance_actions=%d prior_seed=%d guide_seed=%d"
        % (
            args.checkpoint,
            args.guide_checkpoint,
            args.guidance_scale,
            args.inference_steps,
            args.guide_inference_steps,
            args.guidance_steps,
            args.seed,
            args.guide_seed,
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
