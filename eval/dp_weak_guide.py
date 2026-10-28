"""Optional observation-conditioned weak reference for the DP hand controller."""

import math

from checkpoint_loader import build_policy, configure_policy_sampler, load_checkpoint
from guided_pair_policy import (
    bind_conditional_sample_seed,
    extract_guide_reference,
    validate_compatible_specs,
)


class WeakTrajectoryGuide:
    def __init__(self, controller, args):
        self.scale = float(args.weak_guide_scale)
        self.steps = int(args.weak_guide_steps)
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("weak guide scale must be finite and non-negative")
        if not 1 <= self.steps <= controller.max_execution_steps:
            raise ValueError("weak guide window exceeds controller prediction horizon")
        if args.weak_guide_inference_steps <= 0:
            raise ValueError("weak guide inference steps must be positive")
        loaded = load_checkpoint(args.weak_guide_checkpoint, allow_salvage=not args.no_salvage)
        self.policy, spec = build_policy(loaded)
        validate_compatible_specs(controller.spec, spec)
        configure_policy_sampler(self.policy, "ddim", args.weak_guide_inference_steps)
        self.policy.to(controller.device).eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        seed = args.seed if args.weak_guide_seed is None else args.weak_guide_seed
        bind_conditional_sample_seed(
            self.policy, seed=seed, device=controller.device,
            fixed_noise=bool(args.fixed_noise),
        )
        self.device = controller.device
        self.start = controller.action_start
        self.normalizer = controller.policy.normalizer["action"]
        print(f"[weak-guide] {args.weak_guide_checkpoint}; scale={self.scale}; "
              f"actions={self.steps}; ddim_steps={args.weak_guide_inference_steps}; seed={seed}",
              flush=True)

    def __call__(self, history):
        import torch

        # Recompute once per controller call from fresh feedback, then reuse
        # this reference throughout all reverse diffusion steps.
        reference = extract_guide_reference(
            self.policy, history, device=self.device,
            action_start=self.start, reference_steps=self.steps,
        )
        reference = torch.as_tensor(reference, device=self.device, dtype=history.dtype)
        reference = self.normalizer.normalize(reference)
        return ((reference, slice(self.start, self.start + self.steps), self.scale),)
