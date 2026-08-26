from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from diffusion_policy.guidance.guided_ddim import (
    ZeroGuidanceReport,
    create_ddim_scheduler,
    sample_guided_trajectory,
    verify_zero_guidance_equivalence,
)
from diffusion_policy.guidance.sim_adapter import SimPolicyAdapter

NoiseFactory = Callable[
    [tuple[int, ...], torch.device, torch.dtype, torch.Generator | None],
    torch.Tensor,
]


@dataclass(frozen=True)
class SimHandGuidanceConfig:
    execution_steps: int = 5
    guidance_scale: float = 1.0
    num_inference_steps: int = 12
    eta: float = 0.0


def default_noise_factory(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    # torch.randn requires generator.device == sample device. Keep CPU-seeded
    # generators valid for CUDA targets by drawing on the generator device,
    # then moving (also keeps seeds reproducible across CPU/CUDA).
    if generator is None:
        return torch.randn(shape, device=device, dtype=dtype)
    noise = torch.randn(
        shape,
        device=generator.device,
        dtype=torch.float32,
        generator=generator,
    )
    return noise.to(device=device, dtype=dtype)


class SimHandGuidance:
    def __init__(
        self,
        adapter: SimPolicyAdapter,
        config: SimHandGuidanceConfig,
        noise_factory: NoiseFactory | None = None,
    ) -> None:
        if config.guidance_scale < 0:
            raise ValueError(
                f"guidance_scale must be >= 0, got {config.guidance_scale}"
            )
        if config.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {config.num_inference_steps}"
            )
        if config.eta != 0.0:
            raise ValueError(f"eta must be 0.0, got {config.eta}")

        self.adapter = adapter
        self.config = config
        self.noise_factory: NoiseFactory = (
            noise_factory if noise_factory is not None else default_noise_factory
        )
        self.guidance_slice = adapter.guided_slice(config.execution_steps)
        self.trajectory_shape = (1, adapter.horizon, adapter.action_dim)

    def guide_segment(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        adapter = self.adapter
        config = self.config
        global_cond = adapter.global_condition(hand_state_history)
        reference_norm = adapter.normalize_reference(hand_reference)
        noise = self.noise_factory(
            self.trajectory_shape,
            global_cond.device,
            global_cond.dtype,
            generator,
        )
        scheduler = create_ddim_scheduler(adapter.policy.noise_scheduler)
        sample = sample_guided_trajectory(
            model=adapter.predict_epsilon,
            scheduler=scheduler,
            initial_noise=noise,
            global_cond=global_cond,
            reference=reference_norm,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            guidance_slice=self.guidance_slice,
            eta=config.eta,
        )
        guided_norm = sample.trajectory[:, self.guidance_slice, :]
        return adapter.unnormalize_action(guided_norm)

    def verify_zero_guidance(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
    ) -> ZeroGuidanceReport:
        adapter = self.adapter
        global_cond = adapter.global_condition(hand_state_history)
        reference_norm = adapter.normalize_reference(hand_reference)
        return verify_zero_guidance_equivalence(
            model=adapter.predict_epsilon,
            training_scheduler=adapter.policy.noise_scheduler,
            initial_noise=initial_noise,
            global_cond=global_cond,
            reference=reference_norm,
            guidance_slice=self.guidance_slice,
            num_inference_steps=self.config.num_inference_steps,
        )
