"""Compose check and fake dry-run services for guided Sim hand inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from diffusion_policy.guidance.checkpoint_loader import (
    LoadedPolicy,
    load_workspace_policy,
)
from diffusion_policy.guidance.coordinator import GuidedCoordinator, GuidedRunResult
from diffusion_policy.guidance.executor import FakeSegmentExecutor
from diffusion_policy.guidance.real_adapter import RealPolicyAdapter
from diffusion_policy.guidance.sim_adapter import SimPolicyAdapter
from diffusion_policy.guidance.sim_hand_guidance import (
    SimHandGuidance,
    SimHandGuidanceConfig,
)

HAND_SLICE = slice(9, 31)
HAND_DIM = 22


@dataclass(frozen=True)
class LoadedGuidedPolicies:
    real: LoadedPolicy
    sim: LoadedPolicy
    real_adapter: RealPolicyAdapter
    sim_adapter: SimPolicyAdapter
    guidance: SimHandGuidance


@dataclass(frozen=True)
class CheckReport:
    real_action_proposal: torch.Tensor
    real_action_shape: tuple[int, ...]
    guided_hand_shape: tuple[int, ...]
    segment_count: int
    sim_horizon: int
    sim_obs_steps: int
    sim_pred_action_steps: int
    guided_slice: tuple[int, int]
    timesteps: tuple[int, ...]
    max_x0_error: float
    max_prev_error: float


@dataclass(frozen=True)
class DryRunReport:
    check: CheckReport
    guided: GuidedRunResult


def load_guided_policies(
    real_checkpoint: Path,
    sim_checkpoint: Path,
    device: torch.device,
    guidance_config: SimHandGuidanceConfig,
) -> LoadedGuidedPolicies:
    real = load_workspace_policy(real_checkpoint, device)
    sim = load_workspace_policy(sim_checkpoint, device)
    real_adapter = RealPolicyAdapter(real)
    sim_adapter = SimPolicyAdapter(sim)
    guidance = SimHandGuidance(sim_adapter, guidance_config)
    return LoadedGuidedPolicies(
        real=real,
        sim=sim,
        real_adapter=real_adapter,
        sim_adapter=sim_adapter,
        guidance=guidance,
    )


def _policy_device_dtype(adapter: SimPolicyAdapter) -> tuple[torch.device, torch.dtype]:
    policy = adapter.policy
    device = getattr(policy, "device", torch.device("cpu"))
    dtype = getattr(policy, "dtype", torch.float32)
    return device, dtype


def _seeded_history(
    adapter: SimPolicyAdapter,
    *,
    seed: int,
) -> torch.Tensor:
    device, dtype = _policy_device_dtype(adapter)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = torch.randn(
        (1, int(adapter.n_obs_steps), HAND_DIM),
        generator=generator,
        dtype=torch.float32,
    )
    return history.to(device=device, dtype=dtype)


def _seeded_trajectory_noise(
    adapter: SimPolicyAdapter,
    *,
    seed: int,
) -> torch.Tensor:
    device, dtype = _policy_device_dtype(adapter)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, int(adapter.horizon), int(adapter.action_dim)),
        generator=generator,
        dtype=torch.float32,
    )
    return noise.to(device=device, dtype=dtype)


def run_check(loaded: LoadedGuidedPolicies, *, seed: int) -> CheckReport:
    proposal = loaded.real_adapter.predict_proposal()
    execution_steps = int(loaded.guidance.config.execution_steps)
    total_steps = int(proposal.shape[1])
    if total_steps % execution_steps != 0:
        raise ValueError(
            f"Real proposal length {total_steps} is not divisible by "
            f"execution_steps {execution_steps}"
        )
    segment_count = total_steps // execution_steps

    adapter = loaded.sim_adapter
    device, dtype = _policy_device_dtype(adapter)
    history = _seeded_history(adapter, seed=seed)
    reference = proposal[:, :execution_steps, HAND_SLICE].to(
        device=device, dtype=dtype
    )

    guide_generator = torch.Generator(device="cpu").manual_seed(seed)
    guided_hand = loaded.guidance.guide_segment(
        history,
        reference,
        generator=guide_generator,
    )

    # Separate seeded full-trajectory noise for the zero-guidance oracle.
    initial_noise = _seeded_trajectory_noise(adapter, seed=seed)
    oracle = loaded.guidance.verify_zero_guidance(
        history,
        reference,
        initial_noise=initial_noise,
    )

    guided_slice = adapter.guided_slice(execution_steps)
    return CheckReport(
        real_action_proposal=proposal,
        real_action_shape=tuple(int(v) for v in proposal.shape),
        guided_hand_shape=tuple(int(v) for v in guided_hand.shape),
        segment_count=segment_count,
        sim_horizon=int(adapter.horizon),
        sim_obs_steps=int(adapter.n_obs_steps),
        sim_pred_action_steps=int(adapter.n_pred_action_steps),
        guided_slice=(int(guided_slice.start), int(guided_slice.stop)),
        timesteps=tuple(int(t) for t in oracle.timesteps),
        max_x0_error=float(oracle.max_x0_error),
        max_prev_error=float(oracle.max_prev_error),
    )


def run_dry_run(loaded: LoadedGuidedPolicies, *, seed: int) -> DryRunReport:
    check = run_check(loaded, seed=seed)
    history = _seeded_history(loaded.sim_adapter, seed=seed)
    executor = FakeSegmentExecutor(history)
    coordinator = GuidedCoordinator(loaded.guidance)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    guided = coordinator.run(
        check.real_action_proposal,
        executor,
        generator=generator,
    )
    return DryRunReport(check=check, guided=guided)
