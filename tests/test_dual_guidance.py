"""Two references must contribute independent gradients without changing legacy DDIM."""

import sys
from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_policy.guidance.guided_ddim import guided_ddim_step, sample_guided_trajectory


def test_independent_guidance_weights_and_windows():
    scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=True)
    scheduler.set_timesteps(8)
    sample = torch.linspace(-2, 2, 12 * 22).reshape(1, 12, 22)
    dp_ref = torch.ones(1, 9, 22) * 0.2
    weak_ref = torch.ones(1, 2, 22) * -0.4

    def step(dp_scale, weak_scale):
        return guided_ddim_step(
            scheduler, torch.zeros_like(sample), scheduler.timesteps[0],
            sample.clone().requires_grad_(True), dp_ref, dp_scale, slice(3, 12),
            extra_guidance=((weak_ref, slice(3, 5), weak_scale),),
        ).prev_sample

    base = step(0, 0)
    dp_only = step(100, 0)
    weak_only = step(0, 25)
    both = step(100, 25)
    torch.testing.assert_close(both - base, (dp_only - base) + (weak_only - base))
    torch.testing.assert_close(weak_only[:, 5:], base[:, 5:], rtol=0, atol=0)
    assert not torch.allclose(weak_only[:, 3:5], base[:, 3:5])


def test_disabled_extra_guidance_preserves_legacy_sampling():
    def sample(extras):
        return sample_guided_trajectory(
            model=lambda x, t, global_cond: torch.zeros_like(x),
            scheduler=DDIMScheduler(num_train_timesteps=100),
            initial_noise=torch.ones(1, 12, 22),
            global_cond=torch.zeros(1, 264),
            reference=torch.zeros(1, 9, 22),
            num_inference_steps=4, guidance_scale=100, guidance_slice=slice(3, 12),
            extra_guidance=extras,
        ).trajectory

    torch.testing.assert_close(
        sample(()), sample(((torch.ones(1, 2, 22), slice(3, 5), 0),)), rtol=0, atol=0,
    )


def test_weak_only_sampling_under_inference_mode():
    with torch.inference_mode():
        result = sample_guided_trajectory(
            model=lambda x, t, global_cond: torch.zeros_like(x),
            scheduler=DDIMScheduler(num_train_timesteps=100),
            initial_noise=torch.ones(1, 12, 22), global_cond=torch.zeros(1, 264),
            reference=torch.zeros(1, 9, 22), num_inference_steps=4,
            guidance_scale=0, guidance_slice=slice(3, 12),
            extra_guidance=((torch.zeros(1, 2, 22), slice(3, 5), 25),),
        )
    assert torch.isfinite(result.trajectory).all()
    assert len(result.steps) == 4
