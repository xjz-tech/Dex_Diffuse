import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from trt_unet import (  # noqa: E402
    fused_conditional_sample, fused_ddim_coeffs_from_scheduler, fused_ddim_update,
)


def _scheduler():
    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=0,
        timestep_spacing="leading",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(4)
    return scheduler


def test_fused_ddim_first_step_matches_huggingface():
    scheduler = _scheduler()
    sample = torch.randn(2, 12, 22)
    epsilon = torch.randn_like(sample)
    timestep = int(scheduler.timesteps[0])
    official = scheduler.step(epsilon, timestep, sample, eta=0.0).prev_sample
    coeffs = fused_ddim_coeffs_from_scheduler(scheduler, sample.device, sample.dtype)
    fused = fused_ddim_update(sample, epsilon, 0, coeffs)
    assert torch.allclose(official, fused, atol=1e-5, rtol=1e-5)


def test_fused_ddim_last_step_matches_huggingface():
    scheduler = _scheduler()
    sample = torch.randn(1, 12, 22)
    epsilon = torch.randn_like(sample)
    last = len(scheduler.timesteps) - 1
    timestep = int(scheduler.timesteps[last])
    official = scheduler.step(epsilon, timestep, sample, eta=0.0).prev_sample
    coeffs = fused_ddim_coeffs_from_scheduler(scheduler, sample.device, sample.dtype)
    fused = fused_ddim_update(sample, epsilon, last, coeffs)
    assert torch.allclose(official, fused, atol=1e-5, rtol=1e-5)


def test_fused_visual_ddim_matches_full_clipped_trajectory():
    scheduler = DDIMScheduler(
        num_train_timesteps=100, clip_sample=True,
        set_alpha_to_one=True, steps_offset=0, prediction_type="epsilon",
    )

    def model(sample, timestep, local_cond=None, global_cond=None):
        return sample * 0.2 + global_cond[:, None, :]

    policy = SimpleNamespace(
        model=model, noise_scheduler=scheduler, num_inference_steps=4,
        obs_as_global_cond=True,
    )
    condition = torch.zeros(1, 8, 22)
    mask = torch.zeros_like(condition, dtype=torch.bool)
    global_cond = torch.randn(1, 22)
    torch.manual_seed(17)
    fused = fused_conditional_sample(policy, condition, mask, global_cond=global_cond)
    torch.manual_seed(17)
    original = torch.randn(condition.shape)
    scheduler.set_timesteps(4)
    for timestep in scheduler.timesteps:
        epsilon = model(original, timestep, global_cond=global_cond)
        original = scheduler.step(epsilon, timestep, original).prev_sample
    torch.testing.assert_close(fused, original, rtol=1e-5, atol=1e-5)
