import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from checkpoint_loader import configure_policy_sampler  # noqa: E402


def test_configure_policy_sampler_switches_ddpm_to_ddim_with_step_count():
    policy = SimpleNamespace(
        noise_scheduler=DDPMScheduler(num_train_timesteps=100),
        num_inference_steps=100,
    )

    configure_policy_sampler(policy, "ddim", inference_steps=8)

    assert isinstance(policy.noise_scheduler, DDIMScheduler)
    assert policy.num_inference_steps == 8


def test_configure_policy_sampler_keeps_ddpm_when_requested():
    scheduler = DDPMScheduler(num_train_timesteps=100)
    policy = SimpleNamespace(noise_scheduler=scheduler, num_inference_steps=100)

    configure_policy_sampler(policy, "ddpm", inference_steps=16)

    assert policy.noise_scheduler is scheduler
    assert policy.num_inference_steps == 16


def test_configure_policy_sampler_rejects_unknown_name():
    policy = SimpleNamespace(
        noise_scheduler=DDPMScheduler(num_train_timesteps=100),
        num_inference_steps=100,
    )
    with pytest.raises(ValueError, match="sampler"):
        configure_policy_sampler(policy, "heun")
