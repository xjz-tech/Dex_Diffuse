"""Parity and invariants for the production reference edit sampler."""

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler

from diffusion_policy.SDEdit.reference_edit import (
    ddim_transition,
    sample_reference_edit,
    select_edit_timesteps,
)
from diffusion_policy.SDEdit.reference_action_editor import ReferenceActionEditor


def test_explicit_ddim_transition_matches_pinned_scheduler():
    scheduler = DDIMScheduler(
        num_train_timesteps=100, prediction_type="epsilon", clip_sample=True
    )
    scheduler.set_timesteps(4)
    generator = torch.Generator().manual_seed(17)
    for index, timestep in enumerate(scheduler.timesteps):
        sample = torch.randn((2, 12, 22), generator=generator)
        epsilon = torch.randn((2, 12, 22), generator=generator)
        alpha_t = scheduler.alphas_cumprod[timestep]
        next_t = int(scheduler.timesteps[index + 1]) if index + 1 < 4 else -1
        alpha_next = (
            scheduler.alphas_cumprod[next_t]
            if next_t >= 0 else scheduler.final_alpha_cumprod
        )
        actual, _ = ddim_transition(sample, epsilon, alpha_t, alpha_next)
        expected = scheduler.step(
            epsilon, timestep, sample, eta=0.0,
            use_clipped_model_output=True,
        ).prev_sample
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_oracle_epsilon_recovers_reference_on_nonuniform_schedule():
    scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=False)
    times, actual_ratio = select_edit_timesteps(
        scheduler.alphas_cumprod, 0.2, 4
    )
    assert len(times) == 4 and times[-1] == 0 and actual_ratio > 0
    generator = torch.Generator().manual_seed(23)
    clean = torch.rand((2, 12, 22), generator=generator) - 0.5
    noise = torch.randn((2, 12, 22), generator=generator)

    def oracle(sample, timestep, global_cond):
        alpha = scheduler.alphas_cumprod[int(timestep)]
        return (sample - alpha.sqrt() * clean) / (1 - alpha).sqrt()

    output = sample_reference_edit(
        oracle, scheduler.alphas_cumprod, clean, noise,
        torch.zeros((2, 1)), times, known_history_steps=3,
        clip_sample=False,
    )
    torch.testing.assert_close(output.trajectory, clean, rtol=1e-5, atol=1e-6)
    assert output.history_mask_max_error == 0.0
    assert output.predicted_x0_clip_fraction == 0.0


@pytest.mark.parametrize("horizon", [8, 12])
def test_zero_edit_returns_exact_reference_without_running_model(horizon):
    editor = object.__new__(ReferenceActionEditor)
    editor.noise_ratio = 0.0
    editor.spec = {
        "n_obs_steps": 4, "obs_dim": 66,
        "horizon": horizon, "n_pred_action_steps": horizon - 3,
    }
    editor.future_steps = horizon - 3
    editor.execution_steps = 2
    future = np.arange(2 * (horizon - 3) * 22, dtype=np.float32).reshape(
        2, horizon - 3, 22
    )
    history = np.zeros((2, 4, 66), dtype=np.float32)
    short, stats = editor.predict(history, future, [41, 42])
    full, _ = editor.predict(history, future, [41, 42], return_full_plan=True)
    np.testing.assert_array_equal(short, future[:, :2])
    np.testing.assert_array_equal(full, future)
    assert stats["zero_edit_exact"]
