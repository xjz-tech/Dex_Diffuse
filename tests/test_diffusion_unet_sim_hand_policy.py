from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.diffusion_unet_sim_hand_policy import (
    DiffusionUnetSimHandPolicy,
)


OBS_DIM = 66
HAND_DIM = 22


def _scheduler():
    return DDPMScheduler(
        num_train_timesteps=10,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )


def _normalizer():
    normalizer = LinearNormalizer()
    normalizer["obs"] = SingleFieldLinearNormalizer.create_identity()
    normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    return normalizer


def _make_policy(return_full_prediction=False):
    model = ConditionalUnet1D(
        input_dim=HAND_DIM,
        local_cond_dim=None,
        global_cond_dim=4 * OBS_DIM,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64, 128),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True,
    )
    policy = DiffusionUnetSimHandPolicy(
        model=model,
        noise_scheduler=_scheduler(),
        horizon=12,
        obs_dim=OBS_DIM,
        action_dim=HAND_DIM,
        n_obs_steps=4,
        n_pred_action_steps=9,
        n_action_steps=4,
        num_inference_steps=2,
        obs_as_global_cond=True,
        return_full_prediction=return_full_prediction,
    )
    policy.set_normalizer(_normalizer())
    return policy


def test_predict_action_returns_execution_and_usable_actions_only():
    policy = _make_policy(return_full_prediction=False)
    obs = torch.randn(2, 4, OBS_DIM)

    result = policy.predict_action({"obs": obs})

    assert set(result) == {"action", "action_usable"}
    assert result["action"].shape == (2, 4, HAND_DIM)
    assert result["action_usable"].shape == (2, 9, HAND_DIM)
    torch.testing.assert_close(
        result["action"],
        result["action_usable"][:, :4],
    )
    assert torch.isfinite(result["action"]).all()
    assert torch.isfinite(result["action_usable"]).all()


def test_predict_action_debug_mode_returns_full_oa_trajectory():
    policy = _make_policy(return_full_prediction=True)

    result = policy.predict_action({"obs": torch.randn(1, 4, OBS_DIM)})

    assert set(result) == {"action", "action_usable", "action_pred"}
    assert result["action_pred"].shape == (1, 12, HAND_DIM)
    torch.testing.assert_close(
        result["action_usable"],
        result["action_pred"][:, 3:12],
    )


def test_compute_loss_is_finite_and_backpropagates():
    policy = _make_policy()
    batch = {
        "obs": torch.randn(2, 12, OBS_DIM),
        "action": torch.randn(2, 12, HAND_DIM),
    }

    loss = policy.compute_loss(batch)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in policy.model.parameters())


def test_forward_exposes_training_loss_for_distributed_wrapping():
    policy = _make_policy()
    batch = {
        "obs": torch.randn(2, 12, OBS_DIM),
        "action": torch.randn(2, 12, HAND_DIM),
    }

    loss = policy(batch)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in policy.model.parameters())


class ShortTemporalModel(nn.Module):
    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        return sample[:, :-1]


def test_policy_rejects_unet_temporal_length_mismatch():
    with pytest.raises(ValueError) as exc_info:
        DiffusionUnetSimHandPolicy(
            model=ShortTemporalModel(),
            noise_scheduler=_scheduler(),
            horizon=12,
            obs_dim=OBS_DIM,
            action_dim=HAND_DIM,
            n_obs_steps=4,
            n_pred_action_steps=9,
            n_action_steps=4,
        )

    message = str(exc_info.value)
    assert "U-Net output temporal length" in message
    assert "Observation steps       : 4" in message
    assert "Prediction action steps : 9" in message
    assert "Execution action steps  : 4" in message
    assert "Derived horizon         : 12" in message


def test_unet_temporal_probe_preserves_training_mode():
    policy = _make_policy()

    assert policy.model.training
