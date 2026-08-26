from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from diffusion_policy.guidance.checkpoint_loader import LoadedPolicy
from diffusion_policy.guidance.guided_ddim import (
    GuidedDDIMSampleOutput,
    ZeroGuidanceReport,
)
from diffusion_policy.guidance.sim_adapter import SimPolicyAdapter
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

HAND_DIM = 22


@dataclass(frozen=True)
class FakeTemporal:
    usable_action_slice: slice


class TrackingSimModel(nn.Module):
    def __init__(self, action_dim: int = HAND_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.calls: list[tuple[torch.Tensor, Any, torch.Tensor]] = []
        self.bias = nn.Parameter(torch.tensor(0.25), requires_grad=False)

    def forward(self, sample, timestep, global_cond=None, local_cond=None):
        self.calls.append((sample, timestep, global_cond))
        return sample + self.bias


class TrackingSimPolicy(nn.Module):
    def __init__(
        self,
        *,
        n_obs_steps: int,
        horizon: int,
        n_pred_action_steps: int,
        usable_start: int,
        obs_dim: int = HAND_DIM,
        action_dim: int = HAND_DIM,
        obs_scale: float = 2.0,
        action_scale: float = 3.0,
        num_train_timesteps: int = 100,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_pred_action_steps = n_pred_action_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.temporal = FakeTemporal(
            usable_action_slice=slice(
                usable_start, usable_start + n_pred_action_steps
            )
        )
        self.model = TrackingSimModel(action_dim=action_dim)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.normalizer = LinearNormalizer()
        self.normalizer["obs"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((obs_dim,), obs_scale, dtype=torch.float32),
            offset=torch.zeros(obs_dim, dtype=torch.float32),
            input_stats_dict={
                "min": torch.full((obs_dim,), -1.0),
                "max": torch.full((obs_dim,), 1.0),
                "mean": torch.zeros(obs_dim),
                "std": torch.ones(obs_dim),
            },
        )
        self.normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((action_dim,), action_scale, dtype=torch.float32),
            offset=torch.zeros(action_dim, dtype=torch.float32),
            input_stats_dict={
                "min": torch.full((action_dim,), -1.0),
                "max": torch.full((action_dim,), 1.0),
                "mean": torch.zeros(action_dim),
                "std": torch.ones(action_dim),
            },
        )
        self.predict_action_calls = 0

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def predict_action(self, obs_dict):
        self.predict_action_calls += 1
        raise AssertionError("SimHandGuidance must never call policy.predict_action")


def make_sim_policy(
    *,
    n_obs_steps: int,
    horizon: int,
    n_pred_action_steps: int,
    usable_start: int,
    obs_dim: int = HAND_DIM,
    action_dim: int = HAND_DIM,
    obs_scale: float = 2.0,
    action_scale: float = 3.0,
    num_train_timesteps: int = 100,
) -> LoadedPolicy:
    policy = TrackingSimPolicy(
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_pred_action_steps=n_pred_action_steps,
        usable_start=usable_start,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_scale=obs_scale,
        action_scale=action_scale,
        num_train_timesteps=num_train_timesteps,
    )
    return LoadedPolicy(
        checkpoint_path=Path("/tmp/fake_sim.ckpt"),
        cfg=OmegaConf.create({}),
        workspace=MagicMock(),
        policy=policy,
        used_ema=False,
    )


def current_adapter() -> SimPolicyAdapter:
    return SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=4,
            horizon=12,
            n_pred_action_steps=9,
            usable_start=3,
        )
    )


def alternate_adapter() -> SimPolicyAdapter:
    return SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=2,
            horizon=8,
            n_pred_action_steps=7,
            usable_start=1,
        )
    )


class RecordingSimAdapter:
    """Thin recording wrapper around a real SimPolicyAdapter."""

    def __init__(self, inner: SimPolicyAdapter):
        self.inner = inner
        self.policy = inner.policy
        self.normalize_history_calls = 0
        self.normalize_reference_calls = 0
        self.unnormalize_action_calls = 0
        self.global_condition_calls = 0
        self.last_unnormalize_input: torch.Tensor | None = None
        self.unnormalize_outputs: list[torch.Tensor] = []

    @property
    def horizon(self) -> int:
        return self.inner.horizon

    @property
    def n_obs_steps(self) -> int:
        return self.inner.n_obs_steps

    @property
    def n_pred_action_steps(self) -> int:
        return self.inner.n_pred_action_steps

    @property
    def action_dim(self) -> int:
        return self.inner.action_dim

    @property
    def oa_start(self) -> int:
        return self.inner.oa_start

    def guided_slice(self, execution_steps: int) -> slice:
        return self.inner.guided_slice(execution_steps)

    def normalize_history(self, value: torch.Tensor) -> torch.Tensor:
        self.normalize_history_calls += 1
        return self.inner.normalize_history(value)

    def normalize_reference(self, value: torch.Tensor) -> torch.Tensor:
        self.normalize_reference_calls += 1
        return self.inner.normalize_reference(value)

    def unnormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        self.unnormalize_action_calls += 1
        self.last_unnormalize_input = value.detach().clone()
        out = self.inner.unnormalize_action(value)
        self.unnormalize_outputs.append(out.detach().clone())
        return out

    def global_condition(self, history: torch.Tensor) -> torch.Tensor:
        self.global_condition_calls += 1
        # Mirror SimPolicyAdapter so normalize_history is recorded once.
        normalized = self.normalize_history(history)
        return normalized[:, : self.n_obs_steps, :].reshape(history.shape[0], -1)

    def predict_epsilon(self, sample, timestep, global_cond):
        return self.inner.predict_epsilon(sample, timestep, global_cond)


def _default_config(**overrides):
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidanceConfig

    return SimHandGuidanceConfig(**overrides)


def test_guidance_derives_slice_and_shape_from_current_adapter():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    guidance = SimHandGuidance(current_adapter(), _default_config())
    assert guidance.guidance_slice == slice(3, 8)
    assert guidance.trajectory_shape == (1, 12, 22)


def test_guidance_derives_slice_and_shape_from_alternate_adapter_without_config_change():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    config = _default_config()
    guidance = SimHandGuidance(alternate_adapter(), config)
    assert guidance.guidance_slice == slice(1, 6)
    assert guidance.trajectory_shape == (1, 8, 22)
    assert config.execution_steps == 5
    assert config.num_inference_steps == 12


def test_guidance_rejects_execution_beyond_usable_prediction():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    with pytest.raises(ValueError, match="n_pred_action_steps|execution_steps"):
        SimHandGuidance(current_adapter(), _default_config(execution_steps=10))


def test_guidance_rejects_slice_beyond_horizon():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    adapter = SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=4,
            horizon=12,
            n_pred_action_steps=9,
            usable_start=4,
        )
    )
    with pytest.raises(ValueError, match="horizon|guided slice"):
        SimHandGuidance(adapter, _default_config(execution_steps=9))


def test_guidance_rejects_negative_scale():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    with pytest.raises(ValueError, match="guidance_scale"):
        SimHandGuidance(current_adapter(), _default_config(guidance_scale=-0.1))


def test_guidance_accepts_zero_guidance_scale():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    guidance = SimHandGuidance(current_adapter(), _default_config(guidance_scale=0.0))
    assert guidance.config.guidance_scale == 0.0


def test_guidance_rejects_nonpositive_inference_steps():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    with pytest.raises(ValueError, match="num_inference_steps"):
        SimHandGuidance(current_adapter(), _default_config(num_inference_steps=0))
    with pytest.raises(ValueError, match="num_inference_steps"):
        SimHandGuidance(current_adapter(), _default_config(num_inference_steps=-3))


def test_guidance_rejects_nonzero_eta():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    with pytest.raises(ValueError, match="eta"):
        SimHandGuidance(current_adapter(), _default_config(eta=0.1))


def test_guide_segment_normalize_sample_unnormalize_pipeline():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    adapter = RecordingSimAdapter(current_adapter())
    config = _default_config(execution_steps=5, num_inference_steps=12, guidance_scale=1.5)
    noise_draws: list[torch.Tensor] = []

    def recording_noise_factory(shape, device, dtype, generator):
        noise = torch.full(shape, 0.5, device=device, dtype=dtype)
        noise_draws.append(noise.detach().clone())
        return noise

    terminal = torch.arange(12 * 22, dtype=torch.float32).reshape(1, 12, 22) / 100.0
    stub_sample = GuidedDDIMSampleOutput(trajectory=terminal, steps=())
    sampler_kwargs: dict[str, Any] = {}

    def fake_sample_guided_trajectory(**kwargs):
        sampler_kwargs.update(kwargs)
        return stub_sample

    with (
        patch(
            "diffusion_policy.guidance.sim_hand_guidance.create_ddim_scheduler",
            return_value=MagicMock(name="ddim"),
        ) as create_sched,
        patch(
            "diffusion_policy.guidance.sim_hand_guidance.sample_guided_trajectory",
            side_effect=fake_sample_guided_trajectory,
        ),
    ):
        guidance = SimHandGuidance(adapter, config, noise_factory=recording_noise_factory)
        history = torch.ones(1, 4, 22)
        reference = torch.full((1, 5, 22), 0.25)
        temporal_before = adapter.policy.temporal
        horizon_before = adapter.policy.horizon
        n_obs_before = adapter.policy.n_obs_steps
        n_pred_before = adapter.policy.n_pred_action_steps

        result = guidance.guide_segment(history, reference)

    assert adapter.global_condition_calls == 1
    assert adapter.normalize_history_calls == 1  # via global_condition
    assert adapter.normalize_reference_calls == 1
    assert len(noise_draws) == 1
    assert noise_draws[0].shape == (1, 12, 22)
    assert sampler_kwargs["guidance_slice"] == slice(3, 8)
    assert sampler_kwargs["num_inference_steps"] == 12
    assert sampler_kwargs["guidance_scale"] == 1.5
    assert sampler_kwargs["eta"] == 0.0
    assert sampler_kwargs["model"] == adapter.predict_epsilon
    create_sched.assert_called_once_with(adapter.policy.noise_scheduler)

    expected_guided = terminal[:, slice(3, 8), :]
    assert adapter.unnormalize_action_calls == 1
    torch.testing.assert_close(adapter.last_unnormalize_input, expected_guided)
    assert result.shape == (1, 5, 22)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, adapter.unnormalize_outputs[0])

    assert adapter.policy.predict_action_calls == 0
    assert adapter.policy.temporal is temporal_before
    assert adapter.policy.temporal.usable_action_slice == temporal_before.usable_action_slice
    assert adapter.policy.horizon == horizon_before
    assert adapter.policy.n_obs_steps == n_obs_before
    assert adapter.policy.n_pred_action_steps == n_pred_before


def test_guide_segment_requests_fresh_noise_per_call_and_reproduces_with_seed():
    from diffusion_policy.guidance.sim_hand_guidance import (
        SimHandGuidance,
        default_noise_factory,
    )

    adapter = current_adapter()
    config = _default_config()
    noise_draws: list[torch.Tensor] = []

    def recording_factory(shape, device, dtype, generator):
        noise = default_noise_factory(shape, device, dtype, generator)
        noise_draws.append(noise.detach().clone())
        return noise

    stub = GuidedDDIMSampleOutput(
        trajectory=torch.zeros(1, 12, 22),
        steps=(),
    )

    with (
        patch(
            "diffusion_policy.guidance.sim_hand_guidance.create_ddim_scheduler",
            return_value=MagicMock(),
        ),
        patch(
            "diffusion_policy.guidance.sim_hand_guidance.sample_guided_trajectory",
            return_value=stub,
        ),
    ):
        guidance = SimHandGuidance(adapter, config, noise_factory=recording_factory)
        history = torch.zeros(1, 4, 22)
        reference = torch.zeros(1, 5, 22)

        gen = torch.Generator().manual_seed(7)
        guidance.guide_segment(history, reference, generator=gen)
        guidance.guide_segment(history, reference, generator=gen)
        assert len(noise_draws) == 2
        assert not torch.equal(noise_draws[0], noise_draws[1])

        first_pair = (noise_draws[0].clone(), noise_draws[1].clone())
        noise_draws.clear()

        gen_a = torch.Generator().manual_seed(7)
        guidance.guide_segment(history, reference, generator=gen_a)
        guidance.guide_segment(history, reference, generator=gen_a)
        assert torch.equal(noise_draws[0], first_pair[0])
        assert torch.equal(noise_draws[1], first_pair[1])


def test_verify_zero_guidance_uses_normalized_inputs_and_configured_steps():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    adapter = RecordingSimAdapter(current_adapter())
    config = _default_config(num_inference_steps=12, guidance_scale=0.0)
    guidance = SimHandGuidance(adapter, config)

    history = torch.ones(1, 4, 22) * 0.5
    reference = torch.ones(1, 5, 22) * 0.25
    initial_noise = torch.randn(1, 12, 22)

    captured: dict[str, Any] = {}

    def fake_verify(**kwargs):
        captured.update(kwargs)
        return ZeroGuidanceReport(
            timesteps=(88, 80),
            terminal_shape=(1, 12, 22),
            max_x0_error=0.0,
            max_prev_error=0.0,
        )

    with patch(
        "diffusion_policy.guidance.sim_hand_guidance.verify_zero_guidance_equivalence",
        side_effect=fake_verify,
    ):
        report = guidance.verify_zero_guidance(
            history,
            reference,
            initial_noise=initial_noise,
        )

    assert isinstance(report, ZeroGuidanceReport)
    assert adapter.global_condition_calls == 1
    assert adapter.normalize_reference_calls == 1
    assert captured["num_inference_steps"] == 12
    assert captured["guidance_slice"] == slice(3, 8)
    assert captured["model"] == adapter.predict_epsilon
    assert captured["training_scheduler"] is adapter.policy.noise_scheduler
    torch.testing.assert_close(captured["initial_noise"], initial_noise)
    expected_cond = adapter.inner.global_condition(history)
    expected_ref = adapter.inner.normalize_reference(reference)
    torch.testing.assert_close(captured["global_cond"], expected_cond)
    torch.testing.assert_close(captured["reference"], expected_ref)
    # verify path should not unnormalize / sample
    assert adapter.unnormalize_action_calls == 0
    assert adapter.policy.predict_action_calls == 0


def test_default_noise_factory_cpu_generator_to_cuda():
    from diffusion_policy.guidance.sim_hand_guidance import default_noise_factory

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    gen = torch.Generator(device="cpu").manual_seed(123)
    noise = default_noise_factory(
        (1, 12, 22),
        torch.device("cuda"),
        torch.float32,
        gen,
    )
    assert noise.device.type == "cuda"
    assert noise.shape == (1, 12, 22)
    assert torch.isfinite(noise).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_guide_segment_cuda_smoke_with_cpu_generator():
    from diffusion_policy.guidance.sim_hand_guidance import SimHandGuidance

    loaded = make_sim_policy(
        n_obs_steps=4,
        horizon=12,
        n_pred_action_steps=9,
        usable_start=3,
    )
    loaded.policy.to("cuda")
    adapter = SimPolicyAdapter(loaded)
    guidance = SimHandGuidance(
        adapter,
        _default_config(
            execution_steps=5,
            guidance_scale=0.0,
            num_inference_steps=2,
        ),
    )
    history = torch.zeros(1, 4, 22, device="cuda")
    reference = torch.zeros(1, 5, 22, device="cuda")
    generator = torch.Generator(device="cpu").manual_seed(0)

    result = guidance.guide_segment(history, reference, generator=generator)
    assert result.device.type == "cuda"
    assert result.shape == (1, 5, 22)
    assert torch.isfinite(result).all()
