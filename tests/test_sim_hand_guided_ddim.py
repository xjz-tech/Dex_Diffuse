from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import diffusers
import pytest
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.guidance.guided_ddim import (
    EXPECTED_DIFFUSERS_VERSION,
    assert_pinned_diffusers_version,
    create_ddim_scheduler,
    guided_ddim_step,
    sample_guided_trajectory,
    verify_zero_guidance_equivalence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDED_DDIM_PATH = REPO_ROOT / "diffusion_policy" / "guidance" / "guided_ddim.py"

EXPECTED_CURRENT_TIMESTEPS = (88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)


def _training_scheduler(*, num_train_timesteps: int, prediction_type: str = "epsilon") -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type=prediction_type,
    )


def _assert_dynamic_scheduler(
    training: DDPMScheduler,
    *,
    inference_steps: int,
    expected_timesteps: tuple[int, ...] | None = None,
) -> None:
    ddim = create_ddim_scheduler(training)
    ddim.set_timesteps(inference_steps)
    torch.testing.assert_close(
        ddim.alphas_cumprod,
        training.alphas_cumprod,
        rtol=0.0,
        atol=0.0,
    )
    assert len(ddim.timesteps) == inference_steps
    assert int(ddim.timesteps.max()) < training.config.num_train_timesteps
    if expected_timesteps is not None:
        assert tuple(map(int, ddim.timesteps)) == expected_timesteps


def test_version_assert_pinned_diffusers_version_matches_expected_constant() -> None:
    assert diffusers.__version__ == EXPECTED_DIFFUSERS_VERSION
    assert_pinned_diffusers_version()


def test_version_rejects_wrong_diffusers_version() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    with patch("diffusion_policy.guidance.guided_ddim.diffusers.__version__", "0.12.0"):
        with pytest.raises(RuntimeError, match="0.11.1"):
            assert_pinned_diffusers_version()
        with pytest.raises(RuntimeError, match="0.11.1"):
            create_ddim_scheduler(training)


def test_factory_copies_alphas_cumprod_for_100_step_training() -> None:
    _assert_dynamic_scheduler(
        _training_scheduler(num_train_timesteps=100),
        inference_steps=12,
        expected_timesteps=EXPECTED_CURRENT_TIMESTEPS,
    )


def test_factory_copies_alphas_cumprod_for_60_step_training() -> None:
    _assert_dynamic_scheduler(
        _training_scheduler(num_train_timesteps=60),
        inference_steps=10,
    )


def test_factory_accepts_non_100_train_step_count() -> None:
    training = _training_scheduler(num_train_timesteps=60)
    ddim = create_ddim_scheduler(training)
    assert ddim.config.num_train_timesteps == 60


def test_factory_rejects_non_epsilon_prediction() -> None:
    training = _training_scheduler(
        num_train_timesteps=100,
        prediction_type="sample",
    )
    with pytest.raises(ValueError, match="epsilon"):
        create_ddim_scheduler(training)


def test_factory_rejects_thresholding() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    training.config.thresholding = True
    with pytest.raises(ValueError, match="thresholding"):
        create_ddim_scheduler(training)


def test_factory_production_has_no_hardcoded_timesteps() -> None:
    source = GUIDED_DDIM_PATH.read_text()
    assert "EXPECTED_CURRENT_TIMESTEPS" not in source
    assert str(EXPECTED_CURRENT_TIMESTEPS) not in source


def _prepared_ddim(num_train_timesteps: int, inference_steps: int):
    ddim = create_ddim_scheduler(
        _training_scheduler(num_train_timesteps=num_train_timesteps)
    )
    ddim.set_timesteps(inference_steps)
    return ddim


def _clipping_triggering_pair(
    scheduler,
    timestep: int,
    shape: tuple[int, int, int],
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    alpha_t = scheduler.alphas_cumprod[int(timestep)]
    sample = torch.full(shape, 8.0)
    model_output = torch.randn(shape)
    x0_raw = (sample - (1.0 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
    assert bool((x0_raw.abs() > 1.0).any())
    assert not torch.equal(x0_raw, x0_raw.clamp(-1.0, 1.0))
    return sample, model_output


def _assert_zero_scale_matches_official(
    *,
    num_train_timesteps: int,
    inference_steps: int,
    shape: tuple[int, int, int],
    guidance_slice: slice,
) -> None:
    official = _prepared_ddim(num_train_timesteps, inference_steps)
    custom = _prepared_ddim(num_train_timesteps, inference_steps)
    assert official is not custom
    reference = torch.randn(
        shape[0],
        guidance_slice.stop - guidance_slice.start,
        shape[2],
    )
    for timestep in map(int, custom.timesteps):
        sample, model_output = _clipping_triggering_pair(
            custom,
            timestep,
            shape,
            seed=timestep + 17,
        )
        official_out = official.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=0.0,
        )

        def _forbid_official_step(*args, **kwargs):
            raise AssertionError("zero-scale path must not call official scheduler.step")

        custom.step = _forbid_official_step
        custom_out = guided_ddim_step(
            scheduler=custom,
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            reference=reference,
            guidance_scale=0.0,
            guidance_slice=guidance_slice,
            eta=0.0,
        )
        torch.testing.assert_close(
            custom_out.pred_original_sample,
            official_out.pred_original_sample,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            custom_out.prev_sample,
            official_out.prev_sample,
            rtol=1e-5,
            atol=1e-6,
        )


def test_zero_scale_custom_step_matches_official_for_each_100_12_timestep() -> None:
    _assert_zero_scale_matches_official(
        num_train_timesteps=100,
        inference_steps=12,
        shape=(2, 12, 22),
        guidance_slice=slice(3, 8),
    )


def test_zero_scale_custom_step_matches_official_for_each_60_10_timestep() -> None:
    _assert_zero_scale_matches_official(
        num_train_timesteps=60,
        inference_steps=10,
        shape=(2, 8, 22),
        guidance_slice=slice(1, 4),
    )


def test_x0_vjp_guidance_matches_official_ddim_with_guided_epsilon() -> None:
    scheduler = _prepared_ddim(100, 12)
    timestep = int(scheduler.timesteps[3])
    guidance_slice = slice(3, 8)
    sample = torch.randn(2, 12, 22, requires_grad=True)
    model_scale = 0.25
    model_output = model_scale * sample + 0.03
    reference = torch.full((2, 5, 22), 0.2)
    guidance_scale = 1.5

    alpha_t = scheduler.alphas_cumprod[timestep]
    sqrt_beta_t = (1.0 - alpha_t).sqrt()
    x0_raw = (sample - sqrt_beta_t * model_output) / alpha_t.sqrt()
    expected_gradient = torch.zeros_like(sample)
    expected_gradient[:, guidance_slice] = (
        2.0
        / (reference.shape[1] * reference.shape[2])
        * (x0_raw[:, guidance_slice] - reference)
        * (1.0 - sqrt_beta_t * model_scale)
        / alpha_t.sqrt()
    )
    expected_epsilon = (
        model_output + sqrt_beta_t * guidance_scale * expected_gradient
    )
    expected = scheduler.step(
        model_output=expected_epsilon.detach(),
        timestep=timestep,
        sample=sample.detach(),
        eta=0.0,
    )

    output = guided_ddim_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=guidance_scale,
        guidance_slice=guidance_slice,
        eta=0.0,
    )

    torch.testing.assert_close(output.prev_sample, expected.prev_sample)
    torch.testing.assert_close(
        output.pred_original_sample,
        expected.pred_original_sample,
    )


def test_positive_guidance_sampler_works_inside_inference_mode() -> None:
    scheduler = create_ddim_scheduler(
        _training_scheduler(num_train_timesteps=60)
    )
    with torch.inference_mode():
        result = sample_guided_trajectory(
            model=_DeterministicEpsilonModel(scale=0.2),
            scheduler=scheduler,
            initial_noise=torch.randn(1, 8, 22),
            global_cond=torch.randn(1, 44),
            reference=torch.zeros(1, 3, 22),
            num_inference_steps=5,
            guidance_scale=1.0,
            guidance_slice=slice(1, 4),
            eta=0.0,
        )

    assert torch.isfinite(result.trajectory).all()
    assert not result.trajectory.requires_grad


def test_x0_guidance_remains_active_at_final_ddim_timestep() -> None:
    scheduler = _prepared_ddim(100, 12)
    timestep = int(scheduler.timesteps[-1])
    assert timestep == 0
    guidance_slice = slice(3, 8)
    sample = torch.zeros(2, 12, 22, requires_grad=True)
    model_output = torch.zeros_like(sample)
    reference = torch.ones(2, 5, 22)
    base = scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=0.0,
    )
    output = guided_ddim_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=100.0,
        guidance_slice=guidance_slice,
    )
    assert not torch.allclose(output.prev_sample, base.prev_sample)


def test_zero_x0_reference_distance_has_zero_update() -> None:
    scheduler = _prepared_ddim(60, 10)
    timestep = int(scheduler.timesteps[2])
    guidance_slice = slice(1, 4)
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 8, 22), seed=9
    )
    dummy_reference = torch.zeros(2, 3, 22)
    base = guided_ddim_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=dummy_reference,
        guidance_scale=0.0,
        guidance_slice=guidance_slice,
    )
    alpha_t = scheduler.alphas_cumprod[timestep]
    raw_x0 = (
        sample - (1.0 - alpha_t).sqrt() * model_output
    ) / alpha_t.sqrt()
    matched_reference = raw_x0[:, guidance_slice].clone()
    sample.requires_grad_(True)
    output = guided_ddim_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=matched_reference,
        guidance_scale=6.0,
        guidance_slice=guidance_slice,
    )
    torch.testing.assert_close(
        output.pred_original_sample,
        base.pred_original_sample,
    )
    torch.testing.assert_close(output.base_prev_sample, base.prev_sample)
    torch.testing.assert_close(output.prev_sample, base.prev_sample)


def test_guidance_losses_are_measured_on_raw_x0_estimates() -> None:
    scheduler = _prepared_ddim(100, 12)
    timestep = int(scheduler.timesteps[2])
    guidance_slice = slice(3, 8)
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 12, 22), seed=17
    )
    sample.requires_grad_(True)
    reference = torch.full((2, 5, 22), 0.4)
    output = guided_ddim_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=2.0,
        guidance_slice=guidance_slice,
    )
    alpha_t = scheduler.alphas_cumprod[timestep]
    base_raw_x0 = (
        sample - (1.0 - alpha_t).sqrt() * model_output
    ) / alpha_t.sqrt()
    expected_before = (
        base_raw_x0[:, guidance_slice] - reference
    ).square().mean(dim=(1, 2))
    expected_after = (
        output.raw_pred_original_sample[:, guidance_slice] - reference
    ).square().mean(dim=(1, 2))

    torch.testing.assert_close(output.guidance_loss_before, expected_before)
    torch.testing.assert_close(output.guidance_loss_after, expected_after)


def test_negative_scale_and_nonzero_eta_are_rejected() -> None:
    scheduler = _prepared_ddim(100, 12)
    timestep = int(scheduler.timesteps[0])
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 12, 22)
    )
    reference = torch.zeros(2, 5, 22)
    kwargs = dict(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_slice=slice(3, 8),
    )
    with pytest.raises(ValueError, match="scale"):
        guided_ddim_step(**kwargs, guidance_scale=-0.1, eta=0.0)
    with pytest.raises(ValueError, match="eta"):
        guided_ddim_step(**kwargs, guidance_scale=1.0, eta=0.1)


@pytest.mark.parametrize("guidance_scale", [float("nan"), float("inf")])
def test_step_rejects_non_finite_guidance_scale(guidance_scale: float) -> None:
    scheduler = _prepared_ddim(60, 10)
    timestep = int(scheduler.timesteps[0])
    sample = torch.zeros(1, 8, 22, requires_grad=True)
    with pytest.raises(ValueError, match="finite"):
        guided_ddim_step(
            scheduler=scheduler,
            model_output=torch.zeros_like(sample),
            timestep=timestep,
            sample=sample,
            reference=torch.zeros(1, 3, 22),
            guidance_scale=guidance_scale,
            guidance_slice=slice(1, 4),
            eta=0.0,
        )


def test_step_rejects_shape_dtype_device_and_finite_violations() -> None:
    scheduler = _prepared_ddim(100, 12)
    timestep = int(scheduler.timesteps[1])
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 12, 22)
    )
    reference = torch.zeros(2, 5, 22)
    base_kwargs = dict(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=1.0,
        guidance_slice=slice(3, 8),
        eta=0.0,
    )
    with pytest.raises(ValueError, match="shape"):
        guided_ddim_step(
            **{**base_kwargs, "model_output": torch.randn(2, 12, 21)},
        )
    with pytest.raises(ValueError, match="batch"):
        guided_ddim_step(
            **{**base_kwargs, "reference": torch.zeros(1, 5, 22)},
        )
    with pytest.raises(ValueError, match="action dim"):
        guided_ddim_step(
            **{**base_kwargs, "reference": torch.zeros(2, 5, 21)},
        )
    with pytest.raises(ValueError, match="slice"):
        guided_ddim_step(
            **{**base_kwargs, "reference": torch.zeros(2, 4, 22)},
        )
    with pytest.raises(ValueError, match="dtype"):
        guided_ddim_step(
            **{
                **base_kwargs,
                "reference": reference.to(dtype=torch.float64),
            },
        )
    with pytest.raises(ValueError, match="finite"):
        bad_sample = sample.clone()
        bad_sample[0, 0, 0] = float("nan")
        guided_ddim_step(**{**base_kwargs, "sample": bad_sample})
    with pytest.raises(ValueError, match="finite"):
        bad_reference = reference.clone()
        bad_reference[0, 0, 0] = float("inf")
        guided_ddim_step(**{**base_kwargs, "reference": bad_reference})
    unset = create_ddim_scheduler(_training_scheduler(num_train_timesteps=100))
    with pytest.raises(ValueError, match="inference"):
        guided_ddim_step(**{**base_kwargs, "scheduler": unset})
    if not torch.cuda.is_available():
        return
    with pytest.raises(ValueError, match="device"):
        guided_ddim_step(
            **{
                **base_kwargs,
                "reference": reference.to(device="cuda"),
            },
        )


class _DeterministicEpsilonModel:
    """Deterministic epsilon predictor for full-chain sampler tests."""

    def __init__(self, *, scale: float = 0.01) -> None:
        self.scale = scale

    def __call__(
        self,
        sample: torch.Tensor,
        timestep,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = float(int(timestep))
        out = sample * self.scale + t * 1e-4
        if global_cond is not None:
            out = out + global_cond.mean() * 1e-3
        return out


class _RecordingEpsilonModel(_DeterministicEpsilonModel):
    def __init__(self, *, scale: float = 0.01) -> None:
        super().__init__(scale=scale)
        self.inputs: list[torch.Tensor] = []

    def __call__(self, sample, timestep, global_cond=None):
        self.inputs.append(sample.detach().clone())
        return super().__call__(sample, timestep, global_cond)


class _NonFiniteEpsilonModel(_DeterministicEpsilonModel):
    def __call__(self, sample, timestep, global_cond=None):
        out = super().__call__(sample, timestep, global_cond)
        out = out.clone()
        out[0, 0, 0] = float("nan")
        return out


class _WrongShapeEpsilonModel(_DeterministicEpsilonModel):
    def __call__(self, sample, timestep, global_cond=None):
        out = super().__call__(sample, timestep, global_cond)
        return out[..., :-1]


class _ParamEpsilonModel(nn.Module):
    def __init__(self, *, scale: float = 0.2) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, sample, timestep, global_cond=None):
        return self.scale * sample


def _assert_sampler_chain(
    *,
    num_train_timesteps: int,
    num_inference_steps: int,
    shape: tuple[int, int, int],
    cond_dim: int,
    guidance_slice: slice,
) -> None:
    training = _training_scheduler(num_train_timesteps=num_train_timesteps)
    scheduler = create_ddim_scheduler(training)
    torch.manual_seed(num_train_timesteps + num_inference_steps)
    initial_noise = torch.randn(shape)
    global_cond = torch.randn(shape[0], cond_dim)
    reference = torch.randn(
        shape[0],
        guidance_slice.stop - guidance_slice.start,
        shape[2],
    )
    model = _RecordingEpsilonModel()
    result = sample_guided_trajectory(
        model=model,
        scheduler=scheduler,
        initial_noise=initial_noise,
        global_cond=global_cond,
        reference=reference,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.5,
        guidance_slice=guidance_slice,
        eta=0.0,
    )
    assert len(result.steps) == num_inference_steps
    assert tuple(step.timestep for step in result.steps) == tuple(
        map(int, scheduler.timesteps)
    )
    assert result.trajectory.shape == initial_noise.shape
    assert all(not step.prev_sample.requires_grad for step in result.steps)
    assert len(model.inputs) == num_inference_steps
    torch.testing.assert_close(model.inputs[0], initial_noise)
    for model_input, previous_step in zip(model.inputs[1:], result.steps[:-1]):
        torch.testing.assert_close(model_input, previous_step.prev_sample)
    assert any(
        not torch.allclose(step.prev_sample, step.base_prev_sample)
        for step in result.steps[:-1]
    )
    torch.testing.assert_close(result.trajectory, result.steps[-1].prev_sample)


def test_sampler_chain_matches_dynamic_timesteps_for_100_12() -> None:
    _assert_sampler_chain(
        num_train_timesteps=100,
        num_inference_steps=12,
        shape=(2, 12, 22),
        cond_dim=88,
        guidance_slice=slice(3, 8),
    )


def test_sampler_chain_matches_dynamic_timesteps_for_60_10() -> None:
    _assert_sampler_chain(
        num_train_timesteps=60,
        num_inference_steps=10,
        shape=(2, 8, 22),
        cond_dim=44,
        guidance_slice=slice(1, 4),
    )


def test_sampler_does_not_accumulate_model_parameter_grads() -> None:
    training = _training_scheduler(num_train_timesteps=60)
    scheduler = create_ddim_scheduler(training)
    model = _ParamEpsilonModel()
    result = sample_guided_trajectory(
        model=model,
        scheduler=scheduler,
        initial_noise=torch.randn(1, 8, 22),
        global_cond=torch.randn(1, 44),
        reference=torch.zeros(1, 3, 22),
        num_inference_steps=5,
        guidance_scale=1.5,
        guidance_slice=slice(1, 4),
        eta=0.0,
    )
    assert torch.isfinite(result.trajectory).all()
    assert model.scale.grad is None


def test_sampler_rejects_nonpositive_inference_steps() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    scheduler = create_ddim_scheduler(training)
    shape = (1, 12, 22)
    kwargs = dict(
        model=_DeterministicEpsilonModel(),
        scheduler=scheduler,
        initial_noise=torch.randn(shape),
        global_cond=torch.randn(1, 88),
        reference=torch.zeros(1, 5, 22),
        guidance_scale=1.0,
        guidance_slice=slice(3, 8),
        eta=0.0,
    )
    with pytest.raises(ValueError, match="inference"):
        sample_guided_trajectory(**kwargs, num_inference_steps=0)
    with pytest.raises(ValueError, match="inference"):
        sample_guided_trajectory(**kwargs, num_inference_steps=-3)


def test_sampler_rejects_non_finite_model_output() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    scheduler = create_ddim_scheduler(training)
    with pytest.raises(ValueError, match="finite"):
        sample_guided_trajectory(
            model=_NonFiniteEpsilonModel(),
            scheduler=scheduler,
            initial_noise=torch.randn(1, 12, 22),
            global_cond=torch.randn(1, 88),
            reference=torch.zeros(1, 5, 22),
            num_inference_steps=12,
            guidance_scale=1.0,
            guidance_slice=slice(3, 8),
            eta=0.0,
        )


def test_sampler_rejects_wrong_model_output_shape() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    scheduler = create_ddim_scheduler(training)
    with pytest.raises(ValueError, match="shape"):
        sample_guided_trajectory(
            model=_WrongShapeEpsilonModel(),
            scheduler=scheduler,
            initial_noise=torch.randn(1, 12, 22),
            global_cond=torch.randn(1, 88),
            reference=torch.zeros(1, 5, 22),
            num_inference_steps=12,
            guidance_scale=1.0,
            guidance_slice=slice(3, 8),
            eta=0.0,
        )


def _assert_zero_guidance_full_chain_oracle(
    *,
    num_train_timesteps: int,
    num_inference_steps: int,
    shape: tuple[int, int, int],
    cond_dim: int,
    guidance_slice: slice,
) -> None:
    training = _training_scheduler(num_train_timesteps=num_train_timesteps)
    torch.manual_seed(num_train_timesteps * 10 + num_inference_steps)
    initial_noise = torch.randn(shape)
    global_cond = torch.randn(shape[0], cond_dim)
    reference = torch.randn(
        shape[0],
        guidance_slice.stop - guidance_slice.start,
        shape[2],
    )
    model = _DeterministicEpsilonModel(scale=0.02)
    report = verify_zero_guidance_equivalence(
        model=model,
        training_scheduler=training,
        initial_noise=initial_noise,
        global_cond=global_cond,
        reference=reference,
        guidance_slice=guidance_slice,
        num_inference_steps=num_inference_steps,
    )
    probe = create_ddim_scheduler(training)
    probe.set_timesteps(num_inference_steps)
    expected_timesteps = tuple(map(int, probe.timesteps))
    assert report.timesteps == expected_timesteps
    assert len(report.timesteps) == num_inference_steps
    if (num_train_timesteps, num_inference_steps) != (100, 12):
        assert report.timesteps != EXPECTED_CURRENT_TIMESTEPS
    assert report.terminal_shape == tuple(shape)
    assert report.max_x0_error <= 1e-6
    assert report.max_prev_error <= 1e-6


def test_zero_guidance_report_full_official_custom_chain_for_100_12() -> None:
    _assert_zero_guidance_full_chain_oracle(
        num_train_timesteps=100,
        num_inference_steps=12,
        shape=(2, 12, 22),
        cond_dim=88,
        guidance_slice=slice(3, 8),
    )


def test_zero_guidance_report_full_official_custom_chain_for_60_10() -> None:
    _assert_zero_guidance_full_chain_oracle(
        num_train_timesteps=60,
        num_inference_steps=10,
        shape=(2, 8, 22),
        cond_dim=44,
        guidance_slice=slice(1, 4),
    )
