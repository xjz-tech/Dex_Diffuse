from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import diffusers
import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.guidance.guided_ddpm import (
    EXPECTED_DIFFUSERS_VERSION,
    assert_pinned_diffusers_version,
    create_ddpm_scheduler,
    guided_ddpm_step,
    mse_guidance_gradient,
    sample_guided_trajectory,
    verify_zero_guidance_equivalence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDED_DDPM_PATH = REPO_ROOT / "diffusion_policy" / "guidance" / "guided_ddpm.py"

# diffusers 0.11.1 DDPMScheduler.set_timesteps(100 train, 12 infer):
# arange(0, 100, 100//12)[::-1] == 96, 88, ..., 0 (13 steps).
EXPECTED_CURRENT_TIMESTEPS = (96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)


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
    ddpm = create_ddpm_scheduler(training)
    ddpm.set_timesteps(inference_steps)
    torch.testing.assert_close(
        ddpm.alphas_cumprod,
        training.alphas_cumprod,
        rtol=0.0,
        atol=0.0,
    )
    official = DDPMScheduler.from_config(training.config)
    official.set_timesteps(inference_steps)
    assert len(ddpm.timesteps) == len(official.timesteps)
    assert int(ddpm.timesteps.max()) < training.config.num_train_timesteps
    if expected_timesteps is not None:
        assert tuple(map(int, ddpm.timesteps)) == expected_timesteps


def test_version_assert_pinned_diffusers_version_matches_expected_constant() -> None:
    assert diffusers.__version__ == EXPECTED_DIFFUSERS_VERSION
    assert_pinned_diffusers_version()


def test_version_rejects_wrong_diffusers_version() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    with patch("diffusion_policy.guidance.guided_ddpm.diffusers.__version__", "0.12.0"):
        with pytest.raises(RuntimeError, match="0.11.1"):
            assert_pinned_diffusers_version()
        with pytest.raises(RuntimeError, match="0.11.1"):
            create_ddpm_scheduler(training)


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
    ddpm = create_ddpm_scheduler(training)
    assert ddpm.config.num_train_timesteps == 60
    assert isinstance(ddpm, DDPMScheduler)


def test_factory_rejects_non_epsilon_prediction() -> None:
    training = _training_scheduler(
        num_train_timesteps=100,
        prediction_type="sample",
    )
    with pytest.raises(ValueError, match="epsilon"):
        create_ddpm_scheduler(training)


def test_factory_rejects_thresholding() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    training.config.thresholding = True
    with pytest.raises(ValueError, match="thresholding"):
        create_ddpm_scheduler(training)


def test_factory_production_has_no_hardcoded_timesteps() -> None:
    source = GUIDED_DDPM_PATH.read_text()
    assert "EXPECTED_CURRENT_TIMESTEPS" not in source
    assert str(EXPECTED_CURRENT_TIMESTEPS) not in source


def _prepared_ddpm(num_train_timesteps: int, inference_steps: int) -> DDPMScheduler:
    ddpm = create_ddpm_scheduler(
        _training_scheduler(num_train_timesteps=num_train_timesteps)
    )
    ddpm.set_timesteps(inference_steps)
    return ddpm


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


def _independent_ddpm_mean(
    scheduler,
    timestep: int,
    sample: torch.Tensor,
    model_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = int(timestep)
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[t - 1] if t > 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    x0_raw = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    x0 = x0_raw.clamp(-1, 1) if scheduler.config.clip_sample else x0_raw
    mu = (
        (alpha_prod_t_prev ** (0.5) * scheduler.betas[t]) / beta_prod_t
    ) * x0 + (
        scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
    ) * sample
    return x0_raw, x0, mu


def _independent_posterior_variance(scheduler, timestep: int) -> torch.Tensor:
    return scheduler._get_variance(int(timestep))


def _official_noise_term(
    scheduler,
    timestep: int,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t = int(timestep)
    if t <= 0:
        return torch.zeros(shape, device=device, dtype=dtype)
    variance = scheduler._get_variance(t)
    variance_noise = torch.randn(
        shape, generator=generator, device=device, dtype=dtype
    )
    return (variance ** 0.5) * variance_noise


def _seeded_pair(seed: int) -> tuple[torch.Generator, torch.Generator]:
    official = torch.Generator(device="cpu").manual_seed(seed)
    custom = torch.Generator(device="cpu").manual_seed(seed)
    return official, custom


def _assert_zero_scale_matches_official(
    *,
    num_train_timesteps: int,
    inference_steps: int,
    shape: tuple[int, int, int],
    guidance_slice: slice,
) -> None:
    official = _prepared_ddpm(num_train_timesteps, inference_steps)
    custom = _prepared_ddpm(num_train_timesteps, inference_steps)
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
        official_gen, custom_gen = _seeded_pair(timestep + 101)
        official_out = official.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            generator=official_gen,
        )

        def _forbid_official_step(*args, **kwargs):
            raise AssertionError("zero-scale path must not call official scheduler.step")

        custom.step = _forbid_official_step
        custom_out = guided_ddpm_step(
            scheduler=custom,
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            reference=reference,
            guidance_scale=0.0,
            guidance_slice=guidance_slice,
            generator=custom_gen,
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
        torch.testing.assert_close(
            custom_out.base_prev_sample,
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


def test_positive_guidance_shifts_mean_then_adds_official_noise() -> None:
    scheduler = _prepared_ddpm(100, 12)
    timestep = int(scheduler.timesteps[3])
    guidance_slice = slice(3, 8)
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 12, 22), seed=3
    )
    reference = torch.zeros(2, 5, 22)
    scale = 2.5
    official_gen, custom_gen = _seeded_pair(2024)
    guided = guided_ddpm_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=scale,
        guidance_slice=guidance_slice,
        generator=custom_gen,
    )
    x0_raw, x0, mu = _independent_ddpm_mean(
        scheduler, timestep, sample, model_output
    )
    variance = _independent_posterior_variance(scheduler, timestep)
    gradient = mse_guidance_gradient(mu, reference, guidance_slice)
    noise = _official_noise_term(
        scheduler,
        timestep,
        sample.shape,
        generator=official_gen,
        device=sample.device,
        dtype=sample.dtype,
    )
    expected_guided_prev = mu - scale * variance * gradient + noise
    expected_base_prev = mu + noise

    torch.testing.assert_close(guided.pred_original_sample, x0)
    torch.testing.assert_close(guided.raw_pred_original_sample, x0_raw)
    torch.testing.assert_close(guided.base_prev_sample, expected_base_prev)
    torch.testing.assert_close(guided.prev_sample, expected_guided_prev)
    torch.testing.assert_close(guided.raw_variance, variance)
    assert not torch.allclose(
        guided.prev_sample[:, guidance_slice],
        guided.base_prev_sample[:, guidance_slice],
    )
    torch.testing.assert_close(
        guided.prev_sample[:, : guidance_slice.start],
        guided.base_prev_sample[:, : guidance_slice.start],
    )
    torch.testing.assert_close(
        guided.prev_sample[:, guidance_slice.stop :],
        guided.base_prev_sample[:, guidance_slice.stop :],
    )


def test_last_timestep_has_zero_official_noise_and_uses_clamped_variance() -> None:
    scheduler = _prepared_ddpm(100, 12)
    timestep = int(scheduler.timesteps[-1])
    assert timestep == 0
    guidance_slice = slice(3, 8)
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 12, 22), seed=0
    )
    reference = torch.full((2, 5, 22), 0.25)
    scale = 4.0
    official_gen, custom_gen = _seeded_pair(7)
    output = guided_ddpm_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=scale,
        guidance_slice=guidance_slice,
        generator=custom_gen,
    )
    x0_raw, x0, mu = _independent_ddpm_mean(
        scheduler, timestep, sample, model_output
    )
    variance = _independent_posterior_variance(scheduler, timestep)
    gradient = mse_guidance_gradient(mu, reference, guidance_slice)
    noise = _official_noise_term(
        scheduler,
        timestep,
        sample.shape,
        generator=official_gen,
        device=sample.device,
        dtype=sample.dtype,
    )
    torch.testing.assert_close(noise, torch.zeros_like(noise), atol=0.0, rtol=0.0)
    assert float(variance) > 0.0
    expected = mu - scale * variance * gradient + noise
    torch.testing.assert_close(output.pred_original_sample, x0)
    torch.testing.assert_close(output.raw_pred_original_sample, x0_raw)
    torch.testing.assert_close(output.base_prev_sample, mu)
    torch.testing.assert_close(output.prev_sample, expected)
    torch.testing.assert_close(output.raw_variance, variance)


def test_matched_reference_has_zero_mean_update() -> None:
    scheduler = _prepared_ddpm(60, 10)
    timestep = int(scheduler.timesteps[2])
    guidance_slice = slice(1, 4)
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 8, 22), seed=9
    )
    _, _, mu = _independent_ddpm_mean(scheduler, timestep, sample, model_output)
    matched_reference = mu[:, guidance_slice].clone()
    official_gen, custom_gen = _seeded_pair(33)
    output = guided_ddpm_step(
        scheduler=scheduler,
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        reference=matched_reference,
        guidance_scale=6.0,
        guidance_slice=guidance_slice,
        generator=custom_gen,
    )
    noise = _official_noise_term(
        scheduler,
        timestep,
        sample.shape,
        generator=official_gen,
        device=sample.device,
        dtype=sample.dtype,
    )
    torch.testing.assert_close(output.base_prev_sample, mu + noise)
    torch.testing.assert_close(output.prev_sample, output.base_prev_sample)


def test_negative_scale_is_rejected() -> None:
    scheduler = _prepared_ddpm(100, 12)
    timestep = int(scheduler.timesteps[0])
    sample, model_output = _clipping_triggering_pair(
        scheduler, timestep, (2, 12, 22)
    )
    reference = torch.zeros(2, 5, 22)
    with pytest.raises(ValueError, match="scale"):
        guided_ddpm_step(
            scheduler=scheduler,
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            reference=reference,
            guidance_scale=-0.1,
            guidance_slice=slice(3, 8),
        )


def test_step_rejects_shape_dtype_device_and_finite_violations() -> None:
    scheduler = _prepared_ddpm(100, 12)
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
    )
    with pytest.raises(ValueError, match="shape"):
        guided_ddpm_step(
            **{**base_kwargs, "model_output": torch.randn(2, 12, 21)},
        )
    with pytest.raises(ValueError, match="batch"):
        guided_ddpm_step(
            **{**base_kwargs, "reference": torch.zeros(1, 5, 22)},
        )
    with pytest.raises(ValueError, match="dtype"):
        guided_ddpm_step(
            **{
                **base_kwargs,
                "reference": reference.to(dtype=torch.float64),
            },
        )
    with pytest.raises(ValueError, match="finite"):
        bad_sample = sample.clone()
        bad_sample[0, 0, 0] = float("nan")
        guided_ddpm_step(**{**base_kwargs, "sample": bad_sample})
    unset = create_ddpm_scheduler(_training_scheduler(num_train_timesteps=100))
    with pytest.raises(ValueError, match="inference"):
        guided_ddpm_step(**{**base_kwargs, "scheduler": unset})
    if not torch.cuda.is_available():
        return
    with pytest.raises(ValueError, match="device"):
        guided_ddpm_step(
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
        self.inputs.append(sample.clone())
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


def _assert_sampler_chain(
    *,
    num_train_timesteps: int,
    num_inference_steps: int,
    shape: tuple[int, int, int],
    cond_dim: int,
    guidance_slice: slice,
) -> None:
    training = _training_scheduler(num_train_timesteps=num_train_timesteps)
    scheduler = create_ddpm_scheduler(training)
    torch.manual_seed(num_train_timesteps + num_inference_steps)
    initial_noise = torch.randn(shape)
    global_cond = torch.randn(shape[0], cond_dim)
    reference = torch.randn(
        shape[0],
        guidance_slice.stop - guidance_slice.start,
        shape[2],
    )
    model = _RecordingEpsilonModel()
    generator = torch.Generator(device="cpu").manual_seed(99)
    result = sample_guided_trajectory(
        model=model,
        scheduler=scheduler,
        initial_noise=initial_noise,
        global_cond=global_cond,
        reference=reference,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.5,
        guidance_slice=guidance_slice,
        generator=generator,
    )
    expected_len = len(scheduler.timesteps)
    assert len(result.steps) == expected_len
    assert tuple(step.timestep for step in result.steps) == tuple(
        map(int, scheduler.timesteps)
    )
    assert result.trajectory.shape == initial_noise.shape
    assert all(not step.prev_sample.requires_grad for step in result.steps)
    assert len(model.inputs) == expected_len
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


def test_sampler_rejects_nonpositive_inference_steps() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    scheduler = create_ddpm_scheduler(training)
    shape = (1, 12, 22)
    kwargs = dict(
        model=_DeterministicEpsilonModel(),
        scheduler=scheduler,
        initial_noise=torch.randn(shape),
        global_cond=torch.randn(1, 88),
        reference=torch.zeros(1, 5, 22),
        guidance_scale=1.0,
        guidance_slice=slice(3, 8),
    )
    with pytest.raises(ValueError, match="inference"):
        sample_guided_trajectory(**kwargs, num_inference_steps=0)
    with pytest.raises(ValueError, match="inference"):
        sample_guided_trajectory(**kwargs, num_inference_steps=-3)


def test_sampler_rejects_non_finite_model_output() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    scheduler = create_ddpm_scheduler(training)
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
        )


def test_sampler_rejects_wrong_model_output_shape() -> None:
    training = _training_scheduler(num_train_timesteps=100)
    scheduler = create_ddpm_scheduler(training)
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
    generator = torch.Generator(device="cpu").manual_seed(123)
    report = verify_zero_guidance_equivalence(
        model=model,
        training_scheduler=training,
        initial_noise=initial_noise,
        global_cond=global_cond,
        reference=reference,
        guidance_slice=guidance_slice,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    probe = create_ddpm_scheduler(training)
    probe.set_timesteps(num_inference_steps)
    expected_timesteps = tuple(map(int, probe.timesteps))
    assert report.timesteps == expected_timesteps
    assert len(report.timesteps) == len(expected_timesteps)
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
