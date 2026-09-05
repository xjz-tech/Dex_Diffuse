from __future__ import annotations

from dataclasses import dataclass

import diffusers
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    from diffusers.utils.torch_utils import randn_tensor
except ImportError:  # pragma: no cover - pinned 0.11.1 layout
    from diffusers.utils import randn_tensor

# DDIM guidance still documents 0.11.1, but this environment and the
# official DDPMScheduler.previous_timestep API we match are 0.29.2.
EXPECTED_DIFFUSERS_VERSION = "0.29.2"


def assert_pinned_diffusers_version() -> None:
    if diffusers.__version__ != EXPECTED_DIFFUSERS_VERSION:
        raise RuntimeError(
            f"Guided DDPM requires diffusers=={EXPECTED_DIFFUSERS_VERSION}, "
            f"got {diffusers.__version__}"
        )


def create_ddpm_scheduler(training_scheduler: DDPMScheduler) -> DDPMScheduler:
    assert_pinned_diffusers_version()
    config = training_scheduler.config
    if config.prediction_type != "epsilon":
        raise ValueError("Sim DDPM prediction_type must be epsilon")
    if bool(getattr(config, "thresholding", False)):
        raise ValueError("Dynamic thresholding is not supported")
    return DDPMScheduler.from_config(config)


def mse_guidance_gradient(
    base_sample: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
) -> torch.Tensor:
    if base_sample.shape[0] != reference.shape[0]:
        raise ValueError(
            f"batch mismatch: base sample batch {base_sample.shape[0]} != "
            f"reference batch {reference.shape[0]}"
        )
    if base_sample.shape[2] != reference.shape[2]:
        raise ValueError(
            f"action dim mismatch: base sample dim {base_sample.shape[2]} != "
            f"reference dim {reference.shape[2]}"
        )
    sliced = base_sample[:, guidance_slice]
    if sliced.shape != reference.shape:
        raise ValueError(
            f"slice length mismatch: base_sample[{guidance_slice}] shape "
            f"{tuple(sliced.shape)} != reference shape {tuple(reference.shape)}"
        )
    if base_sample.dtype != reference.dtype:
        raise ValueError(
            f"dtype mismatch: base sample {base_sample.dtype} != "
            f"reference {reference.dtype}"
        )
    if base_sample.device != reference.device:
        raise ValueError(
            f"device mismatch: base sample {base_sample.device} != "
            f"reference {reference.device}"
        )
    if not (
        torch.isfinite(base_sample).all()
        and torch.isfinite(reference).all()
    ):
        raise ValueError("non-finite inputs are not supported")

    gradient = torch.zeros_like(base_sample)
    scale = 2.0 / (reference.shape[1] * reference.shape[2])
    gradient[:, guidance_slice] = scale * (
        base_sample[:, guidance_slice] - reference
    )
    return gradient


@dataclass(frozen=True)
class GuidedDDPMStepOutput:
    timestep: int
    prev_sample: torch.Tensor
    # Unguided official DDPM reverse sample: posterior mean μ plus official noise.
    base_prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    raw_pred_original_sample: torch.Tensor
    alpha_bar_t: torch.Tensor
    alpha_bar_prev: torch.Tensor
    raw_variance: torch.Tensor
    guidance_loss_before: torch.Tensor
    guidance_loss_after: torch.Tensor


def _require_matching_sample_and_epsilon(
    sample: torch.Tensor,
    model_output: torch.Tensor,
) -> None:
    if sample.shape != model_output.shape:
        raise ValueError(
            f"shape mismatch: sample {tuple(sample.shape)} != "
            f"model_output {tuple(model_output.shape)}"
        )
    if sample.dtype != model_output.dtype:
        raise ValueError(
            f"dtype mismatch: sample {sample.dtype} != model_output {model_output.dtype}"
        )
    if sample.device != model_output.device:
        raise ValueError(
            f"device mismatch: sample {sample.device} != model_output {model_output.device}"
        )
    if not (torch.isfinite(sample).all() and torch.isfinite(model_output).all()):
        raise ValueError("non-finite inputs are not supported")


def _slice_mse(
    pred: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
) -> torch.Tensor:
    return (pred[:, guidance_slice] - reference).square().mean(dim=(1, 2))


def _previous_timestep(scheduler: DDPMScheduler, timestep: int) -> int:
    if hasattr(scheduler, "previous_timestep"):
        return int(scheduler.previous_timestep(timestep))
    return int(timestep) - 1


def _alpha_one(scheduler: DDPMScheduler) -> torch.Tensor:
    one = getattr(scheduler, "one", None)
    if one is None:
        return torch.tensor(1.0)
    return one


def _official_ddpm_mean(
    scheduler: DDPMScheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = int(timestep)
    prev_t = _previous_timestep(scheduler, t)
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else _alpha_one(scheduler)
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    x0_raw = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    if getattr(scheduler.config, "thresholding", False):
        raise ValueError("Dynamic thresholding is not supported")
    if scheduler.config.clip_sample:
        clip_range = float(getattr(scheduler.config, "clip_sample_range", 1.0))
        x0 = x0_raw.clamp(-clip_range, clip_range)
    else:
        x0 = x0_raw

    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = (
        current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
    )
    mu = pred_original_sample_coeff * x0 + current_sample_coeff * sample
    return x0_raw, x0, mu, alpha_prod_t, alpha_prod_t_prev


def _official_ddpm_noise(
    scheduler: DDPMScheduler,
    timestep: int,
    model_output: torch.Tensor,
    variance: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    t = int(timestep)
    if t <= 0:
        return torch.zeros_like(model_output)
    variance_noise = randn_tensor(
        model_output.shape,
        generator=generator,
        device=model_output.device,
        dtype=model_output.dtype,
    )
    return (variance ** 0.5) * variance_noise


def guided_ddpm_step(
    scheduler: DDPMScheduler,
    model_output: torch.Tensor,
    timestep: int | torch.Tensor,
    sample: torch.Tensor,
    reference: torch.Tensor,
    guidance_scale: float,
    guidance_slice: slice,
    generator: torch.Generator | None = None,
) -> GuidedDDPMStepOutput:
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run "
            "'set_timesteps' after creating the scheduler"
        )
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")

    _require_matching_sample_and_epsilon(sample, model_output)

    t = int(timestep)
    x0_raw, x0, mu, alpha_t, alpha_prev = _official_ddpm_mean(
        scheduler, model_output, t, sample
    )
    raw_variance = scheduler._get_variance(t)
    if torch.is_tensor(raw_variance):
        raw_variance = raw_variance.to(device=sample.device, dtype=sample.dtype)
    gradient = mse_guidance_gradient(mu, reference, guidance_slice)
    mu_guided = mu - guidance_scale * raw_variance * gradient
    noise_term = _official_ddpm_noise(
        scheduler, t, model_output, raw_variance, generator
    )
    base_prev_sample = mu + noise_term
    prev_sample = mu_guided + noise_term
    return GuidedDDPMStepOutput(
        timestep=t,
        prev_sample=prev_sample,
        base_prev_sample=base_prev_sample,
        pred_original_sample=x0,
        raw_pred_original_sample=x0_raw,
        alpha_bar_t=alpha_t,
        alpha_bar_prev=alpha_prev,
        raw_variance=raw_variance,
        guidance_loss_before=_slice_mse(mu, reference, guidance_slice),
        guidance_loss_after=_slice_mse(mu_guided, reference, guidance_slice),
    )


@dataclass(frozen=True)
class GuidedDDPMSampleOutput:
    trajectory: torch.Tensor
    steps: tuple[GuidedDDPMStepOutput, ...]


@dataclass(frozen=True)
class ZeroGuidanceReport:
    timesteps: tuple[int, ...]
    terminal_shape: tuple[int, ...]
    max_x0_error: float
    max_prev_error: float


def sample_guided_trajectory(
    model,
    scheduler: DDPMScheduler,
    initial_noise: torch.Tensor,
    global_cond: torch.Tensor,
    reference: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_slice: slice,
    generator: torch.Generator | None = None,
) -> GuidedDDPMSampleOutput:
    if num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {num_inference_steps}"
        )
    scheduler.set_timesteps(
        num_inference_steps,
        device=initial_noise.device,
    )
    trajectory = initial_noise.clone()
    outputs: list[GuidedDDPMStepOutput] = []
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_output = model(
                trajectory,
                timestep,
                global_cond=global_cond,
            )
            output = guided_ddpm_step(
                scheduler=scheduler,
                model_output=model_output,
                timestep=timestep,
                sample=trajectory,
                reference=reference,
                guidance_scale=guidance_scale,
                guidance_slice=guidance_slice,
                generator=generator,
            )
            outputs.append(output)
            trajectory = output.prev_sample
    return GuidedDDPMSampleOutput(trajectory, tuple(outputs))


def _paired_generators(
    generator: torch.Generator | None,
) -> tuple[torch.Generator, torch.Generator]:
    if generator is None:
        official = torch.Generator(device="cpu").manual_seed(0)
        custom = torch.Generator(device="cpu").manual_seed(0)
        return official, custom
    state = generator.get_state().clone()
    official = torch.Generator(device=generator.device)
    official.set_state(state.clone())
    custom = torch.Generator(device=generator.device)
    custom.set_state(state.clone())
    return official, custom


def verify_zero_guidance_equivalence(
    model,
    training_scheduler: DDPMScheduler,
    initial_noise: torch.Tensor,
    global_cond: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
    num_inference_steps: int,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    generator: torch.Generator | None = None,
) -> ZeroGuidanceReport:
    if num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {num_inference_steps}"
        )
    official = create_ddpm_scheduler(training_scheduler)
    custom = create_ddpm_scheduler(training_scheduler)
    official.set_timesteps(num_inference_steps, device=initial_noise.device)
    custom.set_timesteps(num_inference_steps, device=initial_noise.device)
    official_gen, custom_gen = _paired_generators(generator)

    official_sample = initial_noise.clone()
    custom_sample = initial_noise.clone()
    max_x0_error = 0.0
    max_prev_error = 0.0
    step_timesteps: list[int] = []

    with torch.no_grad():
        for timestep in custom.timesteps:
            official_epsilon = model(
                official_sample,
                timestep,
                global_cond=global_cond,
            )
            custom_epsilon = model(
                custom_sample,
                timestep,
                global_cond=global_cond,
            )
            official_out = official.step(
                model_output=official_epsilon,
                timestep=timestep,
                sample=official_sample,
                generator=official_gen,
            )
            custom_out = guided_ddpm_step(
                scheduler=custom,
                model_output=custom_epsilon,
                timestep=timestep,
                sample=custom_sample,
                reference=reference,
                guidance_scale=0.0,
                guidance_slice=guidance_slice,
                generator=custom_gen,
            )
            torch.testing.assert_close(
                custom_out.pred_original_sample,
                official_out.pred_original_sample,
                rtol=rtol,
                atol=atol,
            )
            torch.testing.assert_close(
                custom_out.prev_sample,
                official_out.prev_sample,
                rtol=rtol,
                atol=atol,
            )
            max_x0_error = max(
                max_x0_error,
                float(
                    (custom_out.pred_original_sample - official_out.pred_original_sample)
                    .abs()
                    .max()
                    .item()
                ),
            )
            max_prev_error = max(
                max_prev_error,
                float(
                    (custom_out.prev_sample - official_out.prev_sample)
                    .abs()
                    .max()
                    .item()
                ),
            )
            step_timesteps.append(custom_out.timestep)
            official_sample = official_out.prev_sample
            custom_sample = custom_out.prev_sample

    return ZeroGuidanceReport(
        timesteps=tuple(step_timesteps),
        terminal_shape=tuple(custom_sample.shape),
        max_x0_error=max_x0_error,
        max_prev_error=max_prev_error,
    )
