from __future__ import annotations

from dataclasses import dataclass

import diffusers
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

EXPECTED_DIFFUSERS_VERSION = "0.11.1"


def assert_pinned_diffusers_version() -> None:
    if diffusers.__version__ != EXPECTED_DIFFUSERS_VERSION:
        raise RuntimeError(
            f"Guided DDIM requires diffusers=={EXPECTED_DIFFUSERS_VERSION}, "
            f"got {diffusers.__version__}"
        )


def create_ddim_scheduler(training_scheduler: DDPMScheduler) -> DDIMScheduler:
    assert_pinned_diffusers_version()
    config = training_scheduler.config
    if config.prediction_type != "epsilon":
        raise ValueError("Sim DDPM prediction_type must be epsilon")
    if bool(getattr(config, "thresholding", False)):
        raise ValueError("Dynamic thresholding is not supported")
    return DDIMScheduler.from_config(
        config,
        set_alpha_to_one=True,
        steps_offset=0,
    )


def mse_guidance_gradient(
    x0_base: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
) -> torch.Tensor:
    if x0_base.shape[0] != reference.shape[0]:
        raise ValueError(
            f"batch mismatch: x0_base batch {x0_base.shape[0]} != "
            f"reference batch {reference.shape[0]}"
        )
    if x0_base.shape[2] != reference.shape[2]:
        raise ValueError(
            f"action dim mismatch: x0_base dim {x0_base.shape[2]} != "
            f"reference dim {reference.shape[2]}"
        )
    sliced = x0_base[:, guidance_slice]
    if sliced.shape != reference.shape:
        raise ValueError(
            f"slice length mismatch: x0_base[{guidance_slice}] shape "
            f"{tuple(sliced.shape)} != reference shape {tuple(reference.shape)}"
        )
    if x0_base.dtype != reference.dtype:
        raise ValueError(
            f"dtype mismatch: x0_base {x0_base.dtype} != reference {reference.dtype}"
        )
    if x0_base.device != reference.device:
        raise ValueError(
            f"device mismatch: x0_base {x0_base.device} != reference {reference.device}"
        )
    if not (torch.isfinite(x0_base).all() and torch.isfinite(reference).all()):
        raise ValueError("non-finite inputs are not supported")

    gradient = torch.zeros_like(x0_base)
    scale = 2.0 / (reference.shape[1] * reference.shape[2])
    gradient[:, guidance_slice] = scale * (x0_base[:, guidance_slice] - reference)
    return gradient


@dataclass(frozen=True)
class GuidedDDIMStepOutput:
    timestep: int
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    base_pred_original_sample: torch.Tensor
    raw_pred_original_sample: torch.Tensor
    alpha_bar_t: torch.Tensor
    alpha_bar_prev: torch.Tensor
    raw_variance: torch.Tensor
    direction_coefficient: torch.Tensor
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


def guided_ddim_step(
    scheduler: DDIMScheduler,
    model_output: torch.Tensor,
    timestep: int | torch.Tensor,
    sample: torch.Tensor,
    reference: torch.Tensor,
    guidance_scale: float,
    guidance_slice: slice,
    eta: float = 0.0,
) -> GuidedDDIMStepOutput:
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run "
            "'set_timesteps' after creating the scheduler"
        )
    if eta != 0.0:
        raise ValueError("eta must be 0.0 for guided DDIM")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")

    _require_matching_sample_and_epsilon(sample, model_output)

    t = int(timestep)
    step_ratio = (
        scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )
    prev_t = t - step_ratio
    alpha_t = scheduler.alphas_cumprod[t].to(device=sample.device, dtype=sample.dtype)
    alpha_prev = (
        scheduler.alphas_cumprod[prev_t]
        if prev_t >= 0
        else scheduler.final_alpha_cumprod
    ).to(device=sample.device, dtype=sample.dtype)
    beta_t = 1.0 - alpha_t
    x0_raw = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
    x0_base = x0_raw.clamp(-1.0, 1.0) if scheduler.config.clip_sample else x0_raw
    raw_variance = scheduler._get_variance(t, prev_t).to(
        device=sample.device, dtype=sample.dtype
    )
    gradient = mse_guidance_gradient(x0_base, reference, guidance_slice)
    x0_guided = x0_base - guidance_scale * raw_variance * gradient
    direction = (1.0 - alpha_prev).sqrt()
    prev_sample = alpha_prev.sqrt() * x0_guided + direction * model_output
    return GuidedDDIMStepOutput(
        timestep=t,
        prev_sample=prev_sample,
        pred_original_sample=x0_guided,
        base_pred_original_sample=x0_base,
        raw_pred_original_sample=x0_raw,
        alpha_bar_t=alpha_t,
        alpha_bar_prev=alpha_prev,
        raw_variance=raw_variance,
        direction_coefficient=direction,
        guidance_loss_before=_slice_mse(x0_base, reference, guidance_slice),
        guidance_loss_after=_slice_mse(x0_guided, reference, guidance_slice),
    )


@dataclass(frozen=True)
class GuidedDDIMSampleOutput:
    trajectory: torch.Tensor
    steps: tuple[GuidedDDIMStepOutput, ...]


@dataclass(frozen=True)
class ZeroGuidanceReport:
    timesteps: tuple[int, ...]
    terminal_shape: tuple[int, ...]
    max_x0_error: float
    max_prev_error: float


def sample_guided_trajectory(
    model,
    scheduler: DDIMScheduler,
    initial_noise: torch.Tensor,
    global_cond: torch.Tensor,
    reference: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_slice: slice,
    eta: float = 0.0,
) -> GuidedDDIMSampleOutput:
    if num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {num_inference_steps}"
        )
    scheduler.set_timesteps(
        num_inference_steps,
        device=initial_noise.device,
    )
    trajectory = initial_noise.clone()
    outputs: list[GuidedDDIMStepOutput] = []
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_output = model(
                trajectory,
                timestep,
                global_cond=global_cond,
            )
            output = guided_ddim_step(
                scheduler=scheduler,
                model_output=model_output,
                timestep=timestep,
                sample=trajectory,
                reference=reference,
                guidance_scale=guidance_scale,
                guidance_slice=guidance_slice,
                eta=eta,
            )
            outputs.append(output)
            trajectory = output.prev_sample
    return GuidedDDIMSampleOutput(trajectory, tuple(outputs))


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
) -> ZeroGuidanceReport:
    if num_inference_steps <= 0:
        raise ValueError(
            f"num_inference_steps must be positive, got {num_inference_steps}"
        )
    official = create_ddim_scheduler(training_scheduler)
    custom = create_ddim_scheduler(training_scheduler)
    official.set_timesteps(num_inference_steps, device=initial_noise.device)
    custom.set_timesteps(num_inference_steps, device=initial_noise.device)

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
                eta=0.0,
            )
            custom_out = guided_ddim_step(
                scheduler=custom,
                model_output=custom_epsilon,
                timestep=timestep,
                sample=custom_sample,
                reference=reference,
                guidance_scale=0.0,
                guidance_slice=guidance_slice,
                eta=0.0,
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
