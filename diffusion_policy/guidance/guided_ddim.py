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


def _validate_guidance_reference(
    predicted_x0: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
) -> None:
    if predicted_x0.shape[0] != reference.shape[0]:
        raise ValueError(
            f"batch mismatch: predicted x0 batch {predicted_x0.shape[0]} != "
            f"reference batch {reference.shape[0]}"
        )
    if predicted_x0.shape[2] != reference.shape[2]:
        raise ValueError(
            f"action dim mismatch: predicted x0 dim {predicted_x0.shape[2]} != "
            f"reference dim {reference.shape[2]}"
        )
    sliced = predicted_x0[:, guidance_slice]
    if sliced.shape != reference.shape:
        raise ValueError(
            f"slice length mismatch: predicted_x0[{guidance_slice}] shape "
            f"{tuple(sliced.shape)} != reference shape {tuple(reference.shape)}"
        )
    if predicted_x0.dtype != reference.dtype:
        raise ValueError(
            f"dtype mismatch: predicted x0 {predicted_x0.dtype} != "
            f"reference {reference.dtype}"
        )
    if predicted_x0.device != reference.device:
        raise ValueError(
            f"device mismatch: predicted x0 {predicted_x0.device} != "
            f"reference {reference.device}"
        )
    if not (
        torch.isfinite(predicted_x0).all()
        and torch.isfinite(reference).all()
    ):
        raise ValueError("non-finite inputs are not supported")


@dataclass(frozen=True)
class GuidedDDIMStepOutput:
    timestep: int
    prev_sample: torch.Tensor
    base_prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    raw_pred_original_sample: torch.Tensor
    alpha_bar_t: torch.Tensor
    alpha_bar_prev: torch.Tensor
    direction_coefficient: torch.Tensor
    guidance_loss_before: torch.Tensor
    guidance_loss_after: torch.Tensor


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


def _require_finite_guidance_scale(guidance_scale: float) -> None:
    if not torch.isfinite(torch.as_tensor(guidance_scale)):
        raise ValueError("guidance_scale must be finite")


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
    _require_finite_guidance_scale(guidance_scale)
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
    sqrt_alpha_t = alpha_t.sqrt()
    sqrt_beta_t = (1.0 - alpha_t).sqrt()
    x0_raw = (sample - sqrt_beta_t * model_output) / sqrt_alpha_t
    x0_base = x0_raw.clamp(-1.0, 1.0) if scheduler.config.clip_sample else x0_raw
    _validate_guidance_reference(x0_raw, reference, guidance_slice)
    # Scheme B: ε_guided = εθ + b_t * λ * ∇_{x_t} D(x0_hat, y), including ∂x0_hat/∂x_t.
    if guidance_scale > 0:
        if not sample.requires_grad:
            raise ValueError(
                "sample must require gradients when guidance_scale is positive"
            )
        guidance_loss = _slice_mse(x0_raw, reference, guidance_slice).sum()
        gradient = torch.autograd.grad(guidance_loss, sample)[0]
    else:
        gradient = torch.zeros_like(sample)
    guided_epsilon = model_output + sqrt_beta_t * guidance_scale * gradient
    guided_x0_raw = (sample - sqrt_beta_t * guided_epsilon) / sqrt_alpha_t
    guided_x0 = (
        guided_x0_raw.clamp(-1.0, 1.0)
        if scheduler.config.clip_sample
        else guided_x0_raw
    )
    direction = (1.0 - alpha_prev).sqrt()
    base_prev_sample = (
        alpha_prev.sqrt() * x0_base + direction * model_output
    )
    prev_sample = (
        alpha_prev.sqrt() * guided_x0 + direction * guided_epsilon
    )
    return GuidedDDIMStepOutput(
        timestep=t,
        prev_sample=prev_sample.detach(),
        base_prev_sample=base_prev_sample.detach(),
        pred_original_sample=guided_x0.detach(),
        raw_pred_original_sample=guided_x0_raw.detach(),
        alpha_bar_t=alpha_t,
        alpha_bar_prev=alpha_prev,
        direction_coefficient=direction,
        guidance_loss_before=_slice_mse(
            x0_raw,
            reference,
            guidance_slice,
        ).detach(),
        guidance_loss_after=_slice_mse(
            guided_x0_raw,
            reference,
            guidance_slice,
        ).detach(),
    )


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
    trajectory = initial_noise
    outputs: list[GuidedDDIMStepOutput] = []
    for timestep in scheduler.timesteps:
        if guidance_scale > 0:
            with torch.inference_mode(False), torch.enable_grad():
                xt = trajectory.detach().clone().requires_grad_(True)
                model_output = model(
                    xt,
                    timestep,
                    global_cond=global_cond,
                )
                output = guided_ddim_step(
                    scheduler=scheduler,
                    model_output=model_output,
                    timestep=timestep,
                    sample=xt,
                    reference=reference,
                    guidance_scale=guidance_scale,
                    guidance_slice=guidance_slice,
                    eta=eta,
                )
        else:
            with torch.no_grad():
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
        trajectory = output.prev_sample.detach()
    return GuidedDDIMSampleOutput(trajectory.detach(), tuple(outputs))


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
