"""Reference-initialized DDIM editing for a hand-action trajectory.

The known action history is re-noised at every reverse step. The future
reference is only used to initialize the sample and remains editable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass(frozen=True)
class ReferenceEditSampleOutput:
    trajectory: torch.Tensor
    initial_sample: torch.Tensor
    timesteps: tuple[int, ...]
    history_mask_max_error: float
    predicted_x0_clip_fraction: float


@dataclass(frozen=True)
class FusedEditCoeffs:
    timesteps: tuple[int, ...]
    model_timesteps: torch.Tensor
    sqrt_alpha_t: torch.Tensor
    sqrt_one_minus_alpha_t: torch.Tensor
    sqrt_alpha_next: torch.Tensor
    sqrt_one_minus_alpha_next: torch.Tensor


def fused_edit_coeffs(alphas_cumprod: torch.Tensor, timesteps: tuple[int, ...], device, dtype) -> FusedEditCoeffs:
    """Cache the nonuniform SDEdit DDIM coefficients on the inference device."""
    times = tuple(int(t) for t in timesteps)
    alphas = alphas_cumprod.to(device=device, dtype=dtype)
    indices = torch.as_tensor(times, device=device, dtype=torch.long)
    current = alphas[indices]
    following = torch.cat((current[1:], torch.ones_like(current[:1])))
    shape = (-1, 1, 1)
    return FusedEditCoeffs(
        times,
        indices.reshape(-1, 1),
        current.sqrt().reshape(shape),
        (1 - current).sqrt().reshape(shape),
        following.sqrt().reshape(shape),
        (1 - following).sqrt().reshape(shape),
    )


def select_edit_timesteps(
    alphas_cumprod: torch.Tensor,
    noise_ratio: float,
    inference_steps: int,
) -> tuple[tuple[int, ...], float]:
    """Choose the nearest training noise ratio and a nonuniform DDIM path to 0."""
    if alphas_cumprod.ndim != 1 or alphas_cumprod.numel() < 2:
        raise ValueError("alphas_cumprod must be a one-dimensional schedule")
    if not torch.isfinite(alphas_cumprod).all() or not (
        (alphas_cumprod > 0).all() and (alphas_cumprod <= 1).all()
    ):
        raise ValueError("alphas_cumprod must be finite and in (0, 1]")
    if not math.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and nonnegative")
    if inference_steps <= 0 or inference_steps > alphas_cumprod.numel():
        raise ValueError("inference_steps must fit within the training schedule")
    if noise_ratio == 0:
        return (), 0.0

    ratios = ((1 - alphas_cumprod) / alphas_cumprod).sqrt()
    candidates = torch.arange(
        inference_steps - 1, len(ratios), device=ratios.device
    )
    start = int(candidates[(ratios[candidates] - noise_ratio).abs().argmin()])
    # Match the experimental editor's np.rint schedule, including tie handling.
    timesteps = tuple(
        int(t) for t in np.rint(np.linspace(start, 0, inference_steps))
    )
    if len(set(timesteps)) != inference_steps or any(
        a <= b for a, b in zip(timesteps, timesteps[1:])
    ):
        raise ValueError("editing timesteps must be strictly decreasing")
    return timesteps, float(ratios[start])


def ddim_transition(
    sample: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_t: torch.Tensor,
    alpha_next: torch.Tensor,
    clip_sample: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic epsilon DDIM transition for an explicit t -> next_t."""
    predicted_x0 = (sample - (1 - alpha_t).sqrt() * epsilon) / alpha_t.sqrt()
    clean = predicted_x0.clamp(-1, 1) if clip_sample else predicted_x0
    corrected_epsilon = (sample - alpha_t.sqrt() * clean) / (1 - alpha_t).sqrt()
    next_sample = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * corrected_epsilon
    return next_sample, predicted_x0


def sample_reference_edit(
    model,
    alphas_cumprod: torch.Tensor,
    clean_reference: torch.Tensor,
    noise: torch.Tensor,
    global_cond: torch.Tensor,
    timesteps: tuple[int, ...] | list[int],
    known_history_steps: int,
    clip_sample: bool = True,
    fused_coeffs: FusedEditCoeffs | None = None,
) -> ReferenceEditSampleOutput:
    """Edit a normalized full horizon while keeping the history prefix known."""
    if clean_reference.ndim != 3 or clean_reference.shape != noise.shape:
        raise ValueError("clean_reference and noise must share (batch, horizon, action) shape")
    if clean_reference.device != noise.device or clean_reference.dtype != noise.dtype:
        raise ValueError("clean_reference and noise must share device and dtype")
    if not 0 < known_history_steps < clean_reference.shape[1]:
        raise ValueError("known_history_steps must leave an editable future")
    if global_cond.shape[0] != clean_reference.shape[0]:
        raise ValueError("global_cond batch does not match the action batch")
    if not (torch.isfinite(clean_reference).all() and torch.isfinite(noise).all()):
        raise ValueError("reference and noise must be finite")
    times = tuple(int(t) for t in timesteps)
    if any(t < 0 or t >= len(alphas_cumprod) for t in times) or any(
        a <= b for a, b in zip(times, times[1:])
    ):
        raise ValueError("timesteps must be in range and strictly decreasing")
    if not times:
        return ReferenceEditSampleOutput(
            clean_reference.clone(), clean_reference.clone(), (), 0.0, 0.0
        )

    if fused_coeffs is not None and fused_coeffs.timesteps != times:
        raise ValueError("fused edit coefficients do not match timesteps")

    if fused_coeffs is not None:
        coeffs = fused_coeffs
        if coeffs.sqrt_alpha_t.device != noise.device or coeffs.sqrt_alpha_t.dtype != noise.dtype:
            raise ValueError("fused edit coefficients must match the sample device and dtype")
        sample = coeffs.sqrt_alpha_t[0] * clean_reference + coeffs.sqrt_one_minus_alpha_t[0] * noise
        initial_sample = sample.clone()
        clipped = torch.zeros((), device=noise.device, dtype=torch.long)
        for index in range(len(times)):
            a_t = coeffs.sqrt_alpha_t[index]
            b_t = coeffs.sqrt_one_minus_alpha_t[index]
            sample[:, :known_history_steps] = (
                a_t * clean_reference[:, :known_history_steps]
                + b_t * noise[:, :known_history_steps]
            )
            epsilon = model(sample, coeffs.model_timesteps[index], global_cond=global_cond)
            if epsilon.shape != sample.shape:
                raise ValueError("model must return an epsilon matching the sample")
            predicted_x0 = (sample - b_t * epsilon) / a_t
            future_x0 = predicted_x0[:, known_history_steps:]
            clipped += (future_x0.abs() > 1).sum()
            clean = predicted_x0.clamp(-1, 1) if clip_sample else predicted_x0
            corrected_epsilon = (sample - a_t * clean) / b_t
            sample = (
                coeffs.sqrt_alpha_next[index] * clean
                + coeffs.sqrt_one_minus_alpha_next[index] * corrected_epsilon
            )
        sample[:, :known_history_steps] = clean_reference[:, :known_history_steps]
        if not torch.isfinite(sample).all():
            raise ValueError("reference editor produced a non-finite action")
        future_count = len(times) * clean_reference[:, known_history_steps:].numel()
        return ReferenceEditSampleOutput(
            sample, initial_sample, times, 0.0, float(clipped.item() / future_count)
        )

    alphas = alphas_cumprod.to(device=noise.device, dtype=noise.dtype)
    alpha_start = alphas[times[0]]
    sample = alpha_start.sqrt() * clean_reference + (1 - alpha_start).sqrt() * noise
    initial_sample = sample.clone()
    mask_error = 0.0
    clipped = 0
    evaluated = 0
    for index, timestep in enumerate(times):
        alpha_t = alphas[timestep]
        known_at_t = (
            alpha_t.sqrt() * clean_reference[:, :known_history_steps]
            + (1 - alpha_t).sqrt() * noise[:, :known_history_steps]
        )
        sample[:, :known_history_steps] = known_at_t
        mask_error = max(
            mask_error,
            float((sample[:, :known_history_steps] - known_at_t).abs().max()),
        )
        epsilon = model(sample, torch.tensor(timestep, device=noise.device), global_cond=global_cond)
        if epsilon.shape != sample.shape or not torch.isfinite(epsilon).all():
            raise ValueError("model must return a finite epsilon matching the sample")
        next_t = times[index + 1] if index + 1 < len(times) else -1
        alpha_next = alphas[next_t] if next_t >= 0 else torch.ones_like(alpha_t)
        sample, predicted_x0 = ddim_transition(
            sample, epsilon, alpha_t, alpha_next, clip_sample
        )
        future_x0 = predicted_x0[:, known_history_steps:]
        clipped += int((future_x0.abs() > 1).sum())
        evaluated += future_x0.numel()
    sample[:, :known_history_steps] = clean_reference[:, :known_history_steps]
    if not torch.isfinite(sample).all():
        raise ValueError("reference editor produced a non-finite action")
    return ReferenceEditSampleOutput(
        sample, initial_sample, times, mask_error, clipped / evaluated
    )
