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
