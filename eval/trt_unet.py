"""TensorRT UNet plus a GPU fused DDIM loop.

The UNet is compiled with torch-tensorrt.  The 4-step (or N-step) DDIM update
stays in PyTorch but uses precomputed GPU coefficients, matching HuggingFace
``scheduler.step`` for epsilon prediction with ``eta=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn


class ExportableUnet(nn.Module):
    """ONNX / TensorRT-friendly wrapper: no optional kwargs, tensor timestep."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, global_cond):
        return self.unet(sample, timestep, local_cond=None, global_cond=global_cond)


class TensorRTUnetAdapter(nn.Module):
    """Drop-in for ``ConditionalUnet1D.forward`` used by ``predict_action``."""

    def __init__(self, compiled):
        super().__init__()
        self.compiled = compiled

    def forward(self, sample, timestep, local_cond=None, global_cond=None, **kwargs):
        timestep = _cuda_timestep(timestep, sample)
        return self.compiled(sample, timestep, global_cond)


@dataclass(frozen=True)
class FusedDDIMCoeffs:
    timesteps: torch.Tensor
    sqrt_alpha_t: torch.Tensor
    sqrt_one_minus_alpha_t: torch.Tensor
    sqrt_alpha_prev: torch.Tensor
    sqrt_one_minus_alpha_prev: torch.Tensor
    clip_sample: bool
    clip_sample_range: float


def _cuda_timestep(timestep, sample):
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([int(timestep)], device=sample.device, dtype=torch.int64)
    else:
        timestep = timestep.to(device=sample.device, dtype=torch.int64)
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
    if timestep.shape[0] != sample.shape[0]:
        timestep = timestep.expand(sample.shape[0])
    return timestep.contiguous()


def fused_ddim_coeffs_from_scheduler(scheduler, device, dtype):
    """Precompute DDIM eta=0 coefficients, matching HuggingFace ``step``."""
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError(
            "fused DDIM only supports epsilon prediction, got %r"
            % (scheduler.config.prediction_type,)
        )
    n_train = int(scheduler.config.num_train_timesteps)
    n_inf = int(scheduler.num_inference_steps)
    step_size = n_train // n_inf
    alphas = scheduler.alphas_cumprod
    final_alpha = scheduler.final_alpha_cumprod
    timesteps = scheduler.timesteps

    sqrt_alpha_t = []
    sqrt_one_minus_alpha_t = []
    sqrt_alpha_prev = []
    sqrt_one_minus_alpha_prev = []
    for timestep in timesteps.tolist():
        timestep = int(timestep)
        prev_timestep = timestep - step_size
        alpha_t = alphas[timestep]
        alpha_prev = alphas[prev_timestep] if prev_timestep >= 0 else final_alpha
        alpha_t = torch.as_tensor(alpha_t, dtype=torch.float32)
        alpha_prev = torch.as_tensor(alpha_prev, dtype=torch.float32)
        sqrt_alpha_t.append(alpha_t.sqrt())
        sqrt_one_minus_alpha_t.append((1.0 - alpha_t).clamp(min=0.0).sqrt())
        sqrt_alpha_prev.append(alpha_prev.sqrt())
        sqrt_one_minus_alpha_prev.append((1.0 - alpha_prev).clamp(min=0.0).sqrt())

    def _stack(values):
        return torch.stack(values).to(device=device, dtype=dtype).view(-1, 1, 1)

    return FusedDDIMCoeffs(
        timesteps=timesteps.to(device=device, dtype=torch.int64),
        sqrt_alpha_t=_stack(sqrt_alpha_t),
        sqrt_one_minus_alpha_t=_stack(sqrt_one_minus_alpha_t),
        sqrt_alpha_prev=_stack(sqrt_alpha_prev),
        sqrt_one_minus_alpha_prev=_stack(sqrt_one_minus_alpha_prev),
        clip_sample=bool(scheduler.config.clip_sample),
        clip_sample_range=float(scheduler.config.clip_sample_range),
    )


def fused_ddim_update(sample, epsilon, step_index, coeffs):
    pred_x0 = (
        sample - coeffs.sqrt_one_minus_alpha_t[step_index] * epsilon
    ) / coeffs.sqrt_alpha_t[step_index].clamp(min=1e-8)
    if coeffs.clip_sample:
        pred_x0 = pred_x0.clamp(-coeffs.clip_sample_range, coeffs.clip_sample_range)
    return (
        coeffs.sqrt_alpha_prev[step_index] * pred_x0
        + coeffs.sqrt_one_minus_alpha_prev[step_index] * epsilon
    )


def _cached_fused_coeffs(policy, sample):
    n_inf = int(policy.num_inference_steps)
    cache_key = (n_inf, sample.device.type, str(sample.device), str(sample.dtype))
    cached = getattr(policy, "_fused_ddim_cache", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(n_inf, device=sample.device)
    coeffs = fused_ddim_coeffs_from_scheduler(scheduler, sample.device, sample.dtype)
    policy._fused_ddim_cache = (cache_key, coeffs)
    return coeffs


def fused_conditional_sample(
    policy,
    condition_data,
    condition_mask,
    local_cond=None,
    global_cond=None,
    generator=None,
    **kwargs
):
    coeffs = _cached_fused_coeffs(policy, condition_data)
    trajectory = torch.randn(
        size=condition_data.shape,
        dtype=condition_data.dtype,
        device=condition_data.device,
        generator=generator,
    )
    # Global / local cond keeps the action mask all-False; skip the copy.
    apply_cond = not (
        getattr(policy, "obs_as_global_cond", False)
        or getattr(policy, "obs_as_local_cond", False)
    )
    n_steps = int(coeffs.timesteps.shape[0])
    for step_index in range(n_steps):
        if apply_cond:
            trajectory = torch.where(condition_mask, condition_data, trajectory)
        timestep = coeffs.timesteps[step_index]
        epsilon = policy.model(
            trajectory,
            timestep,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        trajectory = fused_ddim_update(trajectory, epsilon, step_index, coeffs)
    if apply_cond:
        trajectory = torch.where(condition_mask, condition_data, trajectory)
    return trajectory


def compile_unet(
    unet,
    *,
    horizon,
    action_dim,
    global_cond_dim,
    device,
    fp16=True,
    max_batch=1024,
):
    import torch_tensorrt

    wrapper = ExportableUnet(unet).to(device).eval()
    horizon = int(horizon)
    action_dim = int(action_dim)
    global_cond_dim = int(global_cond_dim)
    max_batch = int(max_batch)
    if max_batch < 1:
        raise ValueError("max_batch must be >= 1, got %r" % (max_batch,))
    # Real inference uses batch 1; keep a larger max for benchmarks.
    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, horizon, action_dim),
            opt_shape=(max_batch, horizon, action_dim),
            max_shape=(max_batch, horizon, action_dim),
            dtype=torch.float32,
        ),
        torch_tensorrt.Input(
            min_shape=(1,),
            opt_shape=(max_batch,),
            max_shape=(max_batch,),
            dtype=torch.int64,
        ),
        torch_tensorrt.Input(
            min_shape=(1, global_cond_dim),
            opt_shape=(max_batch, global_cond_dim),
            max_shape=(max_batch, global_cond_dim),
            dtype=torch.float32,
        ),
    ]
    precisions = {torch.float16} if fp16 else {torch.float32}
    compiled = torch_tensorrt.compile(
        wrapper,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions=precisions,
        min_block_size=1,
    )
    return compiled


def accelerate_policy_unet(policy, fp16=True, max_batch=1024):
    """Replace the UNet with TensorRT and swap in the fused DDIM loop."""
    device = next(policy.model.parameters()).device
    global_cond_dim = int(policy.n_obs_steps) * int(policy.obs_dim)
    compiled = compile_unet(
        policy.model,
        horizon=policy.horizon,
        action_dim=policy.action_dim,
        global_cond_dim=global_cond_dim,
        device=device,
        fp16=fp16,
        max_batch=max_batch,
    )
    policy.model = TensorRTUnetAdapter(compiled)
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps, device=device)
    policy.conditional_sample = MethodType(fused_conditional_sample, policy)
    return policy


def default_engine_path(checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    return path.with_name(path.stem + ".unet.fp16.trt.pt")
