"""TensorRT UNet plus a GPU fused DDIM loop.

The UNet is compiled with torch-tensorrt.  The 4-step (or N-step) DDIM update
stays in PyTorch but uses precomputed GPU coefficients, matching HuggingFace
``scheduler.step`` for epsilon prediction with ``eta=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
import os
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExportableGroupNorm(nn.Module):
    """Express group normalization as per-group LayerNorm + channel affine.

    Torch-TensorRT 2.4's native GroupNorm conversion produced large errors on
    these checkpoints. Keep the affine outside the normalization converter
    and only transform the export copy of the UNet.
    """

    def __init__(self, norm):
        super().__init__()
        self.num_groups = norm.num_groups
        self.eps = norm.eps
        self.weight = norm.weight
        self.bias = norm.bias

    def forward(self, value):
        grouped = value.reshape(value.shape[0], self.num_groups, -1)
        normalized = F.layer_norm(grouped, (grouped.shape[-1],), eps=self.eps)
        normalized = normalized.reshape_as(value)
        affine_shape = (1, value.shape[1]) + (1,) * (value.ndim - 2)
        if self.weight is not None:
            normalized = normalized * self.weight.reshape(affine_shape)
        if self.bias is not None:
            normalized = normalized + self.bias.reshape(affine_shape)
        return normalized


def _replace_export_group_norm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, ExportableGroupNorm(child))
        else:
            _replace_export_group_norm(child)


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


def fused_guided_ddim_update(
    sample, epsilon, step_index, coeffs, reference, guidance_scale, guidance_slice
):
    """Match guided_ddim_step using the analytic frozen-epsilon MSE gradient.

    The reference implementation clips base x0 to [-1, 1], re-derives epsilon
    even at scale zero, and does not clip guided x0. Ordinary fused_ddim_update
    therefore cannot replace this step. No UNet backward pass is needed.
    """
    sqrt_alpha = coeffs.sqrt_alpha_t[step_index]
    sqrt_beta = coeffs.sqrt_one_minus_alpha_t[step_index]
    x0_raw = (sample - sqrt_beta * epsilon) / sqrt_alpha
    x0_base = x0_raw.clamp(-1.0, 1.0) if coeffs.clip_sample else x0_raw
    guided_epsilon = (sample - sqrt_alpha * x0_base) / sqrt_beta
    if guidance_scale > 0:
        difference = x0_raw[:, guidance_slice] - reference
        # loss = sum_batch(mean_window,action((x0_raw - reference)^2))
        gradient = (2.0 / (difference.shape[1] * difference.shape[2])) * difference / sqrt_alpha
        guided_epsilon[:, guidance_slice] += sqrt_beta * guidance_scale * gradient
    guided_x0 = (sample - sqrt_beta * guided_epsilon) / sqrt_alpha
    return (
        coeffs.sqrt_alpha_prev[step_index] * guided_x0
        + coeffs.sqrt_one_minus_alpha_prev[step_index] * guided_epsilon
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
    capture = getattr(policy, "_debug_capture", None)
    if capture is not None:
        capture["initial_noise"] = trajectory.detach().clone()
        capture["timesteps"] = coeffs.timesteps.detach().clone()
        capture["scheduler_capture_method"] = "deterministic_eta_zero"
        capture["scheduler_noise"] = []
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
    if capture is not None:
        capture["normalized_trajectory"] = trajectory.detach().clone()
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

    export_unet = deepcopy(unet)
    _replace_export_group_norm(export_unet)
    wrapper = ExportableUnet(export_unet).to(device).eval()
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
        # The FP32 comparison should not silently use reduced-mantissa TF32.
        disable_tf32=not fp16,
    )
    return compiled


def _engine_cache_metadata(policy, source_path, device, fp16, max_batch):
    import tensorrt
    import torch_tensorrt

    source = Path(source_path).expanduser().resolve()
    stat = source.stat()
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "export_revision": "groupnorm-layernorm-v1",
        "source": str(source),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "horizon": int(policy.horizon),
        "action_dim": int(policy.action_dim),
        "global_cond_dim": int(policy.n_obs_steps) * int(policy.obs_dim),
        "precision": "fp16" if fp16 else "fp32",
        "max_batch": int(max_batch),
        "torch": torch.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "tensorrt": tensorrt.__version__,
        "cuda_device": properties.name,
        "cuda_capability": [properties.major, properties.minor],
    }


def _example_inputs(policy, device):
    return (
        torch.zeros(
            1, int(policy.horizon), int(policy.action_dim),
            device=device, dtype=torch.float32,
        ),
        torch.zeros(1, device=device, dtype=torch.int64),
        torch.zeros(
            1, int(policy.n_obs_steps) * int(policy.obs_dim),
            device=device, dtype=torch.float32,
        ),
    )


def accelerate_policy_unet(
    policy, fp16=True, max_batch=1024, *, engine_path=None, source_path=None
):
    """Replace the UNet with a validated cached TensorRT engine and fused DDIM."""
    import torch_tensorrt

    device = next(policy.model.parameters()).device
    global_cond_dim = int(policy.n_obs_steps) * int(policy.obs_dim)
    cache_path = Path(engine_path).expanduser().resolve() if engine_path else None
    metadata_path = (
        cache_path.with_name(cache_path.name + ".json") if cache_path else None
    )
    metadata = None
    if cache_path is not None:
        if source_path is None:
            raise ValueError("source_path is required when engine_path is set")
        metadata = _engine_cache_metadata(
            policy, source_path, device, fp16, max_batch
        )

    compiled = None
    if cache_path is not None and cache_path.is_file() and metadata_path.is_file():
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if cached_metadata == metadata:
                compiled = torch.jit.load(str(cache_path), map_location=device).eval()
                print(f"[TensorRT] loaded cached engine: {cache_path}", flush=True)
            else:
                print(
                    f"[TensorRT] cache metadata changed; rebuilding: {cache_path}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[TensorRT] cached engine could not be loaded ({exc}); rebuilding",
                flush=True,
            )

    if compiled is None:
        compiled = compile_unet(
            policy.model,
            horizon=policy.horizon,
            action_dim=policy.action_dim,
            global_cond_dim=global_cond_dim,
            device=device,
            fp16=fp16,
            max_batch=max_batch,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_name(
                f".{cache_path.name}.{os.getpid()}.tmp"
            )
            temporary_metadata_path = metadata_path.with_name(
                f".{metadata_path.name}.{os.getpid()}.tmp"
            )
            try:
                torch_tensorrt.save(
                    compiled,
                    str(temporary_path),
                    output_format="torchscript",
                    inputs=_example_inputs(policy, device),
                )
                temporary_metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary_path.replace(cache_path)
                temporary_metadata_path.replace(metadata_path)
                print(f"[TensorRT] saved engine cache: {cache_path}", flush=True)
            finally:
                temporary_path.unlink(missing_ok=True)
                temporary_metadata_path.unlink(missing_ok=True)
    policy.model = TensorRTUnetAdapter(compiled)
    policy.noise_scheduler.set_timesteps(policy.num_inference_steps, device=device)
    policy.conditional_sample = MethodType(fused_conditional_sample, policy)
    return policy


def default_engine_path(checkpoint, fp16=True):
    path = Path(checkpoint).expanduser().resolve()
    precision = "fp16" if fp16 else "fp32"
    return path.with_name(f"{path.stem}.unet.{precision}.trt.pt")
