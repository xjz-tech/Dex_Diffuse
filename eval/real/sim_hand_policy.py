"""Minimal local Sim-Hand Diffusion Policy implementation for inference.

This is the checkpoint-compatible subset of the training model: normalizers,
the conditional 1-D U-Net, mask metadata, and action sampling.  It is kept here
so the real runner does not import Python code from outside ``eval/real``.
"""

from __future__ import annotations

import math
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class ModuleAttrMixin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Present in the original checkpoint state_dict.
        self._dummy_variable = nn.Parameter()

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype


class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict: nn.ParameterDict | None = None) -> None:
        super().__init__()
        self.params_dict = params_dict if params_dict is not None else nn.ParameterDict()


def _as_flat_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.from_numpy(value)
    # The identity initializer intentionally reuses zero/one source tensors.
    # Own each parameter's storage so loading one checkpoint field cannot
    # overwrite another field through an aliased view.
    return value.flatten().clone()


def _normalize(value, params: nn.ParameterDict, forward: bool) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    scale = params["scale"]
    offset = params["offset"]
    value = value.to(device=scale.device, dtype=scale.dtype)
    source_shape = value.shape
    value = value.reshape(-1, scale.shape[0])
    if forward:
        value = value * scale + offset
    else:
        value = (value - offset) / scale
    return value.reshape(source_shape)


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    @classmethod
    def create_manual(
        cls,
        scale: torch.Tensor | np.ndarray,
        offset: torch.Tensor | np.ndarray,
        input_stats_dict: dict[str, torch.Tensor | np.ndarray],
    ) -> "SingleFieldLinearNormalizer":
        scale = _as_flat_tensor(scale)
        offset = _as_flat_tensor(offset)
        stats = {key: _as_flat_tensor(value) for key, value in input_stats_dict.items()}
        if offset.shape != scale.shape or any(
            value.shape != scale.shape for value in stats.values()
        ):
            raise ValueError("normalizer parameter shapes differ")
        params = nn.ParameterDict(
            {
                "scale": scale,
                "offset": offset,
                "input_stats": nn.ParameterDict(stats),
            }
        )
        return cls(params)

    def normalize(self, value) -> torch.Tensor:
        return _normalize(value, self.params_dict, forward=True)

    def unnormalize(self, value) -> torch.Tensor:
        return _normalize(value, self.params_dict, forward=False)

    def get_input_stats(self) -> nn.ParameterDict:
        return self.params_dict["input_stats"]


class LinearNormalizer(DictOfTensorMixin):
    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        self.params_dict[key] = value.params_dict

    def normalize(self, value):
        if isinstance(value, dict):
            return {
                key: _normalize(item, self.params_dict[key], forward=True)
                for key, item in value.items()
            }
        if "_default" not in self.params_dict:
            raise RuntimeError("normalizer is not initialized")
        return _normalize(value, self.params_dict["_default"], forward=True)

    def unnormalize(self, value):
        if isinstance(value, dict):
            return {
                key: _normalize(item, self.params_dict[key], forward=False)
                for key, item in value.items()
            }
        if "_default" not in self.params_dict:
            raise RuntimeError("normalizer is not initialized")
        return _normalize(value, self.params_dict["_default"], forward=False)

    def get_input_stats(self) -> dict[str, nn.ParameterDict] | nn.ParameterDict:
        if len(self.params_dict) == 0:
            raise RuntimeError("normalizer is not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]
        return {
            key: value["input_stats"]
            for key, value in self.params_dict.items()
            if key != "_default"
        }


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(
            torch.arange(half_dim, device=value.device) * -exponent
        )
        embedding = value[:, None] * embedding[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.GroupNorm(n_groups, output_channels),
            nn.Mish(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = int(out_channels)
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        output = self.blocks[0](value)
        embedding = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embedding = embedding.reshape(
                embedding.shape[0], 2, self.out_channels, 1
            )
            scale = embedding[:, 0, ...]
            bias = embedding[:, 1, ...]
            output = scale * output + bias
        else:
            output = output + embedding
        output = self.blocks[1](output)
        return output + self.residual_conv(value)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_cond_dim: int | None = None,
        global_cond_dim: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        embed_dim = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        cond_dim = embed_dim + (global_cond_dim or 0)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            first_output_dim = in_out[0][1]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        first_output_dim,
                        cond_dim,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                    ConditionalResidualBlock1D(
                        local_cond_dim,
                        first_output_dim,
                        cond_dim,
                        kernel_size,
                        n_groups,
                        cond_predict_scale,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim,
                    kernel_size,
                    n_groups,
                    cond_predict_scale,
                ),
            ]
        )

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        nn.Identity() if is_last else Downsample1d(dim_out),
                    ]
                )
            )

        self.up_modules = nn.ModuleList()
        reversed_in_out = list(reversed(in_out[1:]))
        for index, (dim_in, dim_out) in enumerate(reversed_in_out):
            # Keep the original checkpoint's condition (and module topology).
            is_last = index >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim,
                            kernel_size,
                            n_groups,
                            cond_predict_scale,
                        ),
                        nn.Identity() if is_last else Upsample1d(dim_in),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        **unused: Any,
    ) -> torch.Tensor:
        del unused
        sample = einops.rearrange(sample, "b h t -> b t h")
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        local_features = []
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            assert self.local_cond_encoder is not None
            first, second = self.local_cond_encoder
            local_features.append(first(local_cond, global_feature))
            local_features.append(second(local_cond, global_feature))

        value = sample
        skips = []
        for index, (first, second, downsample) in enumerate(self.down_modules):
            value = first(value, global_feature)
            if index == 0 and local_features:
                value = value + local_features[0]
            value = second(value, global_feature)
            skips.append(value)
            value = downsample(value)

        for middle in self.mid_modules:
            value = middle(value, global_feature)

        for index, (first, second, upsample) in enumerate(self.up_modules):
            value = torch.cat((value, skips.pop()), dim=1)
            value = first(value, global_feature)
            # This intentionally matches the published implementation. The
            # condition is unreachable but changing it breaks old checkpoints.
            if index == len(self.up_modules) and local_features:
                value = value + local_features[1]
            value = second(value, global_feature)
            value = upsample(value)

        value = self.final_conv(value)
        return einops.rearrange(value, "b t h -> b h t")


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int,
        fix_obs_steps: bool,
        action_visible: bool,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.max_n_obs_steps = int(max_n_obs_steps)
        self.fix_obs_steps = bool(fix_obs_steps)
        self.action_visible = bool(action_visible)


class DiffusionUnetLowdimPolicy(ModuleAttrMixin):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int | None = None,
        obs_as_local_cond: bool = False,
        obs_as_global_cond: bool = False,
        pred_action_steps_only: bool = False,
        oa_step_convention: bool = False,
        **scheduler_step_kwargs: Any,
    ) -> None:
        super().__init__()
        if obs_as_local_cond and obs_as_global_cond:
            raise ValueError("local and global observation conditioning conflict")
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim,
            0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            n_obs_steps,
            True,
            False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_as_local_cond = bool(obs_as_local_cond)
        self.obs_as_global_cond = bool(obs_as_global_cond)
        self.pred_action_steps_only = bool(pred_action_steps_only)
        self.oa_step_convention = bool(oa_step_convention)
        self.scheduler_step_kwargs = scheduler_step_kwargs
        self.num_inference_steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else noise_scheduler.config.num_train_timesteps
        )

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory,
                timestep,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                **self.scheduler_step_kwargs,
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if "obs" not in obs_dict or "past_action" in obs_dict:
            raise ValueError("obs_dict must contain only the supported observation inputs")
        normalized_obs = self.normalizer["obs"].normalize(obs_dict["obs"])
        batch, _, observed_dim = normalized_obs.shape
        if observed_dim != self.obs_dim:
            raise ValueError(
                f"observation dim {observed_dim}, expected {self.obs_dim}"
            )
        observed_steps = self.n_obs_steps
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            global_cond = normalized_obs[:, :observed_steps].reshape(batch, -1)
            shape = (batch, self.horizon, self.action_dim)
            if self.pred_action_steps_only:
                shape = (batch, self.n_action_steps, self.action_dim)
        elif self.obs_as_local_cond:
            local_cond = torch.zeros(
                (batch, self.horizon, self.obs_dim),
                device=self.device,
                dtype=self.dtype,
            )
            local_cond[:, :observed_steps] = normalized_obs[:, :observed_steps]
            shape = (batch, self.horizon, self.action_dim)
        else:
            raise ValueError("this local inference build requires conditioned observations")

        condition_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        normalized_prediction = self.conditional_sample(
            condition_data,
            condition_mask,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        action_prediction = self.normalizer["action"].unnormalize(
            normalized_prediction[..., : self.action_dim]
        )
        if self.pred_action_steps_only:
            action = action_prediction
        else:
            start = observed_steps - 1 if self.oa_step_convention else observed_steps
            action = action_prediction[:, start : start + self.n_action_steps]
        return {"action": action, "action_pred": action_prediction}
