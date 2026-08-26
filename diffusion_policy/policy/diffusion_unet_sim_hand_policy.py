from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from diffusion_policy.common.sim_hand_temporal_util import (
    SimHandTemporalConfig,
    validate_sim_hand_temporal_config,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class DiffusionUnetSimHandPolicy(BaseLowdimPolicy):
    """Hand-only low-dimensional Diffusion Policy with explicit chunk semantics."""

    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        horizon: int,
        obs_dim: int,
        action_dim: int,
        n_obs_steps: int,
        n_pred_action_steps: int,
        n_action_steps: int,
        num_inference_steps: int | None = None,
        obs_as_global_cond: bool = True,
        oa_step_convention: bool = True,
        return_full_prediction: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not obs_as_global_cond:
            raise ValueError(
                "Sim-Hand DP requires obs_as_global_cond=True."
            )

        temporal = validate_sim_hand_temporal_config(
            n_obs_steps=n_obs_steps,
            n_pred_action_steps=n_pred_action_steps,
            n_action_steps=n_action_steps,
            horizon=horizon,
            obs_dim=obs_dim,
            action_dim=action_dim,
            oa_step_convention=oa_step_convention,
        )
        self._validate_unet_temporal_shape(model, temporal)

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.temporal = temporal
        self.horizon = temporal.horizon
        self.obs_dim = temporal.obs_dim
        self.action_dim = temporal.action_dim
        self.n_obs_steps = temporal.n_obs_steps
        self.n_pred_action_steps = temporal.n_pred_action_steps
        self.n_action_steps = temporal.n_action_steps
        self.obs_as_global_cond = True
        self.oa_step_convention = True
        self.return_full_prediction = bool(return_full_prediction)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = int(num_inference_steps)

    @staticmethod
    def _temporal_values(temporal: SimHandTemporalConfig) -> str:
        return "\n".join(
            (
                f"Observation steps       : {temporal.n_obs_steps}",
                f"Prediction action steps : {temporal.n_pred_action_steps}",
                f"Execution action steps  : {temporal.n_action_steps}",
                f"Derived horizon         : {temporal.horizon}",
                f"Observation dimension   : {temporal.obs_dim}",
                f"Action dimension        : {temporal.action_dim}",
            )
        )

    @classmethod
    def _validate_unet_temporal_shape(
        cls,
        model: torch.nn.Module,
        temporal: SimHandTemporalConfig,
    ) -> None:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        dtype = parameter.dtype if parameter is not None else torch.float32
        sample = torch.zeros(
            (1, temporal.horizon, temporal.action_dim),
            device=device,
            dtype=dtype,
        )
        global_cond = torch.zeros(
            (1, temporal.n_obs_steps * temporal.obs_dim),
            device=device,
            dtype=dtype,
        )
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                output = model(
                    sample,
                    torch.zeros((1,), device=device, dtype=torch.long),
                    global_cond=global_cond,
                )
        except Exception as exc:
            raise ValueError(
                "Invalid Sim-Hand DP temporal configuration.\n\n"
                f"{cls._temporal_values(temporal)}\n\n"
                "The configured ConditionalUnet1D could not preserve the "
                f"requested temporal shape: {exc}"
            ) from exc
        finally:
            model.train(was_training)

        expected_shape = (1, temporal.horizon, temporal.action_dim)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                "Invalid Sim-Hand DP temporal configuration.\n\n"
                f"{cls._temporal_values(temporal)}\n\n"
                "U-Net output temporal length must equal the requested "
                f"horizon. Expected {expected_shape}, got {tuple(output.shape)}.\n"
                "The requested prediction length will NOT be changed "
                "automatically."
            )

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        global_cond: torch.Tensor,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory,
                timestep,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if "obs" not in obs_dict:
            raise KeyError("Sim-Hand policy observation must contain 'obs'")
        if "past_action" in obs_dict:
            raise ValueError("past_action conditioning is not supported")

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        batch_size, available_steps, obs_dim = nobs.shape
        if available_steps < self.n_obs_steps:
            raise ValueError(
                f"Expected at least {self.n_obs_steps} observation steps, "
                f"got {available_steps}"
            )
        if obs_dim != self.obs_dim:
            raise ValueError(
                f"Expected observation dimension {self.obs_dim}, got {obs_dim}"
            )

        global_cond = nobs[:, : self.n_obs_steps].reshape(batch_size, -1)
        condition_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        nsample = self.conditional_sample(
            condition_data,
            condition_mask,
            global_cond=global_cond,
            **self.kwargs,
        )

        action_pred = self.normalizer["action"].unnormalize(nsample)
        action_usable = action_pred[:, self.temporal.usable_action_slice]
        result = {
            "action": action_usable[:, : self.n_action_steps],
            "action_usable": action_usable,
        }
        if self.return_full_prediction:
            result["action_pred"] = action_pred
        return result

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if set(batch) != {"obs", "action"}:
            raise KeyError("Sim-Hand training batch must contain obs and action")
        normalized = self.normalizer.normalize(batch)
        obs = normalized["obs"]
        trajectory = normalized["action"]
        if obs.shape[1] != self.horizon or trajectory.shape[1] != self.horizon:
            raise ValueError(
                f"Training sequences must have horizon {self.horizon}, got "
                f"obs={obs.shape[1]} and action={trajectory.shape[1]}"
            )

        global_cond = obs[:, : self.n_obs_steps].reshape(obs.shape[0], -1)
        condition_mask = self.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory,
            noise,
            timesteps,
        )
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        pred = self.model(
            noisy_trajectory,
            timesteps,
            global_cond=global_cond,
        )

        prediction_type = self.noise_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(
                f"Unsupported prediction type {prediction_type}"
            )

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).to(loss.dtype)
        return reduce(loss, "b ... -> b (...)", "mean").mean()
