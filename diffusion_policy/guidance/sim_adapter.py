from __future__ import annotations

import torch

from diffusion_policy.guidance.checkpoint_loader import LoadedPolicy


class SimPolicyAdapter:
    def __init__(self, loaded: LoadedPolicy) -> None:
        self.loaded = loaded
        self.policy = loaded.policy
        policy = self.policy

        self._horizon = int(policy.horizon)
        self._n_obs_steps = int(policy.n_obs_steps)
        self._n_pred_action_steps = int(policy.n_pred_action_steps)
        self._action_dim = int(policy.action_dim)
        self._obs_dim = int(policy.obs_dim)
        self._oa_start = int(policy.temporal.usable_action_slice.start)

        if self._obs_dim != 22 or self._action_dim != 22:
            raise ValueError(
                "Sim policy obs_dim and action_dim must both be 22, "
                f"got obs_dim={self._obs_dim}, action_dim={self._action_dim}"
            )

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def n_obs_steps(self) -> int:
        return self._n_obs_steps

    @property
    def n_pred_action_steps(self) -> int:
        return self._n_pred_action_steps

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def oa_start(self) -> int:
        return self._oa_start

    @property
    def guidance_slice(self) -> slice:
        stop = self.oa_start + self.n_pred_action_steps
        if stop > self.horizon:
            raise ValueError(
                f"guidance slice [{self.oa_start}:{stop}] exceeds "
                f"horizon={self.horizon}"
            )
        return slice(self.oa_start, stop)

    def execution_slice(self, execution_steps: int) -> slice:
        if execution_steps <= 0:
            raise ValueError(
                f"execution_steps must be positive, got {execution_steps}"
            )
        if execution_steps > self.n_pred_action_steps:
            raise ValueError(
                f"execution_steps={execution_steps} exceeds "
                f"n_pred_action_steps={self.n_pred_action_steps}"
            )
        stop = self.oa_start + execution_steps
        if stop > self.horizon:
            raise ValueError(
                f"execution slice [{self.oa_start}:{stop}] exceeds "
                f"horizon={self.horizon}"
            )
        return slice(self.oa_start, stop)

    def normalize_history(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(value).all():
            raise ValueError("Sim history is non-finite")
        normalized = self.policy.normalizer["obs"].normalize(value)
        if not torch.isfinite(normalized).all():
            raise ValueError("Normalized Sim history is non-finite")
        return normalized

    def normalize_reference(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(value).all():
            raise ValueError("Sim reference is non-finite")
        normalized = self.policy.normalizer["action"].normalize(value)
        if not torch.isfinite(normalized).all():
            raise ValueError("Normalized Sim reference is non-finite")
        return normalized

    def unnormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(value).all():
            raise ValueError("Normalized Sim action is non-finite")
        unnormalized = self.policy.normalizer["action"].unnormalize(value)
        if not torch.isfinite(unnormalized).all():
            raise ValueError("Unnormalized Sim action is non-finite")
        return unnormalized

    def global_condition(self, history: torch.Tensor) -> torch.Tensor:
        normalized = self.normalize_history(history)
        return normalized[:, : self.n_obs_steps, :].reshape(history.shape[0], -1)

    def predict_epsilon(
        self,
        sample: torch.Tensor,
        timestep: int | torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        output = self.policy.model(sample, timestep, global_cond=global_cond)
        if not torch.isfinite(output).all():
            raise ValueError("Sim epsilon prediction is non-finite")
        return output
