from __future__ import annotations

import torch

from diffusion_policy.guidance.checkpoint_loader import LoadedPolicy


class RealPolicyAdapter:
    def __init__(self, loaded: LoadedPolicy) -> None:
        self.loaded = loaded
        self.policy = loaded.policy
        self.cfg = loaded.cfg

    def build_synthetic_observation(self) -> dict[str, torch.Tensor]:
        obs_meta = self.cfg.shape_meta.obs
        n_obs_steps = int(self.policy.n_obs_steps)
        device = self.policy.device
        dtype = self.policy.dtype
        observation: dict[str, torch.Tensor] = {}
        for key, attr in obs_meta.items():
            shape = tuple(int(v) for v in attr.shape)
            observation[key] = torch.zeros(
                (1, n_obs_steps, *shape),
                device=device,
                dtype=dtype,
            )
        return observation

    def predict_proposal(
        self,
        observation: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if observation is None:
            observation = self.build_synthetic_observation()
        result = self.policy.predict_action(observation)
        if "action" not in result:
            raise KeyError("Real policy prediction has no 'action'")
        proposal = result["action"]
        if proposal.ndim != 3 or proposal.shape[0] != 1 or proposal.shape[-1] != 31:
            raise ValueError(
                f"Real proposal must have shape (1,T,31), got {tuple(proposal.shape)}"
            )
        if not torch.isfinite(proposal).all():
            raise ValueError("Real proposal is non-finite")
        return proposal
