from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion_policy.guidance.checkpoint_loader import LoadedPolicy


REAL_ACTION_DIM = 31
HAND_SLICE = slice(9, 31)


@dataclass(frozen=True)
class RealGuidancePrediction:
    proposal: torch.Tensor
    hand_reference: torch.Tensor


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
        return self._validate_action(result["action"], name="proposal")

    def predict_for_guidance(
        self,
        observation: dict[str, torch.Tensor] | None = None,
    ) -> RealGuidancePrediction:
        if observation is None:
            observation = self.build_synthetic_observation()
        result = self.policy.predict_action(observation)
        if "action" not in result:
            raise KeyError("Real policy prediction has no 'action'")
        if "action_pred" not in result:
            raise KeyError("Real policy prediction has no 'action_pred'")

        proposal = self._validate_action(result["action"], name="proposal")
        action_pred = self._validate_action(
            result["action_pred"],
            name="full prediction",
        )
        action_start = int(self.policy.n_obs_steps) - 1
        aligned = action_pred[:, action_start:, :]
        if aligned.shape[1] < proposal.shape[1]:
            raise ValueError(
                "Aligned Real full prediction is shorter than the public proposal: "
                f"{aligned.shape[1]} < {proposal.shape[1]}"
            )
        if not torch.allclose(
            aligned[:, : proposal.shape[1]],
            proposal,
        ):
            raise ValueError(
                "Real action_pred is not aligned with the public action proposal"
            )
        return RealGuidancePrediction(
            proposal=proposal,
            hand_reference=aligned[..., HAND_SLICE],
        )

    @staticmethod
    def _validate_action(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if (
            value.ndim != 3
            or value.shape[0] != 1
            or value.shape[-1] != REAL_ACTION_DIM
        ):
            raise ValueError(
                f"Real {name} must have shape (1,T,31), got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Real {name} is non-finite")
        return value
