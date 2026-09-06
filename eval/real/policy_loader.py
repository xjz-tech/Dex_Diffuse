"""Load the local 22/66-D Sim-Hand Diffusion Policy checkpoint.

The checkpoint names a training-only workspace that is not needed for
inference.  Build the equivalent low-dimensional policy directly so the real
runner does not import any of the simulator evaluator modules.
"""

from __future__ import annotations

import dill
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from sim_hand_policy import (
    ConditionalUnet1D,
    DiffusionUnetLowdimPolicy,
    SingleFieldLinearNormalizer,
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    cfg: object
    state_dict: Mapping[str, torch.Tensor]
    weight_source: str
    global_step: Optional[int]
    epoch: Optional[int]


def _decode_pickled_scalar(payload: dict, name: str) -> Optional[int]:
    encoded = payload.get("pickles", {}).get(name)
    if encoded is None:
        return None
    value = dill.loads(encoded)
    return int(value) if value is not None else None


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(
        str(checkpoint),
        map_location="cpu",
        pickle_module=dill,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    cfg = payload.get("cfg")
    if cfg is None:
        raise ValueError("checkpoint has no embedded configuration")
    state_dicts = payload.get("state_dicts")
    if not isinstance(state_dicts, dict) or "model" not in state_dicts:
        raise ValueError("checkpoint does not contain a model state")

    use_ema = bool(getattr(getattr(cfg, "training", None), "use_ema", False))
    if use_ema:
        if "ema_model" not in state_dicts:
            raise ValueError("checkpoint requests EMA but has no EMA model state")
        state_dict = state_dicts["ema_model"]
        source = "EMA model"
    else:
        state_dict = state_dicts["model"]
        source = "base model"
    return LoadedCheckpoint(
        cfg=cfg,
        state_dict=state_dict,
        weight_source=source,
        global_step=_decode_pickled_scalar(payload, "global_step"),
        epoch=_decode_pickled_scalar(payload, "epoch"),
    )


def _identity_field_normalizer(dim: int):
    ones = torch.ones(dim, dtype=torch.float32)
    zeros = torch.zeros(dim, dtype=torch.float32)
    return SingleFieldLinearNormalizer.create_manual(
        scale=ones,
        offset=zeros,
        input_stats_dict={
            "min": -ones,
            "max": ones,
            "mean": zeros,
            "std": ones,
        },
    )


def build_policy(loaded: LoadedCheckpoint):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from omegaconf import OmegaConf

    cfg = loaded.cfg
    obs_dim = int(cfg["obs_dim"])
    if obs_dim not in (22, 66):
        raise ValueError(
            f"unsupported Sim-Hand checkpoint obs_dim={obs_dim}; expected 22 or 66"
        )
    expected = {
        "obs_dim": obs_dim,
        "action_dim": 22,
        "n_obs_steps": 4,
        "n_pred_action_steps": 9,
        "n_action_steps": 5,
        "horizon": 12,
    }
    for key, expected_value in expected.items():
        actual = int(cfg[key])
        if actual != expected_value:
            raise ValueError(
                f"unsupported Sim-Hand checkpoint {key}={actual}; "
                f"expected {expected_value}"
            )
    if not bool(cfg.obs_as_global_cond) or not bool(cfg.oa_step_convention):
        raise ValueError(
            "checkpoint must use global observation conditioning and "
            "OA-step convention"
        )

    model_kwargs = OmegaConf.to_container(cfg.policy.model, resolve=True)
    model_kwargs.pop("_target_", None)
    scheduler_kwargs = OmegaConf.to_container(
        cfg.policy.noise_scheduler,
        resolve=True,
    )
    scheduler_kwargs.pop("_target_", None)
    model = ConditionalUnet1D(**model_kwargs)
    scheduler = DDPMScheduler(**scheduler_kwargs)
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=scheduler,
        horizon=expected["horizon"],
        obs_dim=expected["obs_dim"],
        action_dim=expected["action_dim"],
        n_action_steps=expected["n_action_steps"],
        n_obs_steps=expected["n_obs_steps"],
        num_inference_steps=int(cfg.policy.num_inference_steps),
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        oa_step_convention=True,
    )
    policy.normalizer["obs"] = _identity_field_normalizer(expected["obs_dim"])
    policy.normalizer["action"] = _identity_field_normalizer(
        expected["action_dim"]
    )
    incompatible = policy.load_state_dict(loaded.state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict checkpoint load returned key errors")
    return policy, expected


def load_policy(
    checkpoint: str | Path,
    device: torch.device,
    sampler: str,
    inference_steps: Optional[int],
):
    loaded = load_checkpoint(checkpoint)
    policy, spec = build_policy(loaded)
    if sampler == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        policy.noise_scheduler = DDIMScheduler.from_config(
            policy.noise_scheduler.config,
            set_alpha_to_one=True,
            steps_offset=0,
            timestep_spacing="leading",
        )
    elif sampler != "ddpm":
        raise ValueError(f"unsupported sampler: {sampler!r}")
    if inference_steps is not None:
        if inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        policy.num_inference_steps = int(inference_steps)
    policy.to(device).eval()
    return loaded, policy, spec
