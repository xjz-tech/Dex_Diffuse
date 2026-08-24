from __future__ import annotations

import os
from typing import Dict
import warnings

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

EE_DIM = 9
HAND_DIM = 22
ACTION_DIM = EE_DIM + HAND_DIM

_LEGACY_ACTION_NORMALIZER_WARNED = False

LEGACY_ACTION_NORMALIZER_WARNING = (
    "Using a legacy Diffusion Policy checkpoint whose normalizer still has "
    "the 31-D 'action' key (the whole EE+hand vector was create_fit together). "
    "Unnormalize will use that single scaler, not the new split "
    "ee_pose (real) + hand_joint (sim) stats. Do not mix this checkpoint's "
    "normalized actions or scores with a sim-hand DP."
)


def has_legacy_action_normalizer(normalizer: LinearNormalizer) -> bool:
    return "action" in normalizer.params_dict


def warn_legacy_action_normalizer() -> None:
    global _LEGACY_ACTION_NORMALIZER_WARNED
    if _LEGACY_ACTION_NORMALIZER_WARNED:
        return
    _LEGACY_ACTION_NORMALIZER_WARNED = True
    warnings.warn(LEGACY_ACTION_NORMALIZER_WARNING, UserWarning, stacklevel=3)



def _as_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def normalize_action(normalizer: LinearNormalizer, action: torch.Tensor) -> torch.Tensor:
    action = _as_tensor(action)
    last = action.shape[-1]
    if last == HAND_DIM:
        return normalizer["hand_joint"].normalize(action)
    if last == ACTION_DIM:
        if has_legacy_action_normalizer(normalizer):
            warn_legacy_action_normalizer()
            return normalizer["action"].normalize(action)
        ee = normalizer["ee_pose"].normalize(action[..., :EE_DIM])
        hand = normalizer["hand_joint"].normalize(action[..., EE_DIM:])
        return torch.cat([ee, hand], dim=-1)
    raise ValueError(f"action last dim must be 22 or 31, got {last}")


def unnormalize_action(normalizer: LinearNormalizer, naction: torch.Tensor) -> torch.Tensor:
    naction = _as_tensor(naction)
    last = naction.shape[-1]
    if last == HAND_DIM:
        return normalizer["hand_joint"].unnormalize(naction)
    if last == ACTION_DIM:
        if has_legacy_action_normalizer(normalizer):
            warn_legacy_action_normalizer()
            return normalizer["action"].unnormalize(naction)
        ee = normalizer["ee_pose"].unnormalize(naction[..., :EE_DIM])
        hand = normalizer["hand_joint"].unnormalize(naction[..., EE_DIM:])
        return torch.cat([ee, hand], dim=-1)
    raise ValueError(f"action last dim must be 22 or 31, got {last}")


def load_hand_joint_stat(path: str) -> Dict[str, np.ndarray]:
    path = os.path.expanduser(path)
    data = np.load(path)
    if "min" not in data.files or "max" not in data.files:
        raise KeyError(f"{path} must contain 'min' and 'max'")
    mn = np.asarray(data["min"], dtype=np.float32).reshape(HAND_DIM)
    mx = np.asarray(data["max"], dtype=np.float32).reshape(HAND_DIM)
    mean = np.asarray(data["mean"], dtype=np.float32).reshape(HAND_DIM) if "mean" in data.files else (mn + mx) / 2
    std = (
        np.asarray(data["std"], dtype=np.float32).reshape(HAND_DIM)
        if "std" in data.files
        else np.maximum(mx - mn, 1e-6) / np.sqrt(12.0)
    )
    return {"min": mn, "max": mx, "mean": mean, "std": std}


def hand_joint_normalizer_from_stat(stat: Dict[str, np.ndarray]) -> SingleFieldLinearNormalizer:
    return get_range_normalizer_from_stat(stat)
