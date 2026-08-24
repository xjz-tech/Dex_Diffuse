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


def streaming_minmax(array, chunk_size: int = 65536):
    n = int(array.shape[0])
    if n == 0:
        raise ValueError("cannot compute min/max on empty array")
    running_min = None
    running_max = None
    for start in range(0, n, chunk_size):
        block = np.asarray(array[start : start + chunk_size], dtype=np.float32)
        block = block.reshape(block.shape[0], -1)
        bmin = block.min(axis=0)
        bmax = block.max(axis=0)
        running_min = bmin if running_min is None else np.minimum(running_min, bmin)
        running_max = bmax if running_max is None else np.maximum(running_max, bmax)
    return running_min.astype(np.float32), running_max.astype(np.float32)


def _hand_array_from_buffer(replay, key: str):
    arr = replay[key]
    tail = arr.shape[-1]
    if tail == HAND_DIM:
        return arr
    if tail == ACTION_DIM:
        return arr[..., EE_DIM:]
    raise ValueError(f"{key} last dim must be 22 or 31, got {tail}")


def _open_replay_buffer(zarr_path: str, mode: str = "r"):
    import zarr

    from diffusion_policy.common.replay_buffer import ReplayBuffer

    zarr_path = os.path.expanduser(zarr_path)
    if int(zarr.__version__.split(".", maxsplit=1)[0]) >= 3:
        from zarr.storage import LocalStore

        class _ZarrReplayView:
            def __init__(self, group):
                self._data = group["data"]

            def keys(self):
                return self._data.keys()

            def __getitem__(self, key):
                return self._data[key]

        group = zarr.open(store=LocalStore(zarr_path), mode=mode)
        return _ZarrReplayView(group)
    return ReplayBuffer.create_from_path(zarr_path, mode=mode)


def collect_hand_joint_stat(zarr_path: str, chunk_size: int = 65536) -> Dict[str, np.ndarray]:
    replay = _open_replay_buffer(zarr_path, mode="r")
    keys = []
    if "hand_joint" in replay.keys():
        keys.append("hand_joint")
    if "action" in replay.keys():
        keys.append("action")
    if "state" in replay.keys() and "hand_joint" not in replay.keys():
        keys.append("state")
    if not keys:
        raise KeyError("replay buffer needs hand_joint, action, or state")
    running_min = None
    running_max = None
    for key in keys:
        mn, mx = streaming_minmax(_hand_array_from_buffer(replay, key), chunk_size=chunk_size)
        running_min = mn if running_min is None else np.minimum(running_min, mn)
        running_max = mx if running_max is None else np.maximum(running_max, mx)
    return {
        "min": running_min,
        "max": running_max,
        "mean": (running_min + running_max) / 2,
        "std": np.maximum(running_max - running_min, 1e-6) / np.sqrt(12.0),
    }
