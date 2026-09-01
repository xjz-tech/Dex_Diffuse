from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

HAND_DIM = 22
SOURCE_DIM = 31
ARM_DIM = 9
OBS_DIM = 66
REQUIRED_FILES = ("state.npy", "action.npy")


def _episode_id(path: Path) -> int:
    prefix = "episode_"
    if not path.is_dir() or not path.name.startswith(prefix):
        raise ValueError(f"not a numeric bulb episode directory: {path}")
    try:
        return int(path.name[len(prefix):])
    except ValueError as exc:
        raise ValueError(f"not a numeric bulb episode directory: {path}") from exc


def _load_episode(path: Path, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = {}
    for name in REQUIRED_FILES:
        file_path = path / name
        if not file_path.is_file():
            raise FileNotFoundError(f"missing bulb trajectory file: {file_path}")
        array = np.load(file_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32:
            raise TypeError(f"{file_path} must use float32, got {array.dtype}")
        if array.ndim != 2 or array.shape[1] != SOURCE_DIM:
            raise ValueError(
                f"{file_path} must have shape (T, {SOURCE_DIM}), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{file_path} contains NaN or infinity")
        arrays[name] = array
    if len(arrays["state.npy"]) != len(arrays["action.npy"]):
        raise ValueError(f"{path} state/action lengths do not match")
    if len(arrays["state.npy"]) < horizon:
        raise ValueError(
            f"{path} length {len(arrays['state.npy'])} is shorter than horizon {horizon}"
        )
    qpos = np.asarray(arrays["state.npy"][:, ARM_DIM:], dtype=np.float32)
    target_after = np.asarray(arrays["action.npy"][:, ARM_DIM:], dtype=np.float32)
    target_before = np.concatenate((qpos[0:1], target_after[:-1]), axis=0)
    obs = np.concatenate(
        (qpos, target_before, target_before - qpos), axis=-1
    ).astype(np.float32, copy=False)
    return obs, target_after


class BulbTacLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        dataset_path: str,
        horizon: int,
        pad_before: int,
        pad_after: int,
        seed: int = 42,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        root = Path(dataset_path).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"bulb dataset directory not found: {root}")
        episodes = []
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("episode_"):
                episodes.append((_episode_id(child), child))
        episodes.sort(key=lambda item: item[0])
        if not episodes:
            raise ValueError(f"no numeric episode_<id> directories under {root}")
        obs_parts, action_parts, lengths = [], [], []
        for _, path in episodes:
            obs, action = _load_episode(path, int(horizon))
            obs_parts.append(obs)
            action_parts.append(action)
            lengths.append(len(obs))
        replay_buffer = ReplayBuffer(root={
            "data": {
                "obs": np.concatenate(obs_parts, axis=0),
                "action": np.concatenate(action_parts, axis=0),
            },
            "meta": {
                "episode_ends": np.cumsum(lengths, dtype=np.int64),
            },
        })
        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        self.replay_buffer = replay_buffer
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.train_mask = ~val_mask
        self.val_mask = val_mask
        self.sampler = self._make_sampler(self.train_mask)

    def _make_sampler(self, mask):
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=("obs", "action"),
            episode_mask=mask,
        )

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = result._make_sampler(self.val_mask)
        return result

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            {"obs": self.replay_buffer["obs"],
             "action": self.replay_buffer["action"]},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return dict_apply(self.sampler.sample_sequence(index), torch.from_numpy)
