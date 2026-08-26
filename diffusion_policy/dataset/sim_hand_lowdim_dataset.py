from __future__ import annotations

import copy
import os
from typing import Dict

import numpy as np
import torch
import zarr

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


HAND_DIM = 22
REQUIRED_KEYS = ("hand_joint", "action")


class SimHandLowdimDataset(BaseLowdimDataset):
    """Standardized hand-only simulation trajectories.

    The conversion layer is responsible for mapping simulator-specific fields
    to absolute 22-D hand joint states and action targets before this Dataset
    reads them.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 12,
        pad_before: int = 3,
        pad_after: int = 8,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
    ):
        super().__init__()
        zarr_path = os.path.join(
            os.path.expanduser(dataset_path),
            "replay_buffer.zarr",
        )
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(
                f"Sim-Hand replay buffer not found: {zarr_path}"
            )

        root = zarr.open_group(zarr_path, mode="r")
        self._validate_replay_group(root)
        replay_buffer = ReplayBuffer.create_from_group(root)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )

        self.replay_buffer = replay_buffer
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.sampler = self._make_sampler(train_mask)

    @staticmethod
    def _validate_replay_group(root: zarr.Group) -> None:
        if "data" not in root or "meta" not in root:
            raise KeyError("Replay buffer must contain data and meta groups")
        if "episode_ends" not in root["meta"]:
            raise KeyError("Missing Sim-Hand replay-buffer key: episode_ends")

        data = root["data"]
        missing = set(REQUIRED_KEYS).difference(data.keys())
        if missing:
            raise KeyError(
                f"Missing Sim-Hand replay-buffer keys: {sorted(missing)}"
            )

        hand_joint = data["hand_joint"]
        action = data["action"]
        for key, array in (("hand_joint", hand_joint), ("action", action)):
            if np.dtype(array.dtype) != np.dtype(np.float32):
                raise TypeError(
                    f"{key} must use float32, got {array.dtype}"
                )
            if len(array.shape) != 2 or array.shape[1] != HAND_DIM:
                raise ValueError(
                    f"{key} must have shape (N, {HAND_DIM}), got {array.shape}"
                )

        if hand_joint.shape[0] != action.shape[0]:
            raise ValueError(
                "hand_joint and action must contain the same number of steps, "
                f"got {hand_joint.shape[0]} and {action.shape[0]}"
            )

        episode_ends_array = root["meta/episode_ends"]
        if np.dtype(episode_ends_array.dtype) != np.dtype(np.int64):
            raise TypeError(
                "episode_ends must use int64, "
                f"got {episode_ends_array.dtype}"
            )
        episode_ends = np.asarray(episode_ends_array[:])
        if episode_ends.ndim != 1 or len(episode_ends) == 0:
            raise ValueError("episode_ends must be a non-empty 1-D array")
        if episode_ends[0] <= 0 or np.any(np.diff(episode_ends) <= 0):
            raise ValueError("episode_ends must be strictly increasing")
        n_steps = hand_joint.shape[0]
        if int(episode_ends[-1]) != n_steps:
            raise ValueError(
                f"The final episode end must equal {n_steps}, got "
                f"{int(episode_ends[-1])}"
            )

    def _make_sampler(self, episode_mask: np.ndarray) -> SequenceSampler:
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=REQUIRED_KEYS,
            episode_mask=episode_mask,
        )

    def get_validation_dataset(self) -> "SimHandLowdimDataset":
        validation = copy.copy(self)
        validation.sampler = validation._make_sampler(self.val_mask)
        return validation

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "obs": self.replay_buffer["hand_joint"],
                "action": self.replay_buffer["action"],
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(index)
        data = {
            "obs": sample["hand_joint"],
            "action": sample["action"],
        }
        return dict_apply(data, torch.from_numpy)
