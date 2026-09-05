from __future__ import annotations

import copy
import os
from typing import Dict

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


EE_DIM = 9
HAND_DIM = 22
ACTION_DIM = EE_DIM + HAND_DIM


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert TacMP's two-column 6D rotation representation to matrices."""
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    third = np.cross(first, second)
    third = third / np.maximum(np.linalg.norm(third, axis=-1, keepdims=True), 1e-8)
    second = np.cross(third, first)
    return np.stack((first, second, third), axis=-1)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return matrix[..., :3, :2].swapaxes(-1, -2).reshape(matrix.shape[:-2] + (6,))


def ee_pose_relative_to(poses: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Express absolute 9D EE poses in ``base_pose``'s SE(3) frame."""
    poses = np.asarray(poses, dtype=np.float32)
    base_pose = np.asarray(base_pose, dtype=np.float32)
    base_rotation = rotation_6d_to_matrix(base_pose[None, 3:9])[0]
    pose_rotation = rotation_6d_to_matrix(poses[..., 3:9])

    relative = np.empty_like(poses, dtype=np.float32)
    relative[..., :3] = np.einsum(
        "ij,...j->...i", base_rotation.T, poses[..., :3] - base_pose[:3]
    )
    relative_rotation = np.einsum("ij,...jk->...ik", base_rotation.T, pose_rotation)
    relative[..., 3:9] = matrix_to_rotation_6d(relative_rotation)
    return relative


class BulbImageDataset(BaseImageDataset):
    """Bulb demonstrations with configurable EE coordinates.

    When ``relative`` is true, the last observed EE pose in each sampled horizon
    is the reference pose. EE observations and action dimensions ``[:9]`` are
    represented relative to it using proper SE(3) composition. When false, EE
    observations and actions remain absolute. Dimensions ``[9:31]`` always remain
    absolute hand-joint values.
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        n_obs_steps: int = 2,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
        relative: bool = True,
    ):
        zarr_path = os.path.join(os.path.expanduser(dataset_path), "replay_buffer.zarr")
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"Diffusion Policy replay buffer not found: {zarr_path}")
        if not 1 <= n_obs_steps <= horizon:
            raise ValueError("n_obs_steps must be between 1 and horizon")

        # Keep the 6.6 GB image dataset on disk. SequenceSampler reads only the
        # requested observation frames from zarr instead of copying it into RAM.
        replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
        required = {"front_image", "wrist_image", "state", "action"}
        missing = required.difference(replay_buffer.keys())
        if missing:
            raise KeyError(f"missing replay-buffer keys: {sorted(missing)}")
        if replay_buffer["state"].shape[1:] != (ACTION_DIM,):
            raise ValueError(f"state must have shape (N, {ACTION_DIM})")
        if replay_buffer["action"].shape[1:] != (ACTION_DIM,):
            raise ValueError(f"action must have shape (N, {ACTION_DIM})")

        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        train_mask = downsample_mask(~val_mask, max_train_episodes, seed)

        self.replay_buffer = replay_buffer
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.relative = bool(relative)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self._set_sampler(train_mask)

        # Low-dimensional data is small and is used repeatedly while fitting the
        # mixed relative/absolute normalizer. Keeping this copy avoids repeatedly
        # decompressing large zarr chunks.
        lowdim_buffer = ReplayBuffer(
            root={
                "data": {
                    "state": replay_buffer["state"][:],
                    "action": replay_buffer["action"][:],
                },
                "meta": {"episode_ends": replay_buffer.episode_ends[:]},
            }
        )
        self._lowdim_buffer = lowdim_buffer
        self._set_lowdim_sampler(train_mask)

    def _set_sampler(self, episode_mask: np.ndarray) -> None:
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=["front_image", "wrist_image", "state", "action"],
            key_first_k={
                "front_image": self.n_obs_steps,
                "wrist_image": self.n_obs_steps,
                "state": self.n_obs_steps,
            },
            episode_mask=episode_mask,
        )

    def _set_lowdim_sampler(self, episode_mask: np.ndarray) -> None:
        self.lowdim_sampler = SequenceSampler(
            replay_buffer=self._lowdim_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=["state", "action"],
            episode_mask=episode_mask,
        )

    def get_validation_dataset(self) -> "BulbImageDataset":
        validation = copy.copy(self)
        validation._set_sampler(self.val_mask)
        validation._set_lowdim_sampler(self.val_mask)
        validation.train_mask = self.val_mask
        return validation

    def _convert_lowdim(self, state: np.ndarray, action: np.ndarray):
        observed_state = state[: self.n_obs_steps].astype(np.float32, copy=True)
        hand_joint = observed_state[:, EE_DIM:ACTION_DIM]
        mixed_action = action.astype(np.float32, copy=True)

        if self.relative:
            base_ee_pose = observed_state[-1, :EE_DIM].copy()
            ee_pose = ee_pose_relative_to(observed_state[:, :EE_DIM], base_ee_pose)
            mixed_action[:, :EE_DIM] = ee_pose_relative_to(
                mixed_action[:, :EE_DIM], base_ee_pose
            )
        else:
            ee_pose = observed_state[:, :EE_DIM].copy()
        # mixed_action[:, 9:31] intentionally remains absolute.
        return ee_pose, hand_joint, mixed_action

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        relative_ee_obs = []
        absolute_hand_obs = []
        mixed_actions = []
        for index in range(len(self.lowdim_sampler)):
            sample = self.lowdim_sampler.sample_sequence(index)
            ee_pose, hand_joint, action = self._convert_lowdim(
                sample["state"], sample["action"]
            )
            relative_ee_obs.append(ee_pose)
            absolute_hand_obs.append(hand_joint)
            mixed_actions.append(action)

        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            np.concatenate(mixed_actions, axis=0)
        )
        normalizer["ee_pose"] = SingleFieldLinearNormalizer.create_fit(
            np.concatenate(relative_ee_obs, axis=0)
        )
        normalizer["hand_joint"] = SingleFieldLinearNormalizer.create_fit(
            np.concatenate(absolute_hand_obs, axis=0)
        )
        normalizer["front_image"] = get_image_range_normalizer()
        normalizer["wrist_image"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = []
        for index in range(len(self.lowdim_sampler)):
            sample = self.lowdim_sampler.sample_sequence(index)
            actions.append(self._convert_lowdim(sample["state"], sample["action"])[2])
        return torch.from_numpy(np.concatenate(actions, axis=0))

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(index)
        ee_pose, hand_joint, action = self._convert_lowdim(
            sample["state"], sample["action"]
        )
        obs = {
            "front_image": np.moveaxis(
                sample["front_image"][: self.n_obs_steps], -1, 1
            ).astype(np.float32)
            / 255.0,
            "wrist_image": np.moveaxis(
                sample["wrist_image"][: self.n_obs_steps], -1, 1
            ).astype(np.float32)
            / 255.0,
            "ee_pose": ee_pose,
            "hand_joint": hand_joint.astype(np.float32),
        }
        return {
            "obs": dict_apply(obs, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
