from __future__ import annotations

import copy
import os
import warnings
from typing import Dict

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.bulb_action_normalizer import (
    hand_joint_normalizer_from_stat,
    load_hand_joint_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


EE_DIM = 9
HAND_DIM = 22
ACTION_DIM = EE_DIM + HAND_DIM

def _open_replay_buffer_from_path(zarr_path: str, mode: str = "r") -> ReplayBuffer:
    import zarr

    zarr_path = os.path.expanduser(zarr_path)
    if int(zarr.__version__.split(".", maxsplit=1)[0]) >= 3:
        from zarr.storage import LocalStore

        group = zarr.open(store=LocalStore(zarr_path), mode=mode)
        root = {
            "data": {key: group["data"][key] for key in group["data"].keys()},
            "meta": {
                key: np.asarray(group["meta"][key]) for key in group["meta"].keys()
            },
        }
        return ReplayBuffer(root=root)
    return ReplayBuffer.create_from_path(zarr_path, mode=mode)


DEFAULT_IMAGE_SHAPE_META = {
    "obs": {
        "front_image": {"shape": [3, 16, 16], "type": "rgb"},
        "wrist_image": {"shape": [3, 16, 16], "type": "rgb"},
        "ee_pose": {"shape": [9], "type": "low_dim"},
        "hand_joint": {"shape": [22], "type": "low_dim"},
    },
    "action": {"shape": [31]},
}


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
    """Bulb demonstrations with relative EE pose and absolute hand actions.

    For every sampled horizon, the last observed EE pose is the reference pose.
    Action dimensions ``[:9]`` are represented relative to that reference using
    proper SE(3) composition. Dimensions ``[9:31]`` remain absolute hand-joint
    targets. EE pose observations use the same reference frame, while hand-joint
    observations remain absolute.
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
        shape_meta: dict | None = None,
        normalizer_path: str | None = None,
    ):
        zarr_path = os.path.join(os.path.expanduser(dataset_path), "replay_buffer.zarr")
        if not os.path.isdir(zarr_path):
            raise FileNotFoundError(f"Diffusion Policy replay buffer not found: {zarr_path}")
        if not 1 <= n_obs_steps <= horizon:
            raise ValueError("n_obs_steps must be between 1 and horizon")

        self.normalizer_path = normalizer_path
        self.shape_meta = shape_meta if shape_meta is not None else DEFAULT_IMAGE_SHAPE_META

        obs_meta = self.shape_meta["obs"]
        self.rgb_keys = [k for k, a in obs_meta.items() if a.get("type") == "rgb"]
        self.lowdim_keys = [
            k for k, a in obs_meta.items() if a.get("type", "low_dim") == "low_dim"
        ]
        self.action_dim = int(self.shape_meta["action"]["shape"][0])
        if self.action_dim not in (HAND_DIM, ACTION_DIM):
            raise ValueError(
                f"action dim must be {HAND_DIM} or {ACTION_DIM}, got {self.action_dim}"
            )

        # Keep the 6.6 GB image dataset on disk. SequenceSampler reads only the
        # requested observation frames from zarr instead of copying it into RAM.
        replay_buffer = _open_replay_buffer_from_path(zarr_path, mode="r")

        missing_rgb = set(self.rgb_keys).difference(replay_buffer.keys())
        if missing_rgb:
            raise KeyError(f"missing replay-buffer rgb keys: {sorted(missing_rgb)}")
        if "action" not in replay_buffer.keys():
            raise KeyError("missing replay-buffer key: action")

        needs_state = self.action_dim == ACTION_DIM or "ee_pose" in self.lowdim_keys
        if needs_state:
            if "state" not in replay_buffer.keys():
                raise KeyError("missing replay-buffer key: state")
            if replay_buffer["state"].shape[1:] != (ACTION_DIM,):
                raise ValueError(f"state must have shape (N, {ACTION_DIM})")
            self._lowdim_source_keys = ["state"]
        elif "hand_joint" in replay_buffer.keys():
            if replay_buffer["hand_joint"].shape[1:] != (HAND_DIM,):
                raise ValueError(f"hand_joint must have shape (N, {HAND_DIM})")
            self._lowdim_source_keys = ["hand_joint"]
        elif "state" in replay_buffer.keys():
            tail = replay_buffer["state"].shape[-1]
            if tail not in (HAND_DIM, ACTION_DIM):
                raise ValueError(
                    f"state last dim must be {HAND_DIM} or {ACTION_DIM}, got {tail}"
                )
            self._lowdim_source_keys = ["state"]
        else:
            raise KeyError("replay buffer needs hand_joint or state for hand-only data")

        action_tail = replay_buffer["action"].shape[-1]
        if self.action_dim == HAND_DIM:
            if action_tail not in (HAND_DIM, ACTION_DIM):
                raise ValueError(
                    f"action last dim must be {HAND_DIM} or {ACTION_DIM}, got {action_tail}"
                )
        elif action_tail != ACTION_DIM:
            raise ValueError(f"action must have shape (N, {ACTION_DIM})")

        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        train_mask = downsample_mask(~val_mask, max_train_episodes, seed)

        self.replay_buffer = replay_buffer
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.val_mask = val_mask
        self._sampler_keys = list(self.rgb_keys) + self._lowdim_source_keys + ["action"]
        self._key_first_k = {
            key: self.n_obs_steps
            for key in self.rgb_keys + self._lowdim_source_keys
        }
        self._set_sampler(train_mask)

        # Low-dimensional data is small and is used repeatedly while fitting the
        # mixed relative/absolute normalizer. Keeping this copy avoids repeatedly
        # decompressing large zarr chunks.
        lowdim_data = {
            key: replay_buffer[key][:] for key in self._lowdim_source_keys + ["action"]
        }
        lowdim_buffer = ReplayBuffer(
            root={
                "data": lowdim_data,
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
            keys=self._sampler_keys,
            key_first_k=self._key_first_k,
            episode_mask=episode_mask,
        )

    def _set_lowdim_sampler(self, episode_mask: np.ndarray) -> None:
        self.lowdim_sampler = SequenceSampler(
            replay_buffer=self._lowdim_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self._lowdim_source_keys + ["action"],
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
        base_ee_pose = observed_state[-1, :EE_DIM].copy()

        ee_pose = ee_pose_relative_to(observed_state[:, :EE_DIM], base_ee_pose)
        hand_joint = observed_state[:, EE_DIM:ACTION_DIM]

        mixed_action = action.astype(np.float32, copy=True)
        mixed_action[:, :EE_DIM] = ee_pose_relative_to(
            mixed_action[:, :EE_DIM], base_ee_pose
        )
        # mixed_action[:, 9:31] intentionally remains absolute.
        return ee_pose, hand_joint, mixed_action

    def _hand_seq(self, sample: dict):
        if "hand_joint" in sample:
            hand = sample["hand_joint"]
        else:
            state = sample["state"]
            hand = state if state.shape[-1] == HAND_DIM else state[:, EE_DIM:ACTION_DIM]
        action = sample["action"]
        if action.shape[-1] == ACTION_DIM:
            action = action[:, EE_DIM:ACTION_DIM]
        return hand[: self.n_obs_steps].astype(np.float32), action.astype(np.float32)

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        if self.normalizer_path:
            path = os.path.expanduser(self.normalizer_path)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"hand-joint normalizer file not found: {path} "
                    "(omit normalizer_path to fit on this dataset instead)"
                )
            stat = load_hand_joint_stat(path)
            normalizer["hand_joint"] = hand_joint_normalizer_from_stat(stat)
        else:
            warnings.warn(
                "No normalizer_path: fitting hand_joint on this dataset. "
                "This scaler is not shared with the other domain; do not mix "
                "normalized hand actions or scores across sim/real.",
                UserWarning,
                stacklevel=2,
            )
            hands = []
            for index in range(len(self.lowdim_sampler)):
                sample = self.lowdim_sampler.sample_sequence(index)
                if self.action_dim == HAND_DIM:
                    hand, _ = self._hand_seq(sample)
                    hands.append(hand)
                else:
                    _, hand, _ = self._convert_lowdim(sample["state"], sample["action"])
                    hands.append(hand)
            normalizer["hand_joint"] = SingleFieldLinearNormalizer.create_fit(
                np.concatenate(hands, axis=0)
            )
        if "ee_pose" in self.lowdim_keys:
            relative_ee = []
            for index in range(len(self.lowdim_sampler)):
                sample = self.lowdim_sampler.sample_sequence(index)
                ee, _, _ = self._convert_lowdim(sample["state"], sample["action"])
                relative_ee.append(ee)
            normalizer["ee_pose"] = SingleFieldLinearNormalizer.create_fit(
                np.concatenate(relative_ee, axis=0)
            )
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        actions = []
        for index in range(len(self.lowdim_sampler)):
            sample = self.lowdim_sampler.sample_sequence(index)
            if self.action_dim == HAND_DIM:
                _, action = self._hand_seq(sample)
                actions.append(action)
            else:
                actions.append(
                    self._convert_lowdim(sample["state"], sample["action"])[2]
                )
        return torch.from_numpy(np.concatenate(actions, axis=0))

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(index)
        obs: Dict[str, np.ndarray] = {}

        if self.action_dim == HAND_DIM:
            hand_joint, action = self._hand_seq(sample)
            if "hand_joint" in self.lowdim_keys:
                obs["hand_joint"] = hand_joint
        else:
            ee_pose, hand_joint, action = self._convert_lowdim(
                sample["state"], sample["action"]
            )
            for key in self.rgb_keys:
                obs[key] = (
                    np.moveaxis(sample[key][: self.n_obs_steps], -1, 1).astype(np.float32)
                    / 255.0
                )
            if "ee_pose" in self.lowdim_keys:
                obs["ee_pose"] = ee_pose
            if "hand_joint" in self.lowdim_keys:
                obs["hand_joint"] = hand_joint.astype(np.float32)

        return {
            "obs": dict_apply(obs, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
