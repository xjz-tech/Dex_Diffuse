from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.chunked_uniform_sampler import ChunkedUniformSampler
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dataset.sim_hand_hdf5 import (
    HAND_DIM,
    OBS_DIM,
    compose_sim_hand_obs,
)
from diffusion_policy.dataset.sim_hand_multiple_cache import (
    LeafCache,
    ensure_sim_hand_multiple_cache,
    open_leaf_cache,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


def _clamped_pads(horizon: int, pad_before: int, pad_after: int) -> tuple[int, int]:
    pad_before = min(max(int(pad_before), 0), int(horizon) - 1)
    pad_after = min(max(int(pad_after), 0), int(horizon) - 1)
    return pad_before, pad_after


def episode_window_count(
    episode_length: int,
    horizon: int,
    pad_before: int,
    pad_after: int,
) -> int:
    pad_before, pad_after = _clamped_pads(horizon, pad_before, pad_after)
    return int(episode_length - horizon + pad_after + pad_before + 1)


def window_spec(
    episode_start: int,
    episode_length: int,
    local_index: int,
    horizon: int,
    pad_before: int,
    pad_after: int,
) -> tuple[int, int, int, int]:
    pad_before, pad_after = _clamped_pads(horizon, pad_before, pad_after)
    idx = int(local_index) - pad_before
    buffer_start = max(idx, 0) + int(episode_start)
    buffer_end = min(idx + horizon, int(episode_length)) + int(episode_start)
    sample_start = buffer_start - (idx + int(episode_start))
    sample_end = horizon - ((idx + horizon + int(episode_start)) - buffer_end)
    return buffer_start, buffer_end, sample_start, sample_end


def _pad_sequence(
    sample: np.ndarray,
    sample_start: int,
    sample_end: int,
    horizon: int,
) -> np.ndarray:
    if sample_start == 0 and sample_end == horizon:
        return np.asarray(sample)
    data = np.zeros((horizon,) + sample.shape[1:], dtype=sample.dtype)
    if sample_start > 0:
        data[:sample_start] = sample[0]
    if sample_end < horizon:
        data[sample_end:] = sample[-1]
    data[sample_start:sample_end] = sample
    return data


def _normalizer_from_stats(
    minimum: np.ndarray,
    maximum: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mode: str,
    output_min: float,
    output_max: float,
    range_eps: float,
) -> SingleFieldLinearNormalizer:
    if mode == "limits":
        input_range = maximum - minimum
        ignore = input_range < range_eps
        safe_range = input_range.copy()
        safe_range[ignore] = output_max - output_min
        scale = (output_max - output_min) / safe_range
        offset = output_min - scale * minimum
        offset[ignore] = (output_max + output_min) / 2 - minimum[ignore]
    elif mode == "gaussian":
        safe_std = std.copy()
        safe_std[safe_std < range_eps] = 1
        scale = 1 / safe_std
        offset = -mean * scale
    else:
        raise ValueError(f"unsupported normalizer mode: {mode}")
    stats = {
        "min": minimum.astype(np.float32, copy=False),
        "max": maximum.astype(np.float32, copy=False),
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale.astype(np.float32, copy=False),
        offset.astype(np.float32, copy=False),
        stats,
    )


class LazySimHandMultipleDataset(BaseLowdimDataset):
    """Memory-mapped union of every DexGen HDF5 leaf under a sim_data root."""

    def __init__(
        self,
        dataset_path: str,
        cache_path: str | None = None,
        horizon: int = 12,
        pad_before: int = 3,
        pad_after: int = 8,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
        samples_per_epoch: int | None = None,
        rebuild_cache: bool = False,
    ):
        super().__init__()
        if samples_per_epoch is not None and (
            type(samples_per_epoch) is not int or samples_per_epoch <= 0
        ):
            raise ValueError("samples_per_epoch must be null or a positive integer")
        cache_dirs = ensure_sim_hand_multiple_cache(
            dataset_path,
            cache_path,
            rebuild=bool(rebuild_cache),
        )
        self._leaves: list[LeafCache] = [open_leaf_cache(path) for path in cache_dirs]
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.samples_per_epoch = samples_per_epoch
        self._sampling_enabled = True
        self._dataset_path = str(dataset_path)

        leaf_index = []
        local_index = []
        starts = []
        lengths = []
        episode_ids = []
        env_ids = []
        obs_min = []
        obs_max = []
        obs_sum = []
        obs_sum_squares = []
        action_min = []
        action_max = []
        action_sum = []
        action_sum_squares = []
        for source_index, leaf in enumerate(self._leaves):
            episode_starts = np.r_[0, leaf.episode_ends[:-1]]
            episode_lengths = leaf.episode_ends - episode_starts
            n_episodes = len(episode_lengths)
            leaf_index.append(np.full(n_episodes, source_index, dtype=np.int32))
            local_index.append(np.arange(n_episodes, dtype=np.int64))
            starts.append(episode_starts.astype(np.int64, copy=False))
            lengths.append(episode_lengths.astype(np.int64, copy=False))
            episode_ids.append(leaf.episode_ids)
            env_ids.append(leaf.env_ids)
            obs_min.append(leaf.obs_min)
            obs_max.append(leaf.obs_max)
            obs_sum.append(leaf.obs_sum)
            obs_sum_squares.append(leaf.obs_sum_squares)
            action_min.append(leaf.action_min)
            action_max.append(leaf.action_max)
            action_sum.append(leaf.action_sum)
            action_sum_squares.append(leaf.action_sum_squares)

        self._leaf_index = np.concatenate(leaf_index)
        self._local_index = np.concatenate(local_index)
        self._episode_starts = np.concatenate(starts)
        self._episode_lengths = np.concatenate(lengths)
        self.episode_ids = np.concatenate(episode_ids)
        self.env_ids = np.concatenate(env_ids)
        self._obs_min = np.concatenate(obs_min)
        self._obs_max = np.concatenate(obs_max)
        self._obs_sum = np.concatenate(obs_sum)
        self._obs_sum_squares = np.concatenate(obs_sum_squares)
        self._action_min = np.concatenate(action_min)
        self._action_max = np.concatenate(action_max)
        self._action_sum = np.concatenate(action_sum)
        self._action_sum_squares = np.concatenate(action_sum_squares)

        eligible = self._episode_lengths >= self.horizon
        if not np.any(eligible):
            raise ValueError(
                "Sim-Hand multiple dataset contains no episodes with length >= "
                f"horizon={self.horizon}: {self._dataset_path}"
            )
        eligible_idx = np.flatnonzero(eligible)
        val_mask_eligible = get_val_mask(
            n_episodes=len(eligible_idx),
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask_eligible = downsample_mask(
            mask=~val_mask_eligible,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.train_mask = np.zeros(len(self._episode_lengths), dtype=bool)
        self.val_mask = np.zeros(len(self._episode_lengths), dtype=bool)
        self.train_mask[eligible_idx] = train_mask_eligible
        self.val_mask[eligible_idx] = val_mask_eligible
        self._eligible_mask = eligible
        self._set_window_index(self.train_mask, split_name="training")

    def _set_window_index(self, episode_mask: np.ndarray, split_name: str) -> None:
        selected = np.flatnonzero(episode_mask)
        if selected.size == 0:
            raise ValueError(
                f"Sim-Hand multiple {split_name} split has no episodes: "
                f"{self._dataset_path}"
            )
        counts = np.empty(len(selected), dtype=np.int64)
        for offset, episode_i in enumerate(selected):
            counts[offset] = episode_window_count(
                int(self._episode_lengths[episode_i]),
                self.horizon,
                self.pad_before,
                self.pad_after,
            )
        if int(counts.sum()) <= 0:
            raise ValueError(
                f"Sim-Hand multiple {split_name} window space is empty: "
                f"{self._dataset_path}"
            )
        self._index_episodes = selected
        self._window_counts = counts
        self._cumulative = np.cumsum(counts)

    def get_validation_dataset(self) -> "LazySimHandMultipleDataset":
        validation = copy.copy(self)
        validation._sampling_enabled = False
        validation._set_window_index(self.val_mask, split_name="validation")
        return validation

    def get_training_sampler(self, *, seed, num_replicas=1, rank=0):
        if not self._sampling_enabled:
            return None
        samples_per_epoch = self.samples_per_epoch or len(self)
        return ChunkedUniformSampler(
            dataset_length=len(self),
            samples_per_epoch=int(samples_per_epoch),
            seed=int(seed),
            num_replicas=int(num_replicas),
            rank=int(rank),
        )

    def get_normalizer_sample_count(self) -> int:
        return int(self._episode_lengths[self._eligible_mask].sum())

    def get_all_actions(self) -> torch.Tensor:
        raise RuntimeError(
            "LazySimHandMultipleDataset does not materialize all actions; "
            "use cached episode statistics via get_normalizer()"
        )

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        output_max = kwargs.pop("output_max", 1.0)
        output_min = kwargs.pop("output_min", -1.0)
        range_eps = kwargs.pop("range_eps", 1e-4)
        if kwargs:
            raise TypeError(
                f"unsupported multiple normalizer options: {sorted(kwargs)}"
            )
        mask = self._eligible_mask
        lengths = self._episode_lengths[mask].astype(np.float64)
        total = float(lengths.sum())
        normalizer = LinearNormalizer()
        for field, minimum, maximum, sums, sum_squares, width in (
            (
                "obs",
                self._obs_min[mask],
                self._obs_max[mask],
                self._obs_sum[mask],
                self._obs_sum_squares[mask],
                OBS_DIM,
            ),
            (
                "action",
                self._action_min[mask],
                self._action_max[mask],
                self._action_sum[mask],
                self._action_sum_squares[mask],
                HAND_DIM,
            ),
        ):
            assert minimum.shape[1] == width
            field_min = minimum.min(axis=0)
            field_max = maximum.max(axis=0)
            field_sum = sums.sum(axis=0)
            field_sumsq = sum_squares.sum(axis=0)
            mean = field_sum / total
            m2 = np.maximum(field_sumsq - (field_sum ** 2) / total, 0.0)
            std = np.sqrt(m2 / (total - 1.0))
            normalizer[field] = _normalizer_from_stats(
                field_min,
                field_max,
                mean,
                std,
                mode,
                output_min,
                output_max,
                range_eps,
            )
        return normalizer

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_pos = int(np.searchsorted(self._cumulative, index, side="right"))
        previous = 0 if episode_pos == 0 else int(self._cumulative[episode_pos - 1])
        local_window = index - previous
        episode_i = int(self._index_episodes[episode_pos])
        leaf = self._leaves[int(self._leaf_index[episode_i])]
        episode_start = int(self._episode_starts[episode_i])
        episode_length = int(self._episode_lengths[episode_i])
        buffer_start, buffer_end, sample_start, sample_end = window_spec(
            episode_start,
            episode_length,
            local_window,
            self.horizon,
            self.pad_before,
            self.pad_after,
        )
        qpos = np.array(leaf.qpos[buffer_start:buffer_end], dtype=np.float32, copy=True)
        target_before = np.array(
            leaf.target_before[buffer_start:buffer_end],
            dtype=np.float32,
            copy=True,
        )
        action = np.array(
            leaf.action[buffer_start:buffer_end],
            dtype=np.float32,
            copy=True,
        )
        obs = compose_sim_hand_obs(qpos, target_before)
        data = {
            "obs": _pad_sequence(obs, sample_start, sample_end, self.horizon),
            "action": _pad_sequence(action, sample_start, sample_end, self.horizon),
        }
        return dict_apply(data, torch.from_numpy)
