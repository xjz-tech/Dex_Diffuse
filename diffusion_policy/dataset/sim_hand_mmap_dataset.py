from __future__ import annotations

import copy
import os
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.sampler import downsample_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dataset.sim_hand_mmap_cache import (
    resolve_cache_dir,
    validate_sim_hand_mmap_cache,
)
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

HAND_DIM = 22
_ALLOWED_NORMALIZER_KWARGS = frozenset(
    {"output_min", "output_max", "range_eps", "fit_offset"}
)


def _episode_bounds(episode_ends: np.ndarray, episode_index: int) -> tuple[int, int]:
    start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    end = int(episode_ends[episode_index])
    return start, end


def _pad_sequence_from_mmap(
    array: np.ndarray,
    *,
    episode_start: int,
    episode_length: int,
    local_start: int,
    horizon: int,
) -> np.ndarray:
    buffer_start = max(local_start, 0) + episode_start
    buffer_end = min(local_start + horizon, episode_length) + episode_start
    sample_start = buffer_start - (local_start + episode_start)
    sample_end = horizon - ((local_start + horizon + episode_start) - buffer_end)

    sample = np.asarray(array[buffer_start:buffer_end], dtype=array.dtype).copy()
    if sample_start == 0 and sample_end == horizon:
        return sample

    data = np.empty((horizon,) + array.shape[1:], dtype=array.dtype)
    if sample_start > 0:
        data[:sample_start] = sample[0]
    if sample_end < horizon:
        data[sample_end:] = sample[-1]
    data[sample_start:sample_end] = sample
    return data


def _normalizer_from_cached_stats(
    *,
    input_min: np.ndarray,
    input_max: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    mode: str,
    output_min: float,
    output_max: float,
    range_eps: float,
    fit_offset: bool,
) -> SingleFieldLinearNormalizer:
    input_min = np.asarray(input_min, dtype=np.float32)
    input_max = np.asarray(input_max, dtype=np.float32)
    input_mean = np.asarray(input_mean, dtype=np.float32)
    input_std = np.asarray(input_std, dtype=np.float32)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range = input_range.copy()
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset = offset.copy()
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            if not (output_max > 0 and output_min < 0):
                raise ValueError(
                    "fit_offset=False requires output_max > 0 and output_min < 0"
                )
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = np.maximum(np.abs(input_min), np.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs = input_abs.copy()
            input_abs[ignore_dim] = output_abs
            scale = (output_abs / input_abs).astype(np.float32, copy=False)
            offset = np.zeros_like(input_mean)
    elif mode == "gaussian":
        ignore_dim = input_std < range_eps
        scale = input_std.copy()
        scale[ignore_dim] = 1
        scale = (1 / scale).astype(np.float32, copy=False)
        if fit_offset:
            offset = (-input_mean * scale).astype(np.float32, copy=False)
        else:
            offset = np.zeros_like(input_mean)
    else:
        raise ValueError(f"Unsupported normalizer mode: {mode!r}")

    scale = np.asarray(scale, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict={
            "min": input_min,
            "max": input_max,
            "mean": input_mean,
            "std": input_std,
        },
    )


class SimHandMmapDataset(BaseLowdimDataset):
    """Read-only Sim-Hand dataset over a completed ``exp_data_mmap`` cache."""

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
        expanded_path = os.path.expanduser(dataset_path)
        cache_dir = resolve_cache_dir(expanded_path)
        validate_sim_hand_mmap_cache(
            cache_dir,
            dataset_path=expanded_path,
            horizon=int(horizon),
            pad_before=int(pad_before),
            pad_after=int(pad_after),
            val_ratio=float(val_ratio),
            seed=int(seed),
        )

        self.dataset_path = expanded_path
        self.cache_dir = cache_dir
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.max_train_episodes = max_train_episodes

        self.obs = np.load(
            cache_dir / "obs.npy", mmap_mode="r", allow_pickle=False
        )
        self.action = np.load(
            cache_dir / "action.npy", mmap_mode="r", allow_pickle=False
        )
        self.episode_ends = np.asarray(
            np.load(cache_dir / "episode_ends.npy", mmap_mode="r", allow_pickle=False)
        )
        self.val_mask = np.asarray(
            np.load(cache_dir / "val_mask.npy", mmap_mode="r", allow_pickle=False)
        )
        self._normalizer_path = cache_dir / "normalizer.npz"

        train_mask = downsample_mask(
            mask=~self.val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.train_mask = train_mask
        self._set_selected_episodes(train_mask, role="training")

    def _set_selected_episodes(self, episode_mask: np.ndarray, *, role: str) -> None:
        selected = np.flatnonzero(episode_mask).astype(np.int64, copy=False)
        if selected.size == 0:
            if role == "training":
                raise ValueError(f"no {role} episodes selected")
            self.selected_episode_indices = selected
            self.sample_ends = np.empty(0, dtype=np.int64)
            return

        window_counts = np.empty(selected.shape, dtype=np.int64)
        for position, episode_index in enumerate(selected):
            start, end = _episode_bounds(self.episode_ends, int(episode_index))
            length = end - start
            window_counts[position] = max(
                0,
                length - self.horizon + 1 + self.pad_before + self.pad_after,
            )

        self.selected_episode_indices = selected
        self.sample_ends = np.cumsum(window_counts, dtype=np.int64)

    def get_validation_dataset(self) -> "SimHandMmapDataset":
        validation = copy.copy(self)
        validation._set_selected_episodes(self.val_mask, role="validation")
        return validation

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        if mode not in ("limits", "gaussian"):
            raise ValueError(
                f"Unsupported normalizer mode: {mode!r}; expected 'limits' or 'gaussian'"
            )
        unknown = set(kwargs) - _ALLOWED_NORMALIZER_KWARGS
        if unknown:
            raise ValueError(
                f"Unknown normalizer kwargs: {sorted(unknown)}"
            )

        output_min = float(kwargs.get("output_min", -1.0))
        output_max = float(kwargs.get("output_max", 1.0))
        range_eps = float(kwargs.get("range_eps", 1e-4))
        fit_offset = bool(kwargs.get("fit_offset", True))
        if output_max <= output_min:
            raise ValueError("output_max must be greater than output_min")

        with np.load(self._normalizer_path, allow_pickle=False) as stats:
            normalizer = LinearNormalizer()
            normalizer["obs"] = _normalizer_from_cached_stats(
                input_min=stats["obs_min"],
                input_max=stats["obs_max"],
                input_mean=stats["obs_mean"],
                input_std=stats["obs_std"],
                mode=mode,
                output_min=output_min,
                output_max=output_max,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )
            normalizer["action"] = _normalizer_from_cached_stats(
                input_min=stats["action_min"],
                input_max=stats["action_max"],
                input_mean=stats["action_mean"],
                input_std=stats["action_std"],
                mode=mode,
                output_min=output_min,
                output_max=output_max,
                range_eps=range_eps,
                fit_offset=fit_offset,
            )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        raise RuntimeError("get_all_actions would materialize the mmap dataset")

    def __len__(self) -> int:
        if self.sample_ends.size == 0:
            return 0
        return int(self.sample_ends[-1])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if type(index) is not int:
            raise IndexError(f"dataset index must be an int, got {type(index)!r}")
        if index < 0 or index >= len(self):
            raise IndexError(
                f"index {index} out of range for dataset of length {len(self)}"
            )

        episode_position = int(
            np.searchsorted(self.sample_ends, index, side="right")
        )
        previous_end = (
            0
            if episode_position == 0
            else int(self.sample_ends[episode_position - 1])
        )
        local_window = index - previous_end
        local_start = local_window - self.pad_before

        episode_index = int(self.selected_episode_indices[episode_position])
        episode_start, episode_end = _episode_bounds(
            self.episode_ends, episode_index
        )
        episode_length = episode_end - episode_start

        obs = _pad_sequence_from_mmap(
            self.obs,
            episode_start=episode_start,
            episode_length=episode_length,
            local_start=local_start,
            horizon=self.horizon,
        )
        action = _pad_sequence_from_mmap(
            self.action,
            episode_start=episode_start,
            episode_length=episode_length,
            local_start=local_start,
            horizon=self.horizon,
        )
        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action),
        }
