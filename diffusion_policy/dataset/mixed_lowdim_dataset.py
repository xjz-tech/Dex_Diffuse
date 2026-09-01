from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from diffusion_policy.common.probability_mixture_sampler import (
    ProbabilityMixtureSampler,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class MixedLowdimDataset(BaseLowdimDataset):
    def __init__(self, datasets, probabilities, samples_per_epoch=None):
        if not isinstance(datasets, Mapping) or len(datasets) < 2:
            raise ValueError("mixed dataset requires at least two named datasets")
        self.datasets = dict(datasets)
        if not all(isinstance(value, BaseLowdimDataset) for value in self.datasets.values()):
            raise TypeError("all mixed components must inherit BaseLowdimDataset")
        self.source_names = tuple(self.datasets)
        if set(probabilities) != set(self.source_names):
            raise ValueError("probability keys must exactly match dataset names")
        self.probabilities = tuple(float(probabilities[name]) for name in self.source_names)
        probability_array = np.asarray(self.probabilities)
        if (not np.isfinite(probability_array).all()
                or np.any(probability_array <= 0)
                or not np.isclose(probability_array.sum(), 1.0, atol=1e-8, rtol=0.0)):
            raise ValueError("probabilities must be finite, positive, and sum to 1")
        self.source_lengths = tuple(len(value) for value in self.datasets.values())
        if any(length <= 0 for length in self.source_lengths):
            raise ValueError("mixed component training datasets must be non-empty")
        self.offsets = tuple(np.r_[0, np.cumsum(self.source_lengths[:-1])].tolist())
        if samples_per_epoch is not None and (
            type(samples_per_epoch) is not int or samples_per_epoch <= 0
        ):
            raise ValueError("samples_per_epoch must be null or a positive integer")
        self.samples_per_epoch = samples_per_epoch
        self._sampling_enabled = True

    def __len__(self):
        return sum(self.source_lengths)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_id = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local_index = index - self.offsets[source_id]
        sample = dict(self.datasets[self.source_names[source_id]][local_index])
        sample["source_id"] = torch.tensor(source_id, dtype=torch.int64)
        return sample

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.datasets = {
            name: dataset.get_validation_dataset()
            for name, dataset in self.datasets.items()
        }
        result.source_lengths = tuple(len(value) for value in result.datasets.values())
        result.offsets = tuple(np.r_[0, np.cumsum(result.source_lengths[:-1])].tolist())
        result._sampling_enabled = False
        return result

    def get_training_sampler(self, *, seed, num_replicas=1, rank=0):
        if not self._sampling_enabled:
            return None
        return ProbabilityMixtureSampler(
            source_lengths=self.source_lengths,
            probabilities=self.probabilities,
            samples_per_epoch=(self.samples_per_epoch or len(self)),
            seed=int(seed),
            num_replicas=int(num_replicas),
            rank=int(rank),
        )

    def get_normalizer(self, mode="limits", output_max=1.0, output_min=-1.0,
                       range_eps=1e-4, **kwargs):
        if kwargs:
            raise TypeError(f"unsupported mixed normalizer options: {sorted(kwargs)}")
        component_normalizers = [
            dataset.get_normalizer(
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
            )
            for dataset in self.datasets.values()
        ]
        counts = [
            dataset.get_normalizer_sample_count()
            for dataset in self.datasets.values()
        ]
        result = LinearNormalizer()
        for field in ("obs", "action"):
            merged = _merge_field_stats([
                (normalizer[field].get_input_stats(), count)
                for normalizer, count in zip(component_normalizers, counts)
            ])
            result[field] = _normalizer_from_stats(
                merged, mode, output_min, output_max, range_eps
            )
        return result


def _merge_field_stats(stats_and_counts):
    total = sum(count for _, count in stats_and_counts)
    means = [stats["mean"] for stats, _ in stats_and_counts]
    mean = sum(
        stats["mean"] * count for stats, count in stats_and_counts
    ) / total
    minimum = torch.stack([stats["min"] for stats, _ in stats_and_counts]).amin(0)
    maximum = torch.stack([stats["max"] for stats, _ in stats_and_counts]).amax(0)
    m2 = sum(
        (count - 1) * stats["std"].square()
        + count * (component_mean - mean).square()
        for (stats, count), component_mean in zip(stats_and_counts, means)
    )
    std = torch.sqrt(m2 / (total - 1))
    return {"min": minimum, "max": maximum, "mean": mean, "std": std}


def _normalizer_from_stats(stats, mode, output_min, output_max, range_eps):
    if mode == "limits":
        input_range = stats["max"] - stats["min"]
        ignore = input_range < range_eps
        safe_range = input_range.clone()
        safe_range[ignore] = output_max - output_min
        scale = (output_max - output_min) / safe_range
        offset = output_min - scale * stats["min"]
        offset[ignore] = (output_max + output_min) / 2 - stats["min"][ignore]
    elif mode == "gaussian":
        safe_std = stats["std"].clone()
        safe_std[safe_std < range_eps] = 1
        scale = 1 / safe_std
        offset = -stats["mean"] * scale
    else:
        raise ValueError(f"unsupported normalizer mode: {mode}")
    return SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
