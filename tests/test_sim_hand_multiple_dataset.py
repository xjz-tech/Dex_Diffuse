from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from test_sim_hand_hdf5_dataset import _row, _write_rollout
from diffusion_policy.common.chunked_uniform_sampler import ChunkedUniformSampler
from diffusion_policy.dataset.sim_hand_lowdim_dataset import SimHandLowdimDataset
from diffusion_policy.dataset.sim_hand_multiple_dataset import (
    LazySimHandMultipleDataset,
)


def _nested_root(tmp_path: Path) -> Path:
    root = tmp_path / "sim_data"
    first = [_row(10, 0, step) for step in range(4)]
    second = [_row(10, 0, step) for step in range(4)]
    for row in second:
        row["qpos"] += 5_000.0
    short = [_row(99, 1, step) for step in range(2)]
    _write_rollout(root / "brush" / "run_b", shards=[second + short])
    _write_rollout(root / "bulb" / "run_a", shards=[first])
    return root


def _lazy(root: Path, cache_path: Path, **kwargs) -> LazySimHandMultipleDataset:
    options = dict(
        dataset_path=str(root),
        cache_path=str(cache_path),
        horizon=3,
        pad_before=0,
        pad_after=0,
        val_ratio=0.0,
        seed=42,
    )
    options.update(kwargs)
    return LazySimHandMultipleDataset(**options)


def test_lazy_dataset_keeps_duplicate_episode_ids_separate(tmp_path):
    root = _nested_root(tmp_path)
    dataset = _lazy(root, tmp_path / "cache")
    assert len(dataset.episode_ids) == 3
    np.testing.assert_array_equal(dataset.episode_ids[:2], [10, 99])
    sample = dataset[0]
    assert set(sample) == {"obs", "action"}
    assert "source_id" not in sample
    np.testing.assert_allclose(sample["obs"][0, 0].item(), 6_000.0)


def test_lazy_windows_and_normalizer_match_eager_single_leaf(tmp_path):
    leaf = tmp_path / "leaf"
    _write_rollout(
        leaf,
        shards=[[_row(10, 0, step) for step in range(5)]],
    )
    eager = SimHandLowdimDataset(
        dataset_path=str(leaf),
        horizon=3,
        pad_before=1,
        pad_after=1,
        val_ratio=0.0,
        seed=0,
    )
    nested = tmp_path / "sim_data" / "obj" / "run"
    _write_rollout(
        nested,
        shards=[[_row(10, 0, step) for step in range(5)]],
    )
    lazy = _lazy(
        nested.parent.parent,
        tmp_path / "cache",
        horizon=3,
        pad_before=1,
        pad_after=1,
        val_ratio=0.0,
        seed=0,
    )
    assert len(lazy) == len(eager)
    for index in range(len(eager)):
        left = eager[index]
        right = lazy[index]
        torch.testing.assert_close(left["obs"], right["obs"])
        torch.testing.assert_close(left["action"], right["action"])
    eager_norm = eager.get_normalizer()
    lazy_norm = lazy.get_normalizer()
    torch.testing.assert_close(
        eager_norm["obs"].get_input_stats()["min"],
        lazy_norm["obs"].get_input_stats()["min"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        eager_norm["action"].get_input_stats()["max"],
        lazy_norm["action"].get_input_stats()["max"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_validation_mask_is_independent_of_train_downsampling(tmp_path):
    root = _nested_root(tmp_path)
    dataset = _lazy(
        root,
        tmp_path / "cache",
        val_ratio=0.5,
        max_train_episodes=1,
        seed=0,
    )
    validation = dataset.get_validation_dataset()
    assert dataset.train_mask.sum() == 1
    assert dataset.val_mask.sum() >= 1
    assert validation.get_training_sampler(seed=1) is None
    assert len(validation) > 0
    assert "source_id" not in validation[0]


def test_custom_sampler_is_uniform_and_rank_aware(tmp_path):
    root = _nested_root(tmp_path)
    dataset = _lazy(root, tmp_path / "cache", samples_per_epoch=20)
    sampler = dataset.get_training_sampler(seed=7, num_replicas=4, rank=1)
    assert isinstance(sampler, ChunkedUniformSampler)
    assert len(sampler) == 5
    indices = list(sampler)
    assert indices == list(dataset.get_training_sampler(seed=7, num_replicas=4, rank=1))
    assert all(0 <= index < len(dataset) for index in indices)
