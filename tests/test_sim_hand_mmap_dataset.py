from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.sim_hand_hdf5 import load_sim_hand_hdf5
from diffusion_policy.dataset.sim_hand_mmap_converter import build_sim_hand_mmap_cache

HAND_DIM = 22
HORIZON = 4
PAD_BEFORE = 2
PAD_AFTER = 3


def _row(
    episode_id: int,
    env_id: int,
    step: int,
    *,
    done: bool = False,
    success: bool = False,
    failure: bool = False,
    timeout: bool = False,
):
    value = float(episode_id * 100 + step)
    if success:
        reset_reason = 1
    elif failure:
        reset_reason = 2
    elif timeout:
        reset_reason = 3
    elif done:
        reset_reason = 4
    else:
        reset_reason = 0
    return {
        "episode_id": episode_id,
        "env_id": env_id,
        "step": step,
        "done": done,
        "success": success,
        "failure": failure,
        "timeout": timeout,
        "reset_reason": reset_reason,
        "qpos": np.full(HAND_DIM, value, dtype=np.float32),
        "target_after": np.full(HAND_DIM, value + 10_000.0, dtype=np.float32),
    }


def _write_rollout(dataset_dir: Path, shards):
    shard_dir = dataset_dir / "shards"
    shard_dir.mkdir(parents=True)
    manifest_shards = []
    total_rows = 0
    for shard_index, rows in enumerate(shards):
        path = shard_dir / f"shard_{shard_index:06d}.h5"
        with h5py.File(path, "w") as f:
            index = f.create_group("index")
            robot = f.create_group("robot")
            for key, dtype in (
                ("episode_id", np.int64),
                ("env_id", np.int32),
                ("step", np.int64),
                ("reset_reason", np.int8),
            ):
                index.create_dataset(
                    key,
                    data=np.asarray([row[key] for row in rows], dtype=dtype),
                )
            for key in ("done", "success", "failure", "timeout"):
                index.create_dataset(
                    key,
                    data=np.asarray([row[key] for row in rows], dtype=np.bool_),
                )
            for key in ("qpos", "target_after"):
                robot.create_dataset(
                    key,
                    data=np.stack([row[key] for row in rows]),
                )
        manifest_shards.append(
            {
                "path": str(path.relative_to(dataset_dir)),
                "num_transitions": len(rows),
            }
        )
        total_rows += len(rows)
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_transitions": total_rows,
                "dof": HAND_DIM,
                "shards": manifest_shards,
            }
        ),
        encoding="utf-8",
    )


def _write_four_episode_fixture(dataset_dir: Path) -> None:
    # Lengths 4, 5, 6, 7 retained; length-2 episode dropped by horizon=4.
    rows = []
    for episode_id, env_id, length in (
        (10, 0, 4),
        (20, 1, 5),
        (30, 2, 6),
        (40, 3, 7),
        (50, 4, 2),
    ):
        for step in range(length):
            rows.append(_row(episode_id, env_id, step))
    _write_rollout(dataset_dir, shards=[rows])


@pytest.fixture
def cache_fixture(tmp_path_factory):
    def _make(*, val_ratio: float = 0.0):
        dataset_path = tmp_path_factory.mktemp("sim_hand_mmap_ds")
        _write_four_episode_fixture(dataset_path)
        build_sim_hand_mmap_cache(
            dataset_path,
            horizon=HORIZON,
            pad_before=PAD_BEFORE,
            pad_after=PAD_AFTER,
            val_ratio=val_ratio,
            seed=42,
            chunk_rows=8,
        )
        reference_buffer = load_sim_hand_hdf5(
            dataset_path,
            min_episode_length=HORIZON,
        )
        return dataset_path, reference_buffer

    return _make


def test_mmap_dataset_matches_sequence_sampler_for_every_padded_window(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, reference_buffer = cache_fixture(val_ratio=0.0)
    dataset = SimHandMmapDataset(
        str(dataset_path),
        horizon=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        seed=42,
        val_ratio=0.0,
    )
    reference = SequenceSampler(
        reference_buffer,
        sequence_length=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        keys=["hand_joint", "action"],
    )

    assert len(dataset) == len(reference)
    assert isinstance(dataset.obs, np.memmap)
    assert isinstance(dataset.action, np.memmap)
    assert not dataset.obs.flags.writeable
    assert not dataset.action.flags.writeable
    assert dataset.sample_ends.shape == (len(dataset.selected_episode_indices),)
    assert not hasattr(dataset, "indices")
    for index in range(len(dataset)):
        actual = dataset[index]
        expected = reference.sample_sequence(index)
        torch.testing.assert_close(
            actual["obs"], torch.from_numpy(expected["hand_joint"])
        )
        torch.testing.assert_close(
            actual["action"], torch.from_numpy(expected["action"])
        )


def test_mmap_train_validation_views_are_disjoint_and_downsample_metadata(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, _ = cache_fixture(val_ratio=0.25)
    dataset = SimHandMmapDataset(
        str(dataset_path),
        horizon=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        seed=42,
        val_ratio=0.25,
        max_train_episodes=1,
    )
    validation = dataset.get_validation_dataset()

    assert len(dataset.selected_episode_indices) == 1
    assert set(dataset.selected_episode_indices).isdisjoint(
        set(validation.selected_episode_indices)
    )
    assert np.shares_memory(dataset.obs, validation.obs)
    assert np.shares_memory(dataset.action, validation.action)


def test_mmap_normalizer_reconstructs_cached_all_row_statistics(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, reference_buffer = cache_fixture(val_ratio=0.25)
    dataset = SimHandMmapDataset(
        str(dataset_path),
        horizon=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        seed=42,
        val_ratio=0.25,
    )
    normalizer = dataset.get_normalizer()

    torch.testing.assert_close(
        normalizer["obs"].get_input_stats()["mean"],
        torch.from_numpy(reference_buffer["hand_joint"].mean(axis=0)),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        normalizer["action"].get_input_stats()["std"],
        torch.from_numpy(reference_buffer["action"].std(axis=0, ddof=1)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_mmap_dataset_rejects_out_of_range_indices(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, _ = cache_fixture(val_ratio=0.0)
    dataset = SimHandMmapDataset(
        str(dataset_path),
        horizon=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        seed=42,
        val_ratio=0.0,
    )

    with pytest.raises(IndexError):
        _ = dataset[-1]
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_mmap_dataset_rejects_temporal_and_split_mismatch(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_cache import SimHandMmapCacheError
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, _ = cache_fixture(val_ratio=0.25)

    with pytest.raises(SimHandMmapCacheError, match="horizon mismatch"):
        SimHandMmapDataset(
            str(dataset_path),
            horizon=HORIZON + 1,
            pad_before=PAD_BEFORE,
            pad_after=PAD_AFTER,
            seed=42,
            val_ratio=0.25,
        )

    with pytest.raises(SimHandMmapCacheError, match="val_ratio mismatch"):
        SimHandMmapDataset(
            str(dataset_path),
            horizon=HORIZON,
            pad_before=PAD_BEFORE,
            pad_after=PAD_AFTER,
            seed=42,
            val_ratio=0.1,
        )


def test_mmap_dataset_rejects_zero_selected_training_episodes(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, _ = cache_fixture(val_ratio=0.25)
    with pytest.raises(ValueError, match="(?i)no .*train|training episode"):
        SimHandMmapDataset(
            str(dataset_path),
            horizon=HORIZON,
            pad_before=PAD_BEFORE,
            pad_after=PAD_AFTER,
            seed=42,
            val_ratio=0.25,
            max_train_episodes=0,
        )


def test_mmap_get_all_actions_refuses_materialization(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, _ = cache_fixture(val_ratio=0.0)
    dataset = SimHandMmapDataset(
        str(dataset_path),
        horizon=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        seed=42,
        val_ratio=0.0,
    )
    with pytest.raises(
        RuntimeError,
        match="get_all_actions would materialize the mmap dataset",
    ):
        dataset.get_all_actions()


def test_mmap_normalizer_rejects_unknown_mode_and_kwargs(cache_fixture):
    from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

    dataset_path, _ = cache_fixture(val_ratio=0.0)
    dataset = SimHandMmapDataset(
        str(dataset_path),
        horizon=HORIZON,
        pad_before=PAD_BEFORE,
        pad_after=PAD_AFTER,
        seed=42,
        val_ratio=0.0,
    )

    with pytest.raises(ValueError, match="mode"):
        dataset.get_normalizer(mode="unknown")
    with pytest.raises(ValueError, match="(?i)unknown|unexpected|kwargs"):
        dataset.get_normalizer(last_n_dims=1)
