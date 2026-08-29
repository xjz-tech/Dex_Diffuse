from __future__ import annotations

import numpy as np
import pytest
import torch
import zarr

from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.sim_hand_lowdim_dataset import SimHandLowdimDataset


HAND_DIM = 22
OBS_DIM = 66


def _episode_slice(episode_ends: np.ndarray, episode_index: int) -> slice:
    start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    return slice(start, int(episode_ends[episode_index]))


def _write_replay_buffer(
    dataset_dir,
    obs: np.ndarray,
    action: np.ndarray,
    episode_ends: np.ndarray,
    episode_ends_dtype="int64",
):
    root = zarr.open_group(str(dataset_dir / "replay_buffer.zarr"), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.create_dataset(
        "obs",
        data=obs,
        shape=obs.shape,
        chunks=(min(16, len(obs)), OBS_DIM),
    )
    data.create_dataset(
        "action",
        data=action,
        shape=action.shape,
        chunks=(min(16, len(action)), HAND_DIM),
    )
    meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        shape=episode_ends.shape,
        dtype=episode_ends_dtype,
    )


@pytest.fixture
def synthetic_dataset(tmp_path):
    lengths = np.array([14, 15, 16, 17, 18], dtype=np.int64)
    episode_ends = np.cumsum(lengths)
    n_steps = int(episode_ends[-1])
    obs = np.linspace(
        -1.0, 1.0, num=n_steps * OBS_DIM, dtype=np.float32
    ).reshape(n_steps, OBS_DIM)
    action = np.linspace(
        -2.0, 2.0, num=n_steps * HAND_DIM, dtype=np.float32
    ).reshape(n_steps, HAND_DIM)

    val_mask = get_val_mask(len(lengths), val_ratio=0.2, seed=42)
    val_index = int(np.flatnonzero(val_mask)[0])
    val_slice = _episode_slice(episode_ends, val_index)
    obs[val_slice.start, 0] = -1000.0
    action[val_slice.start, 0] = 1000.0

    _write_replay_buffer(tmp_path, obs, action, episode_ends)
    return tmp_path, obs, action, episode_ends, val_index


def _make_dataset(dataset_dir, max_train_episodes=None):
    return SimHandLowdimDataset(
        dataset_path=str(dataset_dir),
        horizon=12,
        pad_before=3,
        pad_after=8,
        seed=42,
        val_ratio=0.2,
        max_train_episodes=max_train_episodes,
    )


def test_dataset_maps_standard_disk_keys_to_training_keys(synthetic_dataset):
    dataset_dir, _, _, _, _ = synthetic_dataset
    dataset = _make_dataset(dataset_dir)

    sample = dataset[0]

    assert set(sample) == {"obs", "action"}
    assert sample["obs"].shape == (12, OBS_DIM)
    assert sample["action"].shape == (12, HAND_DIM)
    assert sample["obs"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32


def test_train_and_validation_use_disjoint_episode_masks(synthetic_dataset):
    dataset_dir, _, _, _, _ = synthetic_dataset
    dataset = _make_dataset(dataset_dir)
    validation = dataset.get_validation_dataset()
    expected_sampler = SequenceSampler(
        replay_buffer=dataset.replay_buffer,
        sequence_length=12,
        pad_before=3,
        pad_after=8,
        keys=["obs", "action"],
        episode_mask=dataset.val_mask,
    )

    assert not np.any(dataset.train_mask & dataset.val_mask)
    np.testing.assert_array_equal(
        validation.sampler.indices,
        expected_sampler.indices,
    )


def test_max_train_episodes_does_not_expand_validation_mask(synthetic_dataset):
    dataset_dir, _, _, _, _ = synthetic_dataset
    dataset = _make_dataset(dataset_dir, max_train_episodes=1)
    validation = dataset.get_validation_dataset()

    assert int(dataset.train_mask.sum()) == 1
    assert int(dataset.val_mask.sum()) == 1
    assert len(validation) < len(
        SequenceSampler(
            replay_buffer=dataset.replay_buffer,
            sequence_length=12,
            pad_before=3,
            pad_after=8,
            keys=["obs", "action"],
            episode_mask=~dataset.train_mask,
        )
    )


def test_normalizer_uses_training_and_validation_episodes(synthetic_dataset):
    dataset_dir, obs, action, _, val_index = synthetic_dataset
    dataset = _make_dataset(dataset_dir)
    assert dataset.val_mask[val_index]

    normalizer = dataset.get_normalizer()

    torch.testing.assert_close(
        normalizer["obs"].get_input_stats()["min"],
        torch.from_numpy(obs.min(axis=0)),
    )
    torch.testing.assert_close(
        normalizer["action"].get_input_stats()["max"],
        torch.from_numpy(action.max(axis=0)),
    )


def test_dataset_rejects_missing_required_key(tmp_path):
    episode_ends = np.array([12], dtype=np.int64)
    obs = np.zeros((12, OBS_DIM), dtype=np.float32)
    action = np.zeros((12, HAND_DIM), dtype=np.float32)
    _write_replay_buffer(tmp_path, obs, action, episode_ends)
    del zarr.open_group(str(tmp_path / "replay_buffer.zarr"), mode="a")[
        "data/action"
    ]

    with pytest.raises(KeyError, match="action"):
        _make_dataset(tmp_path)


def test_dataset_rejects_non_float32_fields(tmp_path):
    episode_ends = np.array([12], dtype=np.int64)
    obs = np.zeros((12, OBS_DIM), dtype=np.float64)
    action = np.zeros((12, HAND_DIM), dtype=np.float32)
    _write_replay_buffer(tmp_path, obs, action, episode_ends)

    with pytest.raises(TypeError, match="obs.*float32"):
        _make_dataset(tmp_path)


def test_dataset_rejects_non_int64_episode_ends(tmp_path):
    episode_ends = np.array([12], dtype=np.int32)
    obs = np.zeros((12, OBS_DIM), dtype=np.float32)
    action = np.zeros((12, HAND_DIM), dtype=np.float32)
    _write_replay_buffer(
        tmp_path,
        obs,
        action,
        episode_ends,
        episode_ends_dtype="int32",
    )

    with pytest.raises(TypeError, match="episode_ends.*int64"):
        _make_dataset(tmp_path)


@pytest.mark.parametrize(
    "obs_shape, action_shape, episode_ends, message",
    [
        ((12, 22), (12, 22), [12], r"obs.*\(N, 66\)"),
        ((12, 66), (11, 22), [11], "same number of steps"),
        ((12, 66), (12, 22), [6, 6, 12], "strictly increasing"),
        ((12, 66), (12, 22), [11], "final episode end.*12"),
    ],
)
def test_dataset_rejects_invalid_shapes_and_episode_ends(
    tmp_path,
    obs_shape,
    action_shape,
    episode_ends,
    message,
):
    obs = np.zeros(obs_shape, dtype=np.float32)
    action = np.zeros(action_shape, dtype=np.float32)
    _write_replay_buffer(
        tmp_path,
        obs,
        action,
        np.asarray(episode_ends, dtype=np.int64),
    )

    with pytest.raises((ValueError, TypeError), match=message):
        _make_dataset(tmp_path)
