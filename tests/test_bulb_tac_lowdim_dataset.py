from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from diffusion_policy.dataset.bulb_tac_lowdim_dataset import (
    BulbTacLowdimDataset,
)

HAND_DIM = 22
OBS_DIM = 66


def _write_episode(root: Path, episode_id: int, length: int, base: float):
    episode = root / f"episode_{episode_id}"
    episode.mkdir(parents=True)
    state = np.zeros((length, 31), dtype=np.float32)
    action = np.zeros((length, 31), dtype=np.float32)
    steps = np.arange(length, dtype=np.float32)[:, None]
    channels = np.arange(HAND_DIM, dtype=np.float32)[None, :]
    state[:, 9:31] = base + steps + channels / 100.0
    action[:, 9:31] = base + 100.0 + steps + channels / 100.0
    np.save(episode / "state.npy", state)
    np.save(episode / "action.npy", action)
    return state, action


def _make_dataset(root: Path, val_ratio: float = 0.0):
    return BulbTacLowdimDataset(
        dataset_path=str(root),
        horizon=3,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=val_ratio,
    )


def test_bulb_mapping_matches_expdata_temporal_contract(tmp_path):
    state, action = _write_episode(tmp_path, 0, 4, 10.0)
    dataset = _make_dataset(tmp_path)

    qpos = state[:, 9:31]
    target_after = action[:, 9:31]
    target_before = np.concatenate((qpos[0:1], target_after[:-1]), axis=0)
    expected_obs = np.concatenate(
        (qpos, target_before, target_before - qpos), axis=-1
    )

    np.testing.assert_array_equal(dataset.replay_buffer.episode_ends, [4])
    np.testing.assert_allclose(dataset.replay_buffer["obs"], expected_obs)
    np.testing.assert_allclose(dataset.replay_buffer["action"], target_after)
    assert dataset.replay_buffer["obs"].shape == (4, OBS_DIM)
    assert dataset.replay_buffer["action"].shape == (4, HAND_DIM)
    assert set(dataset[0]) == {"obs", "action"}
    assert dataset[0]["obs"].dtype == torch.float32


def test_numeric_episode_order_and_source_cardinality_are_preserved(tmp_path):
    _write_episode(tmp_path, 10, 3, 1000.0)
    _write_episode(tmp_path, 2, 4, 200.0)
    dataset = _make_dataset(tmp_path)

    np.testing.assert_array_equal(dataset.replay_buffer.episode_ends, [4, 7])
    assert dataset.replay_buffer.n_steps == 7
    assert dataset.replay_buffer["obs"][0, 0] == 200.0
    assert dataset.replay_buffer["obs"][4, 0] == 1000.0


def test_validation_uses_disjoint_episode_mask_without_downsampling(tmp_path):
    for episode_id in range(10):
        _write_episode(tmp_path, episode_id, 4, float(episode_id * 10))
    dataset = _make_dataset(tmp_path, val_ratio=0.2)
    validation = dataset.get_validation_dataset()
    assert dataset.replay_buffer.n_steps == 40
    assert int(dataset.train_mask.sum()) == 8
    assert int(dataset.val_mask.sum()) == 2
    assert not np.any(dataset.train_mask & dataset.val_mask)
    assert len(validation) > 0


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("missing_action", FileNotFoundError, "action.npy"),
        ("float64", TypeError, "float32"),
        ("wrong_width", ValueError, r"\(T, 31\)"),
        ("length_mismatch", ValueError, "lengths do not match"),
        ("non_finite", ValueError, "NaN or infinity"),
        ("too_short", ValueError, "shorter than horizon"),
    ],
)
def test_bulb_dataset_rejects_invalid_episodes(tmp_path, mutation, error, message):
    episode = tmp_path / "episode_0"
    episode.mkdir()
    state = np.zeros((3, 31), dtype=np.float32)
    action = np.zeros((3, 31), dtype=np.float32)
    if mutation == "float64":
        state = state.astype(np.float64)
    elif mutation == "wrong_width":
        state = state[:, :30]
    elif mutation == "length_mismatch":
        action = action[:2]
    elif mutation == "non_finite":
        state[0, 0] = np.nan
    elif mutation == "too_short":
        state, action = state[:2], action[:2]
    np.save(episode / "state.npy", state)
    if mutation != "missing_action":
        np.save(episode / "action.npy", action)
    with pytest.raises(error, match=message):
        _make_dataset(tmp_path)
