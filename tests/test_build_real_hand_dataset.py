"""Tests for the real SharpA -> Sim-Hand replay-buffer converter."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from scripts.build_real_hand_dataset import (
    build_real_hand_dataset,
    convert_episode_arrays,
    list_real_episodes,
)


def _write_episode(root, rank, ep_id, states, actions):
    ep_dir = root / f"rank_{rank}" / f"id_{ep_id}" / "data" / "chunk-000"
    ep_dir.mkdir(parents=True, exist_ok=True)
    n = states.shape[0]
    table = pyarrow.table(
        {
            "state": [list(map(float, row)) for row in states],
            "actions": [list(map(float, row)) for row in actions],
            "frame_index": list(range(n)),
        }
    )
    pq.write_table(table, ep_dir / "episode_000000.parquet")


def _grasp_episode(length, seed=0):
    """Synthetic episode in radians that sweeps the grasp range.

    ``actions`` deliberately differ from shifted states (commanded targets lead
    the measured pose by a margin), so tests can distinguish target-before from
    target-after semantics.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)[:, None]  # 0=open -> 1=closed
    profile = np.linspace(0.0, 1.4, 22)  # radians per joint at full close
    hand = t * profile + rng.normal(0, 0.005, (length, 22))
    arm = np.zeros((length, 9))
    states = np.concatenate([arm, hand], axis=1)
    # Commanded target leads the measured pose: action[t] drives t -> t+1.
    actions = np.concatenate([arm, hand + 0.05], axis=1)
    return states.astype(np.float32), actions.astype(np.float32)


def test_list_real_episodes(tmp_path):
    states, actions = _grasp_episode(20)
    _write_episode(tmp_path, 0, 0, states, actions)
    _write_episode(tmp_path, 0, 1, states, actions)
    _write_episode(tmp_path, 1, 0, states, actions)
    eps = list_real_episodes(tmp_path)
    assert len(eps) == 3


def test_convert_episode_radians_and_order(tmp_path):
    states, actions = _grasp_episode(30)
    obs, act = convert_episode_arrays(states, actions)
    assert obs.shape == (30, 66)
    assert act.shape == (30, 22)
    qpos, target_before, residual = obs[:, :22], obs[:, 22:44], obs[:, 44:]
    hand_state = states[:, 9:]
    hand_cmd = actions[:, 9:]
    np.testing.assert_allclose(qpos, hand_state, atol=1e-6)
    # target_before[t] = command[t-1]; first frame uses qpos[0] (sim convention)
    np.testing.assert_allclose(target_before[0], hand_state[0], atol=1e-6)
    np.testing.assert_allclose(target_before[1:], hand_cmd[:-1], atol=1e-6)
    np.testing.assert_allclose(residual, target_before - qpos, atol=1e-6)
    # training action = commanded target-after = raw actions[t]
    np.testing.assert_allclose(act, hand_cmd, atol=1e-6)
    # radians sweeping a real grasp range
    assert qpos.min() >= -0.05 and qpos.max() <= 1.6
    assert qpos.max() > 1.0  # actually closes


def test_build_real_hand_dataset_writes_zarr(tmp_path):
    src = tmp_path / "real"
    states, actions = _grasp_episode(25)
    _write_episode(src, 0, 0, states, actions)
    _write_episode(src, 0, 1, states, actions)
    out = tmp_path / "out"
    summary = build_real_hand_dataset(src, out, overwrite=False)
    zarr_dir = out / "replay_buffer.zarr"
    assert zarr_dir.is_dir()
    assert summary["episodes"] == 2
    assert summary["frames"] == 50
    import zarr
    root = zarr.open_group(str(zarr_dir), mode="r")
    assert root["data/obs"].shape == (50, 66)
    assert root["data/action"].shape == (50, 22)
    np.testing.assert_array_equal(
        root["meta/episode_ends"][:], np.array([25, 50], dtype=np.int64)
    )
    # manifest records the mapping for reproducibility
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["unit"] == "radians"
    assert manifest["joint_order"] == "policy_order_passthrough"
