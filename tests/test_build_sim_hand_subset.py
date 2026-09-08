from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import zarr

from scripts.build_sim_hand_subset import build_subset, select_episode_keys


HAND_DIM = 22


def _write_source(root: Path) -> None:
    shards = root / "shards"
    shards.mkdir(parents=True)
    rows = {
        (10, 1): range(4),
        (20, 2): range(5),
        (30, 3): range(6),
    }
    flat = []
    for key, steps in rows.items():
        for step in steps:
            flat.append((key[0], key[1], step))
    # Split episodes across shards and scramble their physical row order.
    shard_rows = [flat[::2], flat[1::2]]
    manifest_shards = []
    for shard_index, records in enumerate(shard_rows):
        path = shards / f"shard_{shard_index:06d}.h5"
        with h5py.File(path, "w") as handle:
            index = handle.create_group("index")
            robot = handle.create_group("robot")
            index.create_dataset(
                "episode_id", data=np.asarray([r[0] for r in records], dtype=np.int64)
            )
            index.create_dataset(
                "env_id", data=np.asarray([r[1] for r in records], dtype=np.int32)
            )
            index.create_dataset(
                "step", data=np.asarray([r[2] for r in records], dtype=np.int64)
            )
            qpos = np.stack(
                [np.full(HAND_DIM, r[0] + r[2], dtype=np.float32) for r in records]
            )
            target_before = qpos + 100.0
            target_after = qpos + 200.0
            robot.create_dataset("qpos", data=qpos)
            robot.create_dataset("target_before", data=target_before)
            robot.create_dataset("target_after", data=target_after)
        manifest_shards.append(
            {"path": str(path.relative_to(root)), "num_transitions": len(records)}
        )

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dof": HAND_DIM,
                "total_transitions": len(flat),
                "shards": manifest_shards,
            }
        ),
        encoding="utf-8",
    )
    mmap_dir = root / "exp_data_mmap"
    mmap_dir.mkdir()
    np.save(mmap_dir / "episode_ids.npy", np.asarray([10, 20, 30], dtype=np.int64))
    np.save(mmap_dir / "env_ids.npy", np.asarray([1, 2, 3], dtype=np.int32))
    np.save(mmap_dir / "episode_ends.npy", np.asarray([4, 9, 15], dtype=np.int64))


def test_select_episode_keys_never_splits_an_episode_and_is_deterministic():
    episode_ids = np.asarray([10, 20, 30], dtype=np.int64)
    env_ids = np.asarray([1, 2, 3], dtype=np.int32)
    episode_ends = np.asarray([4, 9, 15], dtype=np.int64)

    first = select_episode_keys(
        episode_ids, env_ids, episode_ends, max_transitions=10, seed=7
    )
    second = select_episode_keys(
        episode_ids, env_ids, episode_ends, max_transitions=10, seed=7
    )

    assert first == second
    assert sum(item.length for item in first) <= 10
    assert {item.length for item in first}.issubset({4, 5, 6})


def test_build_subset_reconstructs_66d_observations_in_step_order(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)

    summary = build_subset(source, output, max_transitions=10, seed=7)

    root = zarr.open_group(str(output / "replay_buffer.zarr"), mode="r")
    obs = np.asarray(root["data/obs"][:])
    action = np.asarray(root["data/action"][:])
    episode_ends = np.asarray(root["meta/episode_ends"][:])
    assert obs.shape == (summary.actual_transitions, 66)
    assert action.shape == (summary.actual_transitions, 22)
    assert int(episode_ends[-1]) == summary.actual_transitions
    np.testing.assert_allclose(obs[:, 22:44] - obs[:, :22], 100.0)
    np.testing.assert_allclose(obs[:, 44:66], 100.0)
    np.testing.assert_allclose(action - obs[:, :22], 200.0)

    starts = np.r_[0, episode_ends[:-1]]
    for start, end in zip(starts, episode_ends):
        first_joint = obs[start:end, 0]
        np.testing.assert_array_equal(np.diff(first_joint), np.ones(end - start - 1))

    metadata = json.loads((output / "subset_manifest.json").read_text())
    assert metadata["requested_max_transitions"] == 10
    assert metadata["actual_transitions"] == summary.actual_transitions
