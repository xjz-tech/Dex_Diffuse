from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.sim_hand_hdf5 import (
    read_sim_hand_manifest,
    validate_sim_hand_shards,
)
from diffusion_policy.dataset.sim_hand_mmap_converter import scan_episode_inventory

HAND_DIM = 22


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


def _write_rollout(dataset_dir, shards):
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


def test_inventory_counts_interleaved_pairs_filters_and_splits(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[
            [_row(20, 1, 0), _row(10, 4, 0), _row(30, 0, 0)],
            [
                _row(10, 4, 1), _row(20, 1, 1), _row(10, 4, 2),
                _row(20, 1, 2), _row(20, 1, 3),
            ],
        ],
    )
    manifest = read_sim_hand_manifest(tmp_path)
    validate_sim_hand_shards(manifest)

    inventory = scan_episode_inventory(
        manifest,
        horizon=3,
        val_ratio=0.5,
        seed=42,
        chunk_rows=2,
    )

    np.testing.assert_array_equal(
        inventory.keys["episode_id"], np.array([10, 20], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        inventory.keys["env_id"], np.array([4, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(inventory.lengths, [3, 4])
    np.testing.assert_array_equal(inventory.offsets, [0, 3])
    np.testing.assert_array_equal(inventory.episode_ends, [3, 7])
    assert inventory.source_episode_count == 3
    assert inventory.source_transition_count == 8
    assert inventory.dropped_episode_count == 1
    assert inventory.dropped_transition_count == 1
    np.testing.assert_array_equal(
        inventory.val_mask,
        get_val_mask(2, val_ratio=0.5, seed=42),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 3, "val_ratio": 0.5, "seed": 42, "chunk_rows": True},
        {"horizon": 3, "val_ratio": 0.5, "seed": 42, "chunk_rows": False},
        {"horizon": 3, "val_ratio": 0.5, "seed": 42, "chunk_rows": 0},
        {"horizon": 3, "val_ratio": 0.5, "seed": 42, "chunk_rows": -1},
        {"horizon": 0, "val_ratio": 0.5, "seed": 42},
        {"horizon": -1, "val_ratio": 0.5, "seed": 42},
        {"horizon": 3, "val_ratio": -0.1, "seed": 42},
        {"horizon": 3, "val_ratio": 1.0, "seed": 42},
        {"horizon": 3, "val_ratio": 1.5, "seed": 42},
    ],
)
def test_scan_episode_inventory_rejects_invalid_args(tmp_path, kwargs):
    _write_rollout(
        tmp_path,
        shards=[[_row(10, 0, 0), _row(10, 0, 1), _row(10, 0, 2)]],
    )
    manifest = read_sim_hand_manifest(tmp_path)
    validate_sim_hand_shards(manifest)

    with pytest.raises(ValueError):
        scan_episode_inventory(manifest, **kwargs)
