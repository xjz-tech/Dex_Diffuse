from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from diffusion_policy.dataset.sim_hand_lowdim_dataset import SimHandLowdimDataset


HAND_DIM = 22
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _make_dataset(dataset_dir, **kwargs):
    return SimHandLowdimDataset(
        dataset_path=str(dataset_dir),
        horizon=3,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.5,
        **kwargs,
    )


def test_hdf5_shards_reconstruct_interleaved_episodes_in_step_order(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[
            [
                _row(10, 0, 0),
                _row(20, 1, 0),
                _row(10, 0, 1),
                _row(20, 1, 1),
            ],
            [
                _row(20, 1, 2),
                _row(10, 0, 2),
                _row(20, 1, 3, done=True, success=True),
                _row(10, 0, 3, done=True, success=True),
            ],
        ],
    )

    dataset = _make_dataset(tmp_path)

    np.testing.assert_array_equal(
        dataset.replay_buffer.episode_ends,
        np.asarray([4, 8], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer.meta["episode_ids"],
        np.asarray([10, 20], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer.meta["env_ids"],
        np.asarray([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer["hand_joint"][:, 0],
        np.asarray(
            [1000, 1001, 1002, 1003, 2000, 2001, 2002, 2003],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer["action"][:, 0],
        np.asarray(
            [11000, 11001, 11002, 11003, 12000, 12001, 12002, 12003],
            dtype=np.float32,
        ),
    )


def test_hdf5_keeps_failure_and_timeout_episodes(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, failure=True),
            _row(20, 1, 0),
            _row(20, 1, 1),
            _row(20, 1, 2, done=True, timeout=True),
        ]],
    )

    dataset = _make_dataset(tmp_path)  # horizon=3

    assert dataset.replay_buffer.n_episodes == 2
    np.testing.assert_array_equal(
        dataset.replay_buffer.meta["episode_ids"],
        np.asarray([10, 20], dtype=np.int64),
    )


def test_hdf5_keeps_episode_without_terminal_done(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2),
        ]],
    )

    dataset = _make_dataset(tmp_path)
    assert dataset.replay_buffer.n_episodes == 1


def test_hdf5_drops_episodes_shorter_than_horizon(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            # length 2 < horizon 3 → drop
            _row(10, 0, 0),
            _row(10, 0, 1, done=True, success=True),
            # length 3 >= horizon 3 → keep
            _row(20, 1, 0),
            _row(20, 1, 1),
            _row(20, 1, 2, done=True, failure=True),
        ]],
    )

    dataset = _make_dataset(tmp_path)

    assert dataset.replay_buffer.n_episodes == 1
    np.testing.assert_array_equal(
        dataset.replay_buffer.episode_ends,
        np.asarray([3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer.meta["episode_ids"],
        np.asarray([20], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer["hand_joint"][:, 0],
        np.asarray([2000, 2001, 2002], dtype=np.float32),
    )


def test_hdf5_rejects_when_all_episodes_are_too_short(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1, done=True, success=True),
        ]],
    )

    with pytest.raises(ValueError, match="no episodes|min_episode_length|too short"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_invalid_min_episode_length(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2),
        ]],
    )
    from diffusion_policy.dataset.sim_hand_hdf5 import load_sim_hand_hdf5

    with pytest.raises(ValueError, match="min_episode_length"):
        load_sim_hand_hdf5(tmp_path, min_episode_length=0)


def test_hdf5_rejects_non_contiguous_episode_steps(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )

    with pytest.raises(ValueError, match="steps must be contiguous"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_non_float32_training_fields(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    shard_path = tmp_path / "shards" / "shard_000000.h5"
    with h5py.File(shard_path, "a") as f:
        qpos = f["robot/qpos"][:].astype(np.float64)
        del f["robot/qpos"]
        f["robot"].create_dataset("qpos", data=qpos)

    with pytest.raises(TypeError, match="robot/qpos.*float32"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_wrong_training_field_shape(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    shard_path = tmp_path / "shards" / "shard_000000.h5"
    with h5py.File(shard_path, "a") as f:
        qpos = f["robot/qpos"][:, :21]
        del f["robot/qpos"]
        f["robot"].create_dataset("qpos", data=qpos)

    with pytest.raises(ValueError, match=r"robot/qpos.*\(N, 22\)"):
        _make_dataset(tmp_path)


def test_train_script_accepts_hdf5_manifest_dataset(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": "/bin/true",
            "DATASET_PATH": str(tmp_path),
            "WANDB_DIR": str(tmp_path / "wandb"),
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        }
    )

    result = subprocess.run(
        ["bash", str(REPO_ROOT / "dp_train_sim_hand.sh")],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_hdf5_rejects_wrong_index_dtype(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    shard_path = tmp_path / "shards" / "shard_000000.h5"
    with h5py.File(shard_path, "a") as f:
        episode_id = f["index/episode_id"][:].astype(np.int32)
        del f["index/episode_id"]
        f["index"].create_dataset("episode_id", data=episode_id)

    with pytest.raises(TypeError, match="index/episode_id.*int64"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_non_vector_index_field(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    shard_path = tmp_path / "shards" / "shard_000000.h5"
    with h5py.File(shard_path, "a") as f:
        done = f["index/done"][:].reshape(-1, 1)
        del f["index/done"]
        f["index"].create_dataset("done", data=done)

    with pytest.raises(ValueError, match=r"index/done.*\(N,\)"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_manifest_total_transition_mismatch(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["total_transitions"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="total_transitions.*4.*3"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_empty_rollout(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shard_path = shard_dir / "shard_000000.h5"
    with h5py.File(shard_path, "w") as f:
        index = f.create_group("index")
        robot = f.create_group("robot")
        robot.create_dataset("qpos", shape=(0, HAND_DIM), dtype=np.float32)
        robot.create_dataset("target_after", shape=(0, HAND_DIM), dtype=np.float32)
        for key, dtype in (
            ("episode_id", np.int64),
            ("env_id", np.int32),
            ("step", np.int64),
            ("done", np.bool_),
            ("success", np.bool_),
            ("failure", np.bool_),
            ("timeout", np.bool_),
            ("reset_reason", np.int8),
        ):
            index.create_dataset(key, shape=(0,), dtype=dtype)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_transitions": 0,
                "dof": HAND_DIM,
                "shards": [
                    {
                        "path": "shards/shard_000000.h5",
                        "num_transitions": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains no transitions"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_unknown_manifest_schema_version(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version=1.*2"):
        _make_dataset(tmp_path)


@pytest.mark.parametrize(
    ("missing_key", "in_shard"),
    [
        ("dof", False),
        ("total_transitions", False),
        ("path", True),
        ("num_transitions", True),
    ],
)
def test_hdf5_rejects_manifest_missing_required_integrity_field(
    tmp_path,
    missing_key,
    in_shard,
):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    container = manifest["shards"][0] if in_shard else manifest
    del container[missing_key]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"required.*{missing_key}"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_non_object_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest.*JSON object"):
        _make_dataset(tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_message"),
    [
        ("schema_version", "1", "schema_version.*integer"),
        ("dof", 22.5, "dof.*integer"),
        ("total_transitions", "3", "total_transitions.*integer"),
        ("total_transitions", -1, "total_transitions.*non-negative"),
    ],
)
def test_hdf5_rejects_invalid_manifest_field_type_or_range(
    tmp_path,
    field,
    invalid_value,
    expected_message,
):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        _make_dataset(tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid_value", "expected_message"),
    [
        ("path", 123, "path.*string"),
        ("num_transitions", "3", "num_transitions.*integer"),
        ("num_transitions", -1, "num_transitions.*non-negative"),
    ],
)
def test_hdf5_rejects_invalid_shard_manifest_field_type_or_range(
    tmp_path,
    field,
    invalid_value,
    expected_message,
):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0][field] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        _make_dataset(tmp_path)


def test_hdf5_rejects_non_object_shard_entry(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, success=True),
        ]],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="shard 0.*JSON object"):
        _make_dataset(tmp_path)
