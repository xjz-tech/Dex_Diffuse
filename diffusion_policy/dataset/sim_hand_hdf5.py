from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


HAND_DIM = 22
OBS_DIM = 66
SCHEMA_VERSION = 1
HDF5_FIELDS = (
    "robot/qpos",
    "robot/target_before",
    "robot/target_after",
    "index/episode_id",
    "index/env_id",
    "index/step",
    "index/done",
    "index/success",
    "index/failure",
    "index/timeout",
    "index/reset_reason",
)
HDF5_FIELD_DTYPES = {
    "robot/qpos": np.float32,
    "robot/target_before": np.float32,
    "robot/target_after": np.float32,
    "index/episode_id": np.int64,
    "index/env_id": np.int32,
    "index/step": np.int64,
    "index/done": np.bool_,
    "index/success": np.bool_,
    "index/failure": np.bool_,
    "index/timeout": np.bool_,
    "index/reset_reason": np.int8,
}
_ROBOT_TRAINING_FIELDS = (
    "robot/qpos",
    "robot/target_before",
    "robot/target_after",
)


def load_sim_hand_hdf5(
    dataset_path: str | Path,
    *,
    min_episode_length: int,
) -> ReplayBuffer:
    """Load HDF5 rollout shards and make every episode contiguous in memory."""
    if type(min_episode_length) is not int or min_episode_length < 1:
        raise ValueError(
            "min_episode_length must be an integer >= 1, "
            f"got {min_episode_length!r}"
        )
    root_path = Path(dataset_path).expanduser()
    manifest_path = root_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sim-Hand HDF5 manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Sim-Hand HDF5 manifest must be a JSON object")
    required_manifest_fields = (
        "schema_version",
        "dof",
        "total_transitions",
        "shards",
    )
    missing_manifest_fields = [
        key for key in required_manifest_fields if key not in manifest
    ]
    if missing_manifest_fields:
        raise ValueError(
            "Sim-Hand HDF5 manifest is missing required fields: "
            f"{missing_manifest_fields}"
        )
    schema_version = manifest["schema_version"]
    if type(schema_version) is not int:
        raise ValueError(
            "Sim-Hand HDF5 manifest schema_version must be an integer"
        )
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Sim-Hand HDF5 manifest requires schema_version={SCHEMA_VERSION}, "
            f"got {schema_version}"
        )
    dof = manifest["dof"]
    if type(dof) is not int:
        raise ValueError("Sim-Hand HDF5 manifest dof must be an integer")
    if dof != HAND_DIM:
        raise ValueError(
            f"Sim-Hand HDF5 manifest must declare dof={HAND_DIM}, "
            f"got {dof}"
        )
    declared_total = manifest["total_transitions"]
    if type(declared_total) is not int:
        raise ValueError(
            "Sim-Hand HDF5 manifest total_transitions must be an integer"
        )
    if declared_total < 0:
        raise ValueError(
            "Sim-Hand HDF5 manifest total_transitions must be non-negative"
        )
    shards = manifest["shards"]
    if not isinstance(shards, list) or not shards:
        raise ValueError("Sim-Hand HDF5 manifest must contain at least one shard")

    shard_specs: list[tuple[Path, int]] = []
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValueError(
                f"Sim-Hand HDF5 manifest shard {shard_index} must be a JSON object"
            )
        missing_shard_fields = [
            key for key in ("path", "num_transitions") if key not in shard
        ]
        if missing_shard_fields:
            raise ValueError(
                "Sim-Hand HDF5 manifest shard "
                f"{shard_index} is missing required fields: "
                f"{missing_shard_fields}"
            )
        shard_relative_path = shard["path"]
        if not isinstance(shard_relative_path, str) or not shard_relative_path:
            raise ValueError(
                "Sim-Hand HDF5 manifest shard "
                f"{shard_index} path must be a non-empty string"
            )
        expected_rows = shard["num_transitions"]
        if type(expected_rows) is not int:
            raise ValueError(
                "Sim-Hand HDF5 manifest shard "
                f"{shard_index} num_transitions must be an integer"
            )
        if expected_rows < 0:
            raise ValueError(
                "Sim-Hand HDF5 manifest shard "
                f"{shard_index} num_transitions must be non-negative"
            )
        shard_path = root_path / shard_relative_path
        if not shard_path.is_file():
            raise FileNotFoundError(f"Sim-Hand HDF5 shard not found: {shard_path}")
        shard_specs.append((shard_path, expected_rows))

    shard_total = sum(expected_rows for _, expected_rows in shard_specs)
    if declared_total != shard_total:
        raise ValueError(
            "Sim-Hand HDF5 manifest total_transitions mismatch: "
            f"declared {declared_total}, shard metadata {shard_total}"
        )

    for shard_path, expected_rows in shard_specs:
        with h5py.File(shard_path, "r") as f:
            missing = [key for key in HDF5_FIELDS if key not in f]
            if missing:
                raise KeyError(
                    f"Missing Sim-Hand HDF5 fields in {shard_path}: {missing}"
                )
            for key in HDF5_FIELDS:
                dataset = f[key]
                expected_dtype = np.dtype(HDF5_FIELD_DTYPES[key])
                if np.dtype(dataset.dtype) != expected_dtype:
                    raise TypeError(
                        f"HDF5 field {key} must use {expected_dtype}, "
                        f"got {dataset.dtype}"
                    )
                if key in _ROBOT_TRAINING_FIELDS and (
                    dataset.ndim != 2 or dataset.shape[1] != HAND_DIM
                ):
                    raise ValueError(
                        f"HDF5 field {key} must have shape (N, {HAND_DIM}), "
                        f"got {dataset.shape}"
                    )
                if key.startswith("index/") and dataset.ndim != 1:
                    raise ValueError(
                        f"HDF5 field {key} must have shape (N,), "
                        f"got {dataset.shape}"
                    )
                dataset_rows = int(dataset.shape[0])
                if dataset_rows != expected_rows:
                    raise ValueError(
                        f"HDF5 field {key} in {shard_path} has "
                        f"{dataset_rows} rows; expected {expected_rows}"
                    )

    if declared_total == 0:
        raise ValueError("Sim-Hand HDF5 dataset contains no transitions")

    arrays = {}
    for key in HDF5_FIELDS:
        shape = (
            (declared_total, HAND_DIM)
            if key in _ROBOT_TRAINING_FIELDS
            else (declared_total,)
        )
        arrays[key] = np.empty(shape, dtype=HDF5_FIELD_DTYPES[key])

    offset = 0
    for shard_path, shard_rows in shard_specs:
        if shard_rows == 0:
            continue
        end = offset + shard_rows
        with h5py.File(shard_path, "r") as f:
            for key in HDF5_FIELDS:
                destination = arrays[key]
                destination_selection = (
                    np.s_[offset:end, :]
                    if destination.ndim == 2
                    else np.s_[offset:end]
                )
                f[key].read_direct(
                    destination,
                    dest_sel=destination_selection,
                )
        offset = end

    order = np.lexsort(
        (
            arrays["index/step"],
            arrays["index/env_id"],
            arrays["index/episode_id"],
        )
    )
    qpos = arrays.pop("robot/qpos")
    target_before = arrays.pop("robot/target_before")
    target_after = arrays.pop("robot/target_after")
    episode_id = arrays["index/episode_id"][order]
    env_id = arrays["index/env_id"][order]

    episode_starts = np.flatnonzero(
        np.r_[
            True,
            (episode_id[1:] != episode_id[:-1]) | (env_id[1:] != env_id[:-1]),
        ]
    )
    episode_ends = np.r_[episode_starts[1:], len(order)].astype(np.int64)
    ordered_step = arrays["index/step"][order]
    del arrays
    for start, end in zip(episode_starts, episode_ends):
        episode_steps = ordered_step[start:end]
        expected_steps = np.arange(end - start, dtype=episode_steps.dtype)
        if not np.array_equal(episode_steps, expected_steps):
            raise ValueError(
                "HDF5 episode "
                f"(episode_id={int(episode_id[start])}, "
                f"env_id={int(env_id[start])}) steps must be contiguous "
                "and start at zero"
            )

    keep_episodes = (episode_ends - episode_starts) >= min_episode_length
    if not np.any(keep_episodes):
        raise ValueError(
            "Sim-Hand HDF5 dataset contains no episodes with length >= "
            f"min_episode_length={min_episode_length}"
        )

    kept_episode_starts = episode_starts[keep_episodes]
    kept_episode_ends = episode_ends[keep_episodes]
    kept_lengths = kept_episode_ends - kept_episode_starts
    kept_rows = np.empty(int(kept_lengths.sum()), dtype=np.int64)
    kept_offset = 0
    for start, end in zip(kept_episode_starts, kept_episode_ends):
        episode_length = int(end - start)
        kept_rows[kept_offset : kept_offset + episode_length] = np.arange(
            start,
            end,
            dtype=np.int64,
        )
        kept_offset += episode_length
    replay_episode_ends = np.cumsum(kept_lengths, dtype=np.int64)
    source_rows = order[kept_rows]
    qpos = qpos[source_rows]
    target_before = target_before[source_rows]
    action = target_after[source_rows]
    del target_after
    obs = np.concatenate(
        (qpos, target_before, target_before - qpos),
        axis=-1,
    ).astype(np.float32, copy=False)
    del qpos
    del target_before
    assert obs.shape[1] == OBS_DIM

    replay_root = {
        "data": {
            "obs": obs,
            "action": action,
        },
        "meta": {
            "episode_ends": replay_episode_ends,
            "episode_ids": episode_id[kept_episode_starts].astype(
                np.int64,
                copy=False,
            ),
            "env_ids": env_id[kept_episode_starts].astype(
                np.int32,
                copy=False,
            ),
        },
    }
    return ReplayBuffer(root=replay_root)
