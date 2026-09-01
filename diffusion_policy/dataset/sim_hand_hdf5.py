from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SimHandLeafArrays:
    qpos: np.ndarray
    target_before: np.ndarray
    action: np.ndarray
    episode_ends: np.ndarray
    episode_ids: np.ndarray
    env_ids: np.ndarray


def compose_sim_hand_obs(qpos: np.ndarray, target_before: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (qpos, target_before, target_before - qpos),
        axis=-1,
    ).astype(np.float32, copy=False)


def _annotate_error(path: Path, exc: Exception) -> Exception:
    path_text = str(path)
    message = str(exc)
    if path_text in message:
        return exc
    return type(exc)(f"{path_text}: {message}")


def read_sim_hand_manifest_shards(root_path: Path) -> tuple[int, list[tuple[Path, int]]]:
    manifest_path = root_path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{manifest_path}: Sim-Hand HDF5 manifest is not valid JSON"
        ) from exc
    try:
        return _parse_manifest_shards(root_path, manifest_path, manifest)
    except (TypeError, ValueError) as exc:
        raise _annotate_error(manifest_path, exc) from exc


def _parse_manifest_shards(
    root_path: Path,
    manifest_path: Path,
    manifest: object,
) -> tuple[int, list[tuple[Path, int]]]:
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
    return declared_total, shard_specs


def load_sim_hand_leaf_arrays(dataset_path: str | Path) -> SimHandLeafArrays:
    """Load one leaf's ordered source arrays, keeping every reconstructed episode."""
    root_path = Path(dataset_path).expanduser()
    manifest_path = root_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Sim-Hand HDF5 manifest not found: {manifest_path}"
        )
    declared_total, shard_specs = read_sim_hand_manifest_shards(root_path)

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
                        f"{shard_path}: HDF5 field {key} must use "
                        f"{expected_dtype}, got {dataset.dtype}"
                    )
                if key in _ROBOT_TRAINING_FIELDS and (
                    dataset.ndim != 2 or dataset.shape[1] != HAND_DIM
                ):
                    raise ValueError(
                        f"{shard_path}: HDF5 field {key} must have shape "
                        f"(N, {HAND_DIM}), got {dataset.shape}"
                    )
                if key.startswith("index/") and dataset.ndim != 1:
                    raise ValueError(
                        f"{shard_path}: HDF5 field {key} must have shape (N,), "
                        f"got {dataset.shape}"
                    )
                dataset_rows = int(dataset.shape[0])
                if dataset_rows != expected_rows:
                    raise ValueError(
                        f"HDF5 field {key} in {shard_path} has "
                        f"{dataset_rows} rows; expected {expected_rows}"
                    )

    if declared_total == 0:
        raise ValueError(
            f"{root_path}: Sim-Hand HDF5 dataset contains no transitions"
        )

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
    qpos = arrays.pop("robot/qpos")[order]
    target_before = arrays.pop("robot/target_before")[order]
    action = arrays.pop("robot/target_after")[order]
    episode_id = arrays["index/episode_id"][order]
    env_id = arrays["index/env_id"][order]
    ordered_step = arrays["index/step"][order]
    del arrays

    episode_starts = np.flatnonzero(
        np.r_[
            True,
            (episode_id[1:] != episode_id[:-1]) | (env_id[1:] != env_id[:-1]),
        ]
    )
    episode_ends = np.r_[episode_starts[1:], len(order)].astype(np.int64)
    for start, end in zip(episode_starts, episode_ends):
        episode_steps = ordered_step[start:end]
        expected_steps = np.arange(end - start, dtype=episode_steps.dtype)
        if not np.array_equal(episode_steps, expected_steps):
            raise ValueError(
                f"{root_path}: HDF5 episode "
                f"(episode_id={int(episode_id[start])}, "
                f"env_id={int(env_id[start])}) steps must be contiguous "
                "and start at zero"
            )

    return SimHandLeafArrays(
        qpos=qpos.astype(np.float32, copy=False),
        target_before=target_before.astype(np.float32, copy=False),
        action=action.astype(np.float32, copy=False),
        episode_ends=episode_ends,
        episode_ids=episode_id[episode_starts].astype(np.int64, copy=False),
        env_ids=env_id[episode_starts].astype(np.int32, copy=False),
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
    leaf = load_sim_hand_leaf_arrays(dataset_path)
    episode_starts = np.r_[0, leaf.episode_ends[:-1]]
    lengths = leaf.episode_ends - episode_starts
    keep_episodes = lengths >= min_episode_length
    if not np.any(keep_episodes):
        raise ValueError(
            "Sim-Hand HDF5 dataset contains no episodes with length >= "
            f"min_episode_length={min_episode_length}"
        )

    kept_starts = episode_starts[keep_episodes]
    kept_ends = leaf.episode_ends[keep_episodes]
    kept_lengths = kept_ends - kept_starts
    kept_rows = np.empty(int(kept_lengths.sum()), dtype=np.int64)
    kept_offset = 0
    for start, end in zip(kept_starts, kept_ends):
        episode_length = int(end - start)
        kept_rows[kept_offset : kept_offset + episode_length] = np.arange(
            start,
            end,
            dtype=np.int64,
        )
        kept_offset += episode_length
    qpos = leaf.qpos[kept_rows]
    target_before = leaf.target_before[kept_rows]
    action = leaf.action[kept_rows]
    obs = compose_sim_hand_obs(qpos, target_before)
    assert obs.shape[1] == OBS_DIM
    replay_root = {
        "data": {
            "obs": obs,
            "action": action,
        },
        "meta": {
            "episode_ends": np.cumsum(kept_lengths, dtype=np.int64),
            "episode_ids": leaf.episode_ids[keep_episodes],
            "env_ids": leaf.env_ids[keep_episodes],
        },
    }
    return ReplayBuffer(root=replay_root)
