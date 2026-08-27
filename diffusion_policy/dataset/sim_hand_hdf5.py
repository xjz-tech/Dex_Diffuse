from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer


HAND_DIM = 22
SCHEMA_VERSION = 1
HDF5_FIELDS = (
    "robot/qpos",
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


def load_sim_hand_hdf5(
    dataset_path: str | Path,
    *,
    successful_only: bool = True,
) -> ReplayBuffer:
    """Load HDF5 rollout shards and make every episode contiguous in memory."""
    root_path = Path(dataset_path).expanduser()
    manifest_path = root_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Sim-Hand HDF5 manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Sim-Hand HDF5 manifest requires schema_version={SCHEMA_VERSION}, "
            f"got {schema_version}"
        )
    if int(manifest.get("dof", HAND_DIM)) != HAND_DIM:
        raise ValueError(
            f"Sim-Hand HDF5 manifest must declare dof={HAND_DIM}, "
            f"got {manifest.get('dof')}"
        )
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Sim-Hand HDF5 manifest must contain at least one shard")

    chunks: Dict[str, List[np.ndarray]] = {key: [] for key in HDF5_FIELDS}
    for shard in shards:
        shard_path = root_path / shard["path"]
        if not shard_path.is_file():
            raise FileNotFoundError(f"Sim-Hand HDF5 shard not found: {shard_path}")
        with h5py.File(shard_path, "r") as f:
            missing = [key for key in HDF5_FIELDS if key not in f]
            if missing:
                raise KeyError(
                    f"Missing Sim-Hand HDF5 fields in {shard_path}: {missing}"
                )
            shard_rows = int(f["index/episode_id"].shape[0])
            expected_rows = int(shard.get("num_transitions", shard_rows))
            if shard_rows != expected_rows:
                raise ValueError(
                    f"Shard row count mismatch for {shard_path}: "
                    f"manifest={expected_rows}, actual={shard_rows}"
                )
            for key in HDF5_FIELDS:
                dataset = f[key]
                expected_dtype = np.dtype(HDF5_FIELD_DTYPES[key])
                if np.dtype(dataset.dtype) != expected_dtype:
                    raise TypeError(
                        f"HDF5 field {key} must use {expected_dtype}, "
                        f"got {dataset.dtype}"
                    )
                if key in ("robot/qpos", "robot/target_after") and (
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
                value = np.asarray(dataset[:])
                if value.shape[0] != shard_rows:
                    raise ValueError(
                        f"HDF5 field {key} has {value.shape[0]} rows; "
                        f"expected {shard_rows}"
                    )
                chunks[key].append(value)

    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in chunks.items()
    }
    actual_total = int(arrays["index/episode_id"].shape[0])
    declared_total = int(manifest.get("total_transitions", actual_total))
    if declared_total != actual_total:
        raise ValueError(
            "Sim-Hand HDF5 manifest total_transitions mismatch: "
            f"declared {declared_total}, loaded {actual_total}"
        )
    if actual_total == 0:
        raise ValueError("Sim-Hand HDF5 dataset contains no transitions")
    episode_id = arrays["index/episode_id"]
    env_id = arrays["index/env_id"]
    step = arrays["index/step"]
    order = np.lexsort((step, env_id, episode_id))
    episode_id = episode_id[order]
    env_id = env_id[order]

    episode_starts = np.flatnonzero(
        np.r_[
            True,
            (episode_id[1:] != episode_id[:-1]) | (env_id[1:] != env_id[:-1]),
        ]
    )
    episode_ends = np.r_[episode_starts[1:], len(order)].astype(np.int64)
    ordered_step = step[order]
    done = arrays["index/done"][order]
    success = arrays["index/success"][order]
    failure = arrays["index/failure"][order]
    timeout = arrays["index/timeout"][order]
    reset_reason = arrays["index/reset_reason"][order]
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
        if end - start > 1:
            non_terminal = slice(start, end - 1)
            has_early_marker = (
                np.any(done[non_terminal])
                or np.any(success[non_terminal])
                or np.any(failure[non_terminal])
                or np.any(timeout[non_terminal])
                or np.any(reset_reason[non_terminal] != 0)
            )
            if has_early_marker:
                raise ValueError(
                    "HDF5 episode "
                    f"(episode_id={int(episode_id[start])}, "
                    f"env_id={int(env_id[start])}) has a non-terminal row "
                    "with a done, outcome, or reset marker"
                )
        if not bool(done[end - 1]):
            raise ValueError(
                "HDF5 episode "
                f"(episode_id={int(episode_id[start])}, "
                f"env_id={int(env_id[start])}) has no terminal done=True "
                "marker at its final step"
            )
        terminal_outcomes = int(success[end - 1]) + int(failure[end - 1]) + int(
            timeout[end - 1]
        )
        terminal_reset_reason = int(reset_reason[end - 1])
        if terminal_reset_reason == 4 and terminal_outcomes == 0:
            continue
        if terminal_outcomes != 1:
            raise ValueError(
                "HDF5 episode "
                f"(episode_id={int(episode_id[start])}, "
                f"env_id={int(env_id[start])}) must have exactly one terminal "
                "outcome among success, failure, and timeout"
            )
        if success[end - 1]:
            expected_reset_reason = 1
        elif failure[end - 1]:
            expected_reset_reason = 2
        else:
            expected_reset_reason = 3
        if terminal_reset_reason != expected_reset_reason:
            raise ValueError(
                "HDF5 episode "
                f"(episode_id={int(episode_id[start])}, "
                f"env_id={int(env_id[start])}) terminal outcome requires "
                f"reset_reason={expected_reset_reason}, got "
                f"{terminal_reset_reason}"
            )

    keep_episodes = np.ones(len(episode_starts), dtype=bool)
    if successful_only:
        keep_episodes = success[episode_ends - 1].astype(bool)
    if not np.any(keep_episodes):
        raise ValueError("Sim-Hand HDF5 dataset contains no successful episodes")

    kept_episode_starts = episode_starts[keep_episodes]
    kept_episode_ends = episode_ends[keep_episodes]
    kept_rows = np.concatenate(
        [
            np.arange(start, end)
            for start, end in zip(kept_episode_starts, kept_episode_ends)
        ]
    )
    kept_lengths = kept_episode_ends - kept_episode_starts
    replay_episode_ends = np.cumsum(kept_lengths, dtype=np.int64)

    replay_root = {
        "data": {
            "hand_joint": arrays["robot/qpos"][order][kept_rows],
            "action": arrays["robot/target_after"][order][kept_rows],
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
