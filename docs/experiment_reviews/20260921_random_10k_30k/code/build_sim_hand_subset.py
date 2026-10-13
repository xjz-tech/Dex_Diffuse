#!/usr/bin/env python3
"""Build a small, episode-complete 66-D Sim-Hand training subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import zarr


HAND_DIM = 22
OBS_DIM = 66


@dataclass(frozen=True)
class EpisodeSelection:
    episode_id: int
    env_id: int
    length: int


@dataclass(frozen=True)
class SubsetSummary:
    requested_max_transitions: int
    actual_transitions: int
    episode_count: int
    seed: int


def select_episode_keys(
    episode_ids: np.ndarray,
    env_ids: np.ndarray,
    episode_ends: np.ndarray,
    *,
    max_transitions: int,
    seed: int,
) -> list[EpisodeSelection]:
    """Select whole episodes deterministically without exceeding the row cap."""
    episode_ids = np.asarray(episode_ids)
    env_ids = np.asarray(env_ids)
    episode_ends = np.asarray(episode_ends)
    if max_transitions <= 0:
        raise ValueError("max_transitions must be positive")
    if episode_ids.ndim != 1 or env_ids.ndim != 1 or episode_ends.ndim != 1:
        raise ValueError("episode catalog arrays must be one-dimensional")
    if not (len(episode_ids) == len(env_ids) == len(episode_ends)):
        raise ValueError("episode catalog arrays must have equal lengths")
    if len(episode_ends) == 0:
        raise ValueError("episode catalog is empty")
    if episode_ends[0] <= 0 or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing")

    starts = np.r_[0, episode_ends[:-1]]
    lengths = episode_ends - starts
    order = np.random.default_rng(seed).permutation(len(episode_ends))
    selected: list[EpisodeSelection] = []
    total = 0
    for index in order:
        length = int(lengths[index])
        if total + length > max_transitions:
            continue
        selected.append(
            EpisodeSelection(
                episode_id=int(episode_ids[index]),
                env_id=int(env_ids[index]),
                length=length,
            )
        )
        total += length
        if total == max_transitions:
            break
    if not selected:
        raise ValueError(
            "no complete episode fits within max_transitions="
            f"{max_transitions}"
        )
    return selected


def _load_catalog(source_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    catalog_dir = source_dir / "exp_data_mmap"
    paths = {
        name: catalog_dir / f"{name}.npy"
        for name in ("episode_ids", "env_ids", "episode_ends")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing episode catalog arrays: {missing}")
    return tuple(
        np.load(paths[name], mmap_mode="r")
        for name in ("episode_ids", "env_ids", "episode_ends")
    )


def _load_manifest(source_dir: Path) -> dict:
    path = source_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"source manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dof") != HAND_DIM:
        raise ValueError("unsupported Sim-Hand manifest schema or hand dimension")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("source manifest has no shards")
    return manifest


def _collect_episodes(
    source_dir: Path,
    manifest: dict,
    selected: list[EpisodeSelection],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = [(item.episode_id, item.env_id) for item in selected]
    pieces = {
        key: {name: [] for name in ("step", "qpos", "target_before", "action")}
        for key in keys
    }
    required = (
        "index/episode_id",
        "index/env_id",
        "index/step",
        "robot/qpos",
        "robot/target_before",
        "robot/target_after",
    )
    for shard in manifest["shards"]:
        shard_path = source_dir / shard["path"]
        with h5py.File(shard_path, "r") as handle:
            missing = [name for name in required if name not in handle]
            if missing:
                raise KeyError(f"{shard_path} is missing fields: {missing}")
            episode_id = np.asarray(handle["index/episode_id"][:])
            env_id = np.asarray(handle["index/env_id"][:])
            for key in keys:
                indices = np.flatnonzero(
                    (episode_id == key[0]) & (env_id == key[1])
                )
                if indices.size == 0:
                    continue
                destination = pieces[key]
                destination["step"].append(np.asarray(handle["index/step"][indices]))
                destination["qpos"].append(np.asarray(handle["robot/qpos"][indices]))
                destination["target_before"].append(
                    np.asarray(handle["robot/target_before"][indices])
                )
                destination["action"].append(
                    np.asarray(handle["robot/target_after"][indices])
                )

    observation_parts = []
    action_parts = []
    episode_ends = []
    total = 0
    for item in selected:
        key = (item.episode_id, item.env_id)
        episode = pieces[key]
        if not episode["step"]:
            raise ValueError(f"selected episode {key} was not found in HDF5 shards")
        step = np.concatenate(episode["step"])
        order = np.argsort(step, kind="stable")
        expected_steps = np.arange(item.length, dtype=step.dtype)
        if len(order) != item.length or not np.array_equal(step[order], expected_steps):
            raise ValueError(
                f"selected episode {key} is incomplete or has duplicate steps; "
                f"expected {item.length} contiguous rows, found {len(order)}"
            )
        qpos = np.concatenate(episode["qpos"], axis=0)[order].astype(
            np.float32, copy=False
        )
        target_before = np.concatenate(episode["target_before"], axis=0)[order].astype(
            np.float32, copy=False
        )
        action = np.concatenate(episode["action"], axis=0)[order].astype(
            np.float32, copy=False
        )
        obs = np.concatenate(
            (qpos, target_before, target_before - qpos), axis=-1
        ).astype(np.float32, copy=False)
        if obs.shape != (item.length, OBS_DIM) or action.shape != (
            item.length,
            HAND_DIM,
        ):
            raise ValueError(f"selected episode {key} has invalid robot tensor shapes")
        observation_parts.append(obs)
        action_parts.append(action)
        total += item.length
        episode_ends.append(total)

    return (
        np.concatenate(observation_parts, axis=0),
        np.concatenate(action_parts, axis=0),
        np.asarray(episode_ends, dtype=np.int64),
    )


def build_subset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    max_transitions: int,
    seed: int,
) -> SubsetSummary:
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    episode_ids, env_ids, episode_ends = _load_catalog(source_dir)
    selected = select_episode_keys(
        episode_ids,
        env_ids,
        episode_ends,
        max_transitions=max_transitions,
        seed=seed,
    )
    manifest = _load_manifest(source_dir)
    obs, action, subset_episode_ends = _collect_episodes(
        source_dir, manifest, selected
    )

    output_dir.mkdir(parents=True)
    replay = zarr.open_group(str(output_dir / "replay_buffer.zarr"), mode="w")
    data = replay.create_group("data")
    meta = replay.create_group("meta")
    row_chunks = min(4096, len(obs))
    data.array("obs", obs, chunks=(row_chunks, OBS_DIM))
    data.array("action", action, chunks=(row_chunks, HAND_DIM))
    meta.array("episode_ends", subset_episode_ends)
    meta.array(
        "episode_ids",
        np.asarray([item.episode_id for item in selected], dtype=np.int64),
    )
    meta.array(
        "env_ids",
        np.asarray([item.env_id for item in selected], dtype=np.int32),
    )

    summary = SubsetSummary(
        requested_max_transitions=int(max_transitions),
        actual_transitions=int(len(obs)),
        episode_count=len(selected),
        seed=int(seed),
    )
    payload = {
        **asdict(summary),
        "source": str(source_dir),
        "observation": "qpos,target_before,target_before-qpos",
        "episodes": [asdict(item) for item in selected],
    }
    (output_dir / "subset_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-transitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_subset(
        args.source,
        args.output,
        max_transitions=args.max_transitions,
        seed=args.seed,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
