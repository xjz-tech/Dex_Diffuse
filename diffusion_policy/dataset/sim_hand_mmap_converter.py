from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.sim_hand_hdf5 import SimHandManifest

EPISODE_KEY_DTYPE = np.dtype(
    [
        ("episode_id", np.int64),
        ("env_id", np.int32),
    ]
)


@dataclass(frozen=True)
class EpisodeInventory:
    keys: np.ndarray
    lengths: np.ndarray
    offsets: np.ndarray
    episode_ends: np.ndarray
    val_mask: np.ndarray
    source_episode_count: int
    source_transition_count: int
    dropped_episode_count: int
    dropped_transition_count: int


def scan_episode_inventory(
    manifest: SimHandManifest,
    *,
    horizon: int,
    val_ratio: float,
    seed: int,
    chunk_rows: int = 262_144,
) -> EpisodeInventory:
    if type(horizon) is not int or horizon < 1:
        raise ValueError(
            f"horizon must be an integer >= 1, got {horizon!r}"
        )
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError(
            f"chunk_rows must be an integer >= 1, got {chunk_rows!r}"
        )
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, (int, float))
        or not (0.0 <= float(val_ratio) < 1.0)
    ):
        raise ValueError(
            f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio!r}"
        )

    counts: dict[tuple[int, int], int] = {}
    accumulated_transitions = 0

    for shard in manifest.shards:
        if shard.num_transitions == 0:
            continue
        with h5py.File(shard.path, "r") as f:
            episode_ids = f["index/episode_id"]
            env_ids = f["index/env_id"]
            n_rows = int(episode_ids.shape[0])
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                chunk_episode_id = np.asarray(
                    episode_ids[start:end],
                    dtype=np.int64,
                )
                chunk_env_id = np.asarray(
                    env_ids[start:end],
                    dtype=np.int32,
                )
                chunk_keys = np.empty(end - start, dtype=EPISODE_KEY_DTYPE)
                chunk_keys["episode_id"] = chunk_episode_id
                chunk_keys["env_id"] = chunk_env_id
                unique_keys, unique_counts = np.unique(
                    chunk_keys,
                    return_counts=True,
                )
                for key, count in zip(unique_keys, unique_counts):
                    pair = (int(key["episode_id"]), int(key["env_id"]))
                    counts[pair] = counts.get(pair, 0) + int(count)
                accumulated_transitions += end - start

    if accumulated_transitions != manifest.total_transitions:
        raise ValueError(
            "Sim-Hand HDF5 transition count mismatch during inventory: "
            f"scanned {accumulated_transitions}, "
            f"manifest.total_transitions={manifest.total_transitions}"
        )

    source_episode_count = len(counts)
    source_transition_count = accumulated_transitions

    sorted_pairs = sorted(counts.keys())
    source_keys = np.empty(len(sorted_pairs), dtype=EPISODE_KEY_DTYPE)
    source_lengths = np.empty(len(sorted_pairs), dtype=np.int64)
    for index, pair in enumerate(sorted_pairs):
        source_keys[index] = pair
        source_lengths[index] = counts[pair]

    keep = source_lengths >= horizon
    if not np.any(keep):
        raise ValueError(
            "Sim-Hand dataset contains no episodes with length >= "
            f"horizon={horizon}"
        )

    keys = source_keys[keep]
    lengths = source_lengths[keep].astype(np.int64, copy=False)
    offsets = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
    episode_ends = np.cumsum(lengths, dtype=np.int64)
    val_mask = get_val_mask(len(lengths), val_ratio=val_ratio, seed=seed)

    dropped_lengths = source_lengths[~keep]
    return EpisodeInventory(
        keys=keys,
        lengths=lengths,
        offsets=offsets.astype(np.int64, copy=False),
        episode_ends=episode_ends,
        val_mask=np.asarray(val_mask, dtype=bool),
        source_episode_count=int(source_episode_count),
        source_transition_count=int(source_transition_count),
        dropped_episode_count=int(len(dropped_lengths)),
        dropped_transition_count=int(dropped_lengths.sum())
        if len(dropped_lengths)
        else 0,
    )
