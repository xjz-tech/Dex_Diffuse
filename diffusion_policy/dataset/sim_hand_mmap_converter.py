from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.sim_hand_hdf5 import (
    SimHandManifest,
    read_sim_hand_manifest,
    validate_sim_hand_shards,
)
from diffusion_policy.dataset.sim_hand_mmap_cache import (
    FORMAT_NAME,
    FORMAT_VERSION,
    HAND_DIM,
    SimHandMmapCacheError,
    resolve_cache_dir,
    sha256_file,
    validate_sim_hand_mmap_cache,
)

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


class _StreamingStats:
    def __init__(self, dim: int) -> None:
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def update(self, chunk) -> None:
        chunk64 = np.asarray(chunk, dtype=np.float64)
        chunk_count = len(chunk64)
        if chunk_count == 0:
            return
        self.minimum = np.minimum(self.minimum, chunk64.min(axis=0))
        self.maximum = np.maximum(self.maximum, chunk64.max(axis=0))
        chunk_mean = chunk64.mean(axis=0)
        chunk_m2 = ((chunk64 - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - self.mean
        total = self.count + chunk_count
        self.mean += delta * (chunk_count / total)
        self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / total
        self.count = total

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError(
                f"normalizer statistics require count >= 2, got {self.count}"
            )
        std = np.sqrt(self.m2 / (self.count - 1))
        return (
            self.minimum.astype(np.float32, copy=False),
            self.maximum.astype(np.float32, copy=False),
            self.mean.astype(np.float32, copy=False),
            std.astype(np.float32, copy=False),
        )


def _validate_build_args(
    *,
    horizon: int,
    pad_before: int,
    pad_after: int,
    val_ratio: float,
    seed: int,
    chunk_rows: int,
) -> None:
    if type(horizon) is not int or horizon < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {horizon!r}")
    if type(pad_before) is not int or not (0 <= pad_before < horizon):
        raise ValueError(
            "pad_before must be an integer satisfying "
            f"0 <= pad_before < horizon, got {pad_before!r}"
        )
    if type(pad_after) is not int or not (0 <= pad_after < horizon):
        raise ValueError(
            "pad_after must be an integer satisfying "
            f"0 <= pad_after < horizon, got {pad_after!r}"
        )
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, (int, float))
        or not (0.0 <= float(val_ratio) < 1.0)
    ):
        raise ValueError(
            f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio!r}"
        )
    if type(seed) is not int:
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError(
            f"chunk_rows must be an integer >= 1, got {chunk_rows!r}"
        )


def _raise_existing_output(output_path: Path) -> None:
    raise FileExistsError(
        f"Sim-Hand mmap cache already exists at {output_path}; "
        "move it aside before rebuilding"
    )


def _close_memmap(array: np.memmap) -> None:
    array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _retained_mask(
    chunk_keys: np.ndarray,
    inventory_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(inventory_keys, chunk_keys)
    n_keys = inventory_keys.shape[0]
    in_bounds = positions < n_keys
    matched = np.zeros(chunk_keys.shape[0], dtype=bool)
    if np.any(in_bounds):
        matched[in_bounds] = (
            inventory_keys[positions[in_bounds]] == chunk_keys[in_bounds]
        )
    return matched, positions


def _scatter_source_chunks(
    *,
    manifest: SimHandManifest,
    inventory: EpisodeInventory,
    obs_mm: np.memmap,
    action_mm: np.memmap,
    seen_mm: np.memmap,
    obs_stats: _StreamingStats,
    action_stats: _StreamingStats,
    chunk_rows: int,
    n_steps: int,
) -> None:
    for shard in manifest.shards:
        if shard.num_transitions == 0:
            continue
        with h5py.File(shard.path, "r") as handle:
            episode_ids = handle["index/episode_id"]
            env_ids = handle["index/env_id"]
            steps = handle["index/step"]
            qpos = handle["robot/qpos"]
            target_after = handle["robot/target_after"]
            n_rows = int(episode_ids.shape[0])
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                chunk_keys = np.empty(end - start, dtype=EPISODE_KEY_DTYPE)
                chunk_keys["episode_id"] = np.asarray(
                    episode_ids[start:end],
                    dtype=np.int64,
                )
                chunk_keys["env_id"] = np.asarray(
                    env_ids[start:end],
                    dtype=np.int32,
                )
                matched, positions = _retained_mask(chunk_keys, inventory.keys)
                if not np.any(matched):
                    continue

                key_position = positions[matched]
                step = np.asarray(steps[start:end], dtype=np.int64)[matched]
                dest = inventory.offsets[key_position] + step
                if np.any(step < 0) or np.any(step >= inventory.lengths[key_position]):
                    raise ValueError(
                        "Sim-Hand conversion found a negative or out-of-range step"
                    )
                if np.any(dest < 0) or np.any(dest >= n_steps):
                    raise ValueError(
                        "Sim-Hand conversion found a negative or out-of-range step"
                    )
                if np.unique(dest).size != dest.size:
                    raise ValueError(
                        "Sim-Hand conversion found duplicate destination rows"
                    )
                if np.any(seen_mm[dest] != 0):
                    raise ValueError(
                        "Sim-Hand conversion found duplicate destination rows"
                    )

                obs_chunk = np.asarray(qpos[start:end], dtype=np.float32)[matched]
                action_chunk = np.asarray(
                    target_after[start:end],
                    dtype=np.float32,
                )[matched]
                if not np.all(np.isfinite(obs_chunk)):
                    raise ValueError("Sim-Hand obs values must be finite")
                if not np.all(np.isfinite(action_chunk)):
                    raise ValueError("Sim-Hand action values must be finite")

                obs_mm[dest] = obs_chunk
                action_mm[dest] = action_chunk
                seen_mm[dest] = 1
                obs_stats.update(obs_chunk)
                action_stats.update(action_chunk)


def _scan_seen(seen_mm: np.memmap, *, chunk_rows: int) -> None:
    n_steps = int(seen_mm.shape[0])
    for start in range(0, n_steps, chunk_rows):
        end = min(start + chunk_rows, n_steps)
        if np.any(seen_mm[start:end] == 0):
            raise ValueError(
                "Sim-Hand conversion missed destination rows (seen.npy has zeros)"
            )


def _array_meta(filename: str, dtype: str, shape: list[int]) -> dict:
    return {"filename": filename, "dtype": dtype, "shape": shape}


def _build_metadata(
    *,
    manifest: SimHandManifest,
    inventory: EpisodeInventory,
    horizon: int,
    pad_before: int,
    pad_after: int,
    val_ratio: float,
    seed: int,
    n_steps: int,
    n_episodes: int,
) -> dict:
    return {
        "format_name": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "source": {
            "manifest_sha256": sha256_file(manifest.manifest_path),
            "schema_version": int(manifest.schema_version),
            "dof": int(manifest.dof),
            "declared_transitions": int(manifest.total_transitions),
        },
        "temporal": {
            "horizon": int(horizon),
            "pad_before": int(pad_before),
            "pad_after": int(pad_after),
        },
        "split": {
            "val_ratio": float(val_ratio),
            "seed": int(seed),
        },
        "counts": {
            "source_episodes": int(inventory.source_episode_count),
            "retained_episodes": int(n_episodes),
            "dropped_episodes": int(inventory.dropped_episode_count),
            "retained_transitions": int(n_steps),
            "dropped_transitions": int(inventory.dropped_transition_count),
        },
        "arrays": {
            "obs": _array_meta("obs.npy", "float32", [n_steps, HAND_DIM]),
            "action": _array_meta("action.npy", "float32", [n_steps, HAND_DIM]),
            "episode_ends": _array_meta(
                "episode_ends.npy", "int64", [n_episodes]
            ),
            "episode_ids": _array_meta(
                "episode_ids.npy", "int64", [n_episodes]
            ),
            "env_ids": _array_meta("env_ids.npy", "int32", [n_episodes]),
            "val_mask": _array_meta("val_mask.npy", "bool", [n_episodes]),
        },
        "normalizer": {
            "filename": "normalizer.npz",
            "dtype": "float32",
            "shape": [HAND_DIM],
            "row_count": int(n_steps),
            "std_correction": 1,
        },
    }


def _write_temp_cache(
    tmp_dir: Path,
    *,
    dataset_path: Path,
    manifest: SimHandManifest,
    inventory: EpisodeInventory,
    horizon: int,
    pad_before: int,
    pad_after: int,
    val_ratio: float,
    seed: int,
    chunk_rows: int,
) -> None:
    n_episodes = int(inventory.lengths.shape[0])
    n_steps = int(inventory.episode_ends[-1])

    np.save(
        tmp_dir / "episode_ends.npy",
        np.asarray(inventory.episode_ends, dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        tmp_dir / "episode_ids.npy",
        np.asarray(inventory.keys["episode_id"], dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        tmp_dir / "env_ids.npy",
        np.asarray(inventory.keys["env_id"], dtype=np.int32),
        allow_pickle=False,
    )
    np.save(
        tmp_dir / "val_mask.npy",
        np.asarray(inventory.val_mask, dtype=bool),
        allow_pickle=False,
    )

    obs_mm = np.lib.format.open_memmap(
        tmp_dir / "obs.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_steps, HAND_DIM),
    )
    action_mm = np.lib.format.open_memmap(
        tmp_dir / "action.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n_steps, HAND_DIM),
    )
    seen_mm = np.lib.format.open_memmap(
        tmp_dir / "seen.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(n_steps,),
    )
    seen_mm[:] = 0

    obs_stats = _StreamingStats(HAND_DIM)
    action_stats = _StreamingStats(HAND_DIM)
    try:
        _scatter_source_chunks(
            manifest=manifest,
            inventory=inventory,
            obs_mm=obs_mm,
            action_mm=action_mm,
            seen_mm=seen_mm,
            obs_stats=obs_stats,
            action_stats=action_stats,
            chunk_rows=chunk_rows,
            n_steps=n_steps,
        )
        _scan_seen(seen_mm, chunk_rows=chunk_rows)
    finally:
        _close_memmap(obs_mm)
        _close_memmap(action_mm)
        _close_memmap(seen_mm)

    (tmp_dir / "seen.npy").unlink()

    obs_min, obs_max, obs_mean, obs_std = obs_stats.finalize()
    action_min, action_max, action_mean, action_std = action_stats.finalize()
    np.savez(
        tmp_dir / "normalizer.npz",
        obs_min=obs_min,
        obs_max=obs_max,
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_min=action_min,
        action_max=action_max,
        action_mean=action_mean,
        action_std=action_std,
    )

    metadata = _build_metadata(
        manifest=manifest,
        inventory=inventory,
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        val_ratio=val_ratio,
        seed=seed,
        n_steps=n_steps,
        n_episodes=n_episodes,
    )
    (tmp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    validate_sim_hand_mmap_cache(
        tmp_dir,
        dataset_path=dataset_path,
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        val_ratio=val_ratio,
        seed=seed,
        require_ready=False,
    )
    (tmp_dir / "READY").write_bytes(b"")


def build_sim_hand_mmap_cache(
    dataset_path: str | Path,
    *,
    output_path: str | Path | None = None,
    horizon: int = 12,
    pad_before: int = 3,
    pad_after: int = 8,
    val_ratio: float = 0.1,
    seed: int = 42,
    chunk_rows: int = 262_144,
) -> Path:
    _validate_build_args(
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        val_ratio=val_ratio,
        seed=seed,
        chunk_rows=chunk_rows,
    )
    dataset_path = Path(dataset_path)
    output_path = resolve_cache_dir(dataset_path, output_path)

    if output_path.exists():
        try:
            validate_sim_hand_mmap_cache(
                output_path,
                dataset_path=dataset_path,
                horizon=horizon,
                pad_before=pad_before,
                pad_after=pad_after,
                val_ratio=val_ratio,
                seed=seed,
            )
        except SimHandMmapCacheError:
            _raise_existing_output(output_path)
        return output_path

    manifest = read_sim_hand_manifest(dataset_path)
    validate_sim_hand_shards(manifest)
    inventory = scan_episode_inventory(
        manifest,
        horizon=horizon,
        val_ratio=val_ratio,
        seed=seed,
        chunk_rows=chunk_rows,
    )

    tmp_dir: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4()}"
        tmp_dir.mkdir(exist_ok=False)
        _write_temp_cache(
            tmp_dir,
            dataset_path=dataset_path,
            manifest=manifest,
            inventory=inventory,
            horizon=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            val_ratio=val_ratio,
            seed=seed,
            chunk_rows=chunk_rows,
        )
        if output_path.exists():
            _raise_existing_output(output_path)
        tmp_dir.rename(output_path)
        tmp_dir = None
    except Exception:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise

    return output_path
