from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diffusion_policy.dataset.sim_hand_mmap_cache import (
    CACHE_DIRNAME,
    SimHandMmapCacheError,
    sha256_file,
    validate_sim_hand_mmap_cache,
)

HAND_DIM = 22
N_STEPS = 6
N_EPISODES = 2

DEFAULT_HORIZON = 3
DEFAULT_PAD_BEFORE = 1
DEFAULT_PAD_AFTER = 2
DEFAULT_VAL_RATIO = 0.5
DEFAULT_SEED = 7

NORMALIZER_KEYS = (
    "obs_min",
    "obs_max",
    "obs_mean",
    "obs_std",
    "action_min",
    "action_max",
    "action_mean",
    "action_std",
)


def _compute_normalizer_stats(obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    stats: dict[str, np.ndarray] = {}
    for prefix, data in (("obs", obs), ("action", action)):
        stats[f"{prefix}_min"] = data.min(axis=0).astype(np.float32)
        stats[f"{prefix}_max"] = data.max(axis=0).astype(np.float32)
        stats[f"{prefix}_mean"] = data.mean(axis=0).astype(np.float32)
        stats[f"{prefix}_std"] = data.std(axis=0, ddof=1).astype(np.float32)
    return stats


def _write_cache(
    base_path: Path,
    *,
    ready: bool = True,
    manifest_path: Path | None = None,
) -> Path:
    cache_dir = base_path / CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)

    obs = np.arange(N_STEPS * HAND_DIM, dtype=np.float32).reshape(N_STEPS, HAND_DIM)
    action = obs + 100.0
    episode_ends = np.asarray([3, 6], dtype=np.int64)
    episode_ids = np.asarray([10, 20], dtype=np.int64)
    env_ids = np.asarray([0, 1], dtype=np.int32)
    val_mask = np.asarray([False, True], dtype=bool)

    np.save(cache_dir / "obs.npy", obs, allow_pickle=False)
    np.save(cache_dir / "action.npy", action, allow_pickle=False)
    np.save(cache_dir / "episode_ends.npy", episode_ends, allow_pickle=False)
    np.save(cache_dir / "episode_ids.npy", episode_ids, allow_pickle=False)
    np.save(cache_dir / "env_ids.npy", env_ids, allow_pickle=False)
    np.save(cache_dir / "val_mask.npy", val_mask, allow_pickle=False)

    normalizer_stats = _compute_normalizer_stats(obs, action)
    np.savez(cache_dir / "normalizer.npz", **normalizer_stats)

    if manifest_path is not None:
        manifest_sha256 = sha256_file(manifest_path)
    else:
        manifest_sha256 = "0" * 64

    metadata = {
        "format_name": "sim_hand_mmap",
        "format_version": 1,
        "source": {
            "manifest_sha256": manifest_sha256,
            "schema_version": 1,
            "dof": HAND_DIM,
            "declared_transitions": N_STEPS,
        },
        "temporal": {
            "horizon": DEFAULT_HORIZON,
            "pad_before": DEFAULT_PAD_BEFORE,
            "pad_after": DEFAULT_PAD_AFTER,
        },
        "split": {
            "val_ratio": DEFAULT_VAL_RATIO,
            "seed": DEFAULT_SEED,
        },
        "counts": {
            "source_episodes": N_EPISODES,
            "retained_episodes": N_EPISODES,
            "dropped_episodes": 0,
            "retained_transitions": N_STEPS,
            "dropped_transitions": 0,
        },
        "arrays": {
            "obs": {"filename": "obs.npy", "dtype": "float32", "shape": [N_STEPS, HAND_DIM]},
            "action": {
                "filename": "action.npy",
                "dtype": "float32",
                "shape": [N_STEPS, HAND_DIM],
            },
            "episode_ends": {
                "filename": "episode_ends.npy",
                "dtype": "int64",
                "shape": [N_EPISODES],
            },
            "episode_ids": {
                "filename": "episode_ids.npy",
                "dtype": "int64",
                "shape": [N_EPISODES],
            },
            "env_ids": {"filename": "env_ids.npy", "dtype": "int32", "shape": [N_EPISODES]},
            "val_mask": {"filename": "val_mask.npy", "dtype": "bool", "shape": [N_EPISODES]},
        },
        "normalizer": {
            "filename": "normalizer.npz",
            "dtype": "float32",
            "shape": [HAND_DIM],
            "row_count": N_STEPS,
            "std_correction": 1,
        },
    }
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if ready:
        (cache_dir / "READY").touch()

    return cache_dir


def _mutate_cache(cache_dir: Path, mutation: str) -> None:
    if mutation == "wrong_obs_dtype":
        obs = np.load(cache_dir / "obs.npy", allow_pickle=False).astype(np.float64)
        np.save(cache_dir / "obs.npy", obs, allow_pickle=False)
    elif mutation == "truncated_obs":
        obs = np.load(cache_dir / "obs.npy", allow_pickle=False)[:3]
        np.save(cache_dir / "obs.npy", obs, allow_pickle=False)
    elif mutation == "wrong_episode_ends":
        np.save(
            cache_dir / "episode_ends.npy",
            np.asarray([3, 3], dtype=np.int64),
            allow_pickle=False,
        )
    elif mutation == "missing_normalizer_key":
        with np.load(cache_dir / "normalizer.npz", allow_pickle=False) as stats:
            payload = {key: stats[key] for key in stats.files if key != "action_std"}
        np.savez(cache_dir / "normalizer.npz", **payload)
    else:
        raise ValueError(f"unknown mutation: {mutation}")


def test_validate_completed_cache_and_optional_source_fingerprint(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    manifest_path = dataset_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    cache_dir = _write_cache(dataset_path, ready=True, manifest_path=manifest_path)

    info = validate_sim_hand_mmap_cache(
        cache_dir,
        dataset_path=dataset_path,
        horizon=3,
        pad_before=1,
        pad_after=2,
        val_ratio=0.5,
        seed=7,
    )

    assert info.cache_dir == cache_dir
    assert info.n_steps == 6
    assert info.n_episodes == 2


def test_validate_cache_rejects_missing_ready(tmp_path):
    cache_dir = _write_cache(tmp_path, ready=False)
    with pytest.raises(SimHandMmapCacheError, match="READY"):
        validate_sim_hand_mmap_cache(cache_dir)


def test_validate_cache_rejects_temporal_mismatch(tmp_path):
    cache_dir = _write_cache(tmp_path, ready=True)
    with pytest.raises(SimHandMmapCacheError, match="horizon.*4.*3"):
        validate_sim_hand_mmap_cache(cache_dir, horizon=4)


def test_validate_cache_rejects_manifest_fingerprint_mismatch(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    manifest_path = dataset_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    cache_dir = _write_cache(dataset_path, ready=True, manifest_path=manifest_path)
    manifest_path.write_text('{"changed": true}\n')

    with pytest.raises(SimHandMmapCacheError, match="manifest.*fingerprint"):
        validate_sim_hand_mmap_cache(cache_dir, dataset_path=dataset_path)


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("wrong_obs_dtype", "obs.*float32"),
        ("truncated_obs", "obs"),
        ("wrong_episode_ends", "episode_ends.*strictly increasing"),
        ("missing_normalizer_key", "action_std"),
    ],
)
def test_validate_cache_rejects_corrupt_artifacts(tmp_path, mutation, message):
    cache_dir = _write_cache(tmp_path, ready=True)
    _mutate_cache(cache_dir, mutation)
    with pytest.raises(SimHandMmapCacheError, match=message):
        validate_sim_hand_mmap_cache(cache_dir)
