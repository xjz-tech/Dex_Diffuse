from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from diffusion_policy.dataset.sim_hand_hdf5 import (
    HAND_DIM,
    OBS_DIM,
    compose_sim_hand_obs,
    load_sim_hand_leaf_arrays,
    read_sim_hand_manifest_shards,
)


CACHE_FORMAT_VERSION = 1
CACHE_DIRNAME = ".sim_hand_multiple_cache"
CACHE_VERSION_NAME = "v1"
MAIN_ARRAY_BYTES_PER_ROW = 3 * HAND_DIM * 4
SPACE_MARGIN = 1.10
READY_NAME = "READY"
LOCK_NAME = "build.lock"
TRANSITION_FILES = ("qpos.npy", "target_before.npy", "action.npy")
EPISODE_FILES = ("episode_ends.npy", "episode_ids.npy", "env_ids.npy")
STAT_FILES = (
    "obs_min.npy",
    "obs_max.npy",
    "obs_sum.npy",
    "obs_sum_squares.npy",
    "action_min.npy",
    "action_max.npy",
    "action_sum.npy",
    "action_sum_squares.npy",
)


@dataclass(frozen=True)
class SourcePreflight:
    relative_path: str
    manifest_path: str
    declared_transitions: int
    fingerprint: str
    cache_dir: str
    ready: bool


@dataclass(frozen=True)
class CachePreflight:
    dataset_path: str
    cache_path: str
    sources: tuple[SourcePreflight, ...]
    declared_transitions: int
    estimated_cache_bytes: int
    available_bytes: int
    ready_sources: int
    missing_sources: int


@dataclass
class LeafCache:
    relative_path: str
    cache_dir: Path
    qpos: np.ndarray
    target_before: np.ndarray
    action: np.ndarray
    episode_ends: np.ndarray
    episode_ids: np.ndarray
    env_ids: np.ndarray
    obs_min: np.ndarray
    obs_max: np.ndarray
    obs_sum: np.ndarray
    obs_sum_squares: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    action_sum: np.ndarray
    action_sum_squares: np.ndarray


def default_cache_path(dataset_path: str | Path) -> Path:
    return Path(dataset_path).expanduser() / CACHE_DIRNAME / CACHE_VERSION_NAME


def discover_sim_hand_manifests(dataset_path: str | Path) -> list[Path]:
    root = Path(dataset_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(
            "Sim-Hand recursive dataset root is not a directory: "
            f"{root}"
        )
    manifests = sorted(
        (
            path
            for path in root.rglob("manifest.json")
            if path.is_file() and not path.is_symlink()
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not manifests:
        raise FileNotFoundError(
            "Sim-Hand recursive dataset root contains no manifest.json: "
            f"{root}"
        )
    return manifests


def _relative_leaf_path(root: Path, manifest_path: Path) -> str:
    return manifest_path.parent.relative_to(root).as_posix()


def _source_cache_dir_name(relative_path: str, fingerprint: str) -> str:
    path_digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"{path_digest}-{fingerprint}"


def compute_source_fingerprint(root: Path, manifest_path: Path) -> str:
    leaf = manifest_path.parent
    relative_path = _relative_leaf_path(root, manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    _, shard_specs = read_sim_hand_manifest_shards(leaf)
    hasher = hashlib.sha256()
    hasher.update(str(CACHE_FORMAT_VERSION).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(relative_path.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(manifest_bytes)
    for shard_path, _expected_rows in shard_specs:
        stat = shard_path.stat()
        hasher.update(b"\0")
        hasher.update(shard_path.relative_to(leaf).as_posix().encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    return hasher.hexdigest()


def _load_npy(path: Path, mmap_mode: str | None = "r") -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Sim-Hand multiple cache file is missing: {path}")
    try:
        array = np.load(path, mmap_mode=mmap_mode)
    except ValueError as exc:
        raise ValueError(f"Sim-Hand multiple cache file is truncated: {path}") from exc
    return array


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, array)
    _fsync_file(path)


def _episode_starts(episode_ends: np.ndarray) -> np.ndarray:
    return np.r_[0, episode_ends[:-1]]


def _compute_episode_stats(leaf_arrays) -> dict[str, np.ndarray]:
    starts = _episode_starts(leaf_arrays.episode_ends)
    n_episodes = len(leaf_arrays.episode_ends)
    obs_min = np.empty((n_episodes, OBS_DIM), dtype=np.float32)
    obs_max = np.empty((n_episodes, OBS_DIM), dtype=np.float32)
    obs_sum = np.empty((n_episodes, OBS_DIM), dtype=np.float64)
    obs_sum_squares = np.empty((n_episodes, OBS_DIM), dtype=np.float64)
    action_min = np.empty((n_episodes, HAND_DIM), dtype=np.float32)
    action_max = np.empty((n_episodes, HAND_DIM), dtype=np.float32)
    action_sum = np.empty((n_episodes, HAND_DIM), dtype=np.float64)
    action_sum_squares = np.empty((n_episodes, HAND_DIM), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, leaf_arrays.episode_ends)):
        qpos = leaf_arrays.qpos[start:end]
        target_before = leaf_arrays.target_before[start:end]
        action = leaf_arrays.action[start:end]
        obs = compose_sim_hand_obs(qpos, target_before)
        obs64 = obs.astype(np.float64, copy=False)
        action64 = action.astype(np.float64, copy=False)
        obs_min[index] = obs.min(axis=0)
        obs_max[index] = obs.max(axis=0)
        obs_sum[index] = obs64.sum(axis=0)
        obs_sum_squares[index] = np.square(obs64).sum(axis=0)
        action_min[index] = action.min(axis=0)
        action_max[index] = action.max(axis=0)
        action_sum[index] = action64.sum(axis=0)
        action_sum_squares[index] = np.square(action64).sum(axis=0)
    return {
        "obs_min": obs_min,
        "obs_max": obs_max,
        "obs_sum": obs_sum,
        "obs_sum_squares": obs_sum_squares,
        "action_min": action_min,
        "action_max": action_max,
        "action_sum": action_sum,
        "action_sum_squares": action_sum_squares,
    }


def _is_ready_cache(cache_dir: Path, expected_fingerprint: str) -> bool:
    ready_path = cache_dir / READY_NAME
    metadata_path = cache_dir / "metadata.json"
    if not ready_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(metadata, dict):
        return False
    if metadata.get("format_version") != CACHE_FORMAT_VERSION:
        return False
    if metadata.get("fingerprint") != expected_fingerprint:
        return False
    required = ("qpos.npy", *TRANSITION_FILES[1:], *EPISODE_FILES, *STAT_FILES)
    return all((cache_dir / name).is_file() for name in required)


def _available_bytes(cache_path: Path) -> int:
    cache_path.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(cache_path).free)


def preflight_sim_hand_multiple_cache(
    dataset_path: str | Path,
    cache_path: str | Path | None = None,
) -> CachePreflight:
    root = Path(dataset_path).expanduser()
    resolved_cache = (
        Path(cache_path).expanduser()
        if cache_path is not None
        else default_cache_path(root)
    )
    manifests = discover_sim_hand_manifests(root)
    sources = []
    declared_transitions = 0
    for manifest_path in manifests:
        leaf = manifest_path.parent
        relative_path = _relative_leaf_path(root, manifest_path)
        declared_total, _shard_specs = read_sim_hand_manifest_shards(leaf)
        fingerprint = compute_source_fingerprint(root, manifest_path)
        cache_dir = (
            resolved_cache / "sources" / _source_cache_dir_name(relative_path, fingerprint)
        )
        sources.append(
            SourcePreflight(
                relative_path=relative_path,
                manifest_path=str(manifest_path),
                declared_transitions=declared_total,
                fingerprint=fingerprint,
                cache_dir=str(cache_dir),
                ready=_is_ready_cache(cache_dir, fingerprint),
            )
        )
        declared_transitions += declared_total
    estimated_cache_bytes = declared_transitions * MAIN_ARRAY_BYTES_PER_ROW
    ready_sources = sum(1 for source in sources if source.ready)
    try:
        available_bytes = _available_bytes(resolved_cache)
    except OSError:
        available_bytes = 0
    return CachePreflight(
        dataset_path=str(root),
        cache_path=str(resolved_cache),
        sources=tuple(sources),
        declared_transitions=declared_transitions,
        estimated_cache_bytes=estimated_cache_bytes,
        available_bytes=available_bytes,
        ready_sources=ready_sources,
        missing_sources=len(sources) - ready_sources,
    )


def _write_leaf_cache(
    root: Path,
    manifest_path: Path,
    cache_dir: Path,
    fingerprint: str,
) -> None:
    leaf = manifest_path.parent
    relative_path = _relative_leaf_path(root, manifest_path)
    arrays = load_sim_hand_leaf_arrays(leaf)
    current_fingerprint = compute_source_fingerprint(root, manifest_path)
    if current_fingerprint != fingerprint:
        raise RuntimeError(
            "Sim-Hand multiple source changed during cache construction: "
            f"{leaf}"
        )
    stats = _compute_episode_stats(arrays)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(cache_dir / "qpos.npy", arrays.qpos)
    _save_npy(cache_dir / "target_before.npy", arrays.target_before)
    _save_npy(cache_dir / "action.npy", arrays.action)
    _save_npy(cache_dir / "episode_ends.npy", arrays.episode_ends)
    _save_npy(cache_dir / "episode_ids.npy", arrays.episode_ids)
    _save_npy(cache_dir / "env_ids.npy", arrays.env_ids)
    for name, value in stats.items():
        _save_npy(cache_dir / f"{name}.npy", value)
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "source_path": str(leaf),
        "relative_path": relative_path,
        "fingerprint": fingerprint,
        "n_transitions": int(arrays.qpos.shape[0]),
        "n_episodes": int(len(arrays.episode_ends)),
        "qpos_shape": list(arrays.qpos.shape),
        "action_shape": list(arrays.action.shape),
    }
    metadata_path = cache_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _fsync_file(metadata_path)
    ready_path = cache_dir / READY_NAME
    ready_path.write_text("", encoding="utf-8")
    _fsync_file(ready_path)
    _fsync_directory(cache_dir)


def _build_one_source(
    root: Path,
    source: SourcePreflight,
    rebuild: bool,
) -> Path:
    cache_dir = Path(source.cache_dir)
    if rebuild and cache_dir.exists():
        shutil.rmtree(cache_dir)
    if _is_ready_cache(cache_dir, source.fingerprint):
        return cache_dir
    parent = cache_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = parent / f".tmp-{uuid.uuid4().hex}"
    try:
        _write_leaf_cache(
            root,
            Path(source.manifest_path),
            tmp_dir,
            source.fingerprint,
        )
        current_fingerprint = compute_source_fingerprint(
            root,
            Path(source.manifest_path),
        )
        if current_fingerprint != source.fingerprint:
            raise RuntimeError(
                "Sim-Hand multiple source changed during cache construction: "
                f"{source.manifest_path}"
            )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.rename(tmp_dir, cache_dir)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return cache_dir


def ensure_sim_hand_multiple_cache(
    dataset_path: str | Path,
    cache_path: str | Path | None = None,
    *,
    rebuild: bool = False,
) -> list[Path]:
    root = Path(dataset_path).expanduser()
    resolved_cache = (
        Path(cache_path).expanduser()
        if cache_path is not None
        else default_cache_path(root)
    )
    resolved_cache.mkdir(parents=True, exist_ok=True)
    lock_path = resolved_cache / LOCK_NAME
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        preflight = preflight_sim_hand_multiple_cache(root, resolved_cache)
        missing = [
            source for source in preflight.sources if rebuild or not source.ready
        ]
        if missing:
            required_rows = sum(source.declared_transitions for source in missing)
            required_bytes = int(math.ceil(required_rows * MAIN_ARRAY_BYTES_PER_ROW * SPACE_MARGIN))
            available_bytes = _available_bytes(resolved_cache)
            if available_bytes < required_bytes:
                raise OSError(
                    "Insufficient cache disk space: "
                    f"required {required_bytes} bytes, "
                    f"available {available_bytes} bytes, "
                    f"cache path {resolved_cache}"
                )
        cache_dirs = [
            _build_one_source(root, source, rebuild=rebuild)
            for source in preflight.sources
        ]
        return cache_dirs
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def open_leaf_cache(cache_dir: Path) -> LeafCache:
    metadata_path = cache_dir / "metadata.json"
    if not (cache_dir / READY_NAME).is_file():
        raise FileNotFoundError(
            f"Sim-Hand multiple cache is not ready: {cache_dir}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Sim-Hand multiple cache metadata is not valid JSON: {metadata_path}"
        ) from exc
    qpos = _load_npy(cache_dir / "qpos.npy")
    target_before = _load_npy(cache_dir / "target_before.npy")
    action = _load_npy(cache_dir / "action.npy")
    episode_ends = np.asarray(_load_npy(cache_dir / "episode_ends.npy", mmap_mode=None))
    episode_ids = np.asarray(_load_npy(cache_dir / "episode_ids.npy", mmap_mode=None))
    env_ids = np.asarray(_load_npy(cache_dir / "env_ids.npy", mmap_mode=None))
    for name, array, expected_dtype, expected_width in (
        ("qpos.npy", qpos, np.float32, HAND_DIM),
        ("target_before.npy", target_before, np.float32, HAND_DIM),
        ("action.npy", action, np.float32, HAND_DIM),
    ):
        path = cache_dir / name
        if array.dtype != np.dtype(expected_dtype) or array.ndim != 2 or array.shape[1] != expected_width:
            raise ValueError(
                f"{path}: expected float32 [N, {expected_width}], "
                f"got dtype={array.dtype} shape={array.shape}"
            )
    n_rows = int(qpos.shape[0])
    if target_before.shape[0] != n_rows or action.shape[0] != n_rows:
        raise ValueError(
            f"{cache_dir}: transition arrays must share the same row count"
        )
    if episode_ends.dtype != np.int64 or episode_ends.ndim != 1 or len(episode_ends) == 0:
        raise ValueError(
            f"{cache_dir / 'episode_ends.npy'}: expected non-empty int64 vector"
        )
    if int(episode_ends[-1]) != n_rows:
        raise ValueError(
            f"{cache_dir / 'episode_ends.npy'}: final episode end must equal "
            f"{n_rows}, got {int(episode_ends[-1])}"
        )
    n_episodes = len(episode_ends)
    stats = {}
    for name, dtype, width in (
        ("obs_min", np.float32, OBS_DIM),
        ("obs_max", np.float32, OBS_DIM),
        ("obs_sum", np.float64, OBS_DIM),
        ("obs_sum_squares", np.float64, OBS_DIM),
        ("action_min", np.float32, HAND_DIM),
        ("action_max", np.float32, HAND_DIM),
        ("action_sum", np.float64, HAND_DIM),
        ("action_sum_squares", np.float64, HAND_DIM),
    ):
        path = cache_dir / f"{name}.npy"
        array = np.asarray(_load_npy(path, mmap_mode=None))
        if array.dtype != np.dtype(dtype) or array.shape != (n_episodes, width):
            raise ValueError(
                f"{path}: expected {dtype} [{n_episodes}, {width}], "
                f"got dtype={array.dtype} shape={array.shape}"
            )
        stats[name] = array
    return LeafCache(
        relative_path=str(metadata.get("relative_path", cache_dir.name)),
        cache_dir=cache_dir,
        qpos=qpos,
        target_before=target_before,
        action=action,
        episode_ends=episode_ends,
        episode_ids=episode_ids.astype(np.int64, copy=False),
        env_ids=env_ids.astype(np.int32, copy=False),
        **stats,
    )
