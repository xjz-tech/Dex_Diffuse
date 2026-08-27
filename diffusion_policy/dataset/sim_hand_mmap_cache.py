from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIRNAME = "exp_data_mmap"
FORMAT_NAME = "sim_hand_mmap"
FORMAT_VERSION = 1

HAND_DIM = 22

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

ARRAY_NAMES = (
    "obs",
    "action",
    "episode_ends",
    "episode_ids",
    "env_ids",
    "val_mask",
)

REQUIRED_TOP_KEYS = (
    "format_name",
    "format_version",
    "source",
    "temporal",
    "split",
    "counts",
    "arrays",
    "normalizer",
)

REQUIRED_SOURCE_KEYS = (
    "manifest_sha256",
    "schema_version",
    "dof",
    "declared_transitions",
)

REQUIRED_TEMPORAL_KEYS = ("horizon", "pad_before", "pad_after")
REQUIRED_SPLIT_KEYS = ("val_ratio", "seed")
REQUIRED_COUNTS_KEYS = (
    "source_episodes",
    "retained_episodes",
    "dropped_episodes",
    "retained_transitions",
    "dropped_transitions",
)

REQUIRED_ARRAY_ENTRY_KEYS = ("filename", "dtype", "shape")
REQUIRED_NORMALIZER_META_KEYS = (
    "filename",
    "dtype",
    "shape",
    "row_count",
    "std_correction",
)


class SimHandMmapCacheError(ValueError):
    """The completed mmap cache violates its versioned disk contract."""


@dataclass(frozen=True)
class SimHandMmapCacheInfo:
    cache_dir: Path
    metadata: dict
    n_steps: int
    n_episodes: int


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cache_dir(
    dataset_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    if output_path is not None:
        return Path(output_path)
    return Path(dataset_path) / CACHE_DIRNAME


def _raise_cache_error(artifact: str, message: str) -> None:
    raise SimHandMmapCacheError(f"{artifact}: {message}")


def _require_mapping(value: Any, *, artifact: str, label: str) -> dict:
    if not isinstance(value, dict):
        _raise_cache_error(artifact, f"{label} must be a JSON object")
    return value


def _require_keys(mapping: dict, keys: tuple[str, ...], *, artifact: str, label: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        _raise_cache_error(
            artifact,
            f"{label} missing required keys: {', '.join(missing)}",
        )


def _require_int(value: Any, *, artifact: str, label: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        _raise_cache_error(artifact, f"{label} must be an integer")
    return value


def _require_float(value: Any, *, artifact: str, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        _raise_cache_error(artifact, f"{label} must be a number")
    return float(value)


def _require_str(value: Any, *, artifact: str, label: str) -> str:
    if type(value) is not str:
        _raise_cache_error(artifact, f"{label} must be a string")
    return value


def _require_shape(value: Any, *, artifact: str, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        _raise_cache_error(artifact, f"{label} shape must be a non-empty list")
    shape: list[int] = []
    for index, item in enumerate(value):
        if type(item) is not int or isinstance(item, bool):
            _raise_cache_error(
                artifact,
                f"{label} shape[{index}] must be an integer",
            )
        shape.append(item)
    return tuple(shape)


def _dtype_from_name(name: str) -> np.dtype:
    mapping = {
        "float32": np.dtype("float32"),
        "int64": np.dtype("int64"),
        "int32": np.dtype("int32"),
        "bool": np.dtype("bool"),
    }
    if name not in mapping:
        raise KeyError(name)
    return mapping[name]


def _load_metadata(cache_dir: Path) -> dict:
    metadata_path = cache_dir / "metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SimHandMmapCacheError("metadata.json: file not found") from exc
    except json.JSONDecodeError as exc:
        raise SimHandMmapCacheError("metadata.json: invalid JSON object") from exc

    if not isinstance(payload, dict):
        _raise_cache_error("metadata.json", "root value must be a JSON object")
    return payload


def _validate_metadata_schema(metadata: dict) -> None:
    artifact = "metadata.json"
    _require_keys(metadata, REQUIRED_TOP_KEYS, artifact=artifact, label="root")

    format_name = _require_str(metadata["format_name"], artifact=artifact, label="format_name")
    if format_name != FORMAT_NAME:
        _raise_cache_error(
            artifact,
            f"format_name must be {FORMAT_NAME!r}, got {format_name!r}",
        )

    format_version = _require_int(metadata["format_version"], artifact=artifact, label="format_version")
    if format_version != FORMAT_VERSION:
        _raise_cache_error(
            artifact,
            f"format_version must be {FORMAT_VERSION}, got {format_version}",
        )

    source = _require_mapping(metadata["source"], artifact=artifact, label="source")
    _require_keys(source, REQUIRED_SOURCE_KEYS, artifact=artifact, label="source")
    manifest_sha256 = _require_str(
        source["manifest_sha256"],
        artifact=artifact,
        label="source.manifest_sha256",
    )
    if len(manifest_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in manifest_sha256):
        _raise_cache_error(
            artifact,
            "source.manifest_sha256 must be 64 lowercase hex characters",
        )
    _require_int(source["schema_version"], artifact=artifact, label="source.schema_version")
    dof = _require_int(source["dof"], artifact=artifact, label="source.dof")
    if dof != HAND_DIM:
        _raise_cache_error(artifact, f"source.dof must be {HAND_DIM}, got {dof}")
    _require_int(
        source["declared_transitions"],
        artifact=artifact,
        label="source.declared_transitions",
    )

    temporal = _require_mapping(metadata["temporal"], artifact=artifact, label="temporal")
    _require_keys(temporal, REQUIRED_TEMPORAL_KEYS, artifact=artifact, label="temporal")
    for key in REQUIRED_TEMPORAL_KEYS:
        value = _require_int(temporal[key], artifact=artifact, label=f"temporal.{key}")
        if value < 0:
            _raise_cache_error(artifact, f"temporal.{key} must be non-negative")

    split = _require_mapping(metadata["split"], artifact=artifact, label="split")
    _require_keys(split, REQUIRED_SPLIT_KEYS, artifact=artifact, label="split")
    val_ratio = _require_float(split["val_ratio"], artifact=artifact, label="split.val_ratio")
    if not (0.0 <= val_ratio <= 1.0):
        _raise_cache_error(artifact, "split.val_ratio must be between 0 and 1")
    _require_int(split["seed"], artifact=artifact, label="split.seed")

    counts = _require_mapping(metadata["counts"], artifact=artifact, label="counts")
    _require_keys(counts, REQUIRED_COUNTS_KEYS, artifact=artifact, label="counts")
    for key in REQUIRED_COUNTS_KEYS:
        value = _require_int(counts[key], artifact=artifact, label=f"counts.{key}")
        if value < 0:
            _raise_cache_error(artifact, f"counts.{key} must be non-negative")

    arrays = _require_mapping(metadata["arrays"], artifact=artifact, label="arrays")
    _require_keys(arrays, ARRAY_NAMES, artifact=artifact, label="arrays")
    for array_name in ARRAY_NAMES:
        entry = _require_mapping(
            arrays[array_name],
            artifact=artifact,
            label=f"arrays.{array_name}",
        )
        _require_keys(
            entry,
            REQUIRED_ARRAY_ENTRY_KEYS,
            artifact=artifact,
            label=f"arrays.{array_name}",
        )
        filename = _require_str(entry["filename"], artifact=artifact, label=f"arrays.{array_name}.filename")
        expected_filename = f"{array_name}.npy"
        if filename != expected_filename:
            _raise_cache_error(
                artifact,
                f"arrays.{array_name}.filename must be {expected_filename!r}, got {filename!r}",
            )
        dtype_name = _require_str(entry["dtype"], artifact=artifact, label=f"arrays.{array_name}.dtype")
        try:
            _dtype_from_name(dtype_name)
        except KeyError:
            _raise_cache_error(
                artifact,
                f"arrays.{array_name}.dtype has unsupported value {dtype_name!r}",
            )
        _require_shape(entry["shape"], artifact=artifact, label=f"arrays.{array_name}.shape")

    normalizer = _require_mapping(metadata["normalizer"], artifact=artifact, label="normalizer")
    _require_keys(
        normalizer,
        REQUIRED_NORMALIZER_META_KEYS,
        artifact=artifact,
        label="normalizer",
    )
    filename = _require_str(normalizer["filename"], artifact=artifact, label="normalizer.filename")
    if filename != "normalizer.npz":
        _raise_cache_error(
            artifact,
            f"normalizer.filename must be 'normalizer.npz', got {filename!r}",
        )
    dtype_name = _require_str(normalizer["dtype"], artifact=artifact, label="normalizer.dtype")
    if dtype_name != "float32":
        _raise_cache_error(artifact, "normalizer.dtype must be 'float32'")
    shape = _require_shape(normalizer["shape"], artifact=artifact, label="normalizer.shape")
    if shape != (HAND_DIM,):
        _raise_cache_error(artifact, f"normalizer.shape must be [{HAND_DIM}]")
    row_count = _require_int(normalizer["row_count"], artifact=artifact, label="normalizer.row_count")
    if row_count < 0:
        _raise_cache_error(artifact, "normalizer.row_count must be non-negative")
    std_correction = _require_int(
        normalizer["std_correction"],
        artifact=artifact,
        label="normalizer.std_correction",
    )
    if std_correction != 1:
        _raise_cache_error(artifact, "normalizer.std_correction must be 1")


def _validate_expected_values(
    metadata: dict,
    *,
    horizon: int | None,
    pad_before: int | None,
    pad_after: int | None,
    val_ratio: float | None,
    seed: int | None,
) -> None:
    artifact = "metadata.json"
    temporal = metadata["temporal"]
    split = metadata["split"]

    if horizon is not None:
        cached = temporal["horizon"]
        if cached != horizon:
            _raise_cache_error(
                artifact,
                f"horizon mismatch: expected {horizon}, metadata has {cached}",
            )

    if pad_before is not None:
        cached = temporal["pad_before"]
        if cached != pad_before:
            _raise_cache_error(
                artifact,
                f"pad_before mismatch: expected {pad_before}, metadata has {cached}",
            )

    if pad_after is not None:
        cached = temporal["pad_after"]
        if cached != pad_after:
            _raise_cache_error(
                artifact,
                f"pad_after mismatch: expected {pad_after}, metadata has {cached}",
            )

    if val_ratio is not None:
        cached = float(split["val_ratio"])
        if not math.isclose(cached, val_ratio, rel_tol=0, abs_tol=1e-12):
            _raise_cache_error(
                artifact,
                f"val_ratio mismatch: expected {val_ratio}, metadata has {cached}",
            )

    if seed is not None:
        cached = split["seed"]
        if cached != seed:
            _raise_cache_error(
                artifact,
                f"seed mismatch: expected {seed}, metadata has {cached}",
            )


def _validate_manifest_fingerprint(
    metadata: dict,
    *,
    dataset_path: Path,
) -> None:
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.is_file():
        return

    artifact = "manifest.json"
    expected = metadata["source"]["manifest_sha256"]
    actual = sha256_file(manifest_path)
    if actual != expected:
        _raise_cache_error(
            artifact,
            "manifest fingerprint mismatch with metadata.source.manifest_sha256",
        )


def _validate_array_file(
    cache_dir: Path,
    array_name: str,
    entry: dict,
) -> np.ndarray:
    artifact = entry["filename"]
    path = cache_dir / artifact
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except FileNotFoundError as exc:
        raise SimHandMmapCacheError(f"{artifact}: file not found") from exc
    except ValueError as exc:
        raise SimHandMmapCacheError(f"{artifact}: {exc}") from exc

    expected_dtype = _dtype_from_name(entry["dtype"])
    if array.dtype != expected_dtype:
        _raise_cache_error(
            artifact,
            f"{array_name} dtype must be {entry['dtype']}, got {array.dtype.name}",
        )

    expected_shape = tuple(entry["shape"])
    if array.shape != expected_shape:
        _raise_cache_error(
            artifact,
            f"{array_name} shape must be {list(expected_shape)}, got {list(array.shape)}",
        )

    if array.ndim != len(expected_shape):
        _raise_cache_error(
            artifact,
            f"{array_name} rank must be {len(expected_shape)}, got {array.ndim}",
        )

    return array


def _validate_episode_structure(
    metadata: dict,
    arrays: dict[str, np.ndarray],
) -> int:
    n_steps = arrays["obs"].shape[0]
    if arrays["action"].shape[0] != n_steps:
        _raise_cache_error("action.npy", f"action first dimension must equal obs ({n_steps})")

    if arrays["obs"].shape[-1] != HAND_DIM:
        _raise_cache_error("obs.npy", f"obs last dimension must be {HAND_DIM}")
    if arrays["action"].shape[-1] != HAND_DIM:
        _raise_cache_error("action.npy", f"action last dimension must be {HAND_DIM}")

    episode_ends = arrays["episode_ends"]
    if episode_ends.ndim != 1:
        _raise_cache_error("episode_ends.npy", "episode_ends must be rank 1")
    if episode_ends.size == 0:
        _raise_cache_error("episode_ends.npy", "episode_ends must not be empty")

    ends = np.asarray(episode_ends)
    if not np.all(ends > 0):
        _raise_cache_error("episode_ends.npy", "episode_ends must be strictly positive")
    if not np.all(np.diff(ends) > 0):
        _raise_cache_error("episode_ends.npy", "episode_ends must be strictly increasing")
    if int(ends[-1]) != n_steps:
        _raise_cache_error(
            "episode_ends.npy",
            f"final episode end must equal retained transitions ({n_steps})",
        )

    n_episodes = int(ends.size)
    for name in ("episode_ids", "env_ids", "val_mask"):
        if arrays[name].shape != (n_episodes,):
            _raise_cache_error(
                f"{name}.npy",
                f"{name} length must match episode_ends ({n_episodes})",
            )

    counts = metadata["counts"]
    if counts["retained_transitions"] != n_steps:
        _raise_cache_error(
            "metadata.json",
            "counts.retained_transitions must match obs row count",
        )
    if counts["retained_episodes"] != n_episodes:
        _raise_cache_error(
            "metadata.json",
            "counts.retained_episodes must match episode_ends length",
        )

    return n_episodes


def _validate_normalizer_file(cache_dir: Path, metadata: dict, n_steps: int) -> None:
    artifact = metadata["normalizer"]["filename"]
    path = cache_dir / artifact
    try:
        with np.load(path, allow_pickle=False) as stats:
            keys = set(stats.files)
            if keys != set(NORMALIZER_KEYS):
                missing = sorted(set(NORMALIZER_KEYS) - keys)
                if missing:
                    _raise_cache_error(artifact, f"missing normalizer keys: {', '.join(missing)}")
                extra = sorted(keys - set(NORMALIZER_KEYS))
                _raise_cache_error(artifact, f"unexpected normalizer keys: {', '.join(extra)}")

            for key in NORMALIZER_KEYS:
                array = stats[key]
                if array.dtype != np.dtype("float32"):
                    _raise_cache_error(artifact, f"{key} must be float32")
                if array.shape != (HAND_DIM,):
                    _raise_cache_error(artifact, f"{key} shape must be [{HAND_DIM}]")
                if not np.all(np.isfinite(array)):
                    _raise_cache_error(artifact, f"{key} must be finite")

            for prefix in ("obs", "action"):
                min_key = f"{prefix}_min"
                max_key = f"{prefix}_max"
                std_key = f"{prefix}_std"
                if np.any(stats[min_key] > stats[max_key]):
                    _raise_cache_error(artifact, f"{min_key} must be <= {max_key}")
                if np.any(stats[std_key] < 0):
                    _raise_cache_error(artifact, f"{std_key} must be non-negative")

    except FileNotFoundError as exc:
        raise SimHandMmapCacheError(f"{artifact}: file not found") from exc
    except ValueError as exc:
        raise SimHandMmapCacheError(f"{artifact}: {exc}") from exc

    row_count = metadata["normalizer"]["row_count"]
    if row_count != n_steps:
        _raise_cache_error(
            artifact,
            f"normalizer row_count must equal retained transitions ({n_steps})",
        )


def validate_sim_hand_mmap_cache(
    cache_dir: str | Path,
    *,
    dataset_path: str | Path | None = None,
    horizon: int | None = None,
    pad_before: int | None = None,
    pad_after: int | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
    require_ready: bool = True,
) -> SimHandMmapCacheInfo:
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise SimHandMmapCacheError(f"{cache_dir}: cache directory not found")

    if require_ready:
        ready_path = cache_dir / "READY"
        if not ready_path.is_file():
            raise SimHandMmapCacheError("READY: completion marker not found")

    metadata = _load_metadata(cache_dir)
    _validate_metadata_schema(metadata)
    _validate_expected_values(
        metadata,
        horizon=horizon,
        pad_before=pad_before,
        pad_after=pad_after,
        val_ratio=val_ratio,
        seed=seed,
    )

    if dataset_path is not None:
        _validate_manifest_fingerprint(metadata, dataset_path=Path(dataset_path))

    loaded_arrays: dict[str, np.ndarray] = {}
    for array_name in ARRAY_NAMES:
        loaded_arrays[array_name] = _validate_array_file(
            cache_dir,
            array_name,
            metadata["arrays"][array_name],
        )

    n_steps = int(loaded_arrays["obs"].shape[0])
    n_episodes = _validate_episode_structure(metadata, loaded_arrays)
    _validate_normalizer_file(cache_dir, metadata, n_steps)

    return SimHandMmapCacheInfo(
        cache_dir=cache_dir,
        metadata=metadata,
        n_steps=n_steps,
        n_episodes=n_episodes,
    )
