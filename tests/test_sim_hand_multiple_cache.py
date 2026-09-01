from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_sim_hand_hdf5_dataset import _row, _write_rollout
from diffusion_policy.dataset.sim_hand_multiple_cache import (
    compute_source_fingerprint,
    discover_sim_hand_manifests,
    ensure_sim_hand_multiple_cache,
    open_leaf_cache,
    preflight_sim_hand_multiple_cache,
)


def _nested_root(tmp_path: Path) -> Path:
    root = tmp_path / "sim_data"
    first = [_row(10, 0, step) for step in range(3)]
    second = [_row(10, 0, step) for step in range(3)]
    for row in second:
        row["qpos"] += 5_000.0
    _write_rollout(root / "brush" / "run_b", shards=[second])
    _write_rollout(root / "bulb" / "run_a", shards=[first])
    return root


def test_discovery_is_deterministic_and_recursive(tmp_path):
    root = _nested_root(tmp_path)
    manifests = discover_sim_hand_manifests(root)
    relative = [path.relative_to(root).as_posix() for path in manifests]
    assert relative == [
        "brush/run_b/manifest.json",
        "bulb/run_a/manifest.json",
    ]


def test_discovery_names_root_when_no_manifests(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match=str(root)):
        discover_sim_hand_manifests(root)


def test_invalid_nested_manifest_names_its_path(tmp_path):
    root = tmp_path / "sim_data"
    leaf = root / "brush" / "run_b"
    leaf.mkdir(parents=True)
    manifest_path = leaf / "manifest.json"
    manifest_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match=str(manifest_path)):
        preflight_sim_hand_multiple_cache(root)


def test_cache_reuses_fingerprint_and_invalidates_on_source_change(tmp_path):
    root = _nested_root(tmp_path)
    cache_path = tmp_path / "cache"
    first = ensure_sim_hand_multiple_cache(root, cache_path)
    ready_times = [(path / "READY").stat().st_mtime_ns for path in first]
    second = ensure_sim_hand_multiple_cache(root, cache_path)
    assert [str(path) for path in second] == [str(path) for path in first]
    assert [(path / "READY").stat().st_mtime_ns for path in second] == ready_times

    leaf = first[0]
    opened = open_leaf_cache(leaf)
    assert opened.qpos.shape[1] == 22
    assert opened.action.shape[1] == 22
    np.testing.assert_allclose(
        opened.qpos[0],
        np.full(22, 6_000.0, dtype=np.float32),
    )

    manifest = root / "brush" / "run_b" / "manifest.json"
    original = compute_source_fingerprint(root, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["note"] = "changed"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert compute_source_fingerprint(root, manifest) != original
    rebuilt = ensure_sim_hand_multiple_cache(root, cache_path)
    assert str(rebuilt[0]) != str(first[0])
    assert (first[0] / "READY").is_file()


def test_incomplete_cache_without_ready_is_ignored(tmp_path):
    root = _nested_root(tmp_path)
    cache_path = tmp_path / "cache"
    dirs = ensure_sim_hand_multiple_cache(root, cache_path)
    (dirs[0] / "READY").unlink()
    preflight = preflight_sim_hand_multiple_cache(root, cache_path)
    assert preflight.missing_sources == 1
    ensure_sim_hand_multiple_cache(root, cache_path)
    assert (dirs[0] / "READY").is_file()


def test_cache_builder_reports_insufficient_space(tmp_path, monkeypatch):
    root = _nested_root(tmp_path)
    cache_path = tmp_path / "cache"

    class Usage:
        free = 10

    monkeypatch.setattr(
        "diffusion_policy.dataset.sim_hand_multiple_cache.shutil.disk_usage",
        lambda _path: Usage(),
    )
    with pytest.raises(OSError, match="Insufficient cache disk space"):
        ensure_sim_hand_multiple_cache(root, cache_path)
