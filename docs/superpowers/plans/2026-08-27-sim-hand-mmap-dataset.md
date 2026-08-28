# Sim-Hand Memory-Mapped Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 100-million-transition Sim-Hand HDF5 rollout once into a
read-only 16.4-GiB NumPy mmap cache, then train with bounded RAM and bounded,
deterministic logical epochs.

**Architecture:** Keep the raw HDF5 manifest and shards read-only. Extract their
existing validation into reusable functions, perform a two-pass bounded-memory
conversion into an atomically published cache, and add a dedicated map-style
dataset whose window lookup stores only per-episode cumulative counts. Feed it
with chunked random and deterministic validation samplers so PyTorch never
constructs an all-window permutation.

**Tech Stack:** Python 3.9+, NumPy `.npy` memmaps, h5py, PyTorch Dataset/DataLoader,
Hydra/OmegaConf, Bash, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-sim-hand-mmap-dataset-design.md`

## Global Constraints

- Never open a source HDF5 shard or `manifest.json` in write/append mode.
- Never delete, rewrite, rename, or prune files below the source `shards/`
  directory.
- The final cache defaults to `<dataset-path>/exp_data_mmap`; conversion must
  never happen implicitly during training.
- Conversion memory must be bounded by `chunk_rows` plus `O(E)` episode
  metadata. Do not create a transition-sized sort order.
- Dataset memory must be `O(E)`. Do not instantiate `ReplayBuffer` or
  `SequenceSampler` in the mmap path.
- DataLoader must receive bounded samplers and `shuffle: false`; it must not
  construct a permutation with one entry per logical window.
- Normalizer statistics cover every retained row, including validation rows,
  and use sample standard deviation (`correction=1`) to match `torch.std`.
- The cache stores the clipped temporal values directly; require
  `horizon >= 1` and `0 <= pad_before,pad_after < horizon` rather than silently
  accepting incompatible values.
- Preserve the legacy `SimHandLowdimDataset` Zarr/eager-HDF5 paths and their
  tests, but stop selecting them from the Sim-Hand training configuration.
- Apply TDD: add a focused failing test, observe the intended failure, implement
  the minimum behavior, rerun the focused test, then run the related suite.
- Stage only the files named by the current task. The worktree contains user
  data and unrelated untracked files.

## File and Interface Map

```text
diffusion_policy/dataset/sim_hand_hdf5.py
    Public manifest/shard dataclasses and reusable read-only schema validation;
    existing eager loader delegates to them without changing behavior.

diffusion_policy/dataset/sim_hand_mmap_cache.py
    Cache constants, metadata contract, SHA-256 fingerprinting, path resolution,
    strict completed-cache validation.

diffusion_policy/dataset/sim_hand_mmap_converter.py
    O(E) episode inventory, streaming statistics, two-pass scatter, atomic cache
    publication.

diffusion_policy/scripts/prepare_sim_hand_mmap.py
    Thin explicit CLI over build_sim_hand_mmap_cache().

diffusion_policy/dataset/sim_hand_mmap_dataset.py
    Read-only mmap Dataset, lazy padded-window lookup, validation view, cached
    normalizer reconstruction.

diffusion_policy/common/bounded_sampler.py
    EpochRandomSampler and EvenlySpacedSampler; neither stores emitted indices.

diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py
    Constructs bounded train/validation samplers, seeds each logical epoch, and
    validates positive step budgets.

diffusion_policy/config/task/sim_hand_lowdim.yaml
    Selects SimHandMmapDataset.

diffusion_policy/config/train_diffusion_unet_sim_hand_workspace.yaml
    Finite epoch/validation defaults and mmap-friendly DataLoader settings.

dp_train_sim_hand.sh
    Requires exp_data_mmap/READY and prints the exact converter command.

tests/test_sim_hand_hdf5_dataset.py
tests/test_sim_hand_mmap_cache.py
tests/test_sim_hand_mmap_converter.py
tests/test_sim_hand_mmap_dataset.py
tests/test_bounded_sampler.py
tests/test_sim_hand_train_smoke.py
    Unit, equivalence, launcher, and one-step training coverage.
```

The cache metadata schema is fixed in this implementation; do not invent
alternate key names between tasks:

```json
{
  "format_name": "sim_hand_mmap",
  "format_version": 1,
  "source": {
    "manifest_sha256": "<64 lowercase hex characters>",
    "schema_version": 1,
    "dof": 22,
    "declared_transitions": 100000392
  },
  "temporal": {
    "horizon": 12,
    "pad_before": 3,
    "pad_after": 8
  },
  "split": {
    "val_ratio": 0.1,
    "seed": 42
  },
  "counts": {
    "source_episodes": 159578,
    "retained_episodes": 159574,
    "dropped_episodes": 4,
    "retained_transitions": 100000348,
    "dropped_transitions": 44
  },
  "arrays": {
    "obs": {"filename": "obs.npy", "dtype": "float32", "shape": [100000348, 22]},
    "action": {"filename": "action.npy", "dtype": "float32", "shape": [100000348, 22]},
    "episode_ends": {"filename": "episode_ends.npy", "dtype": "int64", "shape": [159574]},
    "episode_ids": {"filename": "episode_ids.npy", "dtype": "int64", "shape": [159574]},
    "env_ids": {"filename": "env_ids.npy", "dtype": "int32", "shape": [159574]},
    "val_mask": {"filename": "val_mask.npy", "dtype": "bool", "shape": [159574]}
  },
  "normalizer": {
    "filename": "normalizer.npz",
    "dtype": "float32",
    "shape": [22],
    "row_count": 100000348,
    "std_correction": 1
  }
}
```

`normalizer.npz` has exactly these eight `float32[22]` keys:
`obs_min`, `obs_max`, `obs_mean`, `obs_std`, `action_min`, `action_max`,
`action_mean`, and `action_std`.

---

### Task 1: Extract reusable read-only HDF5 validation

**Files:**

- Modify: `diffusion_policy/dataset/sim_hand_hdf5.py`
- Test: `tests/test_sim_hand_hdf5_dataset.py`

**Interfaces produced:**

```python
@dataclass(frozen=True)
class SimHandShardSpec:
    relative_path: str
    path: Path
    num_transitions: int


@dataclass(frozen=True)
class SimHandManifest:
    dataset_path: Path
    manifest_path: Path
    schema_version: int
    dof: int
    total_transitions: int
    shards: tuple[SimHandShardSpec, ...]
```

```text
read_sim_hand_manifest(dataset_path: str | Path) -> SimHandManifest
validate_sim_hand_shards(manifest: SimHandManifest) -> None
```

`load_sim_hand_hdf5()` keeps its current signature and behavior, but replaces
its inline manifest/shard validation with these two calls.

- [ ] **Step 1: Add a direct failing contract test**

Append to `tests/test_sim_hand_hdf5_dataset.py`:

```python
def test_hdf5_manifest_and_shard_validation_are_reusable(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 2, 0),
            _row(10, 2, 1),
            _row(10, 2, 2),
        ]],
    )
    from diffusion_policy.dataset.sim_hand_hdf5 import (
        read_sim_hand_manifest,
        validate_sim_hand_shards,
    )

    manifest = read_sim_hand_manifest(tmp_path)
    validate_sim_hand_shards(manifest)

    assert manifest.dataset_path == tmp_path
    assert manifest.schema_version == 1
    assert manifest.dof == HAND_DIM
    assert manifest.total_transitions == 3
    assert manifest.shards[0].relative_path == "shards/shard_000000.h5"
    assert manifest.shards[0].num_transitions == 3
```

- [ ] **Step 2: Run it and confirm the API is missing**

```bash
pytest tests/test_sim_hand_hdf5_dataset.py::test_hdf5_manifest_and_shard_validation_are_reusable -v
```

Expected: FAIL with `ImportError` for `read_sim_hand_manifest`.

- [ ] **Step 3: Extract the existing validation without weakening it**

Add `dataclass` and the two public functions to `sim_hand_hdf5.py`. Move the
current JSON type checks, schema/dof/total checks, shard path checks, aggregate
row-count check, ten-field presence checks, exact dtype checks, and exact shape
checks into them. Keep the current exception types and message fragments.

The eager loader begins as follows:

```python
def load_sim_hand_hdf5(
    dataset_path: str | Path,
    *,
    min_episode_length: int,
) -> ReplayBuffer:
    if type(min_episode_length) is not int or min_episode_length < 1:
        raise ValueError(
            "min_episode_length must be an integer >= 1, "
            f"got {min_episode_length!r}"
        )
    manifest = read_sim_hand_manifest(dataset_path)
    validate_sim_hand_shards(manifest)
    declared_total = manifest.total_transitions
    shard_specs = [
        (shard.path, shard.num_transitions) for shard in manifest.shards
    ]
    # Existing allocation, reconstruction, filtering, and ReplayBuffer creation
    # continue unchanged from this point.
```

`read_sim_hand_manifest()` must use `Path.read_text()` and only read files;
`validate_sim_hand_shards()` must use `h5py.File(shard.path, "r")`.

- [ ] **Step 4: Run focused and legacy HDF5 tests**

```bash
pytest tests/test_sim_hand_hdf5_dataset.py -q
```

Expected: all tests PASS with the same eager-loader behavior.

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/dataset/sim_hand_hdf5.py tests/test_sim_hand_hdf5_dataset.py
git commit -m "refactor: expose Sim-Hand HDF5 validation"
```

---

### Task 2: Define and validate the completed mmap cache contract

**Files:**

- Create: `diffusion_policy/dataset/sim_hand_mmap_cache.py`
- Create: `tests/test_sim_hand_mmap_cache.py`

**Interfaces produced:**

```python
CACHE_DIRNAME = "exp_data_mmap"
FORMAT_NAME = "sim_hand_mmap"
FORMAT_VERSION = 1


class SimHandMmapCacheError(ValueError):
    """The completed mmap cache violates its versioned disk contract."""


@dataclass(frozen=True)
class SimHandMmapCacheInfo:
    cache_dir: Path
    metadata: dict
    n_steps: int
    n_episodes: int
```

```text
sha256_file(path: str | Path) -> str

resolve_cache_dir(
    dataset_path: str | Path,
    output_path: str | Path | None = None,
) -> Path

validate_sim_hand_mmap_cache(
    cache_dir: str | Path,
    *,
    dataset_path: str | Path | None = None,
    horizon: int | None = None,
    pad_before: int | None = None,
    pad_after: int | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
) -> SimHandMmapCacheInfo
```

- [ ] **Step 1: Write a small completed-cache fixture and failing tests**

Create `tests/test_sim_hand_mmap_cache.py` with a helper that writes all eight
arrays, the exact metadata schema above, and optionally `READY`. Add these tests:

```python
def test_validate_completed_cache_and_optional_source_fingerprint(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    cache_dir = _write_cache(dataset_path, ready=True)

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
    (dataset_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    cache_dir = _write_cache(dataset_path, ready=True)
    (dataset_path / "manifest.json").write_text('{"changed": true}\n')

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
def test_validate_cache_rejects_corrupt_artifacts(
    tmp_path, mutation, message
):
    cache_dir = _write_cache(tmp_path, ready=True)
    _mutate_cache(cache_dir, mutation)
    with pytest.raises(SimHandMmapCacheError, match=message):
        validate_sim_hand_mmap_cache(cache_dir)
```

The helper's two episodes have lengths `[3, 3]`; use float32 `obs/action`,
`episode_ends=[3,6]`, int64 IDs, int32 env IDs, bool validation mask, and stats
computed from all six rows. `_mutate_cache` rewrites only the named artifact and
then adjusts no metadata, ensuring the validator catches disk/metadata drift.

- [ ] **Step 2: Run the new file and confirm import failure**

```bash
pytest tests/test_sim_hand_mmap_cache.py -v
```

Expected: FAIL because `sim_hand_mmap_cache` does not exist.

- [ ] **Step 3: Implement strict validation**

Validation order must be deterministic:

1. Require the directory, `READY`, and parseable object `metadata.json`.
2. Check `format_name`, `format_version`, all nested keys, and scalar types.
3. Check any caller-provided temporal/split values exactly (use
   `math.isclose(..., rel_tol=0, abs_tol=1e-12)` only for `val_ratio`).
4. If `<dataset_path>/manifest.json` exists, compare its SHA-256. Do not require
   source shards here.
5. Open each `.npy` with `np.load(..., mmap_mode="r", allow_pickle=False)` and
   compare filename, dtype, rank, and shape with metadata.
6. Require identical first dimensions for obs/action, exact final dimension 22,
   strictly increasing positive episode ends, final end equal to `N`, and
   equal episode metadata lengths.
7. Open `normalizer.npz` with `allow_pickle=False`; require exactly the eight
   keys, float32 `[22]`, finite min/max/mean/std, `min <= max`, nonnegative std,
   `row_count == N`, and `std_correction == 1`.

Wrap JSON, NumPy, missing-file, and key failures in `SimHandMmapCacheError` with
the artifact name in the message. Close all `.npz` handles with `with np.load(...)`.

- [ ] **Step 4: Run cache tests**

```bash
pytest tests/test_sim_hand_mmap_cache.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/dataset/sim_hand_mmap_cache.py tests/test_sim_hand_mmap_cache.py
git commit -m "feat: validate Sim-Hand mmap caches"
```

---

### Task 3: Build the bounded episode inventory pass

**Files:**

- Create: `diffusion_policy/dataset/sim_hand_mmap_converter.py`
- Create: `tests/test_sim_hand_mmap_converter.py`

**Interfaces produced:**

```python
EPISODE_KEY_DTYPE = np.dtype([
    ("episode_id", np.int64),
    ("env_id", np.int32),
])


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
```

```text
scan_episode_inventory(
    manifest: SimHandManifest,
    *,
    horizon: int,
    val_ratio: float,
    seed: int,
    chunk_rows: int = 262_144,
) -> EpisodeInventory
```

- [ ] **Step 1: Add a failing interleaved inventory test**

In `tests/test_sim_hand_mmap_converter.py`, copy the small `_row` and
`_write_rollout` fixture pattern from `test_sim_hand_hdf5_dataset.py`, but keep
the helper local so legacy tests do not move. Add:

```python
def test_inventory_counts_interleaved_pairs_filters_and_splits(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[
            [_row(20, 1, 0), _row(10, 4, 0), _row(30, 0, 0)],
            [
                _row(10, 4, 1), _row(20, 1, 1), _row(10, 4, 2),
                _row(20, 1, 2), _row(20, 1, 3),
            ],
        ],
    )
    manifest = read_sim_hand_manifest(tmp_path)
    validate_sim_hand_shards(manifest)

    inventory = scan_episode_inventory(
        manifest,
        horizon=3,
        val_ratio=0.5,
        seed=42,
        chunk_rows=2,
    )

    np.testing.assert_array_equal(
        inventory.keys["episode_id"], np.array([10, 20], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        inventory.keys["env_id"], np.array([4, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(inventory.lengths, [3, 4])
    np.testing.assert_array_equal(inventory.offsets, [0, 3])
    np.testing.assert_array_equal(inventory.episode_ends, [3, 7])
    assert inventory.source_episode_count == 3
    assert inventory.source_transition_count == 8
    assert inventory.dropped_episode_count == 1
    assert inventory.dropped_transition_count == 1
    np.testing.assert_array_equal(
        inventory.val_mask,
        get_val_mask(2, val_ratio=0.5, seed=42),
    )
```

Also parametrize invalid arguments: boolean/nonpositive `chunk_rows`,
nonpositive `horizon`, `pad` is not part of this interface, and
`val_ratio < 0` or `val_ratio >= 1`.

- [ ] **Step 2: Confirm inventory API failure**

```bash
pytest tests/test_sim_hand_mmap_converter.py::test_inventory_counts_interleaved_pairs_filters_and_splits -v
```

Expected: FAIL because `scan_episode_inventory` is missing.

- [ ] **Step 3: Implement chunked pair counting**

For each shard, read only `index/episode_id[start:end]` and
`index/env_id[start:end]`. Build a structured chunk, call
`np.unique(chunk_keys, return_counts=True)`, and merge those distinct counts
into a Python `dict[tuple[int, int], int]`. This bounds temporary memory by the
chunk and persistent memory by episode count.

Sort the dictionary keys lexicographically, make typed arrays, filter
`length >= horizon`, reject zero retained episodes, and compute:

```python
offsets = np.r_[0, np.cumsum(lengths[:-1], dtype=np.int64)]
episode_ends = np.cumsum(lengths, dtype=np.int64)
val_mask = get_val_mask(len(lengths), val_ratio=val_ratio, seed=seed)
```

Validate that accumulated transition counts equal
`manifest.total_transitions`. Keep lengths, offsets, ends, and count fields
`int64`; keep `val_mask` bool.

- [ ] **Step 4: Run converter inventory and HDF validation tests**

```bash
pytest tests/test_sim_hand_mmap_converter.py -q
pytest tests/test_sim_hand_hdf5_dataset.py -q
```

Expected: all current tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/dataset/sim_hand_mmap_converter.py tests/test_sim_hand_mmap_converter.py
git commit -m "feat: inventory Sim-Hand episodes in bounded memory"
```

---

### Task 4: Stream, validate, and atomically publish the cache

**Files:**

- Modify: `diffusion_policy/dataset/sim_hand_mmap_converter.py`
- Create: `diffusion_policy/scripts/prepare_sim_hand_mmap.py`
- Modify: `tests/test_sim_hand_mmap_converter.py`

**Interfaces produced:**

```text
build_sim_hand_mmap_cache(
    dataset_path: str | Path,
    *,
    output_path: str | Path | None = None,
    horizon: int = 12,
    pad_before: int = 3,
    pad_after: int = 8,
    val_ratio: float = 0.1,
    seed: int = 42,
    chunk_rows: int = 262_144,
) -> Path


main(argv: Sequence[str] | None = None) -> int
```

- [ ] **Step 1: Add failing conversion and atomicity tests**

Extend `tests/test_sim_hand_mmap_converter.py`:

```python
def test_build_cache_reorders_rows_drops_short_episode_and_saves_stats(tmp_path):
    _write_interleaved_conversion_fixture(tmp_path)
    source_hashes = {
        path: sha256_file(path)
        for path in [tmp_path / "manifest.json", *sorted((tmp_path / "shards").glob("*.h5"))]
    }

    cache_dir = build_sim_hand_mmap_cache(
        tmp_path,
        horizon=3,
        pad_before=1,
        pad_after=2,
        val_ratio=0.5,
        seed=42,
        chunk_rows=2,
    )

    assert cache_dir == tmp_path / "exp_data_mmap"
    assert (cache_dir / "READY").read_bytes() == b""
    obs = np.load(cache_dir / "obs.npy", mmap_mode="r")
    action = np.load(cache_dir / "action.npy", mmap_mode="r")
    np.testing.assert_array_equal(obs[:, 0], [1000, 1001, 1002, 2000, 2001, 2002, 2003])
    np.testing.assert_array_equal(action[:, 0], obs[:, 0] + 10_000)
    with np.load(cache_dir / "normalizer.npz") as stats:
        np.testing.assert_allclose(stats["obs_min"], np.asarray(obs).min(axis=0))
        np.testing.assert_allclose(stats["obs_max"], np.asarray(obs).max(axis=0))
        np.testing.assert_allclose(stats["obs_mean"], np.asarray(obs).mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(stats["obs_std"], np.asarray(obs).std(axis=0, ddof=1), rtol=1e-6)
    assert source_hashes == {path: sha256_file(path) for path in source_hashes}
    validate_sim_hand_mmap_cache(cache_dir, dataset_path=tmp_path)


@pytest.mark.parametrize("corruption", ["duplicate", "out_of_range", "nonfinite_obs", "nonfinite_action"])
def test_build_cache_rejects_invalid_rows_without_publishing(tmp_path, corruption):
    _write_corrupt_rollout(tmp_path, corruption)
    with pytest.raises((ValueError, RuntimeError), match="duplicate|step|finite"):
        build_sim_hand_mmap_cache(
            tmp_path, horizon=3, pad_before=1, pad_after=1, chunk_rows=2
        )
    assert not (tmp_path / "exp_data_mmap").exists()
    assert not list(tmp_path.glob(".exp_data_mmap.tmp-*"))


def test_build_cache_refuses_existing_incomplete_output(tmp_path):
    _write_interleaved_conversion_fixture(tmp_path)
    output = tmp_path / "exp_data_mmap"
    output.mkdir()
    (output / "partial").write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="move.*aside"):
        build_sim_hand_mmap_cache(tmp_path, horizon=3, pad_before=1, pad_after=2)

    assert (output / "partial").read_text() == "keep me"


def test_build_cache_reuses_valid_matching_output(tmp_path):
    _write_interleaved_conversion_fixture(tmp_path)
    first = build_sim_hand_mmap_cache(
        tmp_path, horizon=3, pad_before=1, pad_after=2, val_ratio=0.5
    )
    before = (first / "metadata.json").stat().st_mtime_ns
    second = build_sim_hand_mmap_cache(
        tmp_path, horizon=3, pad_before=1, pad_after=2, val_ratio=0.5
    )
    assert second == first
    assert (first / "metadata.json").stat().st_mtime_ns == before
```

Add a CLI test that calls `main([...])`, asserts return value 0, the printed
final path, and `READY`. It must include every CLI option and `--output-path` so
argument wiring is covered without subprocess overhead.

- [ ] **Step 2: Confirm full conversion tests fail**

```bash
pytest tests/test_sim_hand_mmap_converter.py -v
```

Expected: inventory tests PASS; new tests FAIL because builder/CLI are absent.

- [ ] **Step 3: Implement streaming statistics**

Add a private `_StreamingStats` that holds float64 `minimum`, `maximum`,
`mean`, `m2`, and integer `count`. Merge each finite float32 chunk using the
parallel-Welford equations:

```python
chunk64 = np.asarray(chunk, dtype=np.float64)
chunk_count = len(chunk64)
chunk_mean = chunk64.mean(axis=0)
chunk_m2 = ((chunk64 - chunk_mean) ** 2).sum(axis=0)
delta = chunk_mean - self.mean
total = self.count + chunk_count
self.mean += delta * (chunk_count / total)
self.m2 += chunk_m2 + delta * delta * self.count * chunk_count / total
self.count = total
```

Finalize with `sqrt(m2 / (count - 1))`; require `count >= 2`, then cast all
four arrays to float32 before `np.savez`.

- [ ] **Step 4: Implement pass 2 and atomic publication**

`build_sim_hand_mmap_cache()` must:

1. Validate scalar arguments and resolve the output path.
2. If output exists, validate it against source and all conversion arguments;
   return it if valid, otherwise raise `FileExistsError` instructing the user to
   move it aside.
3. Read/validate source manifest and shards, compute inventory, and create one
   unique sibling `.exp_data_mmap.tmp-<uuid>` directory.
4. Write small metadata arrays with `np.save(..., allow_pickle=False)` and open
   `obs.npy`, `action.npy`, and `seen.npy` with `np.lib.format.open_memmap`.
   `seen.npy` is `uint8[N]` and exists only in the temp directory.
5. For each source chunk, form structured keys and use `np.searchsorted` against
   `inventory.keys`. Ignore nonmatching (dropped-episode) rows. For retained
   rows compute `dest = offsets[key_position] + step`.
6. Reject negative/out-of-range step values, duplicate `dest` within the chunk,
   destinations already marked in `seen`, and nonfinite obs/action before any
   chunk assignment.
7. Scatter qpos/action, mark seen, and update both statistics objects.
8. Scan `seen` in `chunk_rows` slices and reject any zero. Flush and close all
   memmaps, unlink only `seen.npy`, save the exact normalizer keys, then write
   `metadata.json`.
9. Validate every artifact in the temp directory with
   `validate_sim_hand_mmap_cache` except for the not-yet-written marker. To make
   this possible, add private `require_ready=False` support inside the cache
   module; keep the public default true.
10. Write empty `READY` last, close files, recheck that final output does not
    exist, and rename the temp directory to final on the same filesystem.
11. On any exception, `shutil.rmtree()` only the UUID temp directory created by
    that call, then re-raise.

Write metadata with sorted keys and a trailing newline. The source manifest hash
comes from `sha256_file(manifest.manifest_path)`. Counts and shapes come only
from the validated manifest/inventory, not from filesystem guesses.

- [ ] **Step 5: Implement the thin CLI**

`prepare_sim_hand_mmap.py` uses `argparse` with:

```text
--dataset-path   required Path
--output-path    optional Path
--horizon        int, default 12
--pad-before     int, default 3
--pad-after      int, default 8
--val-ratio      float, default 0.1
--seed           int, default 42
--chunk-rows     int, default 262144
```

`main()` calls the builder, prints `Sim-Hand mmap cache ready: <path>`, and
returns 0. Do not catch conversion errors into a success exit code.

- [ ] **Step 6: Run converter and cache suites**

```bash
pytest tests/test_sim_hand_mmap_converter.py tests/test_sim_hand_mmap_cache.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add diffusion_policy/dataset/sim_hand_mmap_cache.py \
  diffusion_policy/dataset/sim_hand_mmap_converter.py \
  diffusion_policy/scripts/prepare_sim_hand_mmap.py \
  tests/test_sim_hand_mmap_converter.py
git commit -m "feat: convert Sim-Hand HDF5 to mmap cache"
```

---

### Task 5: Read padded windows lazily from mmap

**Files:**

- Create: `diffusion_policy/dataset/sim_hand_mmap_dataset.py`
- Create: `tests/test_sim_hand_mmap_dataset.py`

**Interface produced:**

```text
SimHandMmapDataset.__init__(
        self,
        dataset_path: str,
        horizon: int = 12,
        pad_before: int = 3,
        pad_after: int = 8,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: int | None = None,
    ) -> None

SimHandMmapDataset.get_validation_dataset() -> SimHandMmapDataset
SimHandMmapDataset.get_normalizer(mode: str = "limits", **kwargs) -> LinearNormalizer
SimHandMmapDataset.get_all_actions() -> torch.Tensor
SimHandMmapDataset.__len__() -> int
SimHandMmapDataset.__getitem__(index: int) -> dict[str, torch.Tensor]
```

`get_all_actions()` preserves the base interface but must raise
`RuntimeError("get_all_actions would materialize the mmap dataset")`; no current
Sim-Hand workspace caller needs it.

- [ ] **Step 1: Add failing dataset-equivalence tests**

Create a small HDF fixture with at least four retained episodes of different
lengths, build its cache during the fixture, and add:

```python
def test_mmap_dataset_matches_sequence_sampler_for_every_padded_window(cache_fixture):
    dataset_path, reference_buffer = cache_fixture(val_ratio=0.0)
    dataset = SimHandMmapDataset(
        str(dataset_path), horizon=4, pad_before=2, pad_after=3,
        seed=42, val_ratio=0.0,
    )
    reference = SequenceSampler(
        reference_buffer, sequence_length=4, pad_before=2, pad_after=3,
        keys=["hand_joint", "action"],
    )

    assert len(dataset) == len(reference)
    assert isinstance(dataset.obs, np.memmap)
    assert isinstance(dataset.action, np.memmap)
    assert not dataset.obs.flags.writeable
    assert not dataset.action.flags.writeable
    assert dataset.sample_ends.shape == (len(dataset.selected_episode_indices),)
    assert not hasattr(dataset, "indices")
    for index in range(len(dataset)):
        actual = dataset[index]
        expected = reference.sample_sequence(index)
        torch.testing.assert_close(actual["obs"], torch.from_numpy(expected["hand_joint"]))
        torch.testing.assert_close(actual["action"], torch.from_numpy(expected["action"]))


def test_mmap_train_validation_views_are_disjoint_and_downsample_metadata(cache_fixture):
    dataset_path, _ = cache_fixture(val_ratio=0.25)
    dataset = SimHandMmapDataset(
        str(dataset_path), horizon=4, pad_before=2, pad_after=3,
        seed=42, val_ratio=0.25, max_train_episodes=1,
    )
    validation = dataset.get_validation_dataset()

    assert len(dataset.selected_episode_indices) == 1
    assert set(dataset.selected_episode_indices).isdisjoint(
        set(validation.selected_episode_indices)
    )
    assert np.shares_memory(dataset.obs, validation.obs)
    assert np.shares_memory(dataset.action, validation.action)


def test_mmap_normalizer_reconstructs_cached_all_row_statistics(cache_fixture):
    dataset_path, reference_buffer = cache_fixture(val_ratio=0.25)
    dataset = SimHandMmapDataset(
        str(dataset_path), horizon=4, pad_before=2, pad_after=3,
        seed=42, val_ratio=0.25,
    )
    normalizer = dataset.get_normalizer()

    torch.testing.assert_close(
        normalizer["obs"].get_input_stats()["mean"],
        torch.from_numpy(reference_buffer["hand_joint"].mean(axis=0)),
        rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        normalizer["action"].get_input_stats()["std"],
        torch.from_numpy(reference_buffer["action"].std(axis=0, ddof=1)),
        rtol=1e-5, atol=1e-6,
    )
```

Also test negative and `index == len(dataset)` raise `IndexError`, temporal and
split mismatch fail in `__init__`, and zero selected training episodes raises a
clear `ValueError`.

- [ ] **Step 2: Confirm dataset import failure**

```bash
pytest tests/test_sim_hand_mmap_dataset.py -v
```

Expected: FAIL because `SimHandMmapDataset` is missing.

- [ ] **Step 3: Implement selected-episode metadata and lazy lookup**

Open arrays using `np.load(path, mmap_mode="r", allow_pickle=False)`. Keep the
global `episode_ends`, `val_mask`, and selected episode indices. Build train mask
with existing `downsample_mask(~val_mask, max_train_episodes, seed)`; the
validation shallow copy selects exactly `val_mask` and does not apply the train
cap.

For a selected episode length `L`, compute:

```python
window_count = max(
    0,
    L - self.horizon + 1 + self.pad_before + self.pad_after,
)
```

Store only `sample_ends = np.cumsum(window_counts, dtype=np.int64)`. Resolve a
logical index with:

```python
episode_position = int(np.searchsorted(self.sample_ends, index, side="right"))
previous_end = 0 if episode_position == 0 else int(self.sample_ends[episode_position - 1])
local_window = index - previous_end
local_start = local_window - self.pad_before
```

Use the selected global episode index to obtain its cached row start/end. Clamp
the source slice to the episode, copy only that small slice into writable NumPy
memory, and edge-fill a `[horizon, 22]` output exactly like `SequenceSampler`.
Return `torch.from_numpy()` tensors. Never return a writable view into the mmap.

- [ ] **Step 4: Rebuild normalizers from cached stats**

Support `mode="limits"` and `mode="gaussian"`; reject other modes and unknown
kwargs. Consume `output_min=-1.0`, `output_max=1.0`, `range_eps=1e-4`, and
`fit_offset=True` with the same formulas as `_fit()` in `normalizer.py`. Build
each field with `SingleFieldLinearNormalizer.create_manual()` and assign it to a
new `LinearNormalizer` under `obs` and `action`. Use the cached min/max/mean/std
as `input_stats_dict`; do not rescan mmap rows.

- [ ] **Step 5: Run dataset, converter, and legacy-equivalence suites**

```bash
pytest tests/test_sim_hand_mmap_dataset.py \
  tests/test_sim_hand_mmap_converter.py \
  tests/test_sim_hand_lowdim_dataset.py -q
```

Expected: mmap tests PASS; legacy tests remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add diffusion_policy/dataset/sim_hand_mmap_dataset.py tests/test_sim_hand_mmap_dataset.py
git commit -m "feat: read Sim-Hand mmap windows lazily"
```

---

### Task 6: Add bounded train and validation samplers

**Files:**

- Create: `diffusion_policy/common/bounded_sampler.py`
- Create: `tests/test_bounded_sampler.py`

**Interfaces produced:**

```text
EpochRandomSampler.__init__(
        self,
        data_source: Sized,
        *,
        num_samples: int,
        seed: int,
        chunk_size: int = 65_536,
    ) -> None
EpochRandomSampler.set_epoch(epoch: int) -> None
EpochRandomSampler.__iter__() -> Iterator[int]
EpochRandomSampler.__len__() -> int

EvenlySpacedSampler.__init__(self, data_source: Sized, *, num_samples: int) -> None
EvenlySpacedSampler.__iter__() -> Iterator[int]
EvenlySpacedSampler.__len__() -> int
```

- [ ] **Step 1: Add failing bounded-sampler tests**

Create `tests/test_bounded_sampler.py`:

```python
def test_epoch_random_sampler_is_exact_in_range_and_epoch_deterministic():
    sampler = EpochRandomSampler(range(17), num_samples=41, seed=9, chunk_size=5)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    second = list(sampler)
    sampler.set_epoch(4)
    third = list(sampler)

    assert len(sampler) == 41
    assert first == second
    assert first != third
    assert all(0 <= index < 17 for index in first + third)
    assert not hasattr(sampler, "indices")


def test_evenly_spaced_sampler_is_fixed_bounded_and_lazy():
    sampler = EvenlySpacedSampler(range(10), num_samples=4)
    assert len(sampler) == 4
    assert list(sampler) == [1, 3, 6, 8]
    assert list(sampler) == [1, 3, 6, 8]
    assert not hasattr(sampler, "indices")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EpochRandomSampler([], num_samples=1, seed=0),
        lambda: EpochRandomSampler(range(2), num_samples=0, seed=0),
        lambda: EpochRandomSampler(range(2), num_samples=1, seed=0, chunk_size=0),
        lambda: EvenlySpacedSampler([], num_samples=1),
        lambda: EvenlySpacedSampler(range(2), num_samples=0),
    ],
)
def test_bounded_samplers_reject_empty_sources_and_nonpositive_sizes(factory):
    with pytest.raises(ValueError):
        factory()
```

- [ ] **Step 2: Confirm import failure**

```bash
pytest tests/test_bounded_sampler.py -v
```

Expected: FAIL because `bounded_sampler` does not exist.

- [ ] **Step 3: Implement lazy generators**

`EpochRandomSampler.__iter__()` creates a local `torch.Generator`, seeds it with
`seed + epoch`, then repeatedly calls `torch.randint` for at most `chunk_size`
values and yields Python integers. It must never concatenate chunks or retain a
generated chunk after yielding it.

`EvenlySpacedSampler.__iter__()` yields midpoint-bin indices without an array:

```python
length = len(self.data_source)
for i in range(self.num_samples):
    yield ((2 * i + 1) * length) // (2 * self.num_samples)
```

Duplicates when `num_samples > len(data_source)` are intentional and preserve
the exact validation budget for small/debug datasets.

- [ ] **Step 4: Run sampler tests**

```bash
pytest tests/test_bounded_sampler.py -q
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/common/bounded_sampler.py tests/test_bounded_sampler.py
git commit -m "feat: add bounded logical-epoch samplers"
```

---

### Task 7: Switch Sim-Hand training and launcher to mmap

**Files:**

- Modify: `diffusion_policy/config/task/sim_hand_lowdim.yaml`
- Modify: `diffusion_policy/config/train_diffusion_unet_sim_hand_workspace.yaml`
- Modify: `diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py`
- Modify: `dp_train_sim_hand.sh`
- Modify: `tests/test_sim_hand_train_smoke.py`
- Modify: `tests/test_sim_hand_hdf5_dataset.py`

**Interfaces consumed:** `SimHandMmapDataset`, `EpochRandomSampler`,
`EvenlySpacedSampler`.

- [ ] **Step 1: Update expected config and launcher behavior first**

In `tests/test_sim_hand_train_smoke.py`, change the Hydra-default assertions to:

```python
assert config.task.dataset._target_.endswith("SimHandMmapDataset")
assert config.training.num_epochs == 10000
assert config.training.steps_per_epoch == 2000
assert config.training.validation_steps == 200
assert config.training.checkpoint_every == 100000
assert config.dataloader.shuffle is False
assert config.dataloader.persistent_workers is True
assert config.dataloader.prefetch_factor == 2
```

Change `_small_cpu_config()` to set both step budgets to 1, both worker counts to
0, both `persistent_workers` to false, and both `prefetch_factor` to null. Replace
the Zarr/eager-HDF smoke fixtures with one HDF fixture followed by:

```python
build_sim_hand_mmap_cache(
    dataset_dir,
    horizon=12,
    pad_before=3,
    pad_after=8,
    val_ratio=0.1,
    seed=42,
    chunk_rows=8,
)
```

Keep one test named
`test_one_step_train_validation_and_periodic_checkpoint_from_mmap`; it must still
assert finite train/validation loss plus `epoch_0001.ckpt` and `latest.ckpt`.

Replace `test_train_script_accepts_hdf5_manifest_dataset` in
`tests/test_sim_hand_hdf5_dataset.py` with:

```python
def test_train_script_requires_completed_mmap_cache(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    result = _run_train_script_with_true_python(tmp_path)
    assert result.returncode != 0
    assert "exp_data_mmap/READY" in result.stderr
    assert "python -m diffusion_policy.scripts.prepare_sim_hand_mmap" in result.stderr


def test_train_script_accepts_ready_mmap_cache(tmp_path):
    cache_dir = tmp_path / "exp_data_mmap"
    cache_dir.mkdir()
    (cache_dir / "READY").touch()
    result = _run_train_script_with_true_python(tmp_path)
    assert result.returncode == 0, result.stderr
```

Factor the existing subprocess environment into
`_run_train_script_with_true_python(dataset_path)`.

- [ ] **Step 2: Run focused tests and observe old-path failures**

```bash
pytest tests/test_sim_hand_train_smoke.py::test_hydra_defaults_have_one_consistent_temporal_configuration \
  tests/test_sim_hand_hdf5_dataset.py::test_train_script_requires_completed_mmap_cache \
  tests/test_sim_hand_hdf5_dataset.py::test_train_script_accepts_ready_mmap_cache -v
```

Expected: FAIL because config and launcher still accept legacy datasets.

- [ ] **Step 3: Update task and training configuration**

Set the task target to:

```yaml
_target_: diffusion_policy.dataset.sim_hand_mmap_dataset.SimHandMmapDataset
```

Use these exact training/DataLoader values:

```yaml
dataloader:
  batch_size: 256
  num_workers: 4
  shuffle: false
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

val_dataloader:
  batch_size: 256
  num_workers: 4
  shuffle: false
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

training:
  num_epochs: 10000
  steps_per_epoch: 2000
  validation_steps: 200
  checkpoint_every: 100000
```

Leave all policy temporal values, optimizer values, and diagnostic
`max_train_steps`/`max_val_steps` unchanged.

- [ ] **Step 4: Wire bounded samplers into the workspace**

In `__init__`, reject nonpositive `steps_per_epoch` and `validation_steps` next
to the checkpoint interval validation.

In `run()`, replace direct DataLoader construction with:

```python
train_samples = int(cfg.training.steps_per_epoch) * int(cfg.dataloader.batch_size)
train_sampler = EpochRandomSampler(
    dataset,
    num_samples=train_samples,
    seed=int(cfg.training.seed),
)
train_dataloader = DataLoader(dataset, sampler=train_sampler, **cfg.dataloader)

val_dataset = dataset.get_validation_dataset()
val_samples = (
    int(cfg.training.validation_steps)
    * int(cfg.val_dataloader.batch_size)
)
val_sampler = EvenlySpacedSampler(val_dataset, num_samples=val_samples)
val_dataloader = DataLoader(
    val_dataset,
    sampler=val_sampler,
    **cfg.val_dataloader,
)
```

`shuffle: false` is deliberately still passed by each config and is compatible
with an explicit sampler. At the top of every logical-epoch loop, before
`_train_epoch`, call:

```python
train_sampler.set_epoch(self.epoch)
```

This makes resumed epoch `k` generate the same window stream as uninterrupted
epoch `k`. The scheduler continues to use `len(train_dataloader) * num_epochs`.

- [ ] **Step 5: Require the completion marker in Bash**

Replace the Zarr/manifest condition in `dp_train_sim_hand.sh` with:

```bash
CACHE_READY="$DATASET_PATH/exp_data_mmap/READY"
if [[ ! -f "$CACHE_READY" ]]; then
    echo "Completed Sim-Hand mmap cache not found: $CACHE_READY" >&2
    echo "Build it first:" >&2
    echo "$PYTHON -m diffusion_policy.scripts.prepare_sim_hand_mmap --dataset-path $DATASET_PATH" >&2
    exit 1
fi
```

Continue passing the raw dataset root through `task.dataset_path`.

- [ ] **Step 6: Run config, launcher, and one-step CPU smoke tests**

```bash
pytest tests/test_sim_hand_hdf5_dataset.py::test_train_script_requires_completed_mmap_cache \
  tests/test_sim_hand_hdf5_dataset.py::test_train_script_accepts_ready_mmap_cache \
  tests/test_sim_hand_train_smoke.py -q
```

Expected: all tests PASS. The smoke test processes one train and one validation
batch, logs finite loss, and writes both checkpoints.

- [ ] **Step 7: Commit**

```bash
git add diffusion_policy/config/task/sim_hand_lowdim.yaml \
  diffusion_policy/config/train_diffusion_unet_sim_hand_workspace.yaml \
  diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py \
  dp_train_sim_hand.sh \
  tests/test_sim_hand_train_smoke.py \
  tests/test_sim_hand_hdf5_dataset.py
git commit -m "feat: train Sim-Hand from bounded mmap data"
```

---

### Task 8: Regression verification and real-data cache build

**Files:**

- No repository edits expected.
- Create outside the repository:
  `/home/carus/Data/exp_data/exp_data_mmap/`

- [ ] **Step 1: Run the complete related regression suite**

```bash
pytest tests/test_sim_hand_hdf5_dataset.py \
  tests/test_sim_hand_lowdim_dataset.py \
  tests/test_sim_hand_mmap_cache.py \
  tests/test_sim_hand_mmap_converter.py \
  tests/test_sim_hand_mmap_dataset.py \
  tests/test_bounded_sampler.py \
  tests/test_sim_hand_train_smoke.py -q
```

Expected: all tests PASS. Existing Zarr-v3 failures outside these files are not
part of this mmap change; do not broaden scope unless one of the named related
tests fails.

- [ ] **Step 2: Record source metadata and confirm disk capacity**

```bash
df -h /home/carus/Data/exp_data
find /home/carus/Data/exp_data/manifest.json /home/carus/Data/exp_data/shards \
  -type f -printf '%p\t%s\t%T@\n' | sort \
  > /tmp/sim-hand-source-before.txt
sha256sum /home/carus/Data/exp_data/manifest.json
```

Require at least 20 GiB free on the filesystem holding `exp_data`; this covers
the approximately 16.4-GiB final arrays, the temporary seen bitmap, NumPy
headers, and safety margin. If `exp_data_mmap` already exists and is invalid,
stop and report it; do not remove or overwrite it.

- [ ] **Step 3: Run the explicit real-data conversion**

```bash
/home/carus/miniforge3/envs/isaaclab/bin/python \
  -m diffusion_policy.scripts.prepare_sim_hand_mmap \
  --dataset-path /home/carus/Data/exp_data \
  --horizon 12 \
  --pad-before 3 \
  --pad-after 8 \
  --val-ratio 0.1 \
  --seed 42 \
  --chunk-rows 262144
```

Expected terminal line:

```text
Sim-Hand mmap cache ready: /home/carus/Data/exp_data/exp_data_mmap
```

This reads the HDF5 shards twice and can take substantial time; continue waiting
for the converter rather than starting training against its temp directory.

- [ ] **Step 4: Verify source untouched and final cache counts**

```bash
find /home/carus/Data/exp_data/manifest.json /home/carus/Data/exp_data/shards \
  -type f -printf '%p\t%s\t%T@\n' | sort \
  > /tmp/sim-hand-source-after.txt
diff -u /tmp/sim-hand-source-before.txt /tmp/sim-hand-source-after.txt
du -sh /home/carus/Data/exp_data/exp_data_mmap
```

Then run:

```bash
/home/carus/miniforge3/envs/isaaclab/bin/python - <<'PY'
import json
from pathlib import Path

import numpy as np

from diffusion_policy.dataset.sim_hand_mmap_cache import validate_sim_hand_mmap_cache
from diffusion_policy.dataset.sim_hand_mmap_dataset import SimHandMmapDataset

root = Path("/home/carus/Data/exp_data")
cache = root / "exp_data_mmap"
info = validate_sim_hand_mmap_cache(
    cache,
    dataset_path=root,
    horizon=12,
    pad_before=3,
    pad_after=8,
    val_ratio=0.1,
    seed=42,
)
metadata = json.loads((cache / "metadata.json").read_text())
assert info.n_steps == 100_000_348
assert info.n_episodes == 159_574
assert metadata["counts"]["dropped_episodes"] == 4
assert metadata["counts"]["dropped_transitions"] == 44
dataset = SimHandMmapDataset(str(root))
assert len(dataset) == 90_133_195
validation = dataset.get_validation_dataset()
assert len(validation) == 9_867_153
for sample in (dataset[0], dataset[len(dataset) - 1], validation[0]):
    assert sample["obs"].shape == (12, 22)
    assert sample["action"].shape == (12, 22)
    assert np.isfinite(sample["obs"].numpy()).all()
    assert np.isfinite(sample["action"].numpy()).all()
print("real Sim-Hand mmap cache validated")
PY
```

Expected: source snapshot diff is empty, assertions pass, and the final line is
`real Sim-Hand mmap cache validated`.

- [ ] **Step 5: Inspect the final repository delta**

```bash
git status --short
git log --oneline -8
```

Expected: only pre-existing user/untracked files remain unstaged; the mmap cache
is outside the repository and is not committed.

## Final Self-Review Checklist

- [ ] Search the implementation and this plan for unresolved placeholders:

```bash
SIM_HAND_PLACEHOLDER_PATTERN='TO''DO|TB''D|FIX''ME|similar'' to|handle'' appropriately'
rg -n "$SIM_HAND_PLACEHOLDER_PATTERN" \
  diffusion_policy/dataset/sim_hand_mmap_cache.py \
  diffusion_policy/dataset/sim_hand_mmap_converter.py \
  diffusion_policy/dataset/sim_hand_mmap_dataset.py \
  diffusion_policy/common/bounded_sampler.py \
  diffusion_policy/scripts/prepare_sim_hand_mmap.py \
  docs/superpowers/plans/2026-08-27-sim-hand-mmap-dataset.md
```

Expected: no matches.

- [ ] Confirm every spec area has a test or operational check: source read-only,
  two-pass bounded conversion, atomic `READY`, stale-cache refusal, exact cache
  schema, lazy window/padding equivalence, all-row normalization, disjoint
  split, metadata-only train cap, bounded random sampler, fixed validation
  sampler, finite scheduler horizon, launcher gate, CPU smoke, and real-data
  counts.

- [ ] Confirm type consistency end to end: obs/action float32; episode ends/IDs
  and steps int64; env IDs int32; validation mask bool; statistics float32 after
  float64 accumulation; Python logical indices returned by samplers.
