# Sim-Hand Multiple Lazy-Cache Design

## Purpose

Add a separately labeled `sim_hand_multiple` training mode that recursively
discovers every DexGen HDF5 rollout below a `sim_data` root and trains one
ordinary Sim-Hand policy from the combined logical dataset. This mode does not
use the mixed mode's 8:2 probabilities, `source_id`, or source-fraction logs.

The current development dataset contains seven manifests, 707 HDF5 shards,
700,004,366 declared transitions, and about 586 GiB of source files. A unified
in-memory replay buffer would exceed 500 GiB at peak in one process and would
be duplicated by every DDP rank. The multiple mode therefore uses a reusable
disk cache, read-only memory maps, formula-based window indexing, and a bounded
uniform sampler.

Existing `sim_hand_lowdim`, `sim_hand_mixed`, their launchers, and their data
contracts remain unchanged.

## Source discovery and ordering

`dataset_path` points at the common root, for example:

```text
/share/generalvision/xiejunzhe/Data/sim_data
├── brush1/20260831204042/manifest.json
├── bulb2/20260827175545/manifest.json
├── bunny07_1/20260901010749/manifest.json
├── cuboid2/20260901002008/manifest.json
├── cylinder2/20260831211529/manifest.json
├── hammer6/20260831171138/manifest.json
└── screwdriver6/20260831231724/manifest.json
```

Discovery recursively finds regular files named `manifest.json` and sorts
their POSIX paths relative to `dataset_path`. Current and future leaves are
included automatically. Cache directories do not contain files named
`manifest.json`, so they cannot become accidental sources.

Every manifest is parsed and validated independently. All manifest, JSON,
schema, shard-path, dtype, shape, row-count, and episode-step errors name the
specific manifest or shard path. Episode identity is local to a leaf: equal
`episode_id` and `env_id` pairs in different leaves remain separate episodes.

## Cache layout

The default cache root is:

```text
<dataset_path>/.sim_hand_multiple_cache/v1/
```

It can be overridden with `task.cache_path`. The default stays beside the
source because the checked `/tmp` filesystem is too small and the shared data
filesystem has sufficient capacity. The expected main-array size for the
current source is about 172 GiB:

```text
700,004,366 rows * 3 fields * 22 float32 channels
```

Each source leaf maps to an immutable fingerprinted directory:

```text
v1/
├── build.lock
└── sources/
    └── <relative-path-digest>-<source-fingerprint>/
        ├── metadata.json
        ├── qpos.npy
        ├── target_before.npy
        ├── action.npy
        ├── episode_ends.npy
        ├── episode_ids.npy
        ├── env_ids.npy
        ├── obs_min.npy
        ├── obs_max.npy
        ├── obs_sum.npy
        ├── obs_sum_squares.npy
        ├── action_min.npy
        ├── action_max.npy
        ├── action_sum.npy
        ├── action_sum_squares.npy
        └── READY
```

The source fingerprint covers the cache format version, relative leaf path,
manifest bytes, and every declared shard's relative path, byte size, and
nanosecond modification time. A source change produces a new immutable cache
directory and leaves the old cache unused. Cache metadata records all array
shapes, dtypes, counts, source path, fingerprint, and format version.

Cache arrays keep every reconstructed episode, including those shorter than
the later training horizon. Horizon filtering, train/validation splitting, and
normalization happen in the lazy dataset so one cache can serve multiple
horizon settings. `get_all_actions()` is not used by Sim-Hand training and
must not materialize the full action tensor.

The three transition arrays contain episode-contiguous data in deterministic
`(episode_id, env_id, step)` order. They store 44 observation source channels
instead of the derived 66-D observation:

```text
qpos          float32 [N, 22]
target_before float32 [N, 22]
action        float32 [N, 22]  # target_after
```

The dataset derives each observation window at read time:

```python
obs = concatenate((qpos, target_before, target_before - qpos), axis=-1)
```

Per-episode statistics allow normalization to exclude episodes shorter than
the configured horizon without scanning the transition files at every process
startup. Sums and squared sums use float64; minima and maxima use float32.

## Cache construction and concurrency

`ensure_sim_hand_multiple_cache()` discovers sources, validates fingerprints,
and builds only missing caches. Construction acquires an exclusive `fcntl`
lock at `build.lock`, then checks again after acquiring it. This makes direct
single-process use and concurrent DDP startup safe.

One leaf is loaded and transformed at a time through the existing validated
single-leaf HDF5 path. Its cache is written to a unique temporary directory
under `sources/`. The source fingerprint is recomputed after writing; a source
that changed during construction aborts with a clear error. `metadata.json`
and `READY` are written last, all files are flushed, and the temporary
directory is atomically renamed to its final fingerprinted name. Incomplete
temporary directories are ignored by readers.

Before building, the cache builder compares available bytes with the declared
main-array requirement plus a ten-percent margin and raises an actionable
error when space is insufficient. It never edits or deletes source data.

`build_sim_hand_multiple_cache.py` exposes the same operation as a CLI. The
multiple launcher runs this preflight once before starting `torchrun`, so GPU
processes do not sit idle while a first cache is built. Dataset construction
also calls the idempotent ensure function, which covers direct Hydra use and
source leaves added after the previous run.

`REBUILD_CACHE=1` / `task.rebuild_cache=true` builds fresh fingerprinted cache
directories even when valid ones exist. Generated cache data is recoverable
from source; rebuild never removes source files.

## Lazy dataset and window mapping

Create `LazySimHandMultipleDataset(BaseLowdimDataset)`. It receives:

```python
LazySimHandMultipleDataset(
    dataset_path: str,
    cache_path: str | None,
    horizon: int,
    pad_before: int,
    pad_after: int,
    seed: int = 42,
    val_ratio: float = 0.1,
    max_train_episodes: int | None = None,
    samples_per_epoch: int | None = None,
    rebuild_cache: bool = False,
)
```

The constructor opens transition arrays with `numpy.load(..., mmap_mode="r")`.
Multiple ranks and DataLoader workers map the same immutable files; the kernel
page cache is shared and the transition arrays are not copied into each
process heap.

Episode metadata from all leaves is concatenated in source-path order.
Episodes shorter than `horizon` are excluded before splitting, matching the
existing Sim-Hand HDF5 behavior. A single global `get_val_mask` call splits
eligible episodes, and `downsample_mask` applies only to the training mask.
The validation mask is preserved independently of training downsampling.

The dataset does not allocate `SequenceSampler.indices`. For each eligible
episode it stores only the count of legal padded windows and a cumulative
window-count array. Given a logical sample index, binary search selects the
episode and arithmetic reconstructs the same four values used by
`SequenceSampler`:

```text
buffer_start, buffer_end, sample_start, sample_end
```

The selected contiguous slices are read from `qpos.npy`,
`target_before.npy`, and `action.npy`; padding and the 66-D observation are
then constructed in a small per-sample array. Windows never cross episode or
source boundaries.

`get_validation_dataset()` returns a shallow dataset view indexed by the
original validation mask. `get_normalizer()` merges cached statistics across
every eligible episode, including both training and validation episodes, and
returns the same `LinearNormalizer` contract as `SimHandLowdimDataset`.

## Uniform non-mixed training sampler

Ordinary PyTorch shuffle and `DistributedSampler` allocate structures
proportional to roughly 700 million windows. The lazy dataset therefore
provides a `ChunkedUniformSampler` through the existing optional
`get_training_sampler()` hook.

The sampler selects a logical window uniformly from the entire training index
space, with replacement, and emits indices in bounded chunks. It does not
choose a source first and has no source weights; a leaf's probability is
exactly proportional to its number of training windows. It emits no
`source_id` and produces no mixed-source metrics.

```python
ChunkedUniformSampler(
    dataset_length: int,
    samples_per_epoch: int,
    seed: int,
    num_replicas: int = 1,
    rank: int = 0,
    chunk_size: int = 65_536,
)
```

`samples_per_epoch=null` means the complete training-window count. Each rank
emits `ceil(samples_per_epoch / num_replicas)` independent deterministic
draws, seeded by `(seed, epoch, rank)`. `set_epoch()` changes the stream.
Validation remains sequential and does not use this sampler.

## Configuration and launcher

Add `diffusion_policy/config/task/sim_hand_multiple.yaml` with:

```yaml
name: sim_hand_multiple
dataset_path: data/sim_data
cache_path: null
samples_per_epoch: null
rebuild_cache: false
```

Its dataset target is `LazySimHandMultipleDataset`. Observation/action sizes,
temporal values, policy, optimizer, workspace, checkpointing, and logging are
unchanged.

`dp_train_sim_hand_multiple.sh`:

- defaults `DATASET_PATH` to the repository sibling `Data/sim_data`;
- defaults `CACHE_PATH` to `<DATASET_PATH>/.sim_hand_multiple_cache/v1`;
- validates at least one recursively nested manifest;
- builds or validates caches once before launching training;
- supports `NUM_GPUS` exactly like the existing Sim-Hand launchers;
- passes `task=sim_hand_multiple`, `exp_name=multiple`, dataset/cache paths,
  rebuild flag, W&B mode, and user overrides.

## Failure behavior

- No recursive manifests: fail before cache allocation and name the root.
- Invalid JSON/schema/shard: fail and name the manifest or shard.
- Insufficient cache disk space: report required, available, and cache path.
- Changed source during build: discard the incomplete temporary output and
  report the source path.
- Missing, truncated, wrongly shaped, or wrongly typed cache file: reject the
  cache and rebuild when allowed; otherwise report the exact cache file.
- No episode at least as long as `horizon`: fail with the horizon and root.
- Empty training or validation window space: fail before starting the model
  loop with the relevant split name.

## Tests

Tests use tiny real HDF5 shards and real memory-mapped cache files. They cover:

- deterministic recursive discovery and path-specific manifest errors;
- duplicate episode IDs across leaves;
- cache field mapping, fingerprint reuse, invalidation, atomic readiness, and
  insufficient-space diagnostics;
- lazy window indexing, padding, episode boundaries, train/validation masks,
  maximum training episodes, and normalizer equivalence to eager data;
- bounded deterministic uniform sampling in single-process and DDP layouts;
- Hydra composition and launcher preflight for single- and multi-GPU modes;
- one-step CPU training with finite train/validation losses, no source-fraction
  keys, JSON logging, and checkpoint creation;
- regression tests for existing single-source and mixed modes.

The production cache build is not run in tests. A read-only preflight command
reports the seven discovered sources, declared transition total, estimated
cache bytes, cache status, and available disk space without reading all shard
payloads.
