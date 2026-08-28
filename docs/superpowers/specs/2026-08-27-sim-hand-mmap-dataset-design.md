# Sim-Hand Memory-Mapped Dataset Design

## Purpose

Replace the eager HDF5 training path for the 100-million-transition Sim-Hand
dataset with a one-time, memory-bounded conversion and a read-only NumPy mmap
cache. The raw rollout remains unchanged at:

```text
/home/carus/Data/exp_data/
├── manifest.json
└── shards/*.h5
```

The converter writes the training cache inside that dataset directory:

```text
/home/carus/Data/exp_data/exp_data_mmap/
```

Training must never build this cache implicitly. A user runs the converter once,
verifies its completion, and then starts training against the completed cache.

## Motivation

The source manifest currently declares 100,000,392 transitions across 101 HDF5
shards. The existing loader allocates all ten required HDF5 fields, globally
sorts every row by `(episode_id, env_id, step)`, creates reordered copies of the
two training arrays, and then materializes one four-integer `SequenceSampler`
index record per training window. The default split produces approximately 90
million training windows and 10 million validation windows. These eager arrays,
sort temporaries, window indices, and DataLoader shuffle permutation create an
unnecessarily high host-memory peak.

The policy consumes only two 22-dimensional float32 fields:

```text
observation <- robot/qpos
action      <- robot/target_after
```

After filtering the four episodes shorter than the 12-step horizon, storing both
arrays without compression requires approximately 16.4 GiB. This fits the host's
page cache and does not need to be copied into the Python heap.

## Goals

- Keep the source manifest and HDF5 shards read-only.
- Bound converter memory independently of the number of transitions.
- Materialize only the fields and metadata required by Sim-Hand training.
- Preserve the current episode reconstruction and temporal padding semantics.
- Avoid an `O(number_of_windows)` in-memory index and global shuffle
  permutation.
- Make incomplete, stale, or mismatched caches fail before training begins.
- Preserve the current deterministic episode split and normalization behavior.
- Define finite, configurable logical epochs for this large dataset.

## Non-goals

- Deleting or rewriting fields in the raw HDF5 shards.
- Supporting remote or multi-node streaming.
- Introducing RLDS, Parquet, Arrow, or WebDataset.
- Changing the policy, diffusion loss, observation/action convention, joint
  ordering, units, or guided-inference behavior.
- Interpreting `done`, `success`, `failure`, `timeout`, or `reset_reason` values.
- Precomputing and storing every 12-step training window.

## Cache layout

The completed cache contains:

```text
exp_data_mmap/
├── metadata.json
├── obs.npy
├── action.npy
├── episode_ends.npy
├── episode_ids.npy
├── env_ids.npy
├── val_mask.npy
├── normalizer.npz
└── READY
```

The files have the following contracts:

| File | Dtype and shape | Meaning |
| --- | --- | --- |
| `obs.npy` | `float32 [N, 22]` | Episode-contiguous `robot/qpos` |
| `action.npy` | `float32 [N, 22]` | Episode-contiguous `robot/target_after` |
| `episode_ends.npy` | `int64 [E]` | Exclusive cumulative episode ends |
| `episode_ids.npy` | `int64 [E]` | Source episode ID for each cached episode |
| `env_ids.npy` | `int32 [E]` | Source environment ID for each cached episode |
| `val_mask.npy` | `bool [E]` | Deterministic held-out episode mask |
| `normalizer.npz` | fixed 22-D arrays | Min, max, mean, and std for obs/action |
| `metadata.json` | JSON object | Format, source, temporal, and count metadata |
| `READY` | empty marker | Written last; cache may be opened for training |

`N` is the number of retained transitions and `E` is the number of retained
episodes. `obs.npy` and `action.npy` are standard NumPy `.npy` files created with
`numpy.lib.format.open_memmap` and opened for training with
`numpy.load(..., mmap_mode="r")`.

`metadata.json` includes:

- `format_name` and `format_version`;
- SHA-256 of the source `manifest.json`;
- source `schema_version`, `dof`, and declared transition count;
- retained and dropped transition/episode counts;
- `horizon`, `pad_before`, and `pad_after` used for conversion;
- validation ratio and seed;
- array filenames, dtypes, and shapes;
- normalizer convention and row count.

The cache is self-contained after creation. Training compares the current source
manifest hash with the recorded hash when the manifest is available. A missing
source shard does not invalidate an otherwise complete cache, but a changed
manifest does.

## Offline conversion command

Add an explicit converter:

```bash
python -m diffusion_policy.scripts.prepare_sim_hand_mmap \
    --dataset-path /home/carus/Data/exp_data
```

The default output is `<dataset-path>/exp_data_mmap`. The command accepts the
temporal values needed to define cache compatibility, defaulting to the training
configuration:

```text
horizon=12
pad_before=3
pad_after=8
val_ratio=0.1
seed=42
```

The converter never overwrites an existing output directory. If a valid matching
cache already exists, it reports that fact and exits successfully. If the output
is incomplete, invalid, or stale, it exits with a clear instruction to move the
old directory aside before rebuilding.

### Atomic creation

Conversion writes to a unique sibling directory such as:

```text
/home/carus/Data/exp_data/.exp_data_mmap.tmp-<uuid>/
```

It flushes all memmaps, writes metadata, writes `READY` last, closes all file
handles, and atomically renames the temporary directory to `exp_data_mmap` on the
same filesystem. An exception removes only that invocation's temporary
directory. It never removes or modifies the source dataset or an existing cache.

## Memory-bounded conversion algorithm

Conversion uses two streaming passes over the source shards.

### Pass 1: episode inventory

1. Read and validate the manifest.
2. For every declared shard, validate the existing Sim-Hand required field
   presence, dtype, shape, and row count without loading whole datasets.
3. Read `episode_id` and `env_id` in bounded chunks and accumulate one count per
   `(episode_id, env_id)` pair.
4. Sort the distinct pairs lexicographically, matching the current loader's
   episode ordering.
5. Drop pairs whose row count is smaller than `horizon`.
6. Compute destination offsets and exclusive `episode_ends` for retained pairs.
7. Generate the validation mask with the existing seed and validation-ratio
   algorithm.

Only `O(E)` counts and metadata remain in memory. No transition-sized index is
retained between chunks.

### Pass 2: scatter into episode-contiguous arrays

1. Create destination `obs.npy` and `action.npy` memmaps at their final shapes.
2. Create a temporary one-byte-per-row `seen` memmap used only during
   conversion.
3. For each bounded source chunk, map `(episode_id, env_id)` to its retained
   episode offset using vectorized lookup against the sorted pair table.
4. For retained rows, compute `destination_row = episode_offset + step`.
5. Reject a negative or out-of-range step, duplicate destination row, non-finite
   observation, or non-finite action before writing that chunk.
6. Scatter `robot/qpos` and `robot/target_after` to their destination rows and
   mark those rows as seen.
7. Accumulate float64 min, max, mean, and sample standard deviation statistics
   for both fields while streaming.
8. At the end, scan `seen` sequentially and reject any missing destination row.
9. Delete the temporary `seen` file before publishing the cache.

This duplicate/missing check is the streaming equivalent of requiring each
episode's sorted steps to equal `0..length-1`; it retains the current structural
validation without a 100-million-row in-memory sort.

Normalizer statistics use all retained cached rows, including validation rows,
to preserve current `SimHandLowdimDataset.get_normalizer()` behavior. The cache
statistics become authoritative at training time, so every restart avoids a full
data scan.

## Memory-mapped dataset

Add a dedicated `SimHandMmapDataset` rather than complicating the existing Zarr
and eager-HDF5 code path. It:

- validates `READY`, metadata, file shapes, dtypes, and source fingerprint;
- opens `obs.npy` and `action.npy` read-only with `mmap_mode="r"`;
- keeps only episode metadata, selected-episode arrays, and cumulative sample
  counts in Python memory;
- returns the existing training interface:

```python
{
    "obs": Tensor[horizon, 22],
    "action": Tensor[horizon, 22],
}
```

- reconstructs the `LinearNormalizer` from `normalizer.npz`;
- creates a validation view by selecting `val_mask` episodes;
- supports `max_train_episodes` by downsampling episode metadata before sample
  counts are built, without loading or copying transition arrays.

### Lazy window lookup

For each selected episode of length `L`, the number of valid padded windows is:

```text
max(0, L - horizon + 1 + pad_before + pad_after)
```

The dataset stores cumulative per-episode window counts, not one record per
window. Given a logical window index, `searchsorted` identifies the episode and
local start. The dataset then reads only the clamped contiguous slice from both
memmaps and applies the same edge-value padding as `SequenceSampler`.

Under the default `horizon=12`, `pad_before=3`, and `pad_after=8`, every retained
episode contributes exactly one logical window per transition, but the formula
and tests must remain general.

## Bounded sampling and logical epochs

Passing a 90-million-element map-style dataset to `DataLoader(shuffle=True)`
would still request a full random permutation. Training therefore uses an
epoch-seeded lazy sampler with replacement:

- it emits random logical window indices in small tensor chunks;
- it never materializes all emitted indices;
- `seed + epoch` makes each logical epoch reproducible;
- its length is exactly `steps_per_epoch * batch_size`;
- the workspace calls `set_epoch(current_epoch)` before each training epoch;
- DataLoader receives this sampler with `shuffle=False`.

Validation uses a deterministic lazy sampler that selects evenly spaced logical
indices across the validation view. Its length is
`validation_steps * validation_batch_size`, so validation cost is bounded and
the same windows are compared across epochs.

The first implementation uses statistically uniform random-window sampling.
Because the compact 16.4-GiB cache fits the host page cache, locality-aware batch
sampling is deferred unless profiling shows data loading stalls the GPU.

## Training defaults

The Sim-Hand workspace gains explicit finite-data-pipeline settings:

```yaml
training:
  num_epochs: 10000
  steps_per_epoch: 2000
  validation_steps: 200
  checkpoint_every: 100000
```

With `batch_size: 1024`, this produces 2,048,000 sampled training windows per
logical epoch and 20,000,000 optimizer steps over the default run. Validation
evaluates a fixed 204,800-window subset each epoch. Periodic checkpoints land
every 100,000 optimizer steps (`step_XXXXXXXX.ckpt`), not every logical epoch.
Existing debug limits still take precedence and checkpoint/resume occurs at
logical epoch boundaries after the step budget is checked.

The learning-rate scheduler uses the bounded DataLoader length and therefore
receives the correct 20,000,000-step training horizon. `max_train_steps` and
`max_val_steps` remain supported as smaller per-epoch diagnostic limits.

## Launcher behavior

`dp_train_sim_hand.sh` continues to accept the raw dataset root through
`DATASET_PATH`, for example:

```bash
DATASET_PATH=/home/carus/Data/exp_data \
PYTHON=/home/carus/miniforge3/envs/isaaclab/bin/python \
bash dp_train_sim_hand.sh
```

Before launching Python, it requires:

```text
$DATASET_PATH/exp_data_mmap/READY
```

If the cache is absent, it prints the exact converter command and exits. The task
configuration targets `SimHandMmapDataset`; it does not silently fall back to the
eager HDF5 loader.

## DataLoader behavior

The mmap files are opened read-only. Linux workers share the same file-backed
pages through the kernel page cache; they do not each own a 16.4-GiB copy.
Training keeps a modest number of workers and prefetch batches:

```yaml
dataloader:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

These are starting values, not correctness requirements. Throughput profiling
may reduce workers if IPC overhead exceeds the cost of slicing this low-dimensional
dataset. The validation loader follows the same file-backed behavior.

## Error handling

Conversion fails without publishing a cache on:

- invalid manifest or shard inventory;
- missing required field, wrong dtype, shape, or row count;
- source total mismatch;
- duplicate, missing, negative, or out-of-range episode step;
- non-finite `qpos` or `target_after`;
- zero retained episodes;
- an existing invalid or stale output directory;
- insufficient disk space or any write failure.

Training fails before dataset iteration on:

- missing `READY` or metadata;
- unsupported cache format version;
- manifest fingerprint mismatch;
- wrong array filename, dtype, shape, or byte size;
- inconsistent episode metadata or split mask;
- temporal values that differ from cache metadata;
- non-positive `steps_per_epoch` or `validation_steps`.

## Testing

### Converter tests

- Reconstruct interleaved episodes across multiple source shards.
- Preserve lexicographic `(episode_id, env_id)` order and ascending steps.
- Drop only episodes shorter than `horizon`.
- Reject duplicates, missing steps, invalid fields, non-finite values, and total
  mismatches.
- Produce deterministic train/validation masks and correct streaming statistics.
- Publish `READY` only after every cache artifact is complete.
- Leave source shards unchanged on success and failure.
- Refuse to overwrite an existing stale or incomplete cache.

### Dataset and sampler tests

- mmap arrays are opened read-only and samples match `SequenceSampler` on a
  small reference dataset, including both boundary-padding cases.
- Train and validation episodes are disjoint.
- `max_train_episodes` changes only the selected metadata.
- Dataset memory structures scale with episode count, not transition/window
  count.
- Lazy random sampler produces in-range indices, exact configured length, and
  deterministic sequences per seed/epoch without a full permutation.
- Validation sampler is fixed, bounded, and deterministic.
- Cached normalizer matches converter statistics.

### Integration tests

- A one-step CPU training smoke test reads the mmap cache and produces finite
  train/validation losses and checkpoints.
- Launcher accepts a complete cache and rejects a missing/incomplete one with the
  converter command in its error message.
- Existing HDF5 and Zarr dataset tests remain available as legacy-path coverage;
  the Sim-Hand launcher no longer selects those paths.

## Acceptance criteria

- The raw HDF5 dataset is byte-for-byte untouched by conversion and training.
- Conversion peak memory is bounded by chunk size plus `O(E)` metadata.
- The completed cache lives at `exp_data/exp_data_mmap` and is approximately
  16.4 GiB for the current retained dataset.
- Training startup performs no global transition sort, full normalizer scan,
  transition-array copy, all-window index construction, or all-window shuffle.
- Train and validation batches retain the existing `(obs, action)` alignment and
  temporal padding behavior.
- Logical epoch and validation work are finite, deterministic, configurable, and
  represented correctly in LR scheduling and checkpoint resume.
