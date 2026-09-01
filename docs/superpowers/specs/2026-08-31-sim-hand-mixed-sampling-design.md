# Sim-Hand Mixed-Source Sampling Design

## Purpose

Add a new Sim-Hand training mode that samples from `exp_data` and
`bulb_tac_80` with probabilities `0.8` and `0.2`. The source datasets remain
unchanged on disk and retain their complete original cardinalities. Mixing is
implemented only by the training sampler; validation is not probability
reweighted.

The existing single-source `sim_hand_lowdim` mode and its launcher remain
available with unchanged behavior. The new mode is named `sim_hand_mixed` and
is launched separately.

## Confirmed source data

The production configuration uses this root:

```text
/home/carus/mnt/bigai/Data
├── exp_data
│   ├── manifest.json
│   └── shards/*.h5
└── bulb_tac_80
    ├── episode_0
    │   ├── state.npy
    │   └── action.npy
    └── episode_79
        ├── state.npy
        └── action.npy
```

The current development host has an equivalent read-only test mirror at
`/share/generalvision/xiejunzhe/Data`. The checked mirror contains:

- `exp_data`: 101 HDF5 shards and 100,000,392 declared transitions;
- `bulb_tac_80`: 80 episodes and exactly 44,142 source steps;
- each bulb episode has equal `state.npy` and `action.npy` lengths;
- every inspected state/action array is `float32` and finite.

Tests and local smoke commands override `data_root` to the mirror. The checked
production default remains `/home/carus/mnt/bigai/Data`.

## Scope

### In scope

- Direct, read-only loading of bulb TacMP trajectory directories.
- Conversion of bulb hand state/action fields to the existing Sim-Hand 66-D
  observation and 22-D absolute-action contract.
- A composite low-dimensional dataset that preserves both component datasets.
- A memory-bounded, deterministic, source-probability training sampler.
- Single-process and Distributed Data Parallel (DDP) training.
- Training logs that report realized source fractions.
- A new mixed task configuration and launcher.
- Unit, configuration, sampler, dataset, and bounded training smoke tests.

### Out of scope

- Rewriting, deleting, downsampling, or converting either source dataset.
- Copying bulb data to HDF5/Zarr or merging both sources into a new replay
  buffer.
- Using bulb images, tactile arrays, torque, or object state.
- Changing the diffusion policy, temporal convention, optimizer, or inference
  behavior.
- Applying the 8:2 ratio to validation or normalization statistics.
- Removing the existing single-source training path.

## Unified trajectory contract

Both component datasets presented to the policy return:

```text
obs:    float32 [horizon, 66]
action: float32 [horizon, 22]
```

The 66-D observation at time `t` is:

```text
concat(qpos[t], target_before[t], target_before[t] - qpos[t])
```

The 22-D action is `target_after[t]`.

### Existing expdata mapping

The existing HDF5 loader keeps its current mapping:

```text
qpos          = robot/qpos
target_before = robot/target_before
target_after  = robot/target_after
```

Observed data confirms that, within an expdata episode,
`target_before[t] == target_after[t - 1]` for `t > 0`.

### New bulb_tac mapping

TacMP stores 31 channels as 9 arm-pose channels followed by 22 hand channels.
Only the hand trajectory is used:

```python
qpos = state[:, 9:31]
target_after = action[:, 9:31]
target_before = np.concatenate(
    (qpos[0:1], target_after[:-1]),
    axis=0,
)
obs = np.concatenate(
    (qpos, target_before, target_before - qpos),
    axis=-1,
)
```

This mapping was explicitly approved. The first step has no preceding command,
so its `target_before` is initialized from the first measured `qpos`.

Episode directories are ordered by the integer suffix of `episode_<integer>`,
not lexicographically. Every episode remains a separate ReplayBuffer episode;
windows never cross episode boundaries.

## Components and interfaces

### BulbTacLowdimDataset

Create `diffusion_policy/dataset/bulb_tac_lowdim_dataset.py` with a
`BulbTacLowdimDataset(BaseLowdimDataset)` implementation.

Constructor contract:

```python
BulbTacLowdimDataset(
    dataset_path: str,
    horizon: int,
    pad_before: int,
    pad_after: int,
    seed: int = 42,
    val_ratio: float = 0.1,
)
```

It loads `state.npy` and `action.npy` from every numeric episode directory,
applies the approved mapping, and creates an in-memory ReplayBuffer. The real
bulb source is small enough for this representation (44,142 transitions in the
checked mirror). It does not load or write any other modality.

Train/validation masks use the same episode-level `get_val_mask` convention as
`SimHandLowdimDataset`. The loader never calls `downsample_mask`; all non-val
episodes remain in the training component. A source episode shorter than the
requested horizon is rejected with a clear error rather than silently removed.

### MixedLowdimDataset

Create `diffusion_policy/dataset/mixed_lowdim_dataset.py` with a composite
`MixedLowdimDataset(BaseLowdimDataset)`.

The composite receives an ordered mapping of named component datasets:

```text
expdata  -> SimHandLowdimDataset
bulb_tac -> BulbTacLowdimDataset
```

Its logical index space is the normal concatenation of both component index
spaces. `len(mixed)` is exactly the sum of component window counts;
`mixed[index]` maps to one unchanged component sample. A mixed sample adds a
small integer `source_id` metadata tensor. The workspace removes this metadata
before invoking the policy, so the policy batch contract remains unchanged.

`get_validation_dataset()` returns another composite containing the two
component validation datasets. A normal sequential DataLoader consumes this
validation composite; it does not use source probabilities.

### ProbabilityMixtureSampler

Create `diffusion_policy/common/probability_mixture_sampler.py` with a sampler
that receives:

```python
ProbabilityMixtureSampler(
    source_lengths: Sequence[int],
    probabilities: Sequence[float],
    samples_per_epoch: int,
    seed: int,
    num_replicas: int = 1,
    rank: int = 0,
)
```

For every yielded index it:

1. selects a source using the configured probability;
2. uniformly selects one index from that source with replacement;
3. adds the source's concatenated-dataset offset;
4. yields the resulting composite index.

Sampling with replacement is intentional: the small bulb source must be
oversampled to receive 20% of training exposure. This does not modify, filter,
or duplicate either source dataset on disk or in the dataset object. An epoch
is a probabilistic exposure schedule, not a promise that every source item is
visited exactly once.

The sampler generates indices in bounded chunks rather than allocating a
weight or index tensor proportional to the roughly 100-million-item expdata
source.

`set_epoch(epoch)` changes the random stream. The stream is deterministic from
`seed`, `epoch`, and `rank`. Each DDP rank receives the same number of indices:

```text
per_rank_samples = ceil(samples_per_epoch / num_replicas)
```

This may add at most `num_replicas - 1` sampled exposures per global epoch; it
never removes component data. Independent rank streams make duplicate draws
possible, as expected for sampling with replacement.

When `samples_per_epoch` is omitted in configuration, the mixed dataset uses
the sum of both component training-window counts. Single-process training
therefore yields exactly that many sampled exposures per epoch.

### Base dataset and workspace integration

Add a backward-compatible optional training-sampler hook to
`BaseLowdimDataset`. The default returns `None`. `MixedLowdimDataset` overrides
it to return `ProbabilityMixtureSampler`.

Update `_make_train_dataloader` in
`train_diffusion_unet_sim_hand_workspace.py`:

- if the dataset supplies a custom sampler, use it in both single-process and
  DDP modes and disable DataLoader shuffle;
- otherwise preserve the existing single-process shuffle and DDP
  `DistributedSampler` behavior exactly.

At the start of every epoch the workspace already calls `set_epoch` when a
sampler exists. The mixed sampler follows that interface.

During training, the workspace counts `source_id` values, removes `source_id`
before model forward, and logs globally reduced DDP fractions as:

```text
train/source_fraction/expdata
train/source_fraction/bulb_tac
```

These values verify realized sampling; they will approach 0.8 and 0.2 and are
not required to equal them exactly for a finite epoch.

## Normalization

The current Sim-Hand configuration uses `mode="limits"`. The mixed normalizer
covers the complete raw transitions of both component datasets, independent of
the 8:2 training probabilities.

To avoid concatenating or copying the giant expdata arrays:

1. fit each component normalizer independently;
2. combine per-field minima and maxima with elementwise min/max;
3. combine means and unbiased variances using component raw-transition counts;
4. create the final manual linear normalizer from the combined statistics.

This preserves complete-source statistics with bounded additional memory. The
8:2 ratio only controls training index selection, as required.

## Configuration and launch contract

Create `diffusion_policy/config/task/sim_hand_mixed.yaml` with:

```yaml
name: sim_hand_mixed
data_root: /home/carus/mnt/bigai/Data

mixing:
  probabilities:
    expdata: 0.8
    bulb_tac: 0.2
  samples_per_epoch: null
```

The task instantiates:

- expdata from `${task.data_root}/exp_data` using
  `SimHandLowdimDataset`;
- bulb TacMP data from `${task.data_root}/bulb_tac_80` using
  `BulbTacLowdimDataset`;
- the composite `MixedLowdimDataset` with the named probabilities.

Probability keys must match component names exactly. Values must be finite,
strictly positive, and sum to 1 within floating-point tolerance.

Create `dp_train_sim_hand_mixed.sh`. It uses the same Python selection, CUDA,
W&B, temporal, optimizer, and pass-through override conventions as the current
`dp_train_sim_hand.sh`, while adding:

```text
task=sim_hand_mixed
exp_name=mixed
```

Its `DATA_ROOT` environment variable defaults to
`/home/carus/mnt/bigai/Data` and overrides `task.data_root`. The existing
launcher and task YAML are not redirected to mixed mode.

## Error handling

Bulb loading fails before training on:

- missing source directory or zero numeric episode directories;
- a missing `state.npy` or `action.npy`;
- an array that is not `float32` with shape `(T, 31)`;
- unequal state/action lengths or an empty episode;
- NaN or infinity in either required array;
- an episode shorter than `horizon`.

Mixed dataset/sampler construction fails on:

- fewer than two named component datasets;
- an empty component training dataset;
- missing, extra, non-finite, zero, or negative source probabilities;
- probabilities that do not sum to 1 within tolerance;
- non-positive explicit `samples_per_epoch`;
- an invalid DDP replica count or rank.

Errors include the source/episode/path involved. No partial converted dataset
or cache is written because this feature has no conversion path.

## Testing

### Bulb adapter tests

Synthetic episode directories prove:

1. numeric episode ordering;
2. exact `state[:, 9:31] -> qpos` mapping;
3. exact shifted `target_before` construction, including the first frame;
4. 66-D observation and 22-D action output;
5. episode boundaries and sequence windows never cross;
6. original transition count is unchanged;
7. missing files, dtype, shape, length, finite-value, and short-episode errors.

### Mixed dataset and sampler tests

Small labeled component datasets prove:

1. composite length equals the untouched sum of component lengths;
2. component samples map without mutation;
3. fixed seed/epoch/rank is reproducible;
4. changing epoch or rank changes the stream;
5. a large synthetic draw is within a statistically safe tolerance of 0.8/0.2;
6. explicit and default epoch sizes are honored;
7. sampler memory does not scale with the total source cardinality;
8. validation concatenates both validation datasets without the mixed sampler.

### Configuration and training tests

1. Existing `sim_hand_lowdim` Hydra composition remains unchanged.
2. `sim_hand_mixed` composes with production defaults and accepts a local
   `task.data_root` override.
3. Existing single-source training smoke tests remain green.
4. A two-source synthetic CPU smoke run completes one train/validation step,
   emits finite losses and source-fraction metrics, and writes a checkpoint.
5. DDP sampler unit tests verify equal per-rank lengths and global source
   counting without requiring GPUs.

### Real mirror read-only check

The real bulb mirror must load 80 episodes and 44,142 steps without writing any
files. The expdata check validates its manifest and representative shard schema;
the normal unit suite does not repeatedly load all 100,000,392 transitions.

## Acceptance criteria

- Existing single-source training behavior and launcher remain available.
- Mixed mode is visibly labeled `sim_hand_mixed` / `mixed` in Hydra and W&B.
- Production defaults resolve to `/home/carus/mnt/bigai/Data/{exp_data,bulb_tac_80}`.
- Bulb mapping matches the approved qpos/target-before/target-after semantics.
- Neither source is rewritten, filtered, downsampled, or assigned a shortened
  dataset length.
- Only the training sampler applies the 0.8/0.2 source probabilities.
- Validation uses both sources without probability reweighting.
- No sampler allocation is proportional to the 100-million-item source.
- Single-process and DDP sampling are deterministic by seed and epoch.
- Logs expose realized per-source training fractions.
- Focused tests and the bounded mixed training smoke test pass.
