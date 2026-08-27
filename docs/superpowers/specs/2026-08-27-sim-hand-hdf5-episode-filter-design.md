# Sim-Hand HDF5 Episode Filter Design

## Purpose

Update the Sim-Hand HDF5 loader so it treats every structurally valid episode as
usable training data. Outcome flags (`success`, `failure`, `timeout`, `done`,
`reset_reason`) are no longer interpreted. Episodes are still reconstructed from
`(episode_id, env_id, step)`, and episodes shorter than the training horizon are
discarded.

This change is confined to the HDF5 load path. The standardized Zarr dataset
path, Policy, temporal validator, Workspace, and guided-inference stack are out
of scope.

## Background

The current loader (`diffusion_policy/dataset/sim_hand_hdf5.py`):

- validates terminal `done=True`;
- requires exactly one of `success` / `failure` / `timeout` at the terminal row;
- checks `reset_reason` against that outcome;
- rejects non-terminal outcome markers;
- optionally keeps only `success` episodes via `successful_only=True`.

New domain information: those outcome fields are unrelated to whether a
trajectory is usable for Sim-Hand diffusion training. All episodes in the
dataset may be used after length filtering.

Training temporal defaults remain:

```text
n_obs_steps          = 4
n_pred_action_steps  = 9   # predict 9
n_action_steps       = 5   # execute 5
horizon              = 12  # 4 + 9 - 1
```

## Scope

**In scope**

- `load_sim_hand_hdf5` behavior and its unit tests
- Dataset/config plumbing that only exists to pass `successful_only` into the
  HDF5 loader
- Confirming train config keeps predict-9 / execute-5

**Out of scope**

- Zarr contract or Zarr-side length filtering
- Changing Policy / U-Net / Workspace / guidance code
- Removing outcome fields from the on-disk HDF5 schema
- Changing how `SequenceSampler` pads short windows (filtering happens before
  the replay buffer is built)

## Loader contract

### Unchanged

1. Manifest schema version, `dof=22`, shard inventory, and
   `total_transitions` checks.
2. Required HDF5 fields, including `index/done`, `index/success`,
   `index/failure`, `index/timeout`, and `index/reset_reason`.
3. Field dtype and shape validation.
4. Lexicographic ordering by `(episode_id, env_id, step)`.
5. Episode boundaries from changes in `(episode_id, env_id)`.
6. Per-episode `step` must be contiguous and start at zero.
7. Mapping `robot/qpos → hand_joint`, `robot/target_after → action`.

Fields in (2) remain **schema requirements** only. The loader must not use their
values for keep/drop or integrity decisions beyond existing type/shape checks.

### Removed behavior

- Terminal `done` requirement
- Terminal outcome exclusivity among success / failure / timeout
- `reset_reason` ↔ outcome consistency checks
- Non-terminal done/outcome/reset marker checks
- `successful_only` filtering (and the parameter itself)

### Added behavior

```text
keep episode ⇔ episode_length >= min_episode_length
```

- `min_episode_length` defaults to the training `horizon` (12 under current
  config).
- The Dataset passes its `horizon` into the loader when loading HDF5.
- Episodes with `length < min_episode_length` are dropped silently (no outcome
  inspection).
- If zero episodes remain after length filtering, raise a clear `ValueError`.

### API

```python
def load_sim_hand_hdf5(
    dataset_path: str | Path,
    *,
    min_episode_length: int,
) -> ReplayBuffer
```

- `min_episode_length` is required and must be `>= 1`.
- `successful_only` is removed from the loader and from
  `SimHandLowdimDataset` / task YAML.

## Dataset / config touchpoints

Only the minimal plumbing that currently forwards `successful_only`:

| File | Change |
|------|--------|
| `diffusion_policy/dataset/sim_hand_hdf5.py` | New filter rules; remove outcome logic |
| `diffusion_policy/dataset/sim_hand_lowdim_dataset.py` | Drop `successful_only`; pass `horizon` as `min_episode_length` for HDF5 |
| `diffusion_policy/config/task/sim_hand_lowdim.yaml` | Remove `successful_only` |
| `tests/test_sim_hand_hdf5_dataset.py` | Rewrite outcome/success cases; add length filter cases |
| `tests/test_sim_hand_train_smoke.py` | Drop `successful_only` assertions if present |

Train workspace YAML already has `n_pred_action_steps: 9` and
`n_action_steps: 5`; no temporal redesign is required—only verify it stays that
way.

## Error handling

Fail loudly on:

- missing/invalid manifest or shards (unchanged);
- missing fields / wrong dtype / wrong shape (unchanged);
- non-contiguous or non-zero-starting `step` within an episode (unchanged);
- `min_episode_length < 1`;
- no transitions, or no episodes remaining after length filtering.

Do **not** fail on:

- missing terminal `done`;
- failure / timeout / success combinations;
- `reset_reason` values;
- early done/outcome markers.

## Testing

Replace outcome-centric tests with:

1. Long episodes with `failure` / `timeout` / no success markers are **kept**.
2. Episodes with `length < min_episode_length` are **dropped**.
3. Mixing short and long episodes keeps only the long ones and preserves
   contiguous `hand_joint` / `action` / `episode_ends`.
4. All episodes shorter than the threshold → `ValueError`.
5. Existing structural tests remain: step continuity, multi-env episode splits,
   manifest/dtype/shape errors.

## Non-goals

- Interpreting task success for imitation filtering
- Changing pad_before / pad_after sampling semantics
- Converting HDF5 to Zarr as a separate offline step
- Altering guided inference execution length (already 5) beyond confirming
  training predict/execute lengths

## Acceptance

- HDF5 load no longer references outcome values for validation or filtering.
- Default training path uses predict 9 / execute 5 / horizon 12.
- Episodes shorter than horizon never enter the replay buffer.
- Unit tests above pass; Zarr-only smoke remains green without schema changes.
