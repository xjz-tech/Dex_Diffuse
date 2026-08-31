# Sim-Hand DexGen-Style Proprioception Observation Design

## Purpose

Update the Sim-Hand **training data contract** so observations match a
DexGen-style proprioception layout, while keeping absolute `target_after`
actions as the diffusion label.

Locked first version:

```text
Observation:  [q, target_before, target_before - q]_{t-n_obs+1 : t}
Action:       target_after_{t : t+n_pred-1}
```

With default temporal lengths this is a 4-step observation history and a
9-step action prediction (`t:t+8` inclusive).

## Background

The current contract is:

```text
obs_t     = q_t                 # 22-D  (robot/qpos → hand_joint)
action_t  = target_after_t      # 22-D
```

Verified on `/home/carus/Data/exp_data`:

- `robot/target_before` exists as a first-class `(N, 22) float32` field.
- Semantics: `target_before_t` is the command that produced `q_t`.
- Within an episode, `target_before_t == target_after_{t-1}` exactly.
- At episode `step=0`, `target_before_t == q_t` (residual is zero).

Residual `target_before - q` is linearly determined by the first two blocks.
It is still retained as an explicit feature channel (DexGen-style), not as
independent information.

## Training I/O assumption

This change targets the **in-memory fread path** used on a server with enough
RAM:

```text
manifest.json + shards/*.h5
  → load_sim_hand_hdf5 (eager numpy / ReplayBuffer)
  → SimHandLowdimDataset
```

Optional Zarr `replay_buffer.zarr` remains supported as the same in-memory
replay contract. **mmap cache builders / `SimHandMmapDataset` are out of
scope** for this design; they are not required for the intended training
setup.

## Scope

**In scope (lock A)**

- HDF5 fread → ReplayBuffer mapping that builds 66-D observations from
  `robot/qpos` and `robot/target_before`
- In-memory Zarr replay arrays that store the composed 66-D `obs` (same
  contract as HDF5 load output)
- Config / temporal validation: `obs_dim=66`, `action_dim=22`
- Dataset, policy wiring, normalizer, and tests required by the new dims

**Out of scope**

- mmap dataset, mmap converter, and any disk-backed obs cache path
- Inference-time construction of `target_before` history
- Guided inference / TAM / object features
- Dual-mode compatibility with old 22-D checkpoints
- Changing action label away from absolute `target_after`
- Deriving `target_before` by time-shifting `target_after` (must read the field)

## Chosen approach

Compose a single 66-D observation **at conversion / load time**.

Do not keep a dual `obs_mode`. Do not store three separate proprio fields in
the training replay buffer for v1. Training stacks continue to see:

```text
obs     # (N, 66)
action  # (N, 22)
```

## Data contract

### Channel layout

```text
obs_t[0:22]   = q_t                         # robot/qpos
obs_t[22:44]  = target_before_t             # robot/target_before
obs_t[44:66]  = target_before_t - q_t       # residual (explicit)
action_t[0:22] = target_after_t             # robot/target_after
```

Units, joint ordering, and absolute-target convention are unchanged from the
existing Sim-Hand hand DOF contract (22-D TAM/Sharpa hand order).

### Temporal model (unchanged)

```text
n_obs_steps
n_pred_action_steps
n_action_steps
horizon = n_obs_steps + n_pred_action_steps - 1
```

OA-step convention remains:

```text
obs condition : s[t-n_obs+1 : t]
usable actions: a[t : t+n_pred]
execute       : a[t : t+n_action]
```

where `a` is absolute `target_after`.

### Source mapping

| Training field | Source |
|---|---|
| `obs[..., 0:22]` | `robot/qpos` |
| `obs[..., 22:44]` | `robot/target_before` (direct read) |
| `obs[..., 44:66]` | compute `target_before - qpos` |
| `action` | `robot/target_after` |

Episode reconstruction, length filtering (`length >= horizon`), train/val
split, and outcome-flag non-interpretation remain as in the HDF5 episode-filter
design.

### In-memory replay contract (fread)

After HDF5 load (or when reading a prepared Zarr), the ReplayBuffer holds:

```text
data/obs      # (N, 66), float32   # preferred key name
data/action   # (N, 22), float32
meta/episode_ends
```

No intermediate mmap materialization is part of this design. Training reads
the composed arrays from RAM via `SimHandLowdimDataset`.

Naming note: current code may still use replay key `hand_joint` for the
observation array. Implementation may either:

1. rename the replay key to `obs` and update Dataset sampling, or
2. keep the key name `hand_joint` but require shape `(N, 66)`.

Prefer renaming to `obs` when touching the loader, so the key matches the
66-D DexGen proprio vector rather than implying raw joints only. If rename
cost is high in an intermediate patch, shape `(N, 66)` under the old key is
acceptable only as a short migration step documented in the plan.

### Normalization

`LinearNormalizer` independently fits:

- full 66-D `obs`
- full 22-D `action`

using every retained timestep in the replay buffer (train + val), same rule as
today. No per-block special casing in v1.

### Dimensions validation

Temporal validation must accept:

```text
obs_dim = 66
action_dim = 22
```

Replace the current hard rule that both dims equal `SIM_HAND_DIM=22`. Keep
`SIM_HAND_ACTION_DIM = 22` (or equivalent) for the action/hand DOF width, and
introduce `SIM_HAND_OBS_DIM = 66` for the composed proprio observation.

Policy `global_cond_dim = obs_dim * n_obs_steps` follows automatically from
config.

## Non-goals / explicit non-claims

- Residual channels do not add independent degrees of freedom.
- Within-episode equality `target_before_t == target_after_{t-1}` is a dataset
  consistency property, not a construction recipe.
- This design does not define how closed-loop inference should populate
  `target_before` after the first command.

## Success criteria

1. `load_sim_hand_hdf5` (fread) emits `(N, 66)` obs and `(N, 22)` action with
   the channel layout above into an in-memory ReplayBuffer.
2. Task config defaults to `obs_dim=66`, `action_dim=22`.
3. Temporal validator, Dataset, Policy smoke tests, and train smoke tests pass
   under the new dims on the fread Dataset path.
4. A unit test asserts channel slices equal `qpos`, `target_before`, and
   `target_before - qpos` on a synthetic shard that includes both fields.
5. A `(N, 22)` observation array is rejected with a clear shape error—no silent
   train-on-wrong-obs.

## Implementation touchpoints (for the later plan)

Expected code surfaces, not an implementation checklist yet:

- `diffusion_policy/dataset/sim_hand_hdf5.py`
- `diffusion_policy/dataset/sim_hand_lowdim_dataset.py`
- `diffusion_policy/common/sim_hand_temporal_util.py`
- `diffusion_policy/config/task/sim_hand_lowdim.yaml`
- related unit and smoke tests for the fread path

## Spec self-review

- No unresolved placeholders.
- Scope limited to training contract (lock A) on the in-memory fread path.
- Consistent with verified HDF5 fields and existing OA-step temporal model.
- mmap explicitly out of scope.
