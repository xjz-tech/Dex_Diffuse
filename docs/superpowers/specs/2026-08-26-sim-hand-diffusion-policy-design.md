# Sim-Hand Diffusion Policy Design

## Purpose

Add an independent, state-only Diffusion Policy that learns the distribution of feasible 22-dimensional dexterous-hand actions from simulation demonstrations. This phase builds and validates only the Sim-Hand training stack. It does not implement TAM guidance, raw HDF5 conversion, simulation rollout, image conditioning, arm conditioning, or real-robot integration.

The learned distribution is:

```text
p_sim(A_hand | S_hand_history)
```

where the default history contains four consecutive hand states.

## Isolation boundary

The Sim-Hand stack uses new Dataset, Policy, Workspace, task config, training config, null runner, launcher, temporal utility, and tests. Existing Bulb/TAM image datasets, policies, workspaces, configs, training launchers, and real-robot inference code remain unchanged.

Shared infrastructure is limited to stable low-dimensional Diffusion Policy primitives:

- `ConditionalUnet1D`, `Downsample1d`, and `Upsample1d`
- `DDPMScheduler` and `LowdimMaskGenerator`
- `LinearNormalizer`
- `ReplayBuffer` and `SequenceSampler`
- episode mask helpers, EMA, LR scheduling, logging, checkpoint serialization, Hydra, and `train.py`

## Standard data contract

The dataset reads only:

```text
<dataset_dir>/replay_buffer.zarr/
  data/
    hand_joint  # (N, 22), float32
    action      # (N, 22), float32
  meta/
    episode_ends  # (E,), int64, exclusive cumulative ends
```

It exposes training samples as:

```python
{
    "obs": Tensor[horizon, 22],
    "action": Tensor[horizon, 22],
}
```

The future HDF5 conversion layer owns raw simulator schema, timestamps, episode reconstruction, and alignment. The expected but unconfirmed mapping is `hand_joint <- robot/qpos` and `action <- robot/target_after`. The normalized Zarr contract represents `(q_t, a_t)`, with `a_t` interpreted as the command issued from `q_t`.

The converter must later preserve the TAM hand joint ordering, units, absolute-action convention, and control timing. Until the simulation schema is inspected, those properties remain requirements on the converter rather than assumptions embedded in the Dataset.

## Episode split and normalization

Training and validation split by complete episodes using a deterministic seed and default `val_ratio=0.1`. The Dataset stores the original validation mask separately from the potentially downsampled training mask. `get_validation_dataset()` uses the original validation mask, never the complement of a `max_train_episodes`-limited mask.

The Dataset verifies required keys, matching step counts, `(N, 22)` field shapes, valid exclusive episode ends, and non-overlapping train/validation episode masks. `LinearNormalizer` independently fits `obs` and `action` using every timestep in the complete replay buffer, including both training and validation episodes. The episode split affects sample selection and held-out loss calculation only; it does not affect normalization statistics.

## Temporal model

The only authoritative temporal parameters are:

```text
n_obs_steps
n_pred_action_steps
n_action_steps
```

Their defaults are 4, 9, and 4. The diffusion horizon is derived once through Hydra:

```text
horizon = n_obs_steps + n_pred_action_steps - 1
```

The Dataset uses:

```text
pad_before = n_obs_steps - 1
pad_after  = n_pred_action_steps - 1
```

Under the OA-step convention, the first usable current action begins at:

```text
start = n_obs_steps - 1
end   = start + n_pred_action_steps
```

For the default configuration, the model denoises a 12-step trajectory and `action_pred[:, 3:12]` represents `a[t:t+9]`.

## Temporal validation

A dependency-light Sim-Hand temporal utility validates and formats the configuration. Training startup and Policy construction both enforce:

- all three temporal lengths are positive;
- `n_action_steps <= n_pred_action_steps`;
- configured `horizon` equals the derived horizon;
- `horizon % 4 == 0`;
- dimensions are exactly `obs_dim=22` and `action_dim=22`.

The Policy additionally performs an actual no-gradient dummy forward through its configured `ConditionalUnet1D` and verifies that the output temporal length equals the requested horizon. An incompatible model raises a descriptive error containing all temporal values. Requested lengths are never silently padded, cropped, or modified.

The current U-Net has two stride-2 downsampling stages and two matching upsampling stages. For the default horizon:

```text
12 -> 6 -> 3 -> 6 -> 12
```

Therefore its skip connections and output length are compatible.

## Startup report

Before model and dataset construction, the Workspace prints a dynamically generated report containing observation steps, usable prediction steps, execution steps, horizon, dimensions, OA-step status, observation slice, usable action slice, and execution slice. No default values are embedded in the formatter.

## Policy architecture and output

`DiffusionUnetSimHandPolicy` is a focused adaptation of the existing low-dimensional U-Net policy. It uses global conditioning from normalized observations shaped `(B, n_obs_steps, 22)`, a 22-dimensional diffusion trajectory, DDPM epsilon prediction, and no image or object features.

Normal inference returns only:

```python
return {
    "action": action_usable[:, :n_action_steps],
    "action_usable": action_usable,
}
```

where `action` has shape `(B, n_action_steps, 22)` and `action_usable` has shape `(B, n_pred_action_steps, 22)`.

The Policy has one Hydra/constructor flag, `return_full_prediction`, which defaults to `false`. When it is configured as `true`, `predict_action()` additionally returns:

```python
result["action_pred"] = action_pred
```

with shape `(B, horizon, 22)`. Full predictions are intended for debugging only and are absent by default.

## Workspace and offline runner

`TrainDiffusionUnetSimHandWorkspace` follows the existing low-dimensional training lifecycle: dataset construction, train/validation loaders, normalizer installation, optimizer, EMA, LR scheduler, train loss, held-out validation loss, optional sample diagnostics, W&B/JSON logging, and checkpoint restoration.

`NullLowdimRunner.run(policy)` returns `{}` because this phase has no rollout environment.

## Checkpoints

`training.checkpoint_every` defaults to 100 and is Hydra-overridable. At each completed interval the Workspace preserves a non-pruned checkpoint:

```text
checkpoints/epoch_0100.ckpt
checkpoints/epoch_0200.ckpt
```

It also maintains `checkpoints/latest.ckpt`, including at the final epoch when the epoch count is not divisible by the interval. Validation loss is logged but is not used as the sole retention or deletion criterion. No Top-K manager deletes periodic Sim-Hand checkpoints.

## Files

The implementation adds:

```text
diffusion_policy/common/sim_hand_temporal_util.py
diffusion_policy/env_runner/null_lowdim_runner.py
diffusion_policy/dataset/sim_hand_lowdim_dataset.py
diffusion_policy/policy/diffusion_unet_sim_hand_policy.py
diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py
diffusion_policy/config/task/sim_hand_lowdim.yaml
diffusion_policy/config/train_diffusion_unet_sim_hand_workspace.yaml
dp_train_sim_hand.sh
tests/test_null_lowdim_runner.py
tests/test_sim_hand_lowdim_dataset.py
tests/test_diffusion_unet_sim_hand_policy.py
tests/test_sim_hand_train_smoke.py
```

## Verification

Tests cover:

- default and invalid temporal configurations;
- dynamic startup report content;
- actual U-Net input/output length at horizon 12;
- Dataset key/shape validation and OA-aligned sampling;
- disjoint episode-level train/validation masks, including `max_train_episodes`;
- normalization statistics that include both training and validation episodes;
- normal and debug Policy output keys and shapes;
- finite forward/backward loss;
- a synthetic multi-episode Zarr train/validation smoke run;
- finite train and validation losses;
- periodic and latest checkpoint creation with `checkpoint_every=1`.

The smoke test uses only the standardized Zarr contract and does not introduce an HDF5 converter or simulator dependency.
