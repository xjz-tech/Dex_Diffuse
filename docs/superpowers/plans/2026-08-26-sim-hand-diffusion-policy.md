# Sim-Hand Diffusion Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build and verify an isolated, hand-only 22-D Diffusion Policy training stack with configurable observe/predict/execute lengths and standardized Zarr input.

**Architecture:** Add focused Sim-Hand Dataset, Policy, Workspace, null runner, temporal utility, Hydra configs, launcher, and tests. Reuse existing low-dimensional diffusion primitives without modifying TAM/image/real-robot paths. Fit normalization statistics on the complete replay buffer while retaining episode-disjoint train and validation samplers.

**Tech Stack:** Python 3.9+, PyTorch, Diffusers DDPMScheduler, Hydra/OmegaConf, Zarr 2, NumPy, pytest, W&B, and existing Diffusion Policy utilities.

**Spec:** docs/superpowers/specs/2026-08-26-sim-hand-diffusion-policy-design.md

## Global Constraints

- Observation and action dimensions are exactly 22; no arm, image, object, language, or tactile inputs.
- Defaults are n_obs_steps=4, n_pred_action_steps=9, n_action_steps=4, and derived horizon=12.
- Derive horizon as n_obs_steps + n_pred_action_steps - 1; reject inconsistent or non-multiple-of-four horizons without padding or cropping.
- Normalization uses every timestep in the complete replay buffer, including training and validation episodes.
- Read only replay_buffer.zarr/data/{hand_joint,action} and meta/episode_ends; add no HDF5 converter.
- Preserve periodic checkpoints and latest.ckpt; validation loss does not prune periodic checkpoints.
- Do not modify Bulb/TAM image datasets, policies, configs, launchers, inference, guidance, or real-robot code.
- Normal Policy output contains only action and action_usable; action_pred is present only when return_full_prediction=true.

## Test Environment

The declared repository environment is absent. Use the existing IsaacLab PyTorch packages through an isolated temporary overlay:

~~~bash
uv venv --python /home/carus/miniforge3/envs/isaaclab/bin/python3.11 --system-site-packages /tmp/dex-diffuse-sim-hand-test
uv pip install --python /tmp/dex-diffuse-sim-hand-test/bin/python 'dill>=0.3.5,<0.4' 'zarr>=2.12,<3' 'numcodecs>=0.10,<0.16'
~~~

All test commands below use /tmp/dex-diffuse-sim-hand-test/bin/python -m pytest.

---

### Task 1: Temporal Configuration and Startup Report

**Files:**
- Create: diffusion_policy/common/sim_hand_temporal_util.py
- Create: tests/test_sim_hand_temporal.py

**Interfaces:**
- Produces: SimHandTemporalConfig, validate_sim_hand_temporal_config(...) -> SimHandTemporalConfig, and format_sim_hand_temporal_config(config) -> str.
- Consumers: Sim-Hand Policy construction and Workspace startup.

- [ ] **Step 1: Write failing valid-default and slice tests**

~~~python
def test_default_temporal_config():
    config = validate_sim_hand_temporal_config(
        n_obs_steps=4,
        n_pred_action_steps=9,
        n_action_steps=4,
        horizon=12,
        obs_dim=22,
        action_dim=22,
    )
    assert config.usable_action_slice == slice(3, 12)
    assert config.execution_action_slice == slice(3, 7)
~~~

- [ ] **Step 2: Write failing invalid-config tests**

Cover non-positive values, predict-four/execute-five, inconsistent derived horizon, horizon=11, non-22 dimensions, and disabled OA convention. Each error must contain every temporal value and the specific cause.

~~~python
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"n_obs_steps": 0}, "must be positive"),
        ({"n_pred_action_steps": 4, "n_action_steps": 5, "horizon": 7},
         "must not exceed prediction action steps"),
        ({"horizon": 16}, "must equal n_obs_steps + n_pred_action_steps - 1"),
        ({"n_pred_action_steps": 8, "horizon": 11},
         "horizon must be a multiple of 4"),
        ({"obs_dim": 21}, "Observation dimension must be 22"),
        ({"action_dim": 21}, "Action dimension must be 22"),
        ({"oa_step_convention": False}, "OA step convention must be enabled"),
    ],
)
def test_invalid_temporal_config(overrides, message):
    values = dict(
        n_obs_steps=4,
        n_pred_action_steps=9,
        n_action_steps=4,
        horizon=12,
        obs_dim=22,
        action_dim=22,
        oa_step_convention=True,
    )
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        validate_sim_hand_temporal_config(**values)
~~~

- [ ] **Step 3: Run tests and observe the missing-module failure**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_temporal.py -q
~~~

- [ ] **Step 4: Implement immutable config, validation, and dynamic report**

~~~python
@dataclass(frozen=True)
class SimHandTemporalConfig:
    n_obs_steps: int
    n_pred_action_steps: int
    n_action_steps: int
    horizon: int
    obs_dim: int
    action_dim: int
    oa_step_convention: bool = True

    @property
    def usable_action_slice(self) -> slice:
        start = self.n_obs_steps - 1
        return slice(start, start + self.n_pred_action_steps)

    @property
    def execution_action_slice(self) -> slice:
        start = self.n_obs_steps - 1
        return slice(start, start + self.n_action_steps)
~~~

The report dynamically emits all required labels and changes from s[t-3:t+1], a[t:t+9], a[t:t+4] to s[t-1:t+1], a[t:t+7], a[t:t+3] for config (2, 7, 3, 8).

- [ ] **Step 5: Run tests and commit**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_temporal.py -q
git add diffusion_policy/common/sim_hand_temporal_util.py tests/test_sim_hand_temporal.py
git commit -m "feat: validate sim-hand temporal configuration"
~~~

---

### Task 2: Standard Zarr Dataset and Null Runner

**Files:**
- Create: diffusion_policy/dataset/sim_hand_lowdim_dataset.py
- Create: diffusion_policy/env_runner/null_lowdim_runner.py
- Create: tests/test_sim_hand_lowdim_dataset.py
- Create: tests/test_null_lowdim_runner.py

**Interfaces:**
- Consumes: ReplayBuffer, SequenceSampler, get_val_mask, downsample_mask, and LinearNormalizer.
- Produces: SimHandLowdimDataset and NullLowdimRunner.run(policy) -> {}.

- [ ] **Step 1: Add a synthetic five-episode Zarr fixture and failing contract tests**

~~~python
root = zarr.open_group(str(dataset_dir / "replay_buffer.zarr"), mode="w")
data = root.create_group("data")
meta = root.create_group("meta")
data.create_dataset("hand_joint", data=hand_joint, chunks=(16, 22))
data.create_dataset("action", data=action, chunks=(16, 22))
meta.create_dataset("episode_ends", data=episode_ends, dtype="int64")
~~~

Test missing keys, non-float32 fields, wrong 22-D shape, unequal step counts, empty episodes, non-increasing ends, and a final end unequal to N.

- [ ] **Step 2: Add failing sample, split, and normalization tests**

Instantiate with horizon 12, pad-before 3, and pad-after 8. Assert samples contain only obs/action shaped (12, 22), masks do not overlap, and max_train_episodes=1 does not change the original validation mask.

Place global extrema in a validation episode and assert:

~~~python
normalizer = dataset.get_normalizer()
torch.testing.assert_close(
    normalizer["obs"].get_input_stats()["min"],
    torch.from_numpy(hand_joint.min(axis=0)),
)
torch.testing.assert_close(
    normalizer["action"].get_input_stats()["max"],
    torch.from_numpy(action.max(axis=0)),
)
~~~

- [ ] **Step 3: Run tests and observe missing implementations**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_lowdim_dataset.py tests/test_null_lowdim_runner.py -q
~~~

- [ ] **Step 4: Implement Dataset validation, masks, mapping, and full-buffer normalization**

~~~python
def _sample_to_data(self, sample):
    return {"obs": sample["hand_joint"], "action": sample["action"]}

def get_normalizer(self, mode="limits", **kwargs):
    normalizer = LinearNormalizer()
    normalizer.fit(
        data={
            "obs": self.replay_buffer["hand_joint"],
            "action": self.replay_buffer["action"],
        },
        last_n_dims=1,
        mode=mode,
        **kwargs,
    )
    return normalizer
~~~

get_validation_dataset() samples self.val_mask directly and preserves both train_mask and val_mask.

- [ ] **Step 5: Implement and test the null runner**

~~~python
class NullLowdimRunner(BaseLowdimRunner):
    def run(self, policy: BaseLowdimPolicy) -> Dict:
        return {}
~~~

- [ ] **Step 6: Run tests and commit**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_lowdim_dataset.py tests/test_null_lowdim_runner.py -q
git add diffusion_policy/dataset/sim_hand_lowdim_dataset.py diffusion_policy/env_runner/null_lowdim_runner.py tests/test_sim_hand_lowdim_dataset.py tests/test_null_lowdim_runner.py
git commit -m "feat: add sim-hand replay dataset"
~~~

---

### Task 3: Independent Sim-Hand Diffusion Policy

**Files:**
- Create: diffusion_policy/policy/diffusion_unet_sim_hand_policy.py
- Create: tests/test_diffusion_unet_sim_hand_policy.py

**Interfaces:**
- Consumes: temporal config, ConditionalUnet1D, DDPMScheduler, LowdimMaskGenerator, and LinearNormalizer.
- Produces: DiffusionUnetSimHandPolicy.predict_action(), compute_loss(), and set_normalizer().

- [ ] **Step 1: Write failing output-contract tests with a small model**

Use down_dims=(32,64,128), horizon 12, 22-D input/output, four observations, nine usable actions, four executed actions, two inference steps, and identity normalizers.

~~~python
assert set(result) == {"action", "action_usable"}
assert result["action"].shape == (2, 4, 22)
assert result["action_usable"].shape == (2, 9, 22)
~~~

With return_full_prediction=True, additionally assert action_pred.shape == (2,12,22).

- [ ] **Step 2: Write failing finite-loss/backward and U-Net mismatch tests**

~~~python
loss = policy.compute_loss({
    "obs": torch.randn(2, 12, 22),
    "action": torch.randn(2, 12, 22),
})
assert torch.isfinite(loss)
loss.backward()
assert any(p.grad is not None for p in policy.model.parameters())
~~~

A deliberately incompatible model returning a shorter temporal dimension must fail construction with the full temporal report.

~~~python
class ShortTemporalModel(nn.Module):
    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        return sample[:, :-1]

def test_rejects_unet_temporal_mismatch(scheduler):
    with pytest.raises(ValueError, match="U-Net output temporal length"):
        DiffusionUnetSimHandPolicy(
            model=ShortTemporalModel(),
            noise_scheduler=scheduler,
            horizon=12,
            obs_dim=22,
            action_dim=22,
            n_obs_steps=4,
            n_pred_action_steps=9,
            n_action_steps=4,
        )
~~~

- [ ] **Step 3: Run tests and observe the missing implementation**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_diffusion_unet_sim_hand_policy.py -q
~~~

- [ ] **Step 4: Implement the focused global-conditioned policy**

Do not subclass the existing lowdim Policy because its n_action_steps means predicted slice length. Store prediction and execution lengths separately. Inference returns:

~~~python
action_usable = action_pred[:, temporal.usable_action_slice]
result = {
    "action": action_usable[:, :self.n_action_steps],
    "action_usable": action_usable,
}
if self.return_full_prediction:
    result["action_pred"] = action_pred
return result
~~~

Training flattens the first n_obs_steps normalized states as global condition and computes DDPM epsilon/sample MSE over the full action horizon.

- [ ] **Step 5: Implement actual U-Net temporal probing**

During construction, pass zeros shaped (1,horizon,22) with zeros shaped (1,n_obs_steps*22) as global condition at timestep zero. Restore prior training state and require output shape (1,horizon,22).

- [ ] **Step 6: Run tests and commit**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_temporal.py tests/test_diffusion_unet_sim_hand_policy.py -q
git add diffusion_policy/policy/diffusion_unet_sim_hand_policy.py tests/test_diffusion_unet_sim_hand_policy.py
git commit -m "feat: add sim-hand diffusion policy"
~~~

---

### Task 4: Workspace, Hydra Configs, Launcher, and Smoke Test

**Files:**
- Create: diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py
- Create: diffusion_policy/config/task/sim_hand_lowdim.yaml
- Create: diffusion_policy/config/train_diffusion_unet_sim_hand_workspace.yaml
- Create: dp_train_sim_hand.sh
- Create: tests/test_sim_hand_train_smoke.py

**Interfaces:**
- Consumes: Dataset, Policy, null runner, temporal report, EMA/LR/logging/checkpoint utilities.
- Produces: Hydra entry train_diffusion_unet_sim_hand_workspace, periodic checkpoints, and latest.ckpt.

- [ ] **Step 1: Write failing Hydra composition and startup-report tests**

~~~python
assert cfg.n_obs_steps == 4
assert cfg.n_pred_action_steps == 9
assert cfg.n_action_steps == 4
assert cfg.horizon == 12
assert cfg.obs_dim == cfg.action_dim == 22
assert cfg.training.checkpoint_every == 100
assert cfg.policy.return_full_prediction is False
~~~

Capture Workspace-construction stdout and assert every temporal label and derived value is present.

- [ ] **Step 2: Write failing synthetic train/validation/checkpoint smoke**

Override to CPU, one epoch, one train step, one validation step, checkpoint interval one, workers zero, batch size two, W&B disabled, inference steps two, and model dims [32,64,128]. After workspace.run(), assert finite train_loss/val_loss in logs.json.txt and:

~~~python
assert (output_dir / "checkpoints/epoch_0001.ckpt").is_file()
assert (output_dir / "checkpoints/latest.ckpt").is_file()
~~~

- [ ] **Step 3: Run smoke test and observe missing files**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_train_smoke.py -q
~~~

- [ ] **Step 4: Add task/training configs with one temporal source**

~~~yaml
n_obs_steps: 4
n_pred_action_steps: 9
n_action_steps: 4
horizon: ${eval:'${n_obs_steps}+${n_pred_action_steps}-1'}
obs_as_global_cond: true
~~~

Task config sets pad_before from n_obs_steps-1 and pad_after from n_pred_action_steps-1.

- [ ] **Step 5: Implement isolated Workspace**

Validate and print temporal config before Policy creation. Mirror existing lowdim dataset/loaders, normalizer, optimizer, EMA, scheduler, train loss, validation loss, W&B, and JSON logging. Do not instantiate Top-K.

After incrementing the completed epoch:

~~~python
should_save_periodic = self.epoch % cfg.training.checkpoint_every == 0
is_final_epoch = local_epoch_idx == cfg.training.num_epochs - 1
if should_save_periodic:
    self.save_checkpoint(
        path=Path(self.output_dir) / "checkpoints" / f"epoch_{self.epoch:04d}.ckpt",
        use_thread=False,
    )
if cfg.checkpoint.save_last_ckpt and (should_save_periodic or is_final_epoch):
    self.save_checkpoint(tag="latest", use_thread=False)
~~~

- [ ] **Step 6: Add portable launcher**

The script validates DATASET_PATH/replay_buffer.zarr, uses overridable PYTHON, DATASET_PATH, CUDA_VISIBLE_DEVICES, and WANDB_MODE, and invokes only the Sim-Hand config.

- [ ] **Step 7: Run all tests and commit**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_temporal.py tests/test_null_lowdim_runner.py tests/test_sim_hand_lowdim_dataset.py tests/test_diffusion_unet_sim_hand_policy.py tests/test_sim_hand_train_smoke.py -q
bash -n dp_train_sim_hand.sh
git add diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py diffusion_policy/config/task/sim_hand_lowdim.yaml diffusion_policy/config/train_diffusion_unet_sim_hand_workspace.yaml dp_train_sim_hand.sh tests/test_sim_hand_train_smoke.py
git commit -m "feat: train sim-hand diffusion policy"
~~~

---

### Task 5: Full Verification and Isolation Audit

**Files:**
- Verify only; no planned production changes.

**Interfaces:**
- Consumes all Sim-Hand components.
- Produces test, syntax, checkpoint, and isolation evidence.

- [ ] **Step 1: Run all Sim-Hand tests from a clean process**

~~~bash
/tmp/dex-diffuse-sim-hand-test/bin/python -m pytest tests/test_sim_hand_temporal.py tests/test_null_lowdim_runner.py tests/test_sim_hand_lowdim_dataset.py tests/test_diffusion_unet_sim_hand_policy.py tests/test_sim_hand_train_smoke.py -q
~~~

- [ ] **Step 2: Run syntax checks**

~~~bash
PYTHONDONTWRITEBYTECODE=1 /tmp/dex-diffuse-sim-hand-test/bin/python -c "import ast,pathlib; files=list(pathlib.Path('.').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(len(files))"
bash -n dp_train_sim_hand.sh
~~~

- [ ] **Step 3: Audit changed paths**

~~~bash
git diff --name-only 1a3e194..HEAD
~~~

Expected production changes are limited to new Sim-Hand/common files, documents, launcher, and tests. The output must not include Bulb/image/TAM configs or code, existing Bulb launchers, inference_dp.py, or real-robot files.

- [ ] **Step 4: Inspect final diff and status**

~~~bash
git diff --check 1a3e194..HEAD
git status --short --branch
~~~

Expected: no whitespace errors and no uncommitted implementation changes.
