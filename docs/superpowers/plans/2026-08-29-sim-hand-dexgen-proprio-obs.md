# Sim-Hand DexGen Proprioception Observation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the Sim-Hand fread training contract so observations are the DexGen-style 66-D proprio vector `[q, target_before, target_before-q]` while actions remain absolute 22-D `target_after`.

**Architecture:** Compose the 66-D observation eagerly inside `load_sim_hand_hdf5` from first-class HDF5 fields `robot/qpos` and `robot/target_before`. Store the composed vector under replay key `obs` (rename from `hand_joint`). Update Dataset validation, temporal dim checks, and task config so `obs_dim=66` / `action_dim=22`. mmap paths stay untouched and out of scope.

**Tech Stack:** Python 3.9+, NumPy, h5py, zarr, Hydra, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-sim-hand-dexgen-proprio-obs-design.md`

## Global Constraints

- fread / in-memory only: `manifest.json + shards` → `load_sim_hand_hdf5` → `SimHandLowdimDataset`; no mmap work.
- Read `robot/target_before` directly; never construct it by shifting `target_after`.
- Channel layout is fixed: `[0:22]=q`, `[22:44]=target_before`, `[44:66]=target_before-q`.
- Replay observation key becomes `obs` with shape `(N, 66)`; action stays `(N, 22)`.
- Temporal lengths / OA-step convention are unchanged; only dims change.
- Apply TDD: failing tests before production edits.
- Do not introduce plan-command soft-newline `+` tokens in shell blocks.
- Do not implement inference-time `target_before` filling.

## File Map

```text
diffusion_policy/common/sim_hand_temporal_util.py
    Split SIM_HAND_DIM into OBS=66 / ACTION=22 validators.

diffusion_policy/dataset/sim_hand_hdf5.py
    Require robot/target_before; compose 66-D obs; emit key "obs".

diffusion_policy/dataset/sim_hand_lowdim_dataset.py
    Validate obs (N,66) + action (N,22); sample key "obs".

diffusion_policy/config/task/sim_hand_lowdim.yaml
    obs_dim: 66

tests/test_sim_hand_temporal.py
    Default obs_dim=66; update error/report asserts.

tests/test_sim_hand_hdf5_dataset.py
    Fixture writes target_before; channel-layout + shape tests.

tests/test_sim_hand_lowdim_dataset.py
    Zarr fixtures use obs (N,66).

tests/test_sim_hand_train_smoke.py
    Synthetic zarr/hdf5 fixtures and hydra obs_dim asserts.

tests/test_diffusion_unet_sim_hand_policy.py
    Policy tests use obs_dim=66 / global_cond_dim=4*66.
```

---

### Task 1: Temporal dims `obs=66`, `action=22`

**Files:**

- Modify: `diffusion_policy/common/sim_hand_temporal_util.py`
- Test: `tests/test_sim_hand_temporal.py`

**Interfaces:**

- Consumes: existing `validate_sim_hand_temporal_config(...)`
- Produces:

```python
SIM_HAND_ACTION_DIM = 22
SIM_HAND_OBS_DIM = 66
# Keep SIM_HAND_DIM = 22 as alias for action/hand DOF if needed by imports,
# but validators must use the split constants above.
```

- [ ] **Step 1: Update temporal tests for the new dims**

Replace `DEFAULTS["obs_dim"]` with `66`. Change invalid-obs message and report asserts:

```python
DEFAULTS = {
    "n_obs_steps": 4,
    "n_pred_action_steps": 9,
    "n_action_steps": 4,
    "horizon": 12,
    "obs_dim": 66,
    "action_dim": 22,
    "oa_step_convention": True,
}

# In parametrize:
({"obs_dim": 22}, "Observation dimension must be 66"),
({"action_dim": 21}, "Action dimension must be 22"),

# In error/report tests, use obs_dim=66 and assert
# "Observation dimension   : 66"
```

Also update `test_startup_report_changes_with_temporal_config` to pass `obs_dim=66`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_sim_hand_temporal.py -v
```

Expected: FAIL because validator still requires `obs_dim == 22`.

- [ ] **Step 3: Implement split dim constants**

In `diffusion_policy/common/sim_hand_temporal_util.py`:

```python
SIM_HAND_ACTION_DIM = 22
SIM_HAND_OBS_DIM = 66
SIM_HAND_DIM = SIM_HAND_ACTION_DIM  # hand/action DOF width
```

Replace the two checks:

```python
if config.obs_dim != SIM_HAND_OBS_DIM:
    raise _invalid(config, "Observation dimension must be 66.")
if config.action_dim != SIM_HAND_ACTION_DIM:
    raise _invalid(config, "Action dimension must be 22.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_sim_hand_temporal.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/common/sim_hand_temporal_util.py tests/test_sim_hand_temporal.py
git commit -m "$(cat <<'EOF'
feat: accept 66-D Sim-Hand proprio observations

EOF
)"
```

---

### Task 2: HDF5 fread composes DexGen 66-D `obs`

**Files:**

- Modify: `diffusion_policy/dataset/sim_hand_hdf5.py`
- Modify: `tests/test_sim_hand_hdf5_dataset.py`

**Interfaces:**

- Consumes: `load_sim_hand_hdf5(dataset_path, *, min_episode_length: int) -> ReplayBuffer`
- Produces: ReplayBuffer data keys:

```python
{
  "obs": float32[N, 66],      # concat(q, target_before, target_before-q)
  "action": float32[N, 22],   # target_after
}
```

Required HDF5 fields gain `"robot/target_before"`.

- [ ] **Step 1: Extend fixtures and add channel-layout test**

In `tests/test_sim_hand_hdf5_dataset.py`:

1. Add to `_row`:

```python
"target_before": np.full(HAND_DIM, value + 1_000.0, dtype=np.float32),
```

2. In `_write_rollout`, write `target_before` alongside `qpos` / `target_after`:

```python
for key in ("qpos", "target_before", "target_after"):
    robot.create_dataset(key, data=np.stack([row[key] for row in rows]))
```

3. Replace assertions that read `dataset.replay_buffer["hand_joint"]` with `["obs"]`.

4. Add:

```python
OBS_DIM = 66

def test_hdf5_obs_channels_are_q_target_before_and_residual(tmp_path):
    rows = [
        _row(0, 0, 0),
        _row(0, 0, 1),
        _row(0, 0, 2),
    ]
    # Override one row with distinct vectors for an exact channel check.
    rows[1]["qpos"] = np.arange(HAND_DIM, dtype=np.float32)
    rows[1]["target_before"] = np.arange(HAND_DIM, dtype=np.float32) + 50.0
    rows[1]["target_after"] = np.arange(HAND_DIM, dtype=np.float32) + 90.0
    _write_rollout(tmp_path, shards=[rows])
    dataset = _make_dataset(tmp_path)

    obs = dataset.replay_buffer["obs"]
    action = dataset.replay_buffer["action"]
    assert obs.shape == (3, OBS_DIM)
    assert action.shape == (3, HAND_DIM)
    np.testing.assert_allclose(obs[1, 0:22], rows[1]["qpos"])
    np.testing.assert_allclose(obs[1, 22:44], rows[1]["target_before"])
    np.testing.assert_allclose(
        obs[1, 44:66],
        rows[1]["target_before"] - rows[1]["qpos"],
    )
    np.testing.assert_allclose(action[1], rows[1]["target_after"])
```

5. Update empty-rollout fixture to also create `robot/target_before`.

6. Keep / adjust the wrong-shape test so missing/wrong `target_before` or wrong `qpos` still raises clearly.

- [ ] **Step 2: Run the new/updated HDF5 tests to verify fail**

Run:

```bash
pytest tests/test_sim_hand_hdf5_dataset.py::test_hdf5_obs_channels_are_q_target_before_and_residual -v
```

Expected: FAIL (`KeyError` for `target_before` and/or missing `obs` key / wrong shape).

- [ ] **Step 3: Implement loader composition**

In `diffusion_policy/dataset/sim_hand_hdf5.py`:

```python
HAND_DIM = 22
OBS_DIM = 66
HDF5_FIELDS = (
    "robot/qpos",
    "robot/target_before",
    "robot/target_after",
    # ... existing index/* fields unchanged ...
)
HDF5_FIELD_DTYPES = {
    "robot/qpos": np.float32,
    "robot/target_before": np.float32,
    "robot/target_after": np.float32,
    # ... existing index dtypes unchanged ...
}
```

Treat all three `robot/*` fields as shape `(N, HAND_DIM)` during validation and allocation.

After episode keep/filter:

```python
qpos = qpos[source_rows]
target_before = target_before[source_rows]
action = target_after[source_rows]
obs = np.concatenate(
    (qpos, target_before, target_before - qpos),
    axis=-1,
).astype(np.float32, copy=False)
assert obs.shape[1] == OBS_DIM

replay_root = {
    "data": {"obs": obs, "action": action},
    "meta": { ... unchanged episode_ends/ids ... },
}
```

Do not time-shift `target_after` to invent `target_before`.

- [ ] **Step 4: Run HDF5 tests**

Run:

```bash
pytest tests/test_sim_hand_hdf5_dataset.py -v
```

Expected: most tests still fail at Dataset validation if Task 3 is not done yet; the loader-level channel test may fail on Dataset init for the same reason. If so, temporarily assert via:

```python
from diffusion_policy.dataset.sim_hand_hdf5 import load_sim_hand_hdf5
buf = load_sim_hand_hdf5(tmp_path, min_episode_length=3)
```

inside the channel test (preferred: make the channel test call `load_sim_hand_hdf5` directly so Task 2 is independently green). Update the plan step if needed so Task 2 does not depend on Dataset rename.

Preferred final form of the channel test:

```python
buf = load_sim_hand_hdf5(tmp_path, min_episode_length=3)
obs = buf["obs"]
action = buf["action"]
```

- [ ] **Step 5: Re-run and commit**

Run:

```bash
pytest tests/test_sim_hand_hdf5_dataset.py::test_hdf5_obs_channels_are_q_target_before_and_residual -v
```

Expected: PASS

```bash
git add diffusion_policy/dataset/sim_hand_hdf5.py tests/test_sim_hand_hdf5_dataset.py
git commit -m "$(cat <<'EOF'
feat: compose DexGen 66-D obs in Sim-Hand HDF5 loader

EOF
)"
```

Also update remaining HDF5 tests in this commit so they write `target_before` and use `load_sim_hand_hdf5` / `["obs"]` where they previously expected `hand_joint`. If Dataset still requires `hand_joint`, keep Dataset-facing tests failing until Task 3, but fixture writing must include `target_before` so loader tests pass.

---

### Task 3: Dataset validates and samples `obs` `(N, 66)`

**Files:**

- Modify: `diffusion_policy/dataset/sim_hand_lowdim_dataset.py`
- Modify: `tests/test_sim_hand_lowdim_dataset.py`
- Modify: `tests/test_sim_hand_hdf5_dataset.py` (any remaining `hand_joint` Dataset asserts)

**Interfaces:**

- Consumes: ReplayBuffer with `obs` / `action`
- Produces:

```python
REQUIRED_KEYS = ("obs", "action")
HAND_DIM = 22
OBS_DIM = 66

# __getitem__ -> {"obs": Tensor[H,66], "action": Tensor[H,22]}
# get_normalizer fits keys "obs" and "action" from replay_buffer["obs"/"action"]
```

- [ ] **Step 1: Rewrite Zarr fixture tests for 66-D `obs`**

In `tests/test_sim_hand_lowdim_dataset.py`:

- Rename fixture helpers from `hand_joint` → `obs`.
- Build `obs` with shape `(N, 66)`.
- Update REQUIRED key expectations to `obs`.
- Sample shape asserts: `(12, 66)` / `(12, 22)`.
- Invalid shape case: `((12, 22), (12, 22), ...)` must raise matching `obs.*(N, 66)`.
- TypeError match becomes `obs.*float32`.

Example writer:

```python
OBS_DIM = 66
HAND_DIM = 22

def _write_replay_buffer(dataset_dir, obs, action, episode_ends):
    root = zarr.open_group(str(dataset_dir / "replay_buffer.zarr"), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.create_dataset(
        "obs", data=obs, shape=obs.shape,
        chunks=(min(16, len(obs)), OBS_DIM),
    )
    data.create_dataset(
        "action", data=action, shape=action.shape,
        chunks=(min(16, len(action)), HAND_DIM),
    )
    meta.create_dataset("episode_ends", data=np.asarray(episode_ends, np.int64))
```

- [ ] **Step 2: Run Dataset tests to verify fail**

Run:

```bash
pytest tests/test_sim_hand_lowdim_dataset.py -v
```

Expected: FAIL on missing `hand_joint` / wrong expected dim in validator.

- [ ] **Step 3: Implement Dataset contract**

In `sim_hand_lowdim_dataset.py`:

```python
HAND_DIM = 22
OBS_DIM = 66
REQUIRED_KEYS = ("obs", "action")
```

Validate:

```python
obs = data["obs"]
action = data["action"]
# obs shape (N, OBS_DIM); action shape (N, HAND_DIM); both float32
```

Sampler keys `REQUIRED_KEYS`. Normalizer:

```python
normalizer.fit(
    data={
        "obs": self.replay_buffer["obs"],
        "action": self.replay_buffer["action"],
    },
    ...
)
```

`__getitem__`:

```python
data = {"obs": sample["obs"], "action": sample["action"]}
```

Update docstring to describe DexGen proprio composition.

- [ ] **Step 4: Run Dataset + HDF5 tests**

Run:

```bash
pytest tests/test_sim_hand_lowdim_dataset.py tests/test_sim_hand_hdf5_dataset.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/dataset/sim_hand_lowdim_dataset.py tests/test_sim_hand_lowdim_dataset.py tests/test_sim_hand_hdf5_dataset.py
git commit -m "$(cat <<'EOF'
feat: train Sim-Hand Dataset from 66-D proprio obs

EOF
)"
```

---

### Task 4: Config + policy/smoke wiring

**Files:**

- Modify: `diffusion_policy/config/task/sim_hand_lowdim.yaml`
- Modify: `tests/test_sim_hand_train_smoke.py`
- Modify: `tests/test_diffusion_unet_sim_hand_policy.py`

**Interfaces:**

- Consumes: Task 1–3 contracts
- Produces: Hydra defaults `obs_dim=66`, `action_dim=22`; smoke/policy tests green

- [ ] **Step 1: Update failing config/smoke/policy tests first**

`sim_hand_lowdim.yaml` change in Step 3; tests first:

In smoke tests:

```python
HAND_DIM = 22
OBS_DIM = 66
```

- Synthetic zarr writer stores `obs` `(N, OBS_DIM)` instead of `hand_joint`.
- Synthetic HDF5 writer includes `target_before` (same as HDF5 fixtures).
- `test_hydra_defaults...` asserts `config.obs_dim == OBS_DIM` and `config.action_dim == HAND_DIM`.
- Any `global_cond_dim = 2 * HAND_DIM` overrides become `2 * OBS_DIM` (or `n_obs * OBS_DIM` as appropriate).

In policy tests:

```python
OBS_DIM = 66
HAND_DIM = 22
# ConditionalUnet1D input_dim=HAND_DIM
# global_cond_dim = n_obs_steps * OBS_DIM  # default 4*66
# policy obs_dim=OBS_DIM, action_dim=HAND_DIM
# batch["obs"] shape (B, n_obs_steps, OBS_DIM)
```

- [ ] **Step 2: Run smoke/policy tests to verify fail**

Run:

```bash
pytest tests/test_sim_hand_train_smoke.py::test_hydra_defaults_have_one_consistent_temporal_configuration tests/test_diffusion_unet_sim_hand_policy.py -v
```

Expected: FAIL on `obs_dim` still 22 and/or policy obs width.

- [ ] **Step 3: Flip task config**

In `diffusion_policy/config/task/sim_hand_lowdim.yaml`:

```yaml
# DexGen-style proprio: [q, target_before, residual] = 66 channels.
# Action remains absolute 22-D target_after.
obs_dim: 66
action_dim: 22
```

Update the comment block accordingly.

- [ ] **Step 4: Run full Sim-Hand fread test suite**

Run:

```bash
pytest tests/test_sim_hand_temporal.py tests/test_sim_hand_hdf5_dataset.py tests/test_sim_hand_lowdim_dataset.py tests/test_diffusion_unet_sim_hand_policy.py tests/test_sim_hand_train_smoke.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/config/task/sim_hand_lowdim.yaml tests/test_sim_hand_train_smoke.py tests/test_diffusion_unet_sim_hand_policy.py
git commit -m "$(cat <<'EOF'
feat: default Sim-Hand training to 66-D DexGen proprio obs

EOF
)"
```

---

## Plan Self-Review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| Compose `[q, tb, tb-q]` at load | Task 2 |
| Direct read of `robot/target_before` | Task 2 |
| Action = `target_after` 22-D | Task 2–3 |
| Replay key `obs` `(N,66)` fread path | Task 2–3 |
| `obs_dim=66` / `action_dim=22` validation | Task 1, 4 |
| Channel-layout unit test | Task 2 |
| Reject wrong obs width | Task 3 |
| mmap out of scope | No mmap tasks |
| Inference out of scope | No inference tasks |

**Placeholder scan:** none.

**Type consistency:** `OBS_DIM=66`, `HAND_DIM`/`ACTION_DIM=22`, replay key `obs` used uniformly across Tasks 2–4.
