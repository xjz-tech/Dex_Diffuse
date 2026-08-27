# Sim-Hand HDF5 Episode Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop interpreting HDF5 outcome flags; keep every episode that is long
enough for the training horizon, while preserving episode reconstruction from
`(episode_id, env_id, step)`.

**Architecture:** Confine behavior changes to `load_sim_hand_hdf5`. Schema still
requires outcome columns, but their values are ignored. `SimHandLowdimDataset`
passes its `horizon` as `min_episode_length` and drops the `successful_only`
knob. Zarr, Policy, Workspace, and guidance stay untouched.

**Tech Stack:** Python 3.9+, NumPy, h5py, pytest, Hydra YAML.

**Spec:** `docs/superpowers/specs/2026-08-27-sim-hand-hdf5-episode-filter-design.md`

## Global Constraints

- Do not modify Policy, temporal util, Workspace, guided inference, or Zarr
  validation.
- Outcome fields remain required on disk for schema compatibility; do not use
  their values for keep/drop or integrity.
- `min_episode_length` is required, must be `>= 1`, and equals Dataset
  `horizon` on the HDF5 path.
- Training temporal defaults stay `n_pred_action_steps=9`, `n_action_steps=5`,
  `horizon=12`.
- Apply TDD: failing tests before production edits.
- Do not introduce plan-command soft-newline `+` tokens in shell blocks.

## File Map

~~~text
diffusion_policy/dataset/sim_hand_hdf5.py
    Remove outcome validation/filtering; keep length filter;
    require min_episode_length.

diffusion_policy/dataset/sim_hand_lowdim_dataset.py
    Remove successful_only; pass horizon as min_episode_length.

diffusion_policy/config/task/sim_hand_lowdim.yaml
    Remove successful_only.

tests/test_sim_hand_hdf5_dataset.py
    Replace outcome tests with keep-all + length-filter tests.

tests/test_sim_hand_train_smoke.py
    Drop successful_only config assertion; keep predict/execute asserts.
~~~

---

### Task 1: Loader keep-all + length filter

**Files:**

- Modify: `diffusion_policy/dataset/sim_hand_hdf5.py`
- Modify: `diffusion_policy/dataset/sim_hand_lowdim_dataset.py`
- Test: `tests/test_sim_hand_hdf5_dataset.py`

**Interfaces:**

- Consumes: existing fixture helpers `_row`, `_write_rollout`, `_make_dataset`
- Produces:

```python
def load_sim_hand_hdf5(
    dataset_path: str | Path,
    *,
    min_episode_length: int,
) -> ReplayBuffer
```

`SimHandLowdimDataset.__init__` no longer accepts `successful_only`. On HDF5
load it calls:

```python
load_sim_hand_hdf5(expanded_path, min_episode_length=int(horizon))
```

- [ ] **Step 1: Rewrite failing / replacement tests**

In `tests/test_sim_hand_hdf5_dataset.py`:

1. Delete these tests (behavior is intentionally removed):
   - `test_hdf5_rejects_episode_without_terminal_done_marker`
   - `test_hdf5_successful_only_discards_valid_failed_episodes`
   - `test_hdf5_rejects_multiple_terminal_outcomes`
   - `test_hdf5_rejects_terminal_reset_reason_mismatch`
   - `test_hdf5_rejects_outcome_marker_before_episode_end`
   - `test_hdf5_accepts_other_done_when_success_filter_is_disabled`

2. Add:

```python
def test_hdf5_keeps_failure_and_timeout_episodes(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2, done=True, failure=True),
            _row(20, 1, 0),
            _row(20, 1, 1),
            _row(20, 1, 2, done=True, timeout=True),
        ]],
    )

    dataset = _make_dataset(tmp_path)  # horizon=3

    assert dataset.replay_buffer.n_episodes == 2
    np.testing.assert_array_equal(
        dataset.replay_buffer.meta["episode_ids"],
        np.asarray([10, 20], dtype=np.int64),
    )


def test_hdf5_keeps_episode_without_terminal_done(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2),
        ]],
    )

    dataset = _make_dataset(tmp_path)
    assert dataset.replay_buffer.n_episodes == 1


def test_hdf5_drops_episodes_shorter_than_horizon(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            # length 2 < horizon 3 → drop
            _row(10, 0, 0),
            _row(10, 0, 1, done=True, success=True),
            # length 3 >= horizon 3 → keep
            _row(20, 1, 0),
            _row(20, 1, 1),
            _row(20, 1, 2, done=True, failure=True),
        ]],
    )

    dataset = _make_dataset(tmp_path)

    assert dataset.replay_buffer.n_episodes == 1
    np.testing.assert_array_equal(
        dataset.replay_buffer.episode_ends,
        np.asarray([3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer.meta["episode_ids"],
        np.asarray([20], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        dataset.replay_buffer["hand_joint"][:, 0],
        np.asarray([2000, 2001, 2002], dtype=np.float32),
    )


def test_hdf5_rejects_when_all_episodes_are_too_short(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1, done=True, success=True),
        ]],
    )

    with pytest.raises(ValueError, match="no episodes|min_episode_length|too short"):
        _make_dataset(tmp_path)


def test_hdf5_rejects_invalid_min_episode_length(tmp_path):
    _write_rollout(
        tmp_path,
        shards=[[
            _row(10, 0, 0),
            _row(10, 0, 1),
            _row(10, 0, 2),
        ]],
    )
    from diffusion_policy.dataset.sim_hand_hdf5 import load_sim_hand_hdf5

    with pytest.raises(ValueError, match="min_episode_length"):
        load_sim_hand_hdf5(tmp_path, min_episode_length=0)
```

3. Keep structural tests unchanged (`contiguous steps`, manifest/dtype/shape,
   interleaved reconstruction). Their fixtures may still set `done`/`success`
   for realism; those flags must not affect pass/fail.

4. Ensure no remaining call sites pass `successful_only=...` in this file.

- [ ] **Step 2: Run the new tests and confirm they fail for the right reason**

```bash
pytest tests/test_sim_hand_hdf5_dataset.py::test_hdf5_keeps_failure_and_timeout_episodes \
  tests/test_sim_hand_hdf5_dataset.py::test_hdf5_keeps_episode_without_terminal_done \
  tests/test_sim_hand_hdf5_dataset.py::test_hdf5_drops_episodes_shorter_than_horizon \
  tests/test_sim_hand_hdf5_dataset.py::test_hdf5_rejects_when_all_episodes_are_too_short \
  tests/test_sim_hand_hdf5_dataset.py::test_hdf5_rejects_invalid_min_episode_length \
  -v
```

Expected: FAIL — still rejecting outcomes / missing `min_episode_length` API,
or still keeping short episodes / filtering by success.

- [ ] **Step 3: Implement loader + dataset wiring**

In `diffusion_policy/dataset/sim_hand_hdf5.py`:

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
    # ... existing manifest + schema validation unchanged ...
```

After building `episode_starts` / `episode_ends` and validating contiguous
`step` only:

```python
keep_episodes = (episode_ends - episode_starts) >= min_episode_length
if not np.any(keep_episodes):
    raise ValueError(
        "Sim-Hand HDF5 dataset contains no episodes with length >= "
        f"min_episode_length={min_episode_length}"
    )
```

Delete the blocks that:

- inspect `done` / `success` / `failure` / `timeout` / `reset_reason` for
  early markers or terminal outcomes;
- apply `successful_only`.

After schema validation, working arrays only need:

- `robot/qpos`, `robot/target_after`
- `index/episode_id`, `index/env_id`, `index/step`

Do not allocate or use outcome arrays for filtering. Outcome fields still
appear in `HDF5_FIELDS` so the existing per-shard dtype/shape loop keeps
enforcing schema presence.

In `diffusion_policy/dataset/sim_hand_lowdim_dataset.py`:

- Remove `successful_only` parameter.
- Call `load_sim_hand_hdf5(expanded_path, min_episode_length=int(horizon))`.
- Update the class docstring to say outcome flags are ignored and episodes
  shorter than `horizon` are dropped.

- [ ] **Step 4: Run HDF5 dataset tests**

```bash
pytest tests/test_sim_hand_hdf5_dataset.py -v
```

Expected: PASS for all cases in that file.

- [ ] **Step 5: Commit**

```bash
git add \
  diffusion_policy/dataset/sim_hand_hdf5.py \
  diffusion_policy/dataset/sim_hand_lowdim_dataset.py \
  tests/test_sim_hand_hdf5_dataset.py
git commit -m "$(cat <<'EOF'
feat: filter Sim-Hand HDF5 by episode length only

Ignore outcome flags; drop episodes shorter than horizon.
EOF
)"
```

---

### Task 2: Config + smoke cleanup

**Files:**

- Modify: `diffusion_policy/config/task/sim_hand_lowdim.yaml`
- Modify: `tests/test_sim_hand_train_smoke.py`
- Verify only: `diffusion_policy/config/train_diffusion_unet_sim_hand_workspace.yaml`

**Interfaces:**

- Consumes: Dataset without `successful_only`
- Produces: task config that no longer declares `successful_only`; smoke still
  asserts `n_pred_action_steps == 9` and `n_action_steps == 5`

- [ ] **Step 1: Write the failing smoke assertion update**

In `tests/test_sim_hand_train_smoke.py`, remove:

```python
assert config.task.dataset.successful_only is True
```

Keep / ensure:

```python
assert config.n_pred_action_steps == 9
assert config.n_action_steps == 5
assert config.horizon == 12
```

If Hydra still injects `successful_only` from YAML, this step alone may still
pass until YAML is edited; that is fine — the next step removes it and Step 3
confirms composition.

- [ ] **Step 2: Remove `successful_only` from task YAML**

In `diffusion_policy/config/task/sim_hand_lowdim.yaml`, delete the line:

```yaml
successful_only: true
```

Confirm train workspace YAML still contains:

```yaml
n_obs_steps: 4
n_pred_action_steps: 9
n_action_steps: 5
```

Do not change those values.

- [ ] **Step 3: Run smoke / config tests**

```bash
pytest tests/test_sim_hand_train_smoke.py tests/test_sim_hand_hdf5_dataset.py -v
```

Expected: PASS. Composition must not mention `successful_only`. Predict 9 /
execute 5 / horizon 12 remain asserted.

- [ ] **Step 4: Commit**

```bash
git add \
  diffusion_policy/config/task/sim_hand_lowdim.yaml \
  tests/test_sim_hand_train_smoke.py
git commit -m "$(cat <<'EOF'
chore: drop successful_only from Sim-Hand task config

HDF5 keep policy is length-only; keep predict-9/execute-5 defaults.
EOF
)"
```

---

## Plan Self-Review

1. **Spec coverage:** outcome ignore, length filter, API change, Dataset wiring,
   YAML/smoke cleanup, temporal confirm — all mapped to Task 1–2.
2. **Placeholders:** none; concrete tests and API snippets included.
3. **Type consistency:** `min_episode_length: int` required; Dataset passes
   `int(horizon)`; `successful_only` removed end-to-end.
