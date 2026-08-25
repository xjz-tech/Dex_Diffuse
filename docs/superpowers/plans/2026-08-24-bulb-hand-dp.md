# Bulb Hand Diffusion Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a proprio-only hand Diffusion Policy (step and chunk, two ckpts) on the existing image-policy stack, with sim-fitted hand min/max shared into real image training, and split EE/hand normalize–unnormalize.

**Architecture:** Keep `DiffusionUnetImagePolicy` + `TrainDiffusionUnetImageWorkspace`. Tasks differ only by `shape_meta` (image 31-D vs hand 22-D). Hand stats are frozen from sim; real EE is fit on real data. Action normalize/unnormalize always splits EE vs hand. Do not implement guidance sampling.

**Tech Stack:** PyTorch, Hydra, zarr ReplayBuffer, `SingleFieldLinearNormalizer.create_manual`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-bulb-hand-dp-design.md`

## Global Constraints

- Hand DP obs is only `hand_joint` `[22]`; action is only absolute hand `[22]`.
- Image DP obs remains images + relative `ee_pose` + `hand_joint`; action remains 31-D (relative EE + absolute hand).
- Sim zarr has no EE and no images; at least `hand_joint` `(N,22)` and `action` `(N,22)` plus `episode_ends`.
- Never `create_fit` on a concatenated 31-D action; new ckpts do not store `normalizer['action']`.
- `normalizer_path` is optional. Unset: fit `hand_joint` (and `ee_pose` if present) on the current dataset. Set to a valid npz: load sim hand stats. Set but missing/corrupt file: raise, do not fall back.
- Independent-fit mode emits a once-per-process warning that this run's hand scaler is local to this dataset.
- If `normalizer['action']` exists, treat as legacy 31-D scaler and emit a once-per-process old-ckpt warning.
- `normalize_action` / `unnormalize_action` split EE vs hand for new models; legacy 31-D uses `normalizer['action']` only.
- `pred_action_steps_only=True` for both hand trainings; default `False` so existing image training is unchanged.
- Step: `horizon=2`, `n_obs_steps=2`, `n_action_steps=1`. Chunk: `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`.
- Noise schedule matches the DINO image yaml (DDIM, epsilon, 100 train steps).
- Do not add `GuidedPolicy`, `inference_dp_lowdim.py`, or `DiffusionUnetLowdimPolicy`.
- Do not convert sim data in this plan (`data2dp.py` unchanged).
- Constants: `EE_DIM=9`, `HAND_DIM=22`.

## File map

| File | Responsibility |
|--|--|
| `diffusion_policy/common/bulb_action_normalizer.py` | Split normalize/unnormalize; load sim hand stats |
| `scripts/fit_bulb_sim_normalizer.py` | Stream sim zarr → freeze `hand_joint` min/max |
| `diffusion_policy/dataset/bulb_image_dataset.py` | `shape_meta` + optional images/EE + split `get_normalizer` |
| `diffusion_policy/model/vision/multi_image_obs_encoder.py` | Allow `rgb_model=None` when no rgb keys |
| `diffusion_policy/policy/diffusion_unet_image_policy.py` | `encode_obs`, `predict_eps`, `pred_action_steps_only`, split action norm |
| `diffusion_policy/config/task/bulb_hand.yaml` | Hand task contract |
| `diffusion_policy/config/task/bulb_image.yaml` | Pass `shape_meta` + `normalizer_path` |
| `diffusion_policy/config/train_diffusion_unet_bulb_hand_step_workspace.yaml` | Step training |
| `diffusion_policy/config/train_diffusion_unet_bulb_hand_chunk_workspace.yaml` | Chunk training |
| `train_bulb_hand_step.sh` / `train_bulb_hand_chunk.sh` | Launchers |
| `train_bulb_dino.sh` | Pass sim hand stats path |
| `inference_dp.py` | Dispatch on ckpt `shape_meta` / `action_dim` |
| `tests/test_bulb_action_normalizer.py` | Split scaler tests |
| `tests/test_bulb_sim_normalizer.py` | Streaming min/max + npz |
| `tests/test_bulb_hand_dataset.py` | Tiny zarr dataset tests |
| `tests/test_multi_image_obs_encoder_lowdim.py` | No-rgb encoder |
| `tests/test_diffusion_unet_image_policy_hand.py` | Step/chunk shapes + encode/eps |

---

### Task 1: Split action normalizer helpers

**Files:**
- Create: `diffusion_policy/common/bulb_action_normalizer.py`
- Test: `tests/test_bulb_action_normalizer.py`

**Interfaces:**
- Consumes: `LinearNormalizer`, `get_range_normalizer_from_stat`
- Produces:
  - `EE_DIM = 9`, `HAND_DIM = 22`
  - `normalize_action(normalizer: LinearNormalizer, action: Tensor) -> Tensor`
  - `unnormalize_action(normalizer: LinearNormalizer, naction: Tensor) -> Tensor`
  - `load_hand_joint_stat(path: str) -> dict` with numpy arrays `min`,`max`,`mean`,`std` shape `(22,)`
  - `hand_joint_normalizer_from_stat(stat: dict) -> SingleFieldLinearNormalizer`
  - `warn_legacy_action_normalizer() -> None`  # `warnings.warn` once per process


- [ ] **Step 1: Write the failing tests**

Create `tests/test_bulb_action_normalizer.py`:

```python
import warnings

import numpy as np
import pytest
import torch

from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.common.bulb_action_normalizer import (
    EE_DIM,
    HAND_DIM,
    has_legacy_action_normalizer,
    warn_legacy_action_normalizer,
    hand_joint_normalizer_from_stat,
    load_hand_joint_stat,
    normalize_action,
    unnormalize_action,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _stat(lo, hi):
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    return {
        "min": lo,
        "max": hi,
        "mean": (lo + hi) / 2,
        "std": np.maximum(hi - lo, 1e-6) / np.sqrt(12.0),
    }


def test_unnormalize_22_uses_only_hand_joint():
    normalizer = LinearNormalizer()
    hand_stat = _stat(np.zeros(HAND_DIM), np.ones(HAND_DIM) * 2)
    ee_stat = _stat(np.full(EE_DIM, -10.0), np.full(EE_DIM, 10.0))
    normalizer["hand_joint"] = get_range_normalizer_from_stat(hand_stat)
    normalizer["ee_pose"] = get_range_normalizer_from_stat(ee_stat)
    naction = torch.zeros(2, 4, HAND_DIM)
    out = unnormalize_action(normalizer, naction)
    assert out.shape == (2, 4, HAND_DIM)
    torch.testing.assert_close(out, torch.ones_like(out))


def test_unnormalize_31_splits_ee_and_hand():
    normalizer = LinearNormalizer()
    hand_stat = _stat(np.zeros(HAND_DIM), np.ones(HAND_DIM) * 2)
    ee_stat = _stat(np.zeros(EE_DIM), np.ones(EE_DIM) * 4)
    normalizer["hand_joint"] = get_range_normalizer_from_stat(hand_stat)
    normalizer["ee_pose"] = get_range_normalizer_from_stat(ee_stat)
    naction = torch.zeros(1, 3, EE_DIM + HAND_DIM)
    out = unnormalize_action(normalizer, naction)
    assert out.shape[-1] == 31
    torch.testing.assert_close(out[..., :EE_DIM], torch.full((1, 3, EE_DIM), 2.0))
    torch.testing.assert_close(out[..., EE_DIM:], torch.ones(1, 3, HAND_DIM))


def test_normalize_roundtrip_31():
    normalizer = LinearNormalizer()
    hand_stat = _stat(np.zeros(HAND_DIM), np.ones(HAND_DIM))
    ee_stat = _stat(-np.ones(EE_DIM), np.ones(EE_DIM))
    normalizer["hand_joint"] = get_range_normalizer_from_stat(hand_stat)
    normalizer["ee_pose"] = get_range_normalizer_from_stat(ee_stat)
    action = torch.cat(
        [torch.zeros(1, 2, EE_DIM), torch.full((1, 2, HAND_DIM), 0.25)], dim=-1
    )
    naction = normalize_action(normalizer, action)
    restored = unnormalize_action(normalizer, naction)
    torch.testing.assert_close(restored, action, atol=1e-5, rtol=1e-5)


def test_legacy_31_uses_action_key_not_split():
    import diffusion_policy.common.bulb_action_normalizer as m
    m._LEGACY_ACTION_NORMALIZER_WARNED = False
    normalizer = LinearNormalizer()
    action = np.concatenate(
        [np.zeros((8, EE_DIM)), np.ones((8, HAND_DIM))], axis=-1
    ).astype(np.float32)
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(action)
    dummy_stat = _stat(np.full(HAND_DIM, 99.0), np.full(HAND_DIM, 100.0))
    normalizer["hand_joint"] = get_range_normalizer_from_stat(dummy_stat)
    dummy_ee = _stat(np.full(EE_DIM, 99.0), np.full(EE_DIM, 100.0))
    normalizer["ee_pose"] = get_range_normalizer_from_stat(dummy_ee)
    assert has_legacy_action_normalizer(normalizer)
    naction = torch.zeros(1, 2, 31)
    with pytest.warns(UserWarning, match="legacy"):
        out = unnormalize_action(normalizer, naction)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        unnormalize_action(normalizer, naction)
        assert record == []
    split = torch.cat(
        [
            normalizer["ee_pose"].unnormalize(naction[..., :EE_DIM]),
            normalizer["hand_joint"].unnormalize(naction[..., EE_DIM:]),
        ],
        dim=-1,
    )
    assert not torch.allclose(out, split)


def test_rejects_bad_last_dim():
    normalizer = LinearNormalizer()
    with pytest.raises(ValueError, match="22 or 31"):
        unnormalize_action(normalizer, torch.zeros(1, 3, 10))


def test_load_hand_joint_stat_roundtrip(tmp_path):
    path = tmp_path / "hand.npz"
    mn = np.arange(HAND_DIM, dtype=np.float32)
    mx = mn + 1
    np.savez(path, min=mn, max=mx)
    stat = load_hand_joint_stat(str(path))
    np.testing.assert_array_equal(stat["min"], mn)
    field = hand_joint_normalizer_from_stat(stat)
    x = torch.from_numpy(mn)
    torch.testing.assert_close(field.normalize(x), torch.full((HAND_DIM,), -1.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bulb_action_normalizer.py -v`

Expected: FAIL with `ModuleNotFoundError: diffusion_policy.common.bulb_action_normalizer`

- [ ] **Step 3: Implement helpers**

Create `diffusion_policy/common/bulb_action_normalizer.py`:

```python
from __future__ import annotations

import os
from typing import Dict
import warnings

import numpy as np
import torch

from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

EE_DIM = 9
HAND_DIM = 22
ACTION_DIM = EE_DIM + HAND_DIM

_LEGACY_ACTION_NORMALIZER_WARNED = False

LEGACY_ACTION_NORMALIZER_WARNING = (
    "Using a legacy Diffusion Policy checkpoint whose normalizer still has "
    "the 31-D 'action' key (the whole EE+hand vector was create_fit together). "
    "Unnormalize will use that single scaler, not the new split "
    "ee_pose (real) + hand_joint (sim) stats. Do not mix this checkpoint's "
    "normalized actions or scores with a sim-hand DP."
)


def has_legacy_action_normalizer(normalizer: LinearNormalizer) -> bool:
    return "action" in normalizer.params_dict


def warn_legacy_action_normalizer() -> None:
    global _LEGACY_ACTION_NORMALIZER_WARNED
    if _LEGACY_ACTION_NORMALIZER_WARNED:
        return
    _LEGACY_ACTION_NORMALIZER_WARNED = True
    warnings.warn(LEGACY_ACTION_NORMALIZER_WARNING, UserWarning, stacklevel=3)



def _as_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def normalize_action(normalizer: LinearNormalizer, action: torch.Tensor) -> torch.Tensor:
    action = _as_tensor(action)
    last = action.shape[-1]
    if last == HAND_DIM:
        return normalizer["hand_joint"].normalize(action)
    if last == ACTION_DIM:
        if has_legacy_action_normalizer(normalizer):
            warn_legacy_action_normalizer()
            return normalizer["action"].normalize(action)
        ee = normalizer["ee_pose"].normalize(action[..., :EE_DIM])
        hand = normalizer["hand_joint"].normalize(action[..., EE_DIM:])
        return torch.cat([ee, hand], dim=-1)
    raise ValueError(f"action last dim must be 22 or 31, got {last}")


def unnormalize_action(normalizer: LinearNormalizer, naction: torch.Tensor) -> torch.Tensor:
    naction = _as_tensor(naction)
    last = naction.shape[-1]
    if last == HAND_DIM:
        return normalizer["hand_joint"].unnormalize(naction)
    if last == ACTION_DIM:
        if has_legacy_action_normalizer(normalizer):
            warn_legacy_action_normalizer()
            return normalizer["action"].unnormalize(naction)
        ee = normalizer["ee_pose"].unnormalize(naction[..., :EE_DIM])
        hand = normalizer["hand_joint"].unnormalize(naction[..., EE_DIM:])
        return torch.cat([ee, hand], dim=-1)
    raise ValueError(f"action last dim must be 22 or 31, got {last}")


def load_hand_joint_stat(path: str) -> Dict[str, np.ndarray]:
    path = os.path.expanduser(path)
    data = np.load(path)
    if "min" not in data.files or "max" not in data.files:
        raise KeyError(f"{path} must contain 'min' and 'max'")
    mn = np.asarray(data["min"], dtype=np.float32).reshape(HAND_DIM)
    mx = np.asarray(data["max"], dtype=np.float32).reshape(HAND_DIM)
    mean = np.asarray(data["mean"], dtype=np.float32).reshape(HAND_DIM) if "mean" in data.files else (mn + mx) / 2
    std = (
        np.asarray(data["std"], dtype=np.float32).reshape(HAND_DIM)
        if "std" in data.files
        else np.maximum(mx - mn, 1e-6) / np.sqrt(12.0)
    )
    return {"min": mn, "max": mx, "mean": mean, "std": std}


def hand_joint_normalizer_from_stat(stat: Dict[str, np.ndarray]) -> SingleFieldLinearNormalizer:
    return get_range_normalizer_from_stat(stat)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bulb_action_normalizer.py -v`

Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/common/bulb_action_normalizer.py tests/test_bulb_action_normalizer.py
git commit -m "$(cat <<'EOF'
Add split EE/hand action normalizer helpers.

EOF
)"
```

---

### Task 2: Stream sim hand min/max and freeze npz

**Files:**
- Create: `scripts/fit_bulb_sim_normalizer.py`
- Modify: `diffusion_policy/common/bulb_action_normalizer.py` (add `streaming_minmax` + `collect_hand_joint_stat`)
- Test: `tests/test_bulb_sim_normalizer.py`

**Interfaces:**
- Consumes: `load_hand_joint_stat`, ReplayBuffer on-disk zarr
- Produces:
  - `streaming_minmax(array, chunk_size=65536) -> tuple[np.ndarray, np.ndarray]`
  - `collect_hand_joint_stat(zarr_path: str, chunk_size: int = 65536) -> dict`
  - CLI writes `hand_joint` npz with `min`/`max`/`mean`/`std`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bulb_sim_normalizer.py`:

```python
import os
import numpy as np
import zarr

from diffusion_policy.common.bulb_action_normalizer import (
    HAND_DIM,
    collect_hand_joint_stat,
    load_hand_joint_stat,
    streaming_minmax,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer


def test_streaming_minmax_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1000, HAND_DIM)).astype(np.float32)
    mn, mx = streaming_minmax(data, chunk_size=128)
    np.testing.assert_allclose(mn, data.min(axis=0))
    np.testing.assert_allclose(mx, data.max(axis=0))


def _write_hand_zarr(root):
    zarr_path = os.path.join(root, "replay_buffer.zarr")
    store = zarr.DirectoryStore(zarr_path)
    buf = ReplayBuffer.create_empty_zarr(storage=store)
    joints = np.stack([np.arange(HAND_DIM, dtype=np.float32), np.arange(HAND_DIM, dtype=np.float32) + 3], axis=0)
    actions = joints + 1
    buf.add_episode({"hand_joint": joints, "action": actions})
    return zarr_path


def test_collect_hand_joint_stat_uses_both_arrays(tmp_path):
    zarr_path = _write_hand_zarr(tmp_path)
    stat = collect_hand_joint_stat(zarr_path, chunk_size=1)
    np.testing.assert_array_equal(stat["min"], np.arange(HAND_DIM, dtype=np.float32))
    np.testing.assert_array_equal(stat["max"], np.arange(HAND_DIM, dtype=np.float32) + 4)


def test_cli_writes_npz(tmp_path):
    zarr_path = _write_hand_zarr(tmp_path)
    out = tmp_path / "hand.npz"
    assert os.system(
        f"python scripts/fit_bulb_sim_normalizer.py --zarr {zarr_path} --output {out}"
    ) == 0
    stat = load_hand_joint_stat(str(out))
    assert stat["min"].shape == (HAND_DIM,)
    assert stat["max"].shape == (HAND_DIM,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bulb_sim_normalizer.py -v`

Expected: FAIL (`streaming_minmax` not defined)

- [ ] **Step 3: Implement streaming stats + CLI**

Append to `diffusion_policy/common/bulb_action_normalizer.py`:

```python
def streaming_minmax(array, chunk_size: int = 65536):
    n = int(array.shape[0])
    if n == 0:
        raise ValueError("cannot compute min/max on empty array")
    running_min = None
    running_max = None
    for start in range(0, n, chunk_size):
        block = np.asarray(array[start : start + chunk_size], dtype=np.float32)
        block = block.reshape(block.shape[0], -1)
        bmin = block.min(axis=0)
        bmax = block.max(axis=0)
        running_min = bmin if running_min is None else np.minimum(running_min, bmin)
        running_max = bmax if running_max is None else np.maximum(running_max, bmax)
    return running_min.astype(np.float32), running_max.astype(np.float32)


def _hand_array_from_buffer(replay, key: str):
    arr = replay[key]
    tail = arr.shape[-1]
    if tail == HAND_DIM:
        return arr
    if tail == ACTION_DIM:
        return arr[..., EE_DIM:]
    raise ValueError(f"{key} last dim must be 22 or 31, got {tail}")


def collect_hand_joint_stat(zarr_path: str, chunk_size: int = 65536) -> Dict[str, np.ndarray]:
    from diffusion_policy.common.replay_buffer import ReplayBuffer

    replay = ReplayBuffer.create_from_path(zarr_path, mode="r")
    keys = []
    if "hand_joint" in replay.keys():
        keys.append("hand_joint")
    if "action" in replay.keys():
        keys.append("action")
    if "state" in replay.keys() and "hand_joint" not in replay.keys():
        keys.append("state")
    if not keys:
        raise KeyError("replay buffer needs hand_joint, action, or state")
    running_min = None
    running_max = None
    for key in keys:
        mn, mx = streaming_minmax(_hand_array_from_buffer(replay, key), chunk_size=chunk_size)
        running_min = mn if running_min is None else np.minimum(running_min, mn)
        running_max = mx if running_max is None else np.maximum(running_max, mx)
    return {
        "min": running_min,
        "max": running_max,
        "mean": (running_min + running_max) / 2,
        "std": np.maximum(running_max - running_min, 1e-6) / np.sqrt(12.0),
    }
```

Create `scripts/fit_bulb_sim_normalizer.py`:

```python
#!/usr/bin/env python3
"""Freeze hand-joint min/max from a sim Diffusion Policy zarr."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from diffusion_policy.common.bulb_action_normalizer import collect_hand_joint_stat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", required=True, help="Path to replay_buffer.zarr")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--chunk-size", type=int, default=65536)
    args = parser.parse_args()
    stat = collect_hand_joint_stat(os.path.expanduser(args.zarr), chunk_size=args.chunk_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **stat)
    print(f"wrote {args.output} min={stat['min'][:3]} max={stat['max'][:3]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bulb_sim_normalizer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/common/bulb_action_normalizer.py scripts/fit_bulb_sim_normalizer.py tests/test_bulb_sim_normalizer.py
git commit -m "$(cat <<'EOF'
Add streaming sim hand-joint min/max freeze script.

EOF
)"
```

---

### Task 3: Allow image encoder with no RGB keys

**Files:**
- Modify: `diffusion_policy/model/vision/multi_image_obs_encoder.py`
- Test: `tests/test_multi_image_obs_encoder_lowdim.py`

**Interfaces:**
- Consumes: `shape_meta['obs']` with only `hand_joint`
- Produces: `MultiImageObsEncoder(..., rgb_model=None)` concatenates low-dim keys; `output_shape() == (22,)`

- [ ] **Step 1: Write the failing test**

```python
import torch

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


def test_lowdim_only_concat_hand_joint():
    shape_meta = {"obs": {"hand_joint": {"shape": [22], "type": "low_dim"}}}
    encoder = MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=None)
    x = {"hand_joint": torch.zeros(4, 22)}
    out = encoder(x)
    assert tuple(out.shape) == (4, 22)
    assert tuple(encoder.output_shape()) == (22,)


def test_rgb_without_model_raises():
    shape_meta = {
        "obs": {
            "front_image": {"shape": [3, 240, 320], "type": "rgb"},
            "hand_joint": {"shape": [22], "type": "low_dim"},
        }
    }
    try:
        MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=None)
    except RuntimeError as exc:
        assert "rgb_model" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multi_image_obs_encoder_lowdim.py -v`

Expected: FAIL (rgb_model required / assert)

- [ ] **Step 3: Minimal encoder change**

In `multi_image_obs_encoder.py`:

1. Change signature: `rgb_model: Union[nn.Module, Dict[str, nn.Module], None] = None`
2. Replace the `if share_rgb_model:` block with:

```python
        if share_rgb_model:
            if rgb_model is None:
                raise RuntimeError("share_rgb_model=True requires rgb_model")
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model
```

3. Inside `if type == 'rgb':`, before using `rgb_model`, if `rgb_model is None` raise `RuntimeError(f"rgb key {key} requires rgb_model")`.

Low-dim branch is already concat; no other change.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_multi_image_obs_encoder_lowdim.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/model/vision/multi_image_obs_encoder.py tests/test_multi_image_obs_encoder_lowdim.py
git commit -m "$(cat <<'EOF'
Allow MultiImageObsEncoder with low-dim keys only.

EOF
)"
```

---

### Task 4: Dataset follows shape_meta and split normalizer

**Files:**
- Modify: `diffusion_policy/dataset/bulb_image_dataset.py`
- Test: `tests/test_bulb_hand_dataset.py`

**Interfaces:**
- Consumes: `shape_meta`, optional `normalizer_path`, Task 1–2 helpers
- Produces: `BulbImageDataset(..., shape_meta=..., normalizer_path=...)` whose `__getitem__` obs keys match `shape_meta['obs']` and whose `get_normalizer()` never `create_fit`s 31-D action when `normalizer_path` is set

Default `shape_meta=None` must preserve current image 31-D + rgb behavior (needed until `bulb_image.yaml` is updated in Task 6).

- [ ] **Step 1: Write the failing tests**

```python
import os
import numpy as np
import torch
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.bulb_image_dataset import BulbImageDataset

HAND_SHAPE_META = {
    "obs": {"hand_joint": {"shape": [22], "type": "low_dim"}},
    "action": {"shape": [22]},
}

IMAGE_SHAPE_META = {
    "obs": {
        "front_image": {"shape": [3, 16, 16], "type": "rgb"},
        "wrist_image": {"shape": [3, 16, 16], "type": "rgb"},
        "ee_pose": {"shape": [9], "type": "low_dim"},
        "hand_joint": {"shape": [22], "type": "low_dim"},
    },
    "action": {"shape": [31]},
}


def _hand_dataset_dir(tmp_path):
    root = tmp_path / "sim"
    zarr_path = root / "replay_buffer.zarr"
    buf = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(zarr_path)))
    T = 12
    buf.add_episode(
        {
            "hand_joint": np.linspace(0, 1, T * 22, dtype=np.float32).reshape(T, 22),
            "action": np.linspace(0.1, 1.1, T * 22, dtype=np.float32).reshape(T, 22),
        }
    )
    return str(root)


def test_hand_dataset_obs_and_action_shapes(tmp_path):
    ds = BulbImageDataset(
        dataset_path=_hand_dataset_dir(tmp_path),
        horizon=2,
        pad_before=1,
        pad_after=0,
        n_obs_steps=2,
        shape_meta=HAND_SHAPE_META,
        val_ratio=0.0,
    )
    sample = ds[0]
    assert set(sample["obs"].keys()) == {"hand_joint"}
    assert sample["obs"]["hand_joint"].shape == (2, 22)
    assert sample["action"].shape[-1] == 22
    assert "front_image" not in sample["obs"]


def test_hand_normalizer_loads_sim_stats(tmp_path):
    from diffusion_policy.common.bulb_action_normalizer import collect_hand_joint_stat

    root = _hand_dataset_dir(tmp_path)
    stat_path = tmp_path / "hand.npz"
    np.savez(stat_path, **collect_hand_joint_stat(os.path.join(root, "replay_buffer.zarr")))
    ds = BulbImageDataset(
        dataset_path=root,
        horizon=2,
        pad_before=1,
        pad_after=0,
        n_obs_steps=2,
        shape_meta=HAND_SHAPE_META,
        normalizer_path=str(stat_path),
        val_ratio=0.0,
    )
    normalizer = ds.get_normalizer()
    assert "hand_joint" in normalizer.params_dict
    assert "ee_pose" not in normalizer.params_dict
    assert "action" not in normalizer.params_dict


def test_image_normalizer_path_does_not_fit_31d_action(tmp_path):
    root = tmp_path / "real"
    zarr_path = root / "replay_buffer.zarr"
    buf = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(zarr_path)))
    T = 16
    state = np.zeros((T, 31), dtype=np.float32)
    state[:, 0] = np.linspace(0, 0.1, T)
    state[:, 9:] = 0.2
    action = state.copy()
    action[:, 0] += 0.01
    buf.add_episode(
        {
            "state": state,
            "action": action,
            "front_image": np.zeros((T, 16, 16, 3), dtype=np.uint8),
            "wrist_image": np.zeros((T, 16, 16, 3), dtype=np.uint8),
        }
    )
    stat_path = tmp_path / "hand.npz"
    np.savez(
        stat_path,
        min=np.zeros(22, dtype=np.float32),
        max=np.ones(22, dtype=np.float32),
    )
    ds = BulbImageDataset(
        dataset_path=str(root),
        horizon=8,
        pad_before=1,
        pad_after=7,
        n_obs_steps=2,
        shape_meta=IMAGE_SHAPE_META,
        normalizer_path=str(stat_path),
        val_ratio=0.0,
    )
    sample = ds[0]
    assert sample["action"].shape[-1] == 31
    assert set(sample["obs"]) >= {"ee_pose", "hand_joint", "front_image", "wrist_image"}
    normalizer = ds.get_normalizer()
    assert "action" not in normalizer.params_dict
    assert "ee_pose" in normalizer.params_dict
    assert "hand_joint" in normalizer.params_dict
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bulb_hand_dataset.py -v`

Expected: FAIL (`shape_meta` unexpected kwarg)

- [ ] **Step 3: Implement dataset changes**

Rewrite `BulbImageDataset.__init__` to take `shape_meta: dict | None = None` and `normalizer_path: str | None = None`.

If `shape_meta is None`, use the current image contract (rgb + ee + hand, action 31).

Parse:

```python
obs_meta = self.shape_meta["obs"]
self.rgb_keys = [k for k, a in obs_meta.items() if a.get("type") == "rgb"]
self.lowdim_keys = [k for k, a in obs_meta.items() if a.get("type", "low_dim") == "low_dim"]
self.action_dim = int(self.shape_meta["action"]["shape"][0])
if self.action_dim not in (22, 31):
    raise ValueError(f"action dim must be 22 or 31, got {self.action_dim}")
```

Buffer keys:

- Always need `action`.
- If `action_dim == 31` or `"ee_pose" in lowdim_keys`: require `state` with last dim 31.
- Else prefer `hand_joint` last dim 22; if missing, `state` last dim 22 or 31 (slice `[9:31]`).
- RGB keys required only when listed; extra image keys in zarr are ignored.

Sampler `keys` = rgb keys + source lowdim keys (`state` and/or `hand_joint`) + `action`.
`key_first_k` only for obs sources (images, `state`, `hand_joint`), not `action`.

`_lowdim_buffer` copies only those low-dim arrays (never images).

`_convert_lowdim` for 31-D stays as today. For 22-D:

```python
def _hand_seq(self, sample):
    if "hand_joint" in sample:
        hand = sample["hand_joint"]
    else:
        state = sample["state"]
        hand = state if state.shape[-1] == 22 else state[:, EE_DIM:ACTION_DIM]
    action = sample["action"]
    if action.shape[-1] == 31:
        action = action[:, EE_DIM:ACTION_DIM]
    return hand[: self.n_obs_steps].astype(np.float32), action.astype(np.float32)
```

`__getitem__`: build `obs` only for keys in `shape_meta['obs']`.

`get_normalizer`:

```python
normalizer = LinearNormalizer()
if self.normalizer_path:
    path = os.path.expanduser(self.normalizer_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"hand-joint normalizer file not found: {path} "
            "(omit normalizer_path to fit on this dataset instead)"
        )
    stat = load_hand_joint_stat(path)
    normalizer["hand_joint"] = hand_joint_normalizer_from_stat(stat)
else:
    warnings.warn(
        "No normalizer_path: fitting hand_joint on this dataset. "
        "This scaler is not shared with the other domain; do not mix "
        "normalized hand actions or scores across sim/real.",
        UserWarning,
        stacklevel=2,
    )
    hands = []
    for index in range(len(self.lowdim_sampler)):
        sample = self.lowdim_sampler.sample_sequence(index)
        if self.action_dim == 22:
            hand, _ = self._hand_seq(sample)
            hands.append(hand)
        else:
            _, hand, _ = self._convert_lowdim(sample["state"], sample["action"])
            hands.append(hand)
    normalizer["hand_joint"] = SingleFieldLinearNormalizer.create_fit(
        np.concatenate(hands, axis=0)
    )
if "ee_pose" in self.lowdim_keys:
    relative_ee = []
    for index in range(len(self.lowdim_sampler)):
        sample = self.lowdim_sampler.sample_sequence(index)
        ee, _, _ = self._convert_lowdim(sample["state"], sample["action"])
        relative_ee.append(ee)
    normalizer["ee_pose"] = SingleFieldLinearNormalizer.create_fit(
        np.concatenate(relative_ee, axis=0)
    )
for key in self.rgb_keys:
    normalizer[key] = get_image_range_normalizer()
return normalizer
```

Do **not** assign `normalizer["action"]`.

Tests:
- `normalizer_path=None` → `get_normalizer()` succeeds, has `hand_joint`, no `action`; `pytest.warns` matching `No normalizer_path`.
- `normalizer_path` pointing at a missing file → `FileNotFoundError`, does not fit.
- valid path → load sim stats, do not fit hands on current data.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bulb_hand_dataset.py tests/test_bulb_action_normalizer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/dataset/bulb_image_dataset.py tests/test_bulb_hand_dataset.py
git commit -m "$(cat <<'EOF'
Teach BulbImageDataset shape_meta and sim hand normalizer loading.

EOF
)"
```

---

### Task 5: Image policy encode/eps, pred_action_steps_only, split action norm

**Files:**
- Modify: `diffusion_policy/policy/diffusion_unet_image_policy.py`
- Test: `tests/test_diffusion_unet_image_policy_hand.py`

**Interfaces:**
- Consumes: Task 1 helpers, Task 3 encoder
- Produces on `DiffusionUnetImagePolicy`:
  - `__init__(..., pred_action_steps_only: bool = False, **kwargs)`
  - `encode_obs(self, obs_dict: Dict[str, Tensor]) -> Tensor`  # global_cond `(B, Do * To)` when `obs_as_global_cond`
  - `predict_eps(self, x_t, t, global_cond, local_cond=None) -> Tensor`
  - `predict_action` output `action` shape `(B, n_action_steps, Da)`; if `pred_action_steps_only` that slice is the full denoised trajectory
  - `compute_loss` uses `normalize_action` / split trajectory when `pred_action_steps_only`

- [ ] **Step 1: Write the failing tests**

```python
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.common.bulb_action_normalizer import HAND_DIM, normalize_action
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


def _hand_policy(pred_action_steps_only, n_action_steps, horizon):
    shape_meta = {
        "obs": {"hand_joint": {"shape": [22], "type": "low_dim"}},
        "action": {"shape": [22]},
    }
    encoder = MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=None)
    scheduler = DDIMScheduler(
        num_train_timesteps=4,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=encoder,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=2,
        num_inference_steps=2,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        pred_action_steps_only=pred_action_steps_only,
    )
    normalizer = LinearNormalizer()
    stat = {
        "min": torch.zeros(HAND_DIM),
        "max": torch.ones(HAND_DIM),
        "mean": torch.full((HAND_DIM,), 0.5),
        "std": torch.ones(HAND_DIM),
    }
    # get_range_normalizer_from_stat wants numpy
    import numpy as np
    np_stat = {k: v.numpy() for k, v in stat.items()}
    from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat as gr
    normalizer["hand_joint"] = gr(np_stat)
    policy.set_normalizer(normalizer)
    policy.num_inference_steps = 2
    return policy


def test_step_predict_action_shape():
    policy = _hand_policy(True, 1, 2)
    obs = {"hand_joint": torch.zeros(1, 2, 22)}
    out = policy.predict_action(obs)
    assert tuple(out["action"].shape) == (1, 1, 22)


def test_chunk_predict_action_shape():
    policy = _hand_policy(True, 8, 16)
    obs = {"hand_joint": torch.zeros(1, 2, 22)}
    out = policy.predict_action(obs)
    assert tuple(out["action"].shape) == (1, 8, 22)


def test_encode_obs_and_predict_eps_shapes():
    policy = _hand_policy(True, 1, 2)
    obs = {"hand_joint": torch.zeros(2, 2, 22)}
    cond = policy.encode_obs(obs)
    assert cond.shape[0] == 2
    x = torch.zeros(2, 1, 22)
    t = torch.zeros(2, dtype=torch.long)
    eps = policy.predict_eps(x, t, cond)
    assert eps.shape == x.shape


def test_compute_loss_step_runs():
    policy = _hand_policy(True, 1, 2)
    batch = {
        "obs": {"hand_joint": torch.zeros(3, 2, 22)},
        "action": torch.zeros(3, 2, 22),
    }
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_diffusion_unet_image_policy_hand.py -v`

Expected: FAIL (`pred_action_steps_only` unexpected / `encode_obs` missing)

- [ ] **Step 3: Implement policy methods**

In `diffusion_unet_image_policy.py`:

1. Import `normalize_action`, `unnormalize_action`.
2. Add `pred_action_steps_only=False` **before** `**kwargs`. If True, assert `obs_as_global_cond`.
3. Store `self.pred_action_steps_only = pred_action_steps_only`.
4. Add:

```python
    def encode_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        To = self.n_obs_steps
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        if self.obs_as_global_cond:
            return nobs_features.reshape(B, -1)
        return nobs_features.reshape(B, To, -1)

    def predict_eps(self, x_t, t, global_cond, local_cond=None):
        return self.model(x_t, t, local_cond=local_cond, global_cond=global_cond)
```

5. In `predict_action`: `global_cond = self.encode_obs(obs_dict)` when `obs_as_global_cond`. Trajectory length `Ta = self.n_action_steps if self.pred_action_steps_only else self.horizon`. `cond_data` zeros of `(B, Ta, Da)`. After sample: `action_pred = unnormalize_action(self.normalizer, nsample[..., :Da])`. If `pred_action_steps_only`: `action = action_pred`; else existing `start = To-1; end = start + n_action_steps`.

6. In `compute_loss`: `nactions = normalize_action(self.normalizer, batch['action'])`. If `pred_action_steps_only`: `start = self.n_obs_steps - 1; trajectory = nactions[:, start:start+self.n_action_steps]`; `condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)`. Else keep mask_generator path. Use `encode_obs` on `batch['obs']` for global_cond.

Do not call `self.normalizer['action']` in the new path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_diffusion_unet_image_policy_hand.py tests/test_bulb_action_normalizer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add diffusion_policy/policy/diffusion_unet_image_policy.py tests/test_diffusion_unet_image_policy_hand.py
git commit -m "$(cat <<'EOF'
Expose encode_obs/predict_eps and hand-only action-step training.

EOF
)"
```

---

### Task 6: Hydra configs and train launchers

**Files:**
- Create: `diffusion_policy/config/task/bulb_hand.yaml`
- Create: `diffusion_policy/config/train_diffusion_unet_bulb_hand_step_workspace.yaml`
- Create: `diffusion_policy/config/train_diffusion_unet_bulb_hand_chunk_workspace.yaml`
- Create: `train_bulb_hand_step.sh`
- Create: `train_bulb_hand_chunk.sh`
- Modify: `diffusion_policy/config/task/bulb_image.yaml`
- Modify: `train_bulb_dino.sh`

**Interfaces:**
- Consumes: Task 4 dataset ctor kwargs `shape_meta`, `normalizer_path`
- Produces: `python train.py --config-name=train_diffusion_unet_bulb_hand_step_workspace` hydra-composes; image yaml passes `shape_meta` and `normalizer_path`

- [ ] **Step 1: Write `bulb_hand.yaml`**

```yaml
name: bulb_hand
dataset_path: /path/to/sim_bulb_hand_dp
normalizer_path: ${oc.env:BULB_HAND_NORMALIZER,null}

shape_meta: &shape_meta
  obs:
    hand_joint:
      shape: [22]
      type: low_dim
  action:
    shape: [22]

env_runner:
  _target_: diffusion_policy.env_runner.null_image_runner.NullImageRunner

dataset:
  _target_: diffusion_policy.dataset.bulb_image_dataset.BulbImageDataset
  dataset_path: ${task.dataset_path}
  shape_meta: ${shape_meta}
  normalizer_path: ${task.normalizer_path}
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  seed: 42
  val_ratio: 0.1
  max_train_episodes: null
```

- [ ] **Step 2: Write step/chunk workspace yamls**

Copy `train_diffusion_unet_dino_image_workspace.yaml` structure with these differences (both files):

```yaml
defaults:
  - _self_
  - task: bulb_hand

name: train_diffusion_unet_bulb_hand_step   # or _chunk
_target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace

pred_action_steps_only: True

policy:
  _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
  shape_meta: ${shape_meta}
  noise_scheduler:   # identical DDIM block as dino image yaml
  obs_encoder:
    _target_: diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
    shape_meta: ${shape_meta}
    rgb_model: null
    resize_shape: null
    crop_shape: null
    random_crop: False
    use_group_norm: False
    share_rgb_model: False
    imagenet_norm: False
  pred_action_steps_only: ${pred_action_steps_only}
  down_dims: [256, 512, 1024]
  diffusion_step_embed_dim: 128
  # remainder same as dino image (horizon refs, n_groups, etc.)

training:
  freeze_encoder: False
```

Step yaml: `horizon: 2`, `n_obs_steps: 2`, `n_action_steps: 1`.

Chunk yaml: `horizon: 16`, `n_obs_steps: 2`, `n_action_steps: 8`.

Keep the same DDIM scheduler fields as `train_diffusion_unet_dino_image_workspace.yaml` (`num_train_timesteps: 100`, `prediction_type: epsilon`, …).

- [ ] **Step 3: Update `bulb_image.yaml` dataset block**

Add:

```yaml
normalizer_path: ${oc.env:BULB_HAND_NORMALIZER,null}

dataset:
  _target_: diffusion_policy.dataset.bulb_image_dataset.BulbImageDataset
  dataset_path: ${task.dataset_path}
  shape_meta: ${shape_meta}
  normalizer_path: ${task.normalizer_path}
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  n_obs_steps: ${dataset_obs_steps}
  seed: 42
  val_ratio: 0.1
  max_train_episodes: null
```

未设环境变量时 Hydra 得到 `null`，dataset 走独立 fit。

- [ ] **Step 4: Write launchers**

`train_bulb_hand_step.sh` (chunk script is the same except config name and wandb name):

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-/home/wangtianyu/miniconda3/envs/dp/bin/python}"
DATASET_PATH="${DATASET_PATH:-/path/to/sim_bulb_hand_dp}"
NORMALIZER_ARGS=()
if [[ -n "${BULB_HAND_NORMALIZER:-}" ]]; then
  if [[ ! -f "$BULB_HAND_NORMALIZER" ]]; then
    echo "BULB_HAND_NORMALIZER file not found: $BULB_HAND_NORMALIZER" >&2
    exit 1
  fi
  NORMALIZER_ARGS+=(task.normalizer_path="$BULB_HAND_NORMALIZER")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-online}"
exec "$PYTHON" train.py \
    --config-name=train_diffusion_unet_bulb_hand_step_workspace \
    task=bulb_hand \
    task.dataset_path="$DATASET_PATH" \
    "${NORMALIZER_ARGS[@]}" \
    training.device=cuda:0 \
    logging.project=tacmp_diffusion_policy \
    logging.name=bulb_hand_step \
    "$@"
```

`train_bulb_dino.sh` 同样：只有设置了 `BULB_HAND_NORMALIZER` 才传入并校验文件存在；未设则不传，图像 DP 在真机数据上自己 fit 手。

`chmod +x` the two new scripts.

- [ ] **Step 5: Compose-check configs (no GPU train)**

Run:

```bash
python -c "
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pathlib
cfg_dir = str(pathlib.Path('diffusion_policy/config').resolve())
OmegaConf.register_new_resolver('eval', eval, replace=True)
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    for name in [
        'train_diffusion_unet_bulb_hand_step_workspace',
        'train_diffusion_unet_bulb_hand_chunk_workspace',
    ]:
        cfg = compose(config_name=name)
        assert cfg.policy.pred_action_steps_only is True
        assert cfg.task.shape_meta.action.shape == [22]
        print(name, 'ok', cfg.horizon, cfg.n_action_steps)
"
```

Expected: prints step `2 1` and chunk `16 8`, no exception.

- [ ] **Step 6: Commit**

```bash
git add diffusion_policy/config/task/bulb_hand.yaml diffusion_policy/config/task/bulb_image.yaml diffusion_policy/config/train_diffusion_unet_bulb_hand_step_workspace.yaml diffusion_policy/config/train_diffusion_unet_bulb_hand_chunk_workspace.yaml train_bulb_hand_step.sh train_bulb_hand_chunk.sh train_bulb_dino.sh
git commit -m "$(cat <<'EOF'
Add bulb_hand Hydra configs and train launchers.

EOF
)"
```

---

### Task 7: Inference dispatches on checkpoint shape_meta

**Files:**
- Modify: `inference_dp.py`

**Interfaces:**
- Consumes: `cfg.shape_meta` / `policy.action_dim` / `cfg.n_action_steps` from `load_policy`
- Produces: one `inference_dp.py` that feeds only obs keys present in the ckpt; 22-D outputs are concatenated with the latest absolute EE before `env.step_single`

- [ ] **Step 1: Add helpers next to `build_policy_obs`**

```python
def obs_keys_from_cfg(cfg) -> list[str]:
    return list(cfg.shape_meta.obs.keys())


def build_policy_obs(obs_history, device: torch.device, obs_keys: list[str]):
    observations = list(obs_history)
    base_ee_pose = observations[-1]["ee_pose"]
    policy_obs = {}
    if "ee_pose" in obs_keys:
        absolute_ee = np.stack([obs["ee_pose"] for obs in observations])
        relative_ee = ee_pose_relative_to(absolute_ee, base_ee_pose)
        policy_obs["ee_pose"] = torch.from_numpy(relative_ee[None]).float().to(device)

    def tensor(key):
        return torch.from_numpy(np.stack([obs[key] for obs in observations])[None]).float().to(device)

    for key in obs_keys:
        if key == "ee_pose":
            continue
        policy_obs[key] = tensor(key)
    return policy_obs, base_ee_pose


def policy_action_to_robot(action: np.ndarray, base_ee_pose: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] == ACTION_DIM:
        return mixed_actions_to_absolute(action, base_ee_pose)
    if action.shape[-1] == ACTION_DIM - EE_DIM:
        out = np.repeat(base_ee_pose[None, :], action.shape[0], axis=0)
        result = np.concatenate([out, action], axis=-1)
        return result
    raise ValueError(f"policy action last dim must be 22 or 31, got {action.shape[-1]}")
```

Keep `capture_obs` collecting full robot obs (images + ee + hand) so both ckpts can slice.

- [ ] **Step 2: Wire `main`**

After `load_policy`:

```python
obs_keys = obs_keys_from_cfg(cfg)
print(f"[policy] obs_keys={obs_keys} action_dim={int(policy.action_dim)}")
```

`--check_only`: build dummy tensors only for `obs_keys` (images `1,n_obs,3,240,320`; `ee_pose` `1,n_obs,9`; `hand_joint` `1,n_obs,22`). Assert `action.shape == (1, trained_action_steps, policy.action_dim)`.

Control loop:

```python
policy_obs, base_ee_pose = build_policy_obs(obs_history, device, obs_keys)
prediction = policy.predict_action(policy_obs)
mixed_or_hand = prediction["action"][0].detach().cpu().numpy()
absolute_actions = policy_action_to_robot(mixed_or_hand, base_ee_pose)[:action_chunk_steps]
```

Do not add a flag that slices a chunk ckpt down to one step as a substitute for the step ckpt.

- [ ] **Step 3: Smoke without hardware**

There is no hand ckpt yet. Run syntax check:

```bash
python -m py_compile inference_dp.py
```

Expected: exit 0.

If an existing image ckpt path is available, run:

```bash
python inference_dp.py --checkpoint <image.ckpt> --check_only --device cpu
```

Expected: `[check] checkpoint smoke inference passed` and action shape `(1, 8, 31)`.

- [ ] **Step 4: Commit**

```bash
git add inference_dp.py
git commit -m "$(cat <<'EOF'
Dispatch inference_dp on ckpt obs keys and 22-D hand actions.

EOF
)"
```

---

## Self-review vs spec

| Spec item | Task |
|--|--|
| Hand obs only `hand_joint`; action 22 | 4, 6 |
| Image DP 31-D chunk unchanged by default `pred_action_steps_only=False` | 5, 6 |
| Two trainings / two losses / two ckpts | 5, 6 |
| Sim zarr no images / no EE | 2, 4 |
| Hand min/max from sim; EE fit on real; no 31-D `create_fit` | 1, 2, 4 |
| Split normalize/unnormalize | 1, 5 |
| Legacy `normalizer['action']` still works | 1 |
| `encode_obs` / `predict_eps` | 5 |
| `rgb_model: null` | 3, 6 |
| Train scripts | 6 |
| Single `inference_dp.py`; hold latest EE for 22-D | 7 |
| No guidance / no second inference script / no lowdim policy / no sim converter | (omitted on purpose) |

No TBD/TODO placeholders remain in task steps.
