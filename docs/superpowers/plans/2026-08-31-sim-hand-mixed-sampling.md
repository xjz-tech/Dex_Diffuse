# Sim-Hand Mixed-Source Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separately labeled Sim-Hand mixed training mode that directly reads expdata and bulb TacMP trajectories and samples them with probabilities 0.8 and 0.2 without changing either source dataset.

**Architecture:** Adapt bulb `episode_*` NumPy trajectories into the existing 66-D observation/22-D action contract, compose the two source datasets without materializing a merged replay buffer, and select training indices through a chunked probability sampler. Keep validation unweighted, combine full-source normalization statistics analytically, preserve the original single-source path, and integrate custom sampling with both single-process and DDP workspace execution.

**Tech Stack:** Python 3.9, NumPy, PyTorch Dataset/DataLoader/Sampler/DDP, Hydra/OmegaConf, Zarr/HDF5, pytest, W&B.

**Spec:** `docs/superpowers/specs/2026-08-31-sim-hand-mixed-sampling-design.md`

## Global Constraints

- Production data root is exactly `/home/carus/mnt/bigai/Data`; local verification overrides it with `/share/generalvision/xiejunzhe/Data`.
- Source probabilities are exactly `expdata=0.8` and `bulb_tac=0.2`, and apply only to training index selection.
- Never rewrite, convert, delete, filter, downsample, or shorten either source dataset.
- Bulb mapping is fixed: `qpos=state[:, 9:31]`, `target_after=action[:, 9:31]`, `target_before=np.concatenate((qpos[0:1], target_after[:-1]))`.
- Observation remains `concat(qpos, target_before, target_before-qpos)` with shape `(T, 66)` and action remains absolute `target_after` with shape `(T, 22)`.
- Validation concatenates the two component validation datasets without probability sampling.
- Sampler auxiliary memory is bounded by a fixed chunk size, not total source cardinality.
- Existing `sim_hand_lowdim` task and `dp_train_sim_hand.sh` behavior must remain unchanged.
- The new mode must be labeled `sim_hand_mixed` and `exp_name=mixed` in Hydra/W&B configuration.
- The project directory currently has no usable Git repository metadata. Do not run `git add` or `git commit`; each task ends with an explicit focused-test checkpoint.

## File structure

| File | Responsibility |
|---|---|
| `diffusion_policy/dataset/bulb_tac_lowdim_dataset.py` | Validate/read bulb episode directories and expose standard Sim-Hand samples. |
| `diffusion_policy/common/probability_mixture_sampler.py` | Generate bounded-memory deterministic source-weighted indices for one or many ranks. |
| `diffusion_policy/dataset/mixed_lowdim_dataset.py` | Compose named source datasets, map indices/source IDs, merge validation datasets and normalizers. |
| `diffusion_policy/dataset/base_dataset.py` | Define the optional custom training-sampler and normalizer-count hooks with backward-compatible defaults. |
| `diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py` | Prefer custom samplers and log globally reduced realized source fractions. |
| `diffusion_policy/config/task/sim_hand_mixed.yaml` | Declare production paths, component datasets, 8:2 probabilities, and mixed label. |
| `dp_train_sim_hand_mixed.sh` | Launch single- or multi-GPU mixed mode while preserving existing W&B credential handling. |
| `tests/test_bulb_tac_lowdim_dataset.py` | Bulb mapping, validation split, cardinality, and error cases. |
| `tests/test_probability_mixture_sampler.py` | Probability, reproducibility, DDP length, validation, and large-cardinality memory behavior. |
| `tests/test_mixed_lowdim_dataset.py` | Composite routing, source metadata, validation, custom sampler, and merged normalization. |
| `tests/test_sim_hand_train_smoke.py` | Workspace custom-sampler/source-metric integration and bounded mixed training. |
| `tests/test_dp_train_sim_hand_mixed_launcher.py` | Launcher paths, label, GPU count, W&B mode, and pass-through overrides. |

---

### Task 1: Direct bulb TacMP trajectory adapter

**Files:**
- Create: `diffusion_policy/dataset/bulb_tac_lowdim_dataset.py`
- Create: `tests/test_bulb_tac_lowdim_dataset.py`

**Interfaces:**
- Consumes: numeric `episode_<id>` directories containing `state.npy` and `action.npy`, each `float32 (T, 31)`.
- Produces: `BulbTacLowdimDataset(BaseLowdimDataset)` with `replay_buffer`, `train_mask`, `val_mask`, `sampler`, `get_validation_dataset()`, `get_normalizer()`, `get_all_actions()`, and standard `{"obs", "action"}` samples.

- [ ] **Step 1: Write failing happy-path mapping and cardinality tests**

Create `tests/test_bulb_tac_lowdim_dataset.py` with these helpers and assertions:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diffusion_policy.dataset.bulb_tac_lowdim_dataset import (
    BulbTacLowdimDataset,
)

HAND_DIM = 22
OBS_DIM = 66


def _write_episode(root: Path, episode_id: int, length: int, base: float):
    episode = root / f"episode_{episode_id}"
    episode.mkdir(parents=True)
    state = np.zeros((length, 31), dtype=np.float32)
    action = np.zeros((length, 31), dtype=np.float32)
    steps = np.arange(length, dtype=np.float32)[:, None]
    channels = np.arange(HAND_DIM, dtype=np.float32)[None, :]
    state[:, 9:31] = base + steps + channels / 100.0
    action[:, 9:31] = base + 100.0 + steps + channels / 100.0
    np.save(episode / "state.npy", state)
    np.save(episode / "action.npy", action)
    return state, action


def _make_dataset(root: Path, val_ratio: float = 0.0):
    return BulbTacLowdimDataset(
        dataset_path=str(root),
        horizon=3,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=val_ratio,
    )


def test_bulb_mapping_matches_expdata_temporal_contract(tmp_path):
    state, action = _write_episode(tmp_path, 0, 4, 10.0)
    dataset = _make_dataset(tmp_path)

    qpos = state[:, 9:31]
    target_after = action[:, 9:31]
    target_before = np.concatenate((qpos[0:1], target_after[:-1]), axis=0)
    expected_obs = np.concatenate(
        (qpos, target_before, target_before - qpos), axis=-1
    )

    np.testing.assert_array_equal(dataset.replay_buffer.episode_ends, [4])
    np.testing.assert_allclose(dataset.replay_buffer["obs"], expected_obs)
    np.testing.assert_allclose(dataset.replay_buffer["action"], target_after)
    assert dataset.replay_buffer["obs"].shape == (4, OBS_DIM)
    assert dataset.replay_buffer["action"].shape == (4, HAND_DIM)
    assert set(dataset[0]) == {"obs", "action"}
    assert dataset[0]["obs"].dtype == torch.float32


def test_numeric_episode_order_and_source_cardinality_are_preserved(tmp_path):
    _write_episode(tmp_path, 10, 3, 1000.0)
    _write_episode(tmp_path, 2, 4, 200.0)
    dataset = _make_dataset(tmp_path)

    np.testing.assert_array_equal(dataset.replay_buffer.episode_ends, [4, 7])
    assert dataset.replay_buffer.n_steps == 7
    assert dataset.replay_buffer["obs"][0, 0] == 200.0
    assert dataset.replay_buffer["obs"][4, 0] == 1000.0
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run:

```bash
.local/miniforge3/envs/robodiff/bin/pytest -q \
  code/Dex_Diffuse-wip-dex-prior-guidance/tests/test_bulb_tac_lowdim_dataset.py
```

Expected: collection fails with `ModuleNotFoundError: diffusion_policy.dataset.bulb_tac_lowdim_dataset`.

- [ ] **Step 3: Implement the minimal bulb loader and dataset**

Create `diffusion_policy/dataset/bulb_tac_lowdim_dataset.py` with these concrete responsibilities and implementation shape:

```python
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

HAND_DIM = 22
SOURCE_DIM = 31
ARM_DIM = 9
OBS_DIM = 66
REQUIRED_FILES = ("state.npy", "action.npy")


def _episode_id(path: Path) -> int:
    prefix = "episode_"
    if not path.is_dir() or not path.name.startswith(prefix):
        raise ValueError(f"not a numeric bulb episode directory: {path}")
    try:
        return int(path.name[len(prefix):])
    except ValueError as exc:
        raise ValueError(f"not a numeric bulb episode directory: {path}") from exc


def _load_episode(path: Path, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = {}
    for name in REQUIRED_FILES:
        file_path = path / name
        if not file_path.is_file():
            raise FileNotFoundError(f"missing bulb trajectory file: {file_path}")
        array = np.load(file_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32:
            raise TypeError(f"{file_path} must use float32, got {array.dtype}")
        if array.ndim != 2 or array.shape[1] != SOURCE_DIM:
            raise ValueError(
                f"{file_path} must have shape (T, {SOURCE_DIM}), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{file_path} contains NaN or infinity")
        arrays[name] = array
    if len(arrays["state.npy"]) != len(arrays["action.npy"]):
        raise ValueError(f"{path} state/action lengths do not match")
    if len(arrays["state.npy"]) < horizon:
        raise ValueError(
            f"{path} length {len(arrays['state.npy'])} is shorter than horizon {horizon}"
        )
    qpos = np.asarray(arrays["state.npy"][:, ARM_DIM:], dtype=np.float32)
    target_after = np.asarray(arrays["action.npy"][:, ARM_DIM:], dtype=np.float32)
    target_before = np.concatenate((qpos[0:1], target_after[:-1]), axis=0)
    obs = np.concatenate(
        (qpos, target_before, target_before - qpos), axis=-1
    ).astype(np.float32, copy=False)
    return obs, target_after


class BulbTacLowdimDataset(BaseLowdimDataset):
    def __init__(
        self,
        dataset_path: str,
        horizon: int,
        pad_before: int,
        pad_after: int,
        seed: int = 42,
        val_ratio: float = 0.1,
    ):
        super().__init__()
        root = Path(dataset_path).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"bulb dataset directory not found: {root}")
        episodes = []
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("episode_"):
                episodes.append((_episode_id(child), child))
        episodes.sort(key=lambda item: item[0])
        if not episodes:
            raise ValueError(f"no numeric episode_<id> directories under {root}")
        obs_parts, action_parts, lengths = [], [], []
        for _, path in episodes:
            obs, action = _load_episode(path, int(horizon))
            obs_parts.append(obs)
            action_parts.append(action)
            lengths.append(len(obs))
        replay_buffer = ReplayBuffer(root={
            "data": {
                "obs": np.concatenate(obs_parts, axis=0),
                "action": np.concatenate(action_parts, axis=0),
            },
            "meta": {
                "episode_ends": np.cumsum(lengths, dtype=np.int64),
            },
        })
        val_mask = get_val_mask(replay_buffer.n_episodes, val_ratio, seed)
        self.replay_buffer = replay_buffer
        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.train_mask = ~val_mask
        self.val_mask = val_mask
        self.sampler = self._make_sampler(self.train_mask)

    def _make_sampler(self, mask):
        return SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=("obs", "action"),
            episode_mask=mask,
        )

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = result._make_sampler(self.val_mask)
        return result

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            {"obs": self.replay_buffer["obs"],
             "action": self.replay_buffer["action"]},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return dict_apply(self.sampler.sample_sequence(index), torch.from_numpy)
```

- [ ] **Step 4: Add exact validation-split and failure tests**

Append parameterized checks that cover every design error instead of relying on implementation exceptions indirectly:

```python
import pytest


def test_validation_uses_disjoint_episode_mask_without_downsampling(tmp_path):
    for episode_id in range(10):
        _write_episode(tmp_path, episode_id, 4, float(episode_id * 10))
    dataset = _make_dataset(tmp_path, val_ratio=0.2)
    validation = dataset.get_validation_dataset()
    assert dataset.replay_buffer.n_steps == 40
    assert int(dataset.train_mask.sum()) == 8
    assert int(dataset.val_mask.sum()) == 2
    assert not np.any(dataset.train_mask & dataset.val_mask)
    assert len(validation) > 0


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("missing_action", FileNotFoundError, "action.npy"),
        ("float64", TypeError, "float32"),
        ("wrong_width", ValueError, r"\(T, 31\)"),
        ("length_mismatch", ValueError, "lengths do not match"),
        ("non_finite", ValueError, "NaN or infinity"),
        ("too_short", ValueError, "shorter than horizon"),
    ],
)
def test_bulb_dataset_rejects_invalid_episodes(tmp_path, mutation, error, message):
    episode = tmp_path / "episode_0"
    episode.mkdir()
    state = np.zeros((3, 31), dtype=np.float32)
    action = np.zeros((3, 31), dtype=np.float32)
    if mutation == "float64":
        state = state.astype(np.float64)
    elif mutation == "wrong_width":
        state = state[:, :30]
    elif mutation == "length_mismatch":
        action = action[:2]
    elif mutation == "non_finite":
        state[0, 0] = np.nan
    elif mutation == "too_short":
        state, action = state[:2], action[:2]
    np.save(episode / "state.npy", state)
    if mutation != "missing_action":
        np.save(episode / "action.npy", action)
    with pytest.raises(error, match=message):
        _make_dataset(tmp_path)
```

- [ ] **Step 5: Run the adapter tests and existing dataset tests**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_bulb_tac_lowdim_dataset.py \
  tests/test_sim_hand_lowdim_dataset.py \
  tests/test_sim_hand_hdf5_dataset.py
```

Expected: all selected tests pass; the implementation writes nothing outside pytest temporary directories.

### Task 2: Bounded-memory probability mixture sampler

**Files:**
- Create: `diffusion_policy/common/probability_mixture_sampler.py`
- Create: `tests/test_probability_mixture_sampler.py`

**Interfaces:**
- Consumes: source lengths, probabilities, composite offsets, epoch sample count, seed, replica count, and rank.
- Produces: `ProbabilityMixtureSampler(torch.utils.data.Sampler[int])` with `__iter__`, `__len__`, and `set_epoch`.

- [ ] **Step 1: Write failing validation, reproducibility, probability, and scale tests**

Create `tests/test_probability_mixture_sampler.py`:

```python
from itertools import islice

import numpy as np
import pytest

from diffusion_policy.common.probability_mixture_sampler import (
    ProbabilityMixtureSampler,
)


def _sampler(**kwargs):
    defaults = dict(
        source_lengths=[100, 20],
        probabilities=[0.8, 0.2],
        samples_per_epoch=10_000,
        seed=42,
        num_replicas=1,
        rank=0,
        chunk_size=257,
    )
    defaults.update(kwargs)
    return ProbabilityMixtureSampler(**defaults)


def test_sampler_is_reproducible_per_seed_epoch_and_rank():
    first = _sampler()
    second = _sampler()
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)
    assert list(_sampler(rank=0, num_replicas=2)) != list(
        _sampler(rank=1, num_replicas=2)
    )


def test_sampler_realized_source_ratio_matches_probability():
    indices = np.asarray(list(_sampler(samples_per_epoch=100_000)))
    exp_fraction = float(np.mean(indices < 100))
    assert exp_fraction == pytest.approx(0.8, abs=0.01)
    assert np.all((indices >= 0) & (indices < 120))


def test_sampler_uses_bounded_chunks_for_huge_sources():
    sampler = _sampler(
        source_lengths=[100_000_000, 44_142],
        samples_per_epoch=100_044_142,
        chunk_size=128,
    )
    first_thousand = list(islice(iter(sampler), 1000))
    assert len(first_thousand) == 1000
    assert len(sampler) == 100_044_142


def test_ddp_ranks_have_equal_lengths_without_dropping_epoch_exposure():
    samplers = [
        _sampler(samples_per_epoch=11, num_replicas=4, rank=rank)
        for rank in range(4)
    ]
    assert [len(sampler) for sampler in samplers] == [3, 3, 3, 3]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"source_lengths": [100, 0]}, "source lengths"),
        ({"probabilities": [0.8]}, "same length"),
        ({"probabilities": [0.8, 0.3]}, "sum to 1"),
        ({"probabilities": [1.0, 0.0]}, "strictly positive"),
        ({"samples_per_epoch": 0}, "samples_per_epoch"),
        ({"num_replicas": 0}, "num_replicas"),
        ({"num_replicas": 2, "rank": 2}, "rank"),
    ],
)
def test_sampler_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _sampler(**kwargs)
```

- [ ] **Step 2: Run the sampler test and verify import failure**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_probability_mixture_sampler.py
```

Expected: collection fails because the sampler module does not exist.

- [ ] **Step 3: Implement validated chunked sampling**

Create `diffusion_policy/common/probability_mixture_sampler.py`:

```python
from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class ProbabilityMixtureSampler(Sampler[int]):
    def __init__(
        self,
        source_lengths: Sequence[int],
        probabilities: Sequence[float],
        samples_per_epoch: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        chunk_size: int = 65_536,
    ):
        lengths = np.asarray(source_lengths, dtype=np.int64)
        probs = np.asarray(probabilities, dtype=np.float64)
        if lengths.ndim != 1 or len(lengths) < 2 or np.any(lengths <= 0):
            raise ValueError("source lengths must contain at least two positive values")
        if probs.shape != lengths.shape:
            raise ValueError("probabilities and source lengths must have the same length")
        if not np.isfinite(probs).all() or np.any(probs <= 0):
            raise ValueError("probabilities must be finite and strictly positive")
        if not np.isclose(probs.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("probabilities must sum to 1")
        if type(samples_per_epoch) is not int or samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be a positive integer")
        if type(num_replicas) is not int or num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer")
        if type(rank) is not int or not 0 <= rank < num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.source_lengths = lengths
        self.probabilities = probs
        self.offsets = np.r_[0, np.cumsum(lengths[:-1])]
        self.samples_per_epoch = samples_per_epoch
        self.num_samples = math.ceil(samples_per_epoch / num_replicas)
        self.seed = int(seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.chunk_size = chunk_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        seed = np.random.SeedSequence([self.seed, self.epoch, self.rank])
        rng = np.random.default_rng(seed)
        remaining = self.num_samples
        while remaining:
            size = min(remaining, self.chunk_size)
            sources = rng.choice(
                len(self.source_lengths), size=size, p=self.probabilities
            )
            indices = np.empty(size, dtype=np.int64)
            for source_id in range(len(self.source_lengths)):
                mask = sources == source_id
                count = int(mask.sum())
                if count:
                    indices[mask] = self.offsets[source_id] + rng.integers(
                        0, self.source_lengths[source_id], size=count
                    )
            yield from indices.tolist()
            remaining -= size
```

- [ ] **Step 4: Run sampler tests twice to prove deterministic stability**

Run the same command twice:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_probability_mixture_sampler.py
```

Expected: both runs pass with identical test outcomes and no large allocation for the 100-million-length test.

### Task 3: Mixed dataset, unweighted validation, and merged normalization

**Files:**
- Create: `diffusion_policy/dataset/mixed_lowdim_dataset.py`
- Create: `tests/test_mixed_lowdim_dataset.py`
- Modify: `diffusion_policy/dataset/base_dataset.py:7-24`

**Interfaces:**
- Consumes: an ordered mapping `datasets: Mapping[str, BaseLowdimDataset]`, matching probability mapping, optional epoch size, and component normalizers/counts.
- Produces: `MixedLowdimDataset`, source-aware samples, a validation composite, a custom training sampler, and full-source combined normalization statistics.

- [ ] **Step 1: Add failing composite routing and validation tests**

Create a small in-memory fake dataset in `tests/test_mixed_lowdim_dataset.py`:

```python
from __future__ import annotations

import pytest
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dataset.mixed_lowdim_dataset import MixedLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class ToyDataset(BaseLowdimDataset):
    def __init__(self, values, val_values):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.val_values = torch.tensor(val_values, dtype=torch.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        return {
            "obs": value.expand(2, 66).clone(),
            "action": value.expand(2, 22).clone(),
        }

    def get_validation_dataset(self):
        return ToyDataset(self.val_values.tolist(), [])

    def get_all_actions(self):
        return self.values[:, None].expand(-1, 22).clone()

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            {
                "obs": self.values[:, None].expand(-1, 66),
                "action": self.values[:, None].expand(-1, 22),
            },
            mode=mode,
            last_n_dims=1,
            **kwargs,
        )
        return normalizer


def _mixed(samples_per_epoch=None):
    return MixedLowdimDataset(
        datasets={
            "expdata": ToyDataset([-2.0, -1.0, 0.0], [10.0]),
            "bulb_tac": ToyDataset([4.0, 8.0], [20.0, 30.0]),
        },
        probabilities={"expdata": 0.8, "bulb_tac": 0.2},
        samples_per_epoch=samples_per_epoch,
    )


def test_mixed_dataset_preserves_lengths_and_routes_source_metadata():
    dataset = _mixed()
    assert len(dataset) == 5
    assert dataset.source_names == ("expdata", "bulb_tac")
    assert dataset.source_lengths == (3, 2)
    assert dataset[0]["source_id"].item() == 0
    assert dataset[2]["obs"][0, 0].item() == 0.0
    assert dataset[3]["source_id"].item() == 1
    assert dataset[4]["action"][0, 0].item() == 8.0


def test_validation_is_plain_concatenation_without_custom_sampler():
    validation = _mixed().get_validation_dataset()
    assert len(validation) == 3
    assert validation[0]["obs"][0, 0].item() == 10.0
    assert validation[1]["obs"][0, 0].item() == 20.0
    assert set(validation[0]) == {"obs", "action", "source_id"}
    assert validation.get_training_sampler(seed=1) is None


def test_default_and_explicit_epoch_size_feed_custom_sampler():
    default_sampler = _mixed().get_training_sampler(seed=7)
    explicit_sampler = _mixed(samples_per_epoch=123).get_training_sampler(seed=7)
    assert len(default_sampler) == 5
    assert len(explicit_sampler) == 123


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({"expdata": 1.0}, "exactly match"),
        (
            {"expdata": 0.8, "bulb_tac": 0.1, "extra": 0.1},
            "exactly match",
        ),
        ({"expdata": 1.0, "bulb_tac": 0.0}, "finite, positive"),
        ({"expdata": 0.8, "bulb_tac": float("nan")}, "finite, positive"),
        ({"expdata": 0.8, "bulb_tac": 0.1}, "sum to 1"),
    ],
)
def test_mixed_dataset_rejects_invalid_probability_maps(
    probabilities,
    message,
):
    with pytest.raises(ValueError, match=message):
        MixedLowdimDataset(
            datasets={
                "expdata": ToyDataset([1.0], []),
                "bulb_tac": ToyDataset([2.0], []),
            },
            probabilities=probabilities,
        )


def test_mixed_dataset_rejects_empty_sources_and_invalid_epoch_size():
    with pytest.raises(ValueError, match="non-empty"):
        MixedLowdimDataset(
            datasets={
                "expdata": ToyDataset([1.0], []),
                "bulb_tac": ToyDataset([], []),
            },
            probabilities={"expdata": 0.8, "bulb_tac": 0.2},
        )
    with pytest.raises(ValueError, match="samples_per_epoch"):
        _mixed(samples_per_epoch=0)
```

- [ ] **Step 2: Add failing merged-statistics tests**

Append:

```python
def test_normalizer_merges_complete_source_statistics_without_ratio_weighting():
    normalizer = _mixed().get_normalizer()
    obs_stats = normalizer["obs"].get_input_stats()
    action_stats = normalizer["action"].get_input_stats()
    expected = torch.tensor([-2.0, -1.0, 0.0, 4.0, 8.0])
    torch.testing.assert_close(obs_stats["min"], torch.full((66,), -2.0))
    torch.testing.assert_close(obs_stats["max"], torch.full((66,), 8.0))
    torch.testing.assert_close(
        obs_stats["mean"], torch.full((66,), expected.mean().item())
    )
    torch.testing.assert_close(
        obs_stats["std"], torch.full((66,), expected.std().item())
    )
    torch.testing.assert_close(action_stats["min"], torch.full((22,), -2.0))
    torch.testing.assert_close(action_stats["max"], torch.full((22,), 8.0))
```

- [ ] **Step 3: Run mixed tests and verify import failure**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_mixed_lowdim_dataset.py
```

Expected: collection fails because `mixed_lowdim_dataset` does not exist.

- [ ] **Step 4: Add backward-compatible BaseLowdimDataset hooks**

Add to `BaseLowdimDataset` in `diffusion_policy/dataset/base_dataset.py`:

```python
    def get_training_sampler(
        self,
        *,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        return None

    def get_normalizer_sample_count(self) -> int:
        return int(self.get_all_actions().shape[0])
```

Do not change existing abstract-like default methods or `BaseImageDataset`.

- [ ] **Step 5: Implement composite routing and sampler construction**

Create `diffusion_policy/dataset/mixed_lowdim_dataset.py` with:

```python
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch

from diffusion_policy.common.probability_mixture_sampler import (
    ProbabilityMixtureSampler,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


class MixedLowdimDataset(BaseLowdimDataset):
    def __init__(self, datasets, probabilities, samples_per_epoch=None):
        if not isinstance(datasets, Mapping) or len(datasets) < 2:
            raise ValueError("mixed dataset requires at least two named datasets")
        self.datasets = dict(datasets)
        if not all(isinstance(value, BaseLowdimDataset) for value in self.datasets.values()):
            raise TypeError("all mixed components must inherit BaseLowdimDataset")
        self.source_names = tuple(self.datasets)
        if set(probabilities) != set(self.source_names):
            raise ValueError("probability keys must exactly match dataset names")
        self.probabilities = tuple(float(probabilities[name]) for name in self.source_names)
        probability_array = np.asarray(self.probabilities)
        if (not np.isfinite(probability_array).all()
                or np.any(probability_array <= 0)
                or not np.isclose(probability_array.sum(), 1.0, atol=1e-8, rtol=0.0)):
            raise ValueError("probabilities must be finite, positive, and sum to 1")
        self.source_lengths = tuple(len(value) for value in self.datasets.values())
        if any(length <= 0 for length in self.source_lengths):
            raise ValueError("mixed component training datasets must be non-empty")
        self.offsets = tuple(np.r_[0, np.cumsum(self.source_lengths[:-1])].tolist())
        if samples_per_epoch is not None and (
            type(samples_per_epoch) is not int or samples_per_epoch <= 0
        ):
            raise ValueError("samples_per_epoch must be null or a positive integer")
        self.samples_per_epoch = samples_per_epoch
        self._sampling_enabled = True

    def __len__(self):
        return sum(self.source_lengths)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_id = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local_index = index - self.offsets[source_id]
        sample = dict(self.datasets[self.source_names[source_id]][local_index])
        sample["source_id"] = torch.tensor(source_id, dtype=torch.int64)
        return sample

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.datasets = {
            name: dataset.get_validation_dataset()
            for name, dataset in self.datasets.items()
        }
        result.source_lengths = tuple(len(value) for value in result.datasets.values())
        result.offsets = tuple(np.r_[0, np.cumsum(result.source_lengths[:-1])].tolist())
        result._sampling_enabled = False
        return result

    def get_training_sampler(self, *, seed, num_replicas=1, rank=0):
        if not self._sampling_enabled:
            return None
        return ProbabilityMixtureSampler(
            source_lengths=self.source_lengths,
            probabilities=self.probabilities,
            samples_per_epoch=(self.samples_per_epoch or len(self)),
            seed=int(seed),
            num_replicas=int(num_replicas),
            rank=int(rank),
        )
```

- [ ] **Step 6: Implement analytic normalizer-stat merging**

Add helpers to the same module. They must merge counts from complete component replay buffers, not source probabilities:

```python
def _merge_field_stats(stats_and_counts):
    total = sum(count for _, count in stats_and_counts)
    means = [stats["mean"] for stats, _ in stats_and_counts]
    mean = sum(
        stats["mean"] * count for stats, count in stats_and_counts
    ) / total
    minimum = torch.stack([stats["min"] for stats, _ in stats_and_counts]).amin(0)
    maximum = torch.stack([stats["max"] for stats, _ in stats_and_counts]).amax(0)
    m2 = sum(
        (count - 1) * stats["std"].square()
        + count * (component_mean - mean).square()
        for (stats, count), component_mean in zip(stats_and_counts, means)
    )
    std = torch.sqrt(m2 / (total - 1))
    return {"min": minimum, "max": maximum, "mean": mean, "std": std}


def _normalizer_from_stats(stats, mode, output_min, output_max, range_eps):
    if mode == "limits":
        input_range = stats["max"] - stats["min"]
        ignore = input_range < range_eps
        safe_range = input_range.clone()
        safe_range[ignore] = output_max - output_min
        scale = (output_max - output_min) / safe_range
        offset = output_min - scale * stats["min"]
        offset[ignore] = (output_max + output_min) / 2 - stats["min"][ignore]
    elif mode == "gaussian":
        safe_std = stats["std"].clone()
        safe_std[safe_std < range_eps] = 1
        scale = 1 / safe_std
        offset = -stats["mean"] * scale
    else:
        raise ValueError(f"unsupported normalizer mode: {mode}")
    return SingleFieldLinearNormalizer.create_manual(scale, offset, stats)


def _get_normalizer(self, mode="limits", output_max=1.0, output_min=-1.0,
                    range_eps=1e-4, **kwargs):
    if kwargs:
        raise TypeError(f"unsupported mixed normalizer options: {sorted(kwargs)}")
    component_normalizers = [
        dataset.get_normalizer(
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
        )
        for dataset in self.datasets.values()
    ]
    counts = [
        dataset.get_normalizer_sample_count()
        for dataset in self.datasets.values()
    ]
    result = LinearNormalizer()
    for field in ("obs", "action"):
        merged = _merge_field_stats([
            (normalizer[field].get_input_stats(), count)
            for normalizer, count in zip(component_normalizers, counts)
        ])
        result[field] = _normalizer_from_stats(
            merged, mode, output_min, output_max, range_eps
        )
    return result


MixedLowdimDataset.get_normalizer = _get_normalizer
```

During implementation, define `get_normalizer` directly inside the class rather than retaining the final assignment; the assignment above only makes the complete method body unambiguous in the plan.

- [ ] **Step 7: Run mixed, sampler, and component dataset tests**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_mixed_lowdim_dataset.py \
  tests/test_probability_mixture_sampler.py \
  tests/test_bulb_tac_lowdim_dataset.py \
  tests/test_sim_hand_lowdim_dataset.py
```

Expected: all selected tests pass and existing single-source samples still contain only `obs` and `action`.

### Task 4: Workspace custom-sampler integration and source-fraction logging

**Files:**
- Modify: `diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py:91-118,199-210,298-314,375-440`
- Modify: `tests/test_sim_hand_train_smoke.py`

**Interfaces:**
- Consumes: optional `dataset.get_training_sampler(seed, num_replicas, rank)` and mixed `source_id` metadata.
- Produces: a DataLoader using the mixed sampler, per-epoch `train/source_fraction/<name>` metrics, and unchanged legacy sampler behavior.

- [ ] **Step 1: Write failing test that a dataset custom sampler wins over shuffle/DDP defaults**

Add to `tests/test_sim_hand_train_smoke.py`:

```python
class CustomSamplerDataset(TensorDataset):
    source_names = ("expdata", "bulb_tac")

    def __init__(self):
        super().__init__(torch.arange(5))
        self.request = None

    def get_training_sampler(self, *, seed, num_replicas=1, rank=0):
        self.request = (seed, num_replicas, rank)
        return torch.utils.data.SequentialSampler(self)


def test_train_loader_prefers_dataset_custom_sampler():
    dataset = CustomSamplerDataset()
    context = workspace_module.DistributedContext(
        rank=1,
        local_rank=1,
        world_size=2,
        device=torch.device("cpu"),
    )
    loader, sampler = workspace_module._make_train_dataloader(
        dataset,
        {"batch_size": 2, "num_workers": 0, "shuffle": True},
        context,
        seed=42,
    )
    assert isinstance(sampler, torch.utils.data.SequentialSampler)
    assert dataset.request == (42, 2, 1)
    assert list(iter(sampler)) == [0, 1, 2, 3, 4]
    assert loader.sampler is sampler
```

- [ ] **Step 2: Run the new test and verify it fails against current loader behavior**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_sim_hand_train_smoke.py::test_train_loader_prefers_dataset_custom_sampler
```

Expected: FAIL because `_make_train_dataloader` creates the existing `DistributedSampler` and never calls the dataset hook.

- [ ] **Step 3: Modify `_make_train_dataloader` to prefer a custom sampler**

After converting the loader config and removing `shuffle`, add:

```python
    custom_sampler_factory = getattr(dataset, "get_training_sampler", None)
    if callable(custom_sampler_factory):
        sampler = custom_sampler_factory(
            seed=int(seed),
            num_replicas=context.world_size,
            rank=context.rank,
        )
    if sampler is not None:
        loader_cfg["sampler"] = sampler
    elif context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            seed=int(seed),
            drop_last=True,
        )
        loader_cfg["sampler"] = sampler
    else:
        loader_cfg["shuffle"] = shuffle
```

Keep the existing `DistributedSampler` arguments unchanged in the fallback.

- [ ] **Step 4: Write failing source-fraction accounting tests**

Add a small pure helper to the workspace API and test it directly before threading it through the training loop:

```python
def test_source_fraction_log_uses_global_counts(monkeypatch):
    counts = torch.tensor([8.0, 2.0], dtype=torch.float64)
    log = workspace_module._source_fraction_log(
        counts,
        source_names=("expdata", "bulb_tac"),
    )
    assert log == {
        "train/source_fraction/expdata": 0.8,
        "train/source_fraction/bulb_tac": 0.2,
    }


def test_source_fraction_log_is_empty_without_source_metadata():
    assert workspace_module._source_fraction_log(None, ()) == {}
```

Run these two tests and expect an `AttributeError` until the helper exists.

- [ ] **Step 5: Implement source counting without changing the policy batch**

Add the helper:

```python
def _source_fraction_log(counts, source_names):
    if counts is None:
        return {}
    total = counts.sum().item()
    if total <= 0:
        return {}
    return {
        f"train/source_fraction/{name}": counts[index].item() / total
        for index, name in enumerate(source_names)
    }
```

In `_run`, derive `source_names = tuple(getattr(dataset, "source_names", ()))`
and pass it to `_train_epoch`.

Extend `_train_epoch(..., source_names)` with:

```python
        source_counts = (
            torch.zeros(len(source_names), device=device, dtype=torch.float64)
            if source_names else None
        )
```

Inside the training batch loop, immediately after device transfer:

```python
                source_id = batch.pop("source_id", None)
                if source_counts is not None:
                    if source_id is None:
                        raise RuntimeError("mixed training batch is missing source_id")
                    source_counts += torch.bincount(
                        source_id.reshape(-1), minlength=len(source_names)
                    ).to(dtype=torch.float64)
```

After loss reduction:

```python
        if context.enabled and source_counts is not None:
            dist.all_reduce(source_counts, op=dist.ReduceOp.SUM)
        result = {
            "train_loss": (loss_stats[0] / loss_stats[1]).item(),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        result.update(_source_fraction_log(source_counts, source_names))
        return result
```

This removes `source_id` before `training_model(batch)` and leaves single-source batches unchanged.

The Sim-Hand policy intentionally rejects extra batch keys, so also remove the
metadata in `_validate_epoch` immediately after device transfer and before
`compute_loss`:

```python
                batch.pop("source_id", None)
```

Validation remains a plain concatenation and does not count or probability-weight
sources; this pop only preserves the existing `{obs, action}` policy contract.

- [ ] **Step 6: Run workspace unit and distributed tests**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_sim_hand_train_smoke.py \
  tests/test_diffusion_unet_sim_hand_policy.py
```

Expected: custom sampler tests, existing disjoint `DistributedSampler` fallback test, CPU DDP smoke tests, resume tests, and policy tests all pass.

### Task 5: Mixed Hydra task and launcher while preserving original mode

**Files:**
- Create: `diffusion_policy/config/task/sim_hand_mixed.yaml`
- Create: `dp_train_sim_hand_mixed.sh`
- Create: `tests/test_dp_train_sim_hand_mixed_launcher.py`
- Modify: `tests/test_sim_hand_train_smoke.py`

**Interfaces:**
- Consumes: `${task.data_root}/{exp_data,bulb_tac_80}`, the existing training workspace, `DATA_ROOT`, `NUM_GPUS`, `PYTHON`, `WANDB_MODE`, and optional private W&B credentials.
- Produces: composed `task=sim_hand_mixed` configuration and an executable mixed launcher with `exp_name=mixed`.

- [ ] **Step 1: Add failing Hydra composition assertions**

Add to `tests/test_sim_hand_train_smoke.py`:

```python
def test_mixed_task_composes_without_changing_single_source_defaults():
    single = _compose_config()
    mixed = _compose_config(overrides=["task=sim_hand_mixed"])
    assert single.task.name == "sim_hand_lowdim"
    assert single.task.dataset._target_.endswith("SimHandLowdimDataset")
    assert mixed.task.name == "sim_hand_mixed"
    assert mixed.task.data_root == "/home/carus/mnt/bigai/Data"
    assert OmegaConf.to_container(mixed.task.mixing.probabilities) == {
        "expdata": 0.8,
        "bulb_tac": 0.2,
    }
    assert mixed.task.dataset._target_.endswith("MixedLowdimDataset")
    assert mixed.task.dataset.datasets.expdata.dataset_path.endswith("/exp_data")
    assert mixed.task.dataset.datasets.bulb_tac.dataset_path.endswith("/bulb_tac_80")
```

Run it and expect Hydra to fail with `Could not find 'task/sim_hand_mixed'`.

- [ ] **Step 2: Create the exact mixed task YAML**

Create `diffusion_policy/config/task/sim_hand_mixed.yaml`:

```yaml
name: sim_hand_mixed

obs_dim: 66
action_dim: 22
data_root: /home/carus/mnt/bigai/Data

mixing:
  probabilities:
    expdata: 0.8
    bulb_tac: 0.2
  samples_per_epoch: null

env_runner:
  _target_: diffusion_policy.env_runner.null_lowdim_runner.NullLowdimRunner

dataset:
  _target_: diffusion_policy.dataset.mixed_lowdim_dataset.MixedLowdimDataset
  datasets:
    expdata:
      _target_: diffusion_policy.dataset.sim_hand_lowdim_dataset.SimHandLowdimDataset
      dataset_path: ${task.data_root}/exp_data
      horizon: ${horizon}
      pad_before: ${eval:'${n_obs_steps}-1'}
      pad_after: ${eval:'${n_pred_action_steps}-1'}
      seed: ${training.seed}
      val_ratio: 0.1
      max_train_episodes: null
    bulb_tac:
      _target_: diffusion_policy.dataset.bulb_tac_lowdim_dataset.BulbTacLowdimDataset
      dataset_path: ${task.data_root}/bulb_tac_80
      horizon: ${horizon}
      pad_before: ${eval:'${n_obs_steps}-1'}
      pad_after: ${eval:'${n_pred_action_steps}-1'}
      seed: ${training.seed}
      val_ratio: 0.1
  probabilities: ${task.mixing.probabilities}
  samples_per_epoch: ${task.mixing.samples_per_epoch}
```

- [ ] **Step 3: Write failing launcher argument-capture tests**

Create `tests/test_dp_train_sim_hand_mixed_launcher.py` using the fake-Python pattern from `test_dp_train_sim_hand_launcher.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "dp_train_sim_hand_mixed.sh"


def _data_root(tmp_path):
    root = tmp_path / "Data"
    (root / "exp_data").mkdir(parents=True)
    (root / "exp_data" / "manifest.json").write_text("{}", encoding="utf-8")
    episode = root / "bulb_tac_80" / "episode_0"
    episode.mkdir(parents=True)
    return root


@pytest.mark.parametrize("num_gpus", [1, 4])
def test_mixed_launcher_labels_mode_and_passes_data_root(tmp_path, num_gpus):
    root = _data_root(tmp_path)
    capture = tmp_path / "argv.txt"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "CAPTURE_PATH": str(capture),
        "DATA_ROOT": str(root),
        "NUM_GPUS": str(num_gpus),
        "PYTHON": str(fake_python),
        "WANDB_MODE": "disabled",
        "WANDB_API_KEY": "test-only",
    })
    subprocess.run(
        [str(LAUNCHER), "training.num_epochs=2"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    argv = capture.read_text(encoding="utf-8").splitlines()
    assert "task=sim_hand_mixed" in argv
    assert "task.data_root=" + str(root) in argv
    assert "exp_name=mixed" in argv
    assert "logging.mode=disabled" in argv
    assert "training.num_epochs=2" in argv
    if num_gpus == 4:
        assert "--nproc_per_node=4" in argv
```

- [ ] **Step 4: Implement the mixed launcher with private W&B handling**

Create executable `dp_train_sim_hand_mixed.sh` by preserving the existing launch mechanics and changing only dataset validation and Hydra overrides:

```bash
#!/usr/bin/env bash
# Train Sim-Hand Diffusion Policy with expdata:bulb_tac sampling at 8:2.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-/home/carus/mnt/bigai/Data}"
NUM_GPUS="${NUM_GPUS:-1}"
if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
    exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    CUDA_VISIBLE_DEVICES=""
    for ((gpu_index = 0; gpu_index < NUM_GPUS; gpu_index++)); do
        CUDA_VISIBLE_DEVICES+="${CUDA_VISIBLE_DEVICES:+,}$gpu_index"
    done
fi
export CUDA_VISIBLE_DEVICES
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$SCRIPT_DIR/data/wandb}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
    WANDB_API_KEY="$(python - <<'PY'
import netrc
from pathlib import Path
auth = netrc.netrc(str(Path.home() / ".netrc")).authenticators("api.wandb.ai")
print((auth[2] if auth else ""), end="")
PY
)"
    export WANDB_API_KEY
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/data/matplotlib}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$DATA_ROOT/exp_data/manifest.json" ]]; then
    echo "expdata manifest not found: $DATA_ROOT/exp_data/manifest.json" >&2
    exit 1
fi
if ! compgen -G "$DATA_ROOT/bulb_tac_80/episode_*" >/dev/null; then
    echo "bulb TacMP episodes not found: $DATA_ROOT/bulb_tac_80/episode_*" >&2
    exit 1
fi
mkdir -p "$WANDB_DIR" "$MPLCONFIGDIR"

LAUNCH_ARGS=()
HYDRA_ARGS=()
if [[ "$NUM_GPUS" -gt 1 ]]; then
    LAUNCH_ARGS=(-m torch.distributed.run --standalone "--nproc_per_node=$NUM_GPUS")
    HYDRA_ARGS=(hydra/job_logging=disabled hydra.output_subdir=null)
fi

exec "$PYTHON" "${LAUNCH_ARGS[@]}" train.py \
    --config-name=train_diffusion_unet_sim_hand_workspace \
    task=sim_hand_mixed \
    task.data_root="$DATA_ROOT" \
    exp_name=mixed \
    logging.mode="$WANDB_MODE" \
    "${HYDRA_ARGS[@]}" \
    "$@"
```

Run `chmod +x dp_train_sim_hand_mixed.sh`; this is a mechanical mode change and does not edit file contents outside `apply_patch`.

- [ ] **Step 5: Run mixed and legacy configuration/launcher tests**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_dp_train_sim_hand_mixed_launcher.py \
  tests/test_dp_train_sim_hand_launcher.py \
  tests/test_sim_hand_train_smoke.py::test_mixed_task_composes_without_changing_single_source_defaults \
  tests/test_sim_hand_train_smoke.py::test_hydra_defaults_have_one_consistent_temporal_configuration
```

Expected: all pass; captured legacy arguments do not contain `task=sim_hand_mixed` or `exp_name=mixed`.

### Task 6: End-to-end mixed training smoke and real-data read-only verification

**Files:**
- Modify: `tests/test_sim_hand_train_smoke.py`
- Read only: `/share/generalvision/xiejunzhe/Data/exp_data/manifest.json`
- Read only: `/share/generalvision/xiejunzhe/Data/bulb_tac_80/episode_*/{state.npy,action.npy}`

**Interfaces:**
- Consumes: all components from Tasks 1-5.
- Produces: a one-step CPU training proof, finite source-fraction metrics/checkpoint, full focused-suite result, and real bulb cardinality/schema evidence.

- [ ] **Step 1: Add synthetic mixed-root writer and CPU configuration helper**

Reuse `_write_synthetic_hdf5_dataset` for `root/exp_data`, and add:

```python
def _write_synthetic_bulb_dataset(dataset_dir: Path):
    for episode_id, length in enumerate([14, 15, 16, 17, 18]):
        episode = dataset_dir / f"episode_{episode_id}"
        episode.mkdir(parents=True)
        state = np.zeros((length, 31), dtype=np.float32)
        action = np.zeros((length, 31), dtype=np.float32)
        values = (
            episode_id * 10
            + np.arange(length, dtype=np.float32)[:, None]
            + np.arange(HAND_DIM, dtype=np.float32)[None, :] / 100
        )
        state[:, 9:31] = values
        action[:, 9:31] = values + 0.25
        np.save(episode / "state.npy", state)
        np.save(episode / "action.npy", action)


def _small_mixed_cpu_config(data_root: Path):
    config = _compose_config(
        overrides=[
            "task=sim_hand_mixed",
            "n_pred_action_steps=9",
            f"task.data_root={data_root}",
            "task.mixing.samples_per_epoch=40",
        ]
    )
    with open_dict(config):
        config.training.device = "cpu"
        config.training.resume = False
        config.training.num_epochs = 1
        config.training.max_train_steps = 2
        config.training.max_val_steps = 1
        config.training.checkpoint_every = 1
        config.training.val_every = 1
        config.training.sample_every = 1
        config.training.rollout_every = 1_000_000
        config.dataloader.batch_size = 4
        config.dataloader.num_workers = 0
        config.dataloader.pin_memory = False
        config.val_dataloader.batch_size = 4
        config.val_dataloader.num_workers = 0
        config.val_dataloader.pin_memory = False
        config.policy.model.down_dims = [32, 64, 128]
        config.policy.model.diffusion_step_embed_dim = 32
        config.policy.model.kernel_size = 3
        config.policy.noise_scheduler.num_train_timesteps = 10
        config.policy.num_inference_steps = 2
        config.logging.mode = "disabled"
        config.logging.name = "sim-hand-mixed-smoke"
    return config
```

- [ ] **Step 2: Write the failing one-epoch mixed workspace test**

```python
def test_one_step_mixed_training_logs_sources_and_checkpoint(tmp_path):
    data_root = tmp_path / "Data"
    expdata = data_root / "exp_data"
    bulb_tac = data_root / "bulb_tac_80"
    expdata.mkdir(parents=True)
    bulb_tac.mkdir(parents=True)
    _write_synthetic_hdf5_dataset(expdata)
    _write_synthetic_bulb_dataset(bulb_tac)
    output_dir = tmp_path / "output"
    workspace = TrainDiffusionUnetSimHandWorkspace(
        _small_mixed_cpu_config(data_root),
        output_dir=str(output_dir),
    )
    workspace.run()

    record = json.loads((output_dir / "logs.json.txt").read_text().splitlines()[-1])
    assert math.isfinite(record["train_loss"])
    assert math.isfinite(record["val_loss"])
    exp_fraction = record["train/source_fraction/expdata"]
    bulb_fraction = record["train/source_fraction/bulb_tac"]
    assert exp_fraction + bulb_fraction == pytest.approx(1.0)
    assert 0.0 <= exp_fraction <= 1.0
    assert 0.0 <= bulb_fraction <= 1.0
    assert (output_dir / "checkpoints" / "latest.ckpt").is_file()
```

Run it before final integration and expect failure at the first incomplete wiring point; fix only that wiring and rerun until the test passes.

- [ ] **Step 3: Run the complete focused mixed and regression suite**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/pytest -q \
  tests/test_bulb_tac_lowdim_dataset.py \
  tests/test_probability_mixture_sampler.py \
  tests/test_mixed_lowdim_dataset.py \
  tests/test_sim_hand_lowdim_dataset.py \
  tests/test_sim_hand_hdf5_dataset.py \
  tests/test_diffusion_unet_sim_hand_policy.py \
  tests/test_sim_hand_train_smoke.py \
  tests/test_dp_train_sim_hand_launcher.py \
  tests/test_dp_train_sim_hand_mixed_launcher.py
```

Expected: all selected tests pass with no warnings indicating dropped source episodes or unexpected dataset writes.

- [ ] **Step 4: Verify the real bulb mirror without writing data**

Record a metadata checksum before loading, run the loader, then record the same
checksum afterward:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
find /share/generalvision/xiejunzhe/Data/bulb_tac_80 -type f \
  -printf '%P %s %T@\n' | sort | sha256sum
../../.local/miniforge3/envs/robodiff/bin/python -c '
from diffusion_policy.dataset.bulb_tac_lowdim_dataset import BulbTacLowdimDataset
dataset = BulbTacLowdimDataset(
    dataset_path="/share/generalvision/xiejunzhe/Data/bulb_tac_80",
    horizon=64,
    pad_before=3,
    pad_after=60,
    seed=42,
    val_ratio=0.1,
)
assert dataset.replay_buffer.n_episodes == 80
assert dataset.replay_buffer.n_steps == 44142
assert dataset.replay_buffer["obs"].shape == (44142, 66)
assert dataset.replay_buffer["action"].shape == (44142, 22)
print("bulb_real_check=pass episodes=80 steps=44142")
'
find /share/generalvision/xiejunzhe/Data/bulb_tac_80 -type f \
  -printf '%P %s %T@\n' | sort | sha256sum
```

Expected: the exact pass line prints and the two metadata checksums match.

- [ ] **Step 5: Verify real mixed configuration without loading all 100M transitions**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/python -c '
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
root = Path("diffusion_policy/config").resolve()
OmegaConf.register_new_resolver("eval", eval, replace=True)
with initialize_config_dir(version_base=None, config_dir=str(root)):
    cfg = compose(
        config_name="train_diffusion_unet_sim_hand_workspace",
        overrides=[
            "task=sim_hand_mixed",
            "task.data_root=/share/generalvision/xiejunzhe/Data",
        ],
    )
OmegaConf.resolve(cfg)
assert cfg.task.dataset.datasets.expdata.dataset_path.endswith("/Data/exp_data")
assert cfg.task.dataset.datasets.bulb_tac.dataset_path.endswith("/Data/bulb_tac_80")
assert cfg.task.mixing.probabilities.expdata == 0.8
assert cfg.task.mixing.probabilities.bulb_tac == 0.2
print("mixed_config_check=pass")
'
```

Expected: `mixed_config_check=pass`; this check deliberately avoids instantiating the 100-million-transition expdata loader.

- [ ] **Step 6: Verify the real expdata manifest and one shard schema read-only**

Run:

```bash
cd code/Dex_Diffuse-wip-dex-prior-guidance
../../.local/miniforge3/envs/robodiff/bin/python -c '
import json
from pathlib import Path
import h5py
root = Path("/share/generalvision/xiejunzhe/Data/exp_data")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
assert manifest["total_transitions"] == 100000392
assert len(manifest["shards"]) == 101
shard_path = root / manifest["shards"][0]["path"]
with h5py.File(shard_path, "r") as shard:
    assert shard["robot/qpos"].shape[1] == 22
    assert shard["robot/target_before"].shape[1] == 22
    assert shard["robot/target_after"].shape[1] == 22
print("expdata_schema_check=pass transitions=100000392 shards=101 dof=22")
'
```

Expected: the exact pass line prints without constructing the expdata replay
buffer or writing beneath the source root.

- [ ] **Step 7: Record final verification evidence**

Capture in the handoff:

- focused pytest command and passed-test count;
- real bulb evidence: 80 episodes, 44,142 steps, `(44142, 66)` observations, `(44142, 22)` actions;
- exact production default and local override paths;
- exact 0.8/0.2 config values;
- mixed smoke train/validation loss finiteness, source fractions, and checkpoint path;
- confirmation that `dp_train_sim_hand.sh` and `sim_hand_lowdim` regression tests stayed green;
- note that no Git commit exists because the supplied project directory has no Git metadata.
