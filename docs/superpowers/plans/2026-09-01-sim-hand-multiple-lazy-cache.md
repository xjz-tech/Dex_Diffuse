# Sim-Hand Multiple Lazy-Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `sim_hand_multiple` training that recursively discovers every DexGen HDF5 leaf under `sim_data` and trains one ordinary Sim-Hand policy from a disk-backed lazy dataset.

**Architecture:** Convert each leaf once into a fingerprinted read-only `.npy` cache, mmap those arrays at train time, map logical windows by formula, and sample uniformly with a bounded chunked sampler. Do not merge 700M transitions into a ReplayBuffer and do not use mixed-mode source weights.

**Tech Stack:** Python 3.9, NumPy memmap, HDF5/h5py, PyTorch Dataset/Sampler/DDP, Hydra, fcntl file locks, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-sim-hand-multiple-lazy-cache-design.md`

## Global Constraints

- Keep `sim_hand_lowdim` and `sim_hand_mixed` behavior unchanged.
- `sim_hand_multiple` has no `source_id`, no source fractions, and no 8:2 weights.
- Episode identity is local to a leaf; duplicate `episode_id`/`env_id` across leaves stay separate.
- Cache default path is `<dataset_path>/.sim_hand_multiple_cache/v1`.
- Tests use tiny HDF5 shards only. Production cache build is not part of unit tests.
- Production verification during development is read-only preflight unless the user later starts training.

## File structure

| File | Responsibility |
|---|---|
| `diffusion_policy/dataset/sim_hand_hdf5.py` | Restore single-leaf loading; add path-bearing errors; expose leaf arrays without 66-D concat. |
| `diffusion_policy/dataset/sim_hand_lowdim_dataset.py` | Restore the non-recursive single-source dataset. |
| `diffusion_policy/dataset/sim_hand_multiple_cache.py` | Recursive discovery, fingerprint, lock, atomic cache publish, preflight. |
| `diffusion_policy/dataset/sim_hand_multiple_dataset.py` | Lazy mmap dataset, formula window mapping, cached normalizer, sampler hook. |
| `diffusion_policy/common/chunked_uniform_sampler.py` | Bounded uniform-with-replacement DDP sampler. |
| `build_sim_hand_multiple_cache.py` | CLI for preflight and cache ensure. |
| `diffusion_policy/config/task/sim_hand_multiple.yaml` | Hydra task targeting `LazySimHandMultipleDataset`. |
| `dp_train_sim_hand_multiple.sh` | Preflight/ensure cache, then launch 1-GPU or DDP training. |
| `tests/test_sim_hand_multiple_cache.py` | Discovery, fingerprint, cache reuse/invalidation, lock, space check. |
| `tests/test_sim_hand_multiple_dataset.py` | Windows, padding, split, normalizer, no source_id. |
| `tests/test_chunked_uniform_sampler.py` | Uniform sampling, seed/rank, bounded chunks. |
| `tests/test_dp_train_sim_hand_multiple_launcher.py` | Preflight + Hydra labels for 1/4 GPU. |
| `tests/test_sim_hand_train_smoke.py` | Hydra compose + one-step multiple training without source fractions. |
| `tests/test_sim_hand_hdf5_dataset.py` | Remove recursive in-memory merge tests. |

---

### Task 1: Revert in-memory recursive merge

Restore single-leaf HDF5 loading and `SimHandLowdimDataset`. Keep path-bearing error messages. Existing single-source HDF5 tests must still pass.

### Task 2: Discovery and cache

Implement recursive manifest discovery, fingerprinting, exclusive `build.lock`, one-leaf-at-a-time conversion through validated HDF5 loading, atomic temp-dir publish, READY marker, rebuild, and insufficient-space errors.

### Task 3: Lazy dataset and sampler

Implement `LazySimHandMultipleDataset` with mmap, formula window mapping matching `SequenceSampler`, global val split after horizon filtering, cached-stat normalizer, and `ChunkedUniformSampler`.

### Task 4: Config, CLI, launcher

Point `task=sim_hand_multiple` at the lazy dataset. Launcher validates nested manifests, ensures cache once, then starts `train.py` or `torchrun`.

### Task 5: Tests and production preflight

Run tiny-cache, lazy-dataset, sampler, launcher, one-step CPU training, and single/mixed regressions. Run read-only production preflight against `/share/generalvision/xiejunzhe/Data/sim_data`.
