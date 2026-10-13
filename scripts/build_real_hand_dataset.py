#!/usr/bin/env python3
"""Convert real SharpA bulb demonstrations (LeRobot v2.1) to a Sim-Hand
replay-buffer zarr compatible with the low-dimensional Diffusion Policy.

Source layout (per episode)::

    <src>/rank_<r>/id_<i>/data/chunk-000/episode_000000.parquet

The parquet ``state``/``actions`` columns are (T, 31) float32:
``[arm(9), hand(22)]``.  The hand block is recorded in **radians** and is
**already in the Isaac/checkpoint (policy) joint order** — the same order as
``bulb_rotate_0902/state.npy``.  No reordering or unit conversion is applied;
the hand block is sliced out directly.

Output ``replay_buffer.zarr``::

    data/obs    (N, 66) float32  [qpos, target_before, target_before - qpos]
    data/action (N, 22) float32  commanded target-after (absolute)
    meta/episode_ends (E,) int64

where ``qpos = state[t].hand`` (measured), ``target_before = actions[t-1].hand``
(the command that drove the hand *into* the current frame; the first frame uses
``qpos[0]``, matching the sim / mixed-loader convention), and
``action = actions[t].hand`` — the commanded target-after driving ``t -> t+1``.
Measured on all 75 episodes, ``raw_action[t]`` is closer to ``qpos[t+1]`` than
to ``qpos[t]`` on 99.2% of frames, confirming ``actions`` is the commanded
target-after, not the current target.

Example::

    python scripts/build_real_hand_dataset.py \
        --src /home/carus/Data/realworld_bulb_sft_260909 \
        --out data/real_hand_sft
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

HAND_DIM = 22
OBS_DIM = 3 * HAND_DIM
ARM_DIM = 9


def list_real_episodes(src) -> list[str]:
    """Return sorted parquet paths for every episode under ``src``."""
    pattern = os.path.join(
        str(src), "rank_*", "id_*", "data", "chunk-*", "episode_*.parquet"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no episode parquet files found under {src}")
    return paths


def _hand_policy_radians(values: np.ndarray) -> np.ndarray:
    """(T, 22) hand block -> (T, 22) float32.

    The parquet hand block is already radians in policy joint order, so this
    is a pass-through slice (no reordering, no unit conversion).
    """
    return np.asarray(values, dtype=np.float32)


def convert_episode_arrays(
    states: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one episode's raw (radians, policy-order) arrays to (obs, action).

    Temporal semantics (verified on all 75 episodes: raw_action[t] is closer
    to qpos[t+1] than to qpos[t] on 99.2% of frames, i.e. it is the commanded
    target-after):

    obs[t]    = [qpos, target_before, target_before - qpos]  (66-D)
        qpos[t]          = state[t].hand            (measured)
        target_before[t] = actions[t-1].hand        (command that drove into t;
                                                     t=0 uses qpos[0], matching
                                                     the sim/mixed convention)
    action[t] = actions[t].hand                     (commanded target-after)
    """
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != ARM_DIM + HAND_DIM:
        raise ValueError(f"state must be (T, 31), got {states.shape}")
    if actions.shape != states.shape:
        raise ValueError(f"actions must match state shape, got {actions.shape}")
    if len(states) < 2:
        raise ValueError("episode must have >= 2 frames")

    qpos = _hand_policy_radians(states[:, ARM_DIM:])
    command = _hand_policy_radians(actions[:, ARM_DIM:])
    # target_before[0] = qpos[0] (sim / mixed-loader convention: no command has
    # driven the hand yet at the first frame); afterwards target_before[t] is
    # the command that drove the hand into frame t.
    target_before = np.concatenate([qpos[:1], command[:-1]], axis=0)
    obs = np.concatenate(
        (qpos, target_before, target_before - qpos), axis=-1
    ).astype(np.float32)
    return obs, command.astype(np.float32)


def _load_episode_arrays(path: str) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["state", "actions"])
    states = np.stack(table.column("state").to_numpy())
    actions = np.stack(table.column("actions").to_numpy())
    return states, actions


def build_real_hand_dataset(src, out, overwrite=False) -> dict:
    import zarr
    from numcodecs import Blosc

    out = os.path.abspath(str(out))
    zarr_path = os.path.join(out, "replay_buffer.zarr")
    if os.path.exists(out):
        if not overwrite:
            raise FileExistsError(f"output exists: {out}; pass --overwrite")
        shutil.rmtree(out)
    os.makedirs(out)

    paths = list_real_episodes(src)
    episodes = []
    total = 0
    for path in paths:
        states, actions = _load_episode_arrays(path)
        obs, act = convert_episode_arrays(states, actions)
        episodes.append((obs, act))
        total += len(obs)
        print(f"[real2dp] {os.path.relpath(path, src)}: {len(obs)} frames")

    root = zarr.open_group(zarr_path, mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    comp = Blosc(cname="zstd", clevel=3, shuffle=1)

    obs_arr = data.create_dataset(
        "obs", shape=(total, OBS_DIM), chunks=(4096, OBS_DIM),
        dtype="float32", compressor=comp,
    )
    act_arr = data.create_dataset(
        "action", shape=(total, HAND_DIM), chunks=(4096, HAND_DIM),
        dtype="float32", compressor=comp,
    )
    cursor = 0
    episode_ends = []
    for obs, act in episodes:
        end = cursor + len(obs)
        obs_arr[cursor:end] = obs
        act_arr[cursor:end] = act
        episode_ends.append(end)
        cursor = end

    ends = np.asarray(episode_ends, dtype=np.int64)
    meta.create_dataset(
        "episode_ends", data=ends, shape=ends.shape, chunks=ends.shape,
        dtype="int64", compressor=None,
    )

    manifest = {
        "schema_version": 1,
        "source": os.path.abspath(str(src)),
        "episodes": len(episodes),
        "frames": total,
        "obs_dim": OBS_DIM,
        "action_dim": HAND_DIM,
        "unit": "radians",
        "joint_order": "policy_order_passthrough",
        "observation": "[qpos, target_before, target_before - qpos]",
        "action": "commanded_target_after (raw actions[t])",
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[real2dp] done: {zarr_path}  episodes={len(episodes)} frames={total}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="real dataset root")
    parser.add_argument("--out", required=True, help="output dir for replay_buffer.zarr")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_real_hand_dataset(args.src, args.out, overwrite=args.overwrite)
