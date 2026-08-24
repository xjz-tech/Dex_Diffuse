import os
import sys

import numpy as np
import zarr
from zarr.storage import LocalStore

from diffusion_policy.common.bulb_action_normalizer import (
    HAND_DIM,
    collect_hand_joint_stat,
    load_hand_joint_stat,
    streaming_minmax,
)


def test_streaming_minmax_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1000, HAND_DIM)).astype(np.float32)
    mn, mx = streaming_minmax(data, chunk_size=128)
    np.testing.assert_allclose(mn, data.min(axis=0))
    np.testing.assert_allclose(mx, data.max(axis=0))


def _write_hand_zarr(root):
    zarr_path = os.path.join(root, "replay_buffer.zarr")
    store = LocalStore(zarr_path)
    root_group = zarr.open(store=store, mode="w")
    meta = root_group.create_group("meta")
    data = root_group.create_group("data")
    joints = np.stack([np.arange(HAND_DIM, dtype=np.float32), np.arange(HAND_DIM, dtype=np.float32) + 3], axis=0)
    actions = joints + 1
    meta.create_array("episode_ends", data=np.array([joints.shape[0]], dtype=np.int64))
    data.create_array("hand_joint", data=joints)
    data.create_array("action", data=actions)
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
        f"{sys.executable} scripts/fit_bulb_sim_normalizer.py --zarr {zarr_path} --output {out}"
    ) == 0
    stat = load_hand_joint_stat(str(out))
    assert stat["min"].shape == (HAND_DIM,)
    assert stat["max"].shape == (HAND_DIM,)
