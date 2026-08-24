import os

import numpy as np
import pytest
import zarr
from zarr.storage import LocalStore

from diffusion_policy.common.bulb_action_normalizer import collect_hand_joint_stat
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


def _write_zarr_episode(zarr_path, episode: dict):
    """Write one episode to replay_buffer.zarr (zarr 2/3 compatible)."""
    if int(zarr.__version__.split(".", maxsplit=1)[0]) >= 3:
        store = LocalStore(str(zarr_path))
        root_group = zarr.open(store=store, mode="w")
        meta = root_group.create_group("meta")
        data = root_group.create_group("data")
        T = next(iter(episode.values())).shape[0]
        meta.create_array("episode_ends", data=np.array([T], dtype=np.int64))
        for key, value in episode.items():
            data.create_array(key, data=value)
        return
    buf = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(zarr_path)))
    buf.add_episode(episode)


def _hand_dataset_dir(tmp_path):
    root = tmp_path / "sim"
    zarr_path = root / "replay_buffer.zarr"
    root.mkdir(parents=True, exist_ok=True)
    T = 12
    _write_zarr_episode(
        zarr_path,
        {
            "hand_joint": np.linspace(0, 1, T * 22, dtype=np.float32).reshape(T, 22),
            "action": np.linspace(0.1, 1.1, T * 22, dtype=np.float32).reshape(T, 22),
        },
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
    root.mkdir(parents=True, exist_ok=True)
    T = 16
    state = np.zeros((T, 31), dtype=np.float32)
    state[:, 0] = np.linspace(0, 0.1, T)
    state[:, 9:] = 0.2
    action = state.copy()
    action[:, 0] += 0.01
    _write_zarr_episode(
        zarr_path,
        {
            "state": state,
            "action": action,
            "front_image": np.zeros((T, 16, 16, 3), dtype=np.uint8),
            "wrist_image": np.zeros((T, 16, 16, 3), dtype=np.uint8),
        },
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


def test_hand_normalizer_fits_locally_when_path_unset(tmp_path):
    ds = BulbImageDataset(
        dataset_path=_hand_dataset_dir(tmp_path),
        horizon=2,
        pad_before=1,
        pad_after=0,
        n_obs_steps=2,
        shape_meta=HAND_SHAPE_META,
        val_ratio=0.0,
    )
    with pytest.warns(UserWarning, match="No normalizer_path"):
        normalizer = ds.get_normalizer()
    assert "hand_joint" in normalizer.params_dict
    assert "action" not in normalizer.params_dict


def test_hand_normalizer_missing_path_raises(tmp_path):
    ds = BulbImageDataset(
        dataset_path=_hand_dataset_dir(tmp_path),
        horizon=2,
        pad_before=1,
        pad_after=0,
        n_obs_steps=2,
        shape_meta=HAND_SHAPE_META,
        normalizer_path=str(tmp_path / "missing.npz"),
        val_ratio=0.0,
    )
    with pytest.raises(FileNotFoundError, match="hand-joint normalizer file not found"):
        ds.get_normalizer()
