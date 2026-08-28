from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from diffusion_policy.workspace.train_diffusion_unet_sim_hand_workspace import (
    TrainDiffusionUnetSimHandWorkspace,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "diffusion_policy" / "config"
HAND_DIM = 22


def _compose_config():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(config_name="train_diffusion_unet_sim_hand_workspace")
    OmegaConf.resolve(config)
    return config


def _write_synthetic_dataset(dataset_dir: Path):
    lengths = np.array([14, 15, 16, 17, 18], dtype=np.int64)
    episode_ends = np.cumsum(lengths)
    n_steps = int(episode_ends[-1])
    rng = np.random.default_rng(8)
    hand_joint = rng.normal(size=(n_steps, HAND_DIM)).astype(np.float32)
    action = (
        0.8 * hand_joint
        + rng.normal(scale=0.05, size=(n_steps, HAND_DIM)).astype(np.float32)
    ).astype(np.float32)

    root = zarr.open_group(str(dataset_dir / "replay_buffer.zarr"), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.create_dataset(
        "hand_joint",
        data=hand_joint,
        shape=hand_joint.shape,
        chunks=(16, HAND_DIM),
    )
    data.create_dataset(
        "action",
        data=action,
        shape=action.shape,
        chunks=(16, HAND_DIM),
    )
    meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        shape=episode_ends.shape,
        dtype="int64",
    )


def _write_synthetic_hdf5_dataset(dataset_dir: Path):
    lengths = np.asarray([14, 15, 16, 17, 18], dtype=np.int64)
    rows = []
    for step in range(int(lengths.max())):
        for episode_id, length in enumerate(lengths):
            if step >= length:
                continue
            value = float(episode_id * 100 + step)
            terminal = step == length - 1
            rows.append(
                {
                    "episode_id": episode_id,
                    "env_id": episode_id,
                    "step": step,
                    "done": terminal,
                    "success": terminal,
                    "failure": False,
                    "timeout": False,
                    "reset_reason": 1 if terminal else 0,
                    "qpos": np.full(HAND_DIM, value, dtype=np.float32),
                    "target_after": np.full(
                        HAND_DIM,
                        0.8 * value + 0.05,
                        dtype=np.float32,
                    ),
                }
            )

    shard_dir = dataset_dir / "shards"
    shard_dir.mkdir()
    shard_path = shard_dir / "shard_000000.h5"
    with h5py.File(shard_path, "w") as f:
        index = f.create_group("index")
        robot = f.create_group("robot")
        for key, dtype in (
            ("episode_id", np.int64),
            ("env_id", np.int32),
            ("step", np.int64),
            ("reset_reason", np.int8),
        ):
            index.create_dataset(
                key,
                data=np.asarray([row[key] for row in rows], dtype=dtype),
            )
        for key in ("done", "success", "failure", "timeout"):
            index.create_dataset(
                key,
                data=np.asarray([row[key] for row in rows], dtype=np.bool_),
            )
        robot.create_dataset(
            "qpos",
            data=np.stack([row["qpos"] for row in rows]),
        )
        robot.create_dataset(
            "target_after",
            data=np.stack([row["target_after"] for row in rows]),
        )

    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "total_transitions": len(rows),
                "dof": HAND_DIM,
                "shards": [
                    {
                        "path": "shards/shard_000000.h5",
                        "num_transitions": len(rows),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _small_cpu_config(dataset_dir: Path):
    config = _compose_config()
    with open_dict(config):
        config.task.dataset.dataset_path = str(dataset_dir)
        config.training.device = "cpu"
        config.training.resume = False
        config.training.num_epochs = 1
        config.training.max_train_steps = 1
        config.training.max_val_steps = 1
        config.training.checkpoint_every = 1
        config.training.val_every = 1
        config.training.sample_every = 1
        config.training.rollout_every = 1
        config.dataloader.batch_size = 2
        config.dataloader.num_workers = 0
        config.dataloader.pin_memory = False
        config.val_dataloader.batch_size = 2
        config.val_dataloader.num_workers = 0
        config.val_dataloader.pin_memory = False
        config.policy.model.down_dims = [32, 64, 128]
        config.policy.model.diffusion_step_embed_dim = 32
        config.policy.model.kernel_size = 3
        config.policy.noise_scheduler.num_train_timesteps = 10
        config.policy.num_inference_steps = 2
        config.logging.mode = "disabled"
        config.logging.name = "sim-hand-smoke"
    return config


def test_hydra_defaults_have_one_consistent_temporal_configuration():
    config = _compose_config()

    assert config.n_obs_steps == 4
    assert config.n_pred_action_steps == 9
    assert config.n_action_steps == 5
    assert config.horizon == 12
    assert config.obs_dim == HAND_DIM
    assert config.action_dim == HAND_DIM
    assert config.training.checkpoint_every == 50
    assert config.policy.return_full_prediction is False
    assert config.task.dataset.pad_before == 3
    assert config.task.dataset.pad_after == 8


def test_workspace_prints_dynamic_temporal_report(tmp_path, capsys):
    config = _compose_config()
    with open_dict(config):
        config.n_obs_steps = 2
        config.n_pred_action_steps = 7
        config.n_action_steps = 3
        config.horizon = 8
        config.policy.n_obs_steps = 2
        config.policy.n_pred_action_steps = 7
        config.policy.n_action_steps = 3
        config.policy.horizon = 8
        config.policy.model.global_cond_dim = 2 * HAND_DIM
        config.policy.model.down_dims = [32, 64, 128]
        config.policy.model.diffusion_step_embed_dim = 32
        config.policy.model.kernel_size = 3

    TrainDiffusionUnetSimHandWorkspace(config, output_dir=str(tmp_path))

    output = capsys.readouterr().out
    assert "Observation steps       : 2" in output
    assert "Prediction action steps : 7" in output
    assert "Execution action steps  : 3" in output
    assert "Diffusion horizon       : 8" in output
    assert "obs condition : s[t-1:t+1]" in output
    assert "usable actions: a[t:t+7]" in output
    assert "execute       : a[t:t+3]" in output


def test_one_step_train_validation_and_periodic_checkpoint(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_synthetic_dataset(dataset_dir)
    output_dir = tmp_path / "output"
    config = _small_cpu_config(dataset_dir)
    workspace = TrainDiffusionUnetSimHandWorkspace(
        config,
        output_dir=str(output_dir),
    )

    workspace.run()

    records = [
        json.loads(line)
        for line in (output_dir / "logs.json.txt").read_text().splitlines()
    ]
    assert records
    final_record = records[-1]
    assert math.isfinite(final_record["train_loss"])
    assert math.isfinite(final_record["val_loss"])
    assert (output_dir / "checkpoints" / "epoch_0001.ckpt").is_file()
    assert (output_dir / "checkpoints" / "latest.ckpt").is_file()


def test_one_step_train_from_hdf5_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_synthetic_hdf5_dataset(dataset_dir)
    output_dir = tmp_path / "output"
    config = _small_cpu_config(dataset_dir)
    workspace = TrainDiffusionUnetSimHandWorkspace(
        config,
        output_dir=str(output_dir),
    )

    workspace.run()

    records = [
        json.loads(line)
        for line in (output_dir / "logs.json.txt").read_text().splitlines()
    ]
    assert records
    final_record = records[-1]
    assert math.isfinite(final_record["train_loss"])
    assert math.isfinite(final_record["val_loss"])
    assert (output_dir / "checkpoints" / "epoch_0001.ckpt").is_file()
    assert (output_dir / "checkpoints" / "latest.ckpt").is_file()
