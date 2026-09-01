from __future__ import annotations

import json
import math
import os
from pathlib import Path
import socket
from datetime import timedelta

import h5py
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from torch.utils.data import TensorDataset

from diffusion_policy.common.sim_hand_temporal_util import (
    validate_sim_hand_temporal_config,
)
from diffusion_policy.workspace import (
    train_diffusion_unet_sim_hand_workspace as workspace_module,
)
from diffusion_policy.workspace.train_diffusion_unet_sim_hand_workspace import (
    TrainDiffusionUnetSimHandWorkspace,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "diffusion_policy" / "config"
HAND_DIM = 22
OBS_DIM = 66


def _compose_config(overrides=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        config = compose(
            config_name="train_diffusion_unet_sim_hand_workspace",
            overrides=overrides or [],
        )
    OmegaConf.resolve(config)
    return config


def _write_synthetic_dataset(dataset_dir: Path):
    lengths = np.array([14, 15, 16, 17, 18], dtype=np.int64)
    episode_ends = np.cumsum(lengths)
    n_steps = int(episode_ends[-1])
    rng = np.random.default_rng(8)
    obs = rng.normal(size=(n_steps, OBS_DIM)).astype(np.float32)
    action = (
        0.8 * obs[:, :HAND_DIM]
        + rng.normal(scale=0.05, size=(n_steps, HAND_DIM)).astype(np.float32)
    ).astype(np.float32)

    root = zarr.open_group(str(dataset_dir / "replay_buffer.zarr"), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")
    data.create_dataset(
        "obs",
        data=obs,
        shape=obs.shape,
        chunks=(16, OBS_DIM),
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
                    "target_before": np.full(
                        HAND_DIM,
                        value + 1_000.0,
                        dtype=np.float32,
                    ),
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
        for key in ("qpos", "target_before", "target_after"):
            robot.create_dataset(
                key,
                data=np.stack([row[key] for row in rows]),
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


def _small_cpu_config(dataset_dir: Path):
    config = _compose_config(overrides=["n_pred_action_steps=9"])
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


def _run_distributed_workspace(
    rank,
    world_size,
    master_port,
    dataset_dir,
    output_dir,
):
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    config = _small_cpu_config(Path(dataset_dir))
    workspace = TrainDiffusionUnetSimHandWorkspace(
        config,
        output_dir=output_dir,
    )
    workspace.run()


def _resume_checkpoint_without_cuda(rank, dataset_dir, output_dir):
    assert not torch.cuda.is_available()
    config = _small_cpu_config(Path(dataset_dir))
    with open_dict(config):
        config.training.resume = True
    workspace = TrainDiffusionUnetSimHandWorkspace(
        config,
        output_dir=output_dir,
    )
    workspace.run()
    assert next(workspace.model.parameters()).device.type == "cpu"


def _unused_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_hydra_defaults_have_one_consistent_temporal_configuration():
    config = _compose_config()

    assert config.n_obs_steps == 4
    assert config.n_pred_action_steps == 61
    assert config.n_action_steps == 5
    assert config.horizon == 64
    assert config.obs_dim == OBS_DIM
    assert config.action_dim == HAND_DIM
    assert config.dataloader.batch_size == 10240
    assert config.val_dataloader.batch_size == 10240
    assert config.training.checkpoint_every == 5
    assert config.training.distributed_timeout_seconds == 7200
    assert config.policy.return_full_prediction is False
    assert config.task.dataset.pad_before == 3
    assert config.task.dataset.pad_after == 60
    validate_sim_hand_temporal_config(
        n_obs_steps=config.n_obs_steps,
        n_pred_action_steps=config.n_pred_action_steps,
        n_action_steps=config.n_action_steps,
        horizon=config.horizon,
        obs_dim=config.obs_dim,
        action_dim=config.action_dim,
        oa_step_convention=config.oa_step_convention,
    )


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


@pytest.mark.parametrize(
    ("prediction_steps", "expected_horizon", "expected_batch_size"),
    [
        (9, 12, 61440),
        (61, 64, 10240),
    ],
)
def test_batch_size_scales_with_prediction_horizon(
    prediction_steps,
    expected_horizon,
    expected_batch_size,
):
    config = _compose_config(
        overrides=[f"n_pred_action_steps={prediction_steps}"]
    )

    assert config.horizon == expected_horizon
    assert config.dataloader.batch_size == expected_batch_size
    assert config.val_dataloader.batch_size == expected_batch_size


def test_distributed_train_sampler_assigns_disjoint_indices():
    dataset = TensorDataset(torch.arange(5))
    loader_config = {
        "batch_size": 2,
        "num_workers": 0,
        "shuffle": True,
    }
    contexts = [
        workspace_module.DistributedContext(
            rank=rank,
            local_rank=rank,
            world_size=2,
            device=torch.device("cpu"),
        )
        for rank in range(2)
    ]

    samplers = [
        workspace_module._make_train_dataloader(
            dataset,
            loader_config,
            context,
            seed=42,
        )[1]
        for context in contexts
    ]
    rank_indices = [set(iter(sampler)) for sampler in samplers]

    assert rank_indices[0].isdisjoint(rank_indices[1])
    assert len(rank_indices[0] | rank_indices[1]) == 4


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


def test_distributed_initialization_uses_configured_timeout(monkeypatch):
    captured = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(workspace_module.dist, "is_initialized", lambda: False)

    def capture_init_process_group(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        workspace_module.dist,
        "init_process_group",
        capture_init_process_group,
    )

    context = workspace_module._initialize_distributed(
        "cpu",
        timeout_seconds=7200,
    )

    assert context.world_size == 2
    assert captured["timeout"] == timedelta(hours=2)


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
        config.policy.model.global_cond_dim = 2 * OBS_DIM
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


def test_two_process_cpu_distributed_training(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_synthetic_dataset(dataset_dir)
    output_dir = tmp_path / "output"

    mp.spawn(
        _run_distributed_workspace,
        args=(
            2,
            _unused_local_port(),
            str(dataset_dir),
            str(output_dir),
        ),
        nprocs=2,
        join=True,
    )

    records = [
        json.loads(line)
        for line in (output_dir / "logs.json.txt").read_text().splitlines()
    ]
    assert len(records) == 1
    assert records[0]["world_size"] == 2
    assert records[0]["per_gpu_batch_size"] == 2
    assert records[0]["global_batch_size"] == 4
    assert (output_dir / "checkpoints" / "latest.ckpt").is_file()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to create a CUDA-tagged checkpoint",
)
def test_cuda_checkpoint_can_resume_on_cpu_only_worker(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_synthetic_dataset(dataset_dir)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    config = _small_cpu_config(dataset_dir)
    workspace = TrainDiffusionUnetSimHandWorkspace(
        config,
        output_dir=str(output_dir),
    )
    workspace.model.to("cuda:0")
    if workspace.ema_model is not None:
        workspace.ema_model.to("cuda:0")
    workspace.epoch = int(config.training.num_epochs)
    workspace.save_checkpoint(tag="latest", use_thread=False)

    previous_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        mp.spawn(
            _resume_checkpoint_without_cuda,
            args=(str(dataset_dir), str(output_dir)),
            nprocs=1,
            join=True,
        )
    finally:
        if previous_visibility is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_visibility
