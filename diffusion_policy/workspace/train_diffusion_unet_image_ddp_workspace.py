"""Independent DDP image training; checkpoints retain the single-GPU format."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from datetime import timedelta
import math
import os
from pathlib import Path
import random
import tempfile

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import tqdm
import wandb

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class _PolicyLoss(torch.nn.Module):
    """DDP must see compute_loss through its forward/backward hooks."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch):
        return self.policy.compute_loss(batch)


def _make_dataloader(dataset, config, *, rank=0, world_size=1, seed=42):
    options = OmegaConf.to_container(config, resolve=True)
    shuffle = options.pop("shuffle", False)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=shuffle, seed=seed, drop_last=True,
        )
        options["sampler"] = sampler
    else:
        options["shuffle"] = shuffle
    if options.get("num_workers", 0) == 0:
        options.pop("prefetch_factor", None)
        options["persistent_workers"] = False
    elif world_size > 1:
        # NCCL and forked DataLoader workers can deadlock.
        options.setdefault("multiprocessing_context", "spawn")
    return DataLoader(dataset, **options), sampler


class TrainDiffusionUnetImageDDPWorkspace(BaseWorkspace):
    include_keys = (
        "global_step", "epoch", "optimizer_step", "ema_step",
        "scheduler_state", "_checkpoint_epoch_completed",
    )

    def __init__(self, cfg, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.model = hydra.utils.instantiate(cfg.policy)
        if cfg.training.freeze_encoder:
            self.model.obs_encoder.requires_grad_(False)
        # ModuleAttrMixin uses empty parameters only to track device/dtype.
        for parameter in self.model.parameters():
            if parameter.numel() == 0:
                parameter.requires_grad_(False)
        self.ema_model = None
        if cfg.training.use_ema and int(os.environ.get("RANK", "0")) == 0:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters(),
        )
        self.global_step = 0
        self.epoch = 0  # The next epoch to train.
        self.optimizer_step = 0
        self.ema_step = 0
        self.scheduler_state = None
        self._checkpoint_epoch_completed = True

    def run(self):
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(self.cfg.training.device)
        initialized_here = False
        if device.type == "cuda":
            if world_size > 1:
                device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                timeout=timedelta(seconds=int(self.cfg.training.distributed_timeout_seconds)),
            )
            initialized_here = True
        try:
            self._run(copy.deepcopy(self.cfg), device, rank, world_size, local_rank)
        finally:
            if initialized_here:
                dist.destroy_process_group()

    def _resume(self, cfg, is_main):
        path = cfg.training.resume_path
        if path is None:
            path = self.get_checkpoint_path()
            if not path.is_file():
                return
        path = Path(path).expanduser().resolve()
        if is_main:
            print(f"Resuming from checkpoint {path}", flush=True)
        # mmap avoids loading a second full CPU copy in every DDP process.
        payload = torch.load(path, map_location="cpu", pickle_module=dill, mmap=True)
        excluded = () if self.ema_model is not None else ("ema_model",)
        self.load_payload(
            payload, exclude_keys=excluded,
            include_keys=self.include_keys,
        )
        if "_checkpoint_epoch_completed" not in payload["pickles"]:
            # The original workspace saves after training but before incrementing.
            self.epoch += 1
            self.global_step += 1
            self.optimizer_step = math.ceil(
                self.global_step / int(cfg.training.gradient_accumulate_every)
            )
            self.ema_step = self.global_step
        self._checkpoint_epoch_completed = True

    def _run(self, cfg, device, rank, world_size, local_rank):
        is_main = rank == 0
        distributed = world_size > 1
        output_dir = str(Path(self.output_dir).resolve())
        if distributed:
            directories = [output_dir if is_main else None]
            dist.broadcast_object_list(directories, src=0)
            output_dir = directories[0]
        self._output_dir = output_dir
        if is_main:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()
        if cfg.training.resume:
            self._resume(cfg, is_main)
        if cfg.training.debug:
            cfg.training.num_epochs = min(int(cfg.training.num_epochs), 2)
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            for name in ("checkpoint_every", "rollout_every", "val_every", "sample_every"):
                cfg.training[name] = 1

        accumulation = int(cfg.training.gradient_accumulate_every)
        if accumulation < 1 or int(cfg.training.num_epochs) < 1:
            raise ValueError("num_epochs and gradient_accumulate_every must be positive")
        for name in ("checkpoint_every", "rollout_every", "val_every", "sample_every"):
            if int(cfg.training[name]) < 1:
                raise ValueError(f"training.{name} must be positive")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("Dataset must inherit BaseImageDataset")
        train_loader, sampler = _make_dataloader(
            dataset, cfg.dataloader, rank=rank, world_size=world_size,
            seed=int(cfg.training.seed),
        )
        train_steps = len(train_loader)
        if cfg.training.max_train_steps is not None:
            train_steps = min(train_steps, int(cfg.training.max_train_steps))
        if train_steps < 1:
            raise ValueError("Training dataset must produce at least one batch per rank")

        # Fit normalization once, using the whole training split, then share it.
        normalizer = dataset.get_normalizer() if is_main else LinearNormalizer()
        if distributed:
            states = [normalizer.state_dict() if is_main else None]
            dist.broadcast_object_list(states, src=0)
            if not is_main:
                normalizer.load_state_dict(states[0])
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
        val_loader = None
        env_runner = None
        if is_main:
            val_loader, _ = _make_dataloader(
                dataset.get_validation_dataset(), cfg.val_dataloader,
            )
            if distributed and int(cfg.val_dataloader.num_workers) > 0:
                val_loader.multiprocessing_context = "spawn"
            env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
            if not isinstance(env_runner, BaseImageRunner):
                raise TypeError("Runner must inherit BaseImageRunner")

        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        scheduler = get_scheduler(
            cfg.training.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            # Account for completed updates if resume changes the GPU count.
            num_training_steps=max(
                1, self.optimizer_step + math.ceil(train_steps / accumulation)
                * max(0, int(cfg.training.num_epochs) - self.epoch),
            ),
            last_epoch=self.optimizer_step - 1,
        )
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)
            # LambdaLR construction can reset the optimizer's restored LR.
            for group, lr in zip(self.optimizer.param_groups, scheduler.get_last_lr()):
                group["lr"] = lr
        ema = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)
            ema.optimization_step = self.ema_step
        training_model = _PolicyLoss(self.model)
        if distributed:
            ddp_options = dict(
                find_unused_parameters=True, broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
            if device.type == "cuda":
                ddp_options.update(device_ids=[local_rank], output_device=local_rank)
            training_model = DistributedDataParallel(training_model, **ddp_options)
        rank_seed = int(cfg.training.seed) + rank
        torch.manual_seed(rank_seed)
        np.random.seed(rank_seed)
        random.seed(rank_seed)

        batch_size = int(cfg.dataloader.batch_size)
        metadata = dict(
            world_size=world_size, per_gpu_batch_size=batch_size,
            global_batch_size=batch_size * world_size,
            effective_batch_size=batch_size * world_size * accumulation,
        )
        wandb_run = None
        manager = None
        if is_main:
            print(f"DINO image training: {metadata}, device={device}", flush=True)
            wandb_run = wandb.init(
                dir=output_dir, config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging,
            )
            wandb_run.config.update(dict(output_dir=output_dir, **metadata))
            manager = TopKCheckpointManager(
                save_dir=str(Path(output_dir) / "checkpoints"), **cfg.checkpoint.topk,
            )
        logger_context = JsonLogger(str(Path(output_dir) / "logs.json.txt")) if is_main else nullcontext()
        sampling_batch = None
        try:
            with logger_context as logger:
                while self.epoch < int(cfg.training.num_epochs):
                    epoch_index = self.epoch
                    if sampler is not None:
                        sampler.set_epoch(epoch_index)
                    self.model.train()
                    if cfg.training.freeze_encoder:
                        self.model.obs_encoder.eval()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss_sum = 0.0
                    sample_count = 0
                    step_log = {}
                    with tqdm.tqdm(
                        train_loader, desc=f"Training epoch {epoch_index}",
                        disable=not is_main, leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as progress:
                        for batch_index, batch in enumerate(progress):
                            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                            if is_main and sampling_batch is None:
                                sampling_batch = batch
                            is_last = batch_index + 1 == train_steps
                            should_step = (batch_index + 1) % accumulation == 0 or is_last
                            window_size = min(
                                accumulation, train_steps - (batch_index // accumulation) * accumulation,
                            )
                            sync_context = training_model.no_sync() if distributed and not should_step else nullcontext()
                            with sync_context:
                                raw_loss = training_model(batch)
                                (raw_loss / window_size).backward()
                            if should_step:
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                                scheduler.step()
                                self.optimizer_step += 1
                                if ema is not None:
                                    ema.step(self.model)
                                    self.ema_step = ema.optimization_step
                            loss_value = raw_loss.detach().item()
                            count = int(batch["action"].shape[0])
                            loss_sum += loss_value * count
                            sample_count += count
                            step_log = dict(
                                train_loss=loss_value, epoch=epoch_index,
                                global_step=self.global_step, lr=scheduler.get_last_lr()[0],
                                **metadata,
                            )
                            if is_main:
                                progress.set_postfix(loss=loss_value, refresh=False)
                                if not is_last:
                                    wandb_run.log(step_log, step=self.global_step)
                                    logger.log(step_log)
                            self.global_step += 1
                            if is_last:
                                break
                    statistics = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
                    if distributed:
                        dist.all_reduce(statistics)
                    step_log["train_loss"] = (statistics[0] / statistics[1]).item()
                    if is_main:
                        self.model.eval()
                        policy = self.ema_model if self.ema_model is not None else self.model
                        policy.eval()
                        if epoch_index % int(cfg.training.rollout_every) == 0:
                            step_log.update(env_runner.run(policy))
                        if epoch_index % int(cfg.training.val_every) == 0:
                            val_loss = self._validate(cfg, val_loader, device)
                            if val_loss is not None:
                                step_log["val_loss"] = val_loss
                        if epoch_index % int(cfg.training.sample_every) == 0:
                            with torch.no_grad():
                                prediction = policy.predict_action(sampling_batch["obs"])
                                step_log["train_action_mse_error"] = torch.nn.functional.mse_loss(
                                    prediction["action_pred"], sampling_batch["action"],
                                ).item()
                    self.epoch += 1
                    self.scheduler_state = scheduler.state_dict()
                    if is_main:
                        if epoch_index % int(cfg.training.checkpoint_every) == 0:
                            if cfg.checkpoint.save_last_ckpt:
                                self.save_checkpoint()
                            topk_path = manager.get_ckpt_path(step_log)
                            if topk_path is not None:
                                self.save_checkpoint(path=topk_path)
                            if cfg.checkpoint.save_last_snapshot:
                                self.save_snapshot()
                        elif self.epoch == int(cfg.training.num_epochs) and cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        wandb_run.log(step_log, step=step_log["global_step"])
                        logger.log(step_log)
                    if distributed:
                        dist.barrier()
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    @torch.no_grad()
    def _validate(self, cfg, loader, device):
        loss_sum = 0.0
        count = 0
        for index, batch in enumerate(loader):
            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
            batch_size = int(batch["action"].shape[0])
            loss_sum += self.model.compute_loss(batch).item() * batch_size
            count += batch_size
            if cfg.training.max_val_steps is not None and index + 1 >= int(cfg.training.max_val_steps):
                break
        return loss_sum / count if count else None

    def save_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None,
        use_thread=False,
    ):
        """Stream one checkpoint synchronously and atomically replace its path."""
        excluded = self.exclude_keys if exclude_keys is None else exclude_keys
        included = tuple(self.include_keys) + ("_output_dir",) if include_keys is None else include_keys
        payload = dict(cfg=self.cfg, state_dicts={}, pickles={})
        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in excluded:
                    payload["state_dicts"][key] = value.state_dict()
            elif key in included:
                payload["pickles"][key] = dill.dumps(value)
        path = Path(path) if path is not None else self.get_checkpoint_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as checkpoint:
                torch.save(payload, checkpoint, pickle_module=dill)
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return str(path.absolute())
