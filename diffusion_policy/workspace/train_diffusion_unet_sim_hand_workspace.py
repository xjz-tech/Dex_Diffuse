from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import math
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.common.sim_hand_temporal_util import (
    format_sim_hand_temporal_config,
    validate_sim_hand_temporal_config,
)
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_sim_hand_policy import (
    DiffusionUnetSimHandPolicy,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _initialize_distributed(
    configured_device: str,
    timeout_seconds: int = 1800,
) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(configured_device)
    initialized_here = False

    if world_size > 1:
        if device.type == "cuda":
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
            backend = "nccl"
        else:
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                timeout=timedelta(seconds=int(timeout_seconds)),
            )
            initialized_here = True

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized_here=initialized_here,
    )


def _shutdown_distributed(context: DistributedContext) -> None:
    if context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def _make_train_dataloader(dataset, dataloader_cfg, context, seed):
    if OmegaConf.is_config(dataloader_cfg):
        loader_cfg = OmegaConf.to_container(dataloader_cfg, resolve=True)
    else:
        loader_cfg = dict(dataloader_cfg)
    shuffle = bool(loader_cfg.pop("shuffle", False))
    sampler = None
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
    return DataLoader(dataset, **loader_cfg), sampler


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


class TrainDiffusionUnetSimHandWorkspace(BaseWorkspace):
    include_keys = ("global_step", "epoch")

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        temporal = validate_sim_hand_temporal_config(
            n_obs_steps=cfg.n_obs_steps,
            n_pred_action_steps=cfg.n_pred_action_steps,
            n_action_steps=cfg.n_action_steps,
            horizon=cfg.horizon,
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            oa_step_convention=cfg.oa_step_convention,
        )
        if int(os.environ.get("RANK", "0")) == 0:
            print(format_sim_hand_temporal_config(temporal))
        if int(cfg.training.checkpoint_every) <= 0:
            raise ValueError("training.checkpoint_every must be positive")

        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetSimHandPolicy = hydra.utils.instantiate(
            cfg.policy
        )
        self.ema_model: DiffusionUnetSimHandPolicy | None = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=self.model.parameters(),
        )
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        context = _initialize_distributed(
            str(cfg.training.device),
            timeout_seconds=cfg.training.distributed_timeout_seconds,
        )
        try:
            self._run(cfg, context)
        finally:
            _shutdown_distributed(context)

    def _run(self, cfg, context: DistributedContext) -> None:
        output_dir = pathlib.Path(self.output_dir)
        if context.enabled:
            shared_output_dir = [str(output_dir) if context.is_main else None]
            dist.broadcast_object_list(shared_output_dir, src=0)
            output_dir = pathlib.Path(shared_output_dir[0])
        self._output_dir = str(output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
        if context.enabled:
            dist.barrier()

        if cfg.training.resume:
            latest_path = self.get_checkpoint_path()
            if latest_path.is_file():
                if context.is_main:
                    print(f"Resuming from checkpoint {latest_path}")
                self.load_checkpoint(path=latest_path, map_location="cpu")

        if cfg.training.debug:
            cfg.training.num_epochs = min(int(cfg.training.num_epochs), 2)
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseLowdimDataset):
            raise TypeError("Sim-Hand dataset must inherit BaseLowdimDataset")
        source_names = tuple(getattr(dataset, "source_names", ()))
        train_dataloader, train_sampler = _make_train_dataloader(
            dataset,
            cfg.dataloader,
            context,
            seed=cfg.training.seed,
        )
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        accumulation = int(cfg.training.gradient_accumulate_every)
        if accumulation <= 0:
            raise ValueError("gradient_accumulate_every must be positive")
        total_optimizer_steps = math.ceil(
            len(train_dataloader) * int(cfg.training.num_epochs) / accumulation
        )
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_optimizer_steps,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel | None = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = None
        wandb_run = None
        if context.is_main:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=str(output_dir),
            )
            if not isinstance(env_runner, BaseLowdimRunner):
                raise TypeError("Sim-Hand runner must inherit BaseLowdimRunner")

            wandb_run = wandb.init(
                dir=str(output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb_run.config.update(
                {
                    "output_dir": str(output_dir),
                    "world_size": context.world_size,
                    "per_gpu_batch_size": int(cfg.dataloader.batch_size),
                    "global_batch_size": (
                        int(cfg.dataloader.batch_size) * context.world_size
                    ),
                }
            )

        device = context.device
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        training_model: torch.nn.Module = self.model
        if context.enabled:
            ddp_kwargs = {"find_unused_parameters": True}
            if device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[context.local_rank],
                    output_device=context.local_rank,
                )
            training_model = DistributedDataParallel(self.model, **ddp_kwargs)

        if context.enabled:
            rank_seed = int(cfg.training.seed) + context.rank
            torch.manual_seed(rank_seed)
            np.random.seed(rank_seed)
            random.seed(rank_seed)

        if context.is_main:
            global_batch_size = (
                int(cfg.dataloader.batch_size) * context.world_size
            )
            print(
                "Sim-Hand training: "
                f"world_size={context.world_size}, "
                f"per_gpu_batch_size={int(cfg.dataloader.batch_size)}, "
                f"global_batch_size={global_batch_size}, "
                f"device={device}"
            )

        train_sampling_batch = None
        log_path = output_dir / "logs.json.txt"
        logger_context = (
            JsonLogger(str(log_path)) if context.is_main else nullcontext()
        )
        try:
            with logger_context as json_logger:
                while self.epoch < int(cfg.training.num_epochs):
                    local_epoch_idx = self.epoch
                    if train_sampler is not None:
                        train_sampler.set_epoch(local_epoch_idx)
                    step_log = self._train_epoch(
                        cfg,
                        train_dataloader,
                        device,
                        lr_scheduler,
                        ema,
                        training_model,
                        context,
                        source_names,
                    )
                    if context.is_main:
                        if train_sampling_batch is None:
                            train_sampling_batch = next(iter(train_dataloader))
                            train_sampling_batch = dict_apply(
                                train_sampling_batch,
                                lambda value: value.to(
                                    device,
                                    non_blocking=True,
                                ),
                            )

                        policy = self.ema_model or self.model
                        policy.eval()
                        if (
                            local_epoch_idx
                            % int(cfg.training.rollout_every)
                            == 0
                        ):
                            step_log.update(env_runner.run(policy))
                        if local_epoch_idx % int(cfg.training.val_every) == 0:
                            val_loss = self._validate_epoch(
                                cfg,
                                val_dataloader,
                                device,
                            )
                            if val_loss is not None:
                                step_log["val_loss"] = val_loss
                        if (
                            local_epoch_idx
                            % int(cfg.training.sample_every)
                            == 0
                        ):
                            with torch.no_grad():
                                result = policy.predict_action(
                                    {"obs": train_sampling_batch["obs"]}
                                )
                                target = train_sampling_batch["action"][
                                    :,
                                    policy.temporal.execution_action_slice,
                                ]
                                step_log["train_action_mse_error"] = (
                                    torch.nn.functional.mse_loss(
                                        result["action"],
                                        target,
                                    ).item()
                                )

                    self.epoch += 1
                    if context.is_main:
                        step_log.update(
                            {
                                "epoch": self.epoch,
                                "global_step": self.global_step,
                                "world_size": context.world_size,
                                "per_gpu_batch_size": int(
                                    cfg.dataloader.batch_size
                                ),
                                "global_batch_size": (
                                    int(cfg.dataloader.batch_size)
                                    * context.world_size
                                ),
                            }
                        )
                        self._save_epoch_checkpoints(cfg)
                        wandb_run.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                    if context.enabled:
                        dist.barrier()
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def _train_epoch(
        self,
        cfg,
        train_dataloader,
        device,
        lr_scheduler,
        ema: EMAModel | None,
        training_model: torch.nn.Module,
        context: DistributedContext,
        source_names,
    ) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        source_counts = (
            torch.zeros(len(source_names), device=device, dtype=torch.float64)
            if source_names else None
        )
        max_steps = cfg.training.max_train_steps
        with tqdm.tqdm(
            train_dataloader,
            desc=f"Training epoch {self.epoch}",
            leave=False,
            mininterval=cfg.training.tqdm_interval_sec,
            disable=not context.is_main,
        ) as progress:
            for batch_idx, batch in enumerate(progress):
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                source_id = batch.pop("source_id", None)
                if source_counts is not None:
                    if source_id is None:
                        raise RuntimeError("mixed training batch is missing source_id")
                    source_counts += torch.bincount(
                        source_id.reshape(-1), minlength=len(source_names)
                    ).to(dtype=torch.float64)
                reached_limit = (
                    max_steps is not None and batch_idx + 1 >= int(max_steps)
                )
                is_last = batch_idx + 1 == len(train_dataloader) or reached_limit
                should_step = (
                    (batch_idx + 1)
                    % int(cfg.training.gradient_accumulate_every)
                    == 0
                    or is_last
                )
                sync_context = nullcontext()
                if context.enabled and not should_step:
                    sync_context = training_model.no_sync()
                with sync_context:
                    raw_loss = training_model(batch)
                    (
                        raw_loss / cfg.training.gradient_accumulate_every
                    ).backward()
                losses.append(raw_loss.item())

                if should_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                    if ema is not None:
                        ema.step(self.model)

                self.global_step += 1
                progress.set_postfix(loss=raw_loss.item(), refresh=False)
                if reached_limit:
                    break

        if not losses:
            raise RuntimeError("Sim-Hand training dataset produced no batches")
        loss_stats = torch.tensor(
            [sum(losses), len(losses)],
            device=device,
            dtype=torch.float64,
        )
        if context.enabled:
            dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
        if context.enabled and source_counts is not None:
            dist.all_reduce(source_counts, op=dist.ReduceOp.SUM)
        result = {
            "train_loss": (loss_stats[0] / loss_stats[1]).item(),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        result.update(_source_fraction_log(source_counts, source_names))
        return result

    def _validate_epoch(self, cfg, val_dataloader, device) -> float | None:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                batch.pop("source_id", None)
                losses.append(self.model.compute_loss(batch))
                if (
                    cfg.training.max_val_steps is not None
                    and batch_idx + 1 >= int(cfg.training.max_val_steps)
                ):
                    break
        if not losses:
            return None
        return torch.stack(losses).mean().item()

    def _save_epoch_checkpoints(self, cfg) -> None:
        checkpoint_every = int(cfg.training.checkpoint_every)
        should_save_periodic = self.epoch % checkpoint_every == 0
        is_final_epoch = self.epoch >= int(cfg.training.num_epochs)
        if should_save_periodic:
            self.save_checkpoint(
                path=pathlib.Path(self.output_dir)
                / "checkpoints"
                / f"epoch_{self.epoch:04d}.ckpt",
                use_thread=False,
            )
            if cfg.checkpoint.save_last_snapshot:
                self.save_snapshot(tag=f"epoch_{self.epoch:04d}")
        if cfg.checkpoint.save_last_ckpt and (
            should_save_periodic or is_final_epoch
        ):
            self.save_checkpoint(tag="latest", use_thread=False)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_diffusion_unet_sim_hand_workspace",
)
def main(cfg):
    workspace = TrainDiffusionUnetSimHandWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
