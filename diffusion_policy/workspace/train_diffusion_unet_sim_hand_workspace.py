from __future__ import annotations

import copy
import math
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.bounded_sampler import (
    EpochRandomSampler,
    EvenlySpacedSampler,
)
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
        print(format_sim_hand_temporal_config(temporal))
        if int(cfg.training.checkpoint_every) <= 0:
            raise ValueError(
                "training.checkpoint_every must be a positive step interval"
            )
        if int(cfg.training.steps_per_epoch) <= 0:
            raise ValueError("training.steps_per_epoch must be positive")
        if int(cfg.training.validation_steps) <= 0:
            raise ValueError("training.validation_steps must be positive")

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
        output_dir = pathlib.Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.training.resume:
            latest_path = self.get_checkpoint_path()
            if latest_path.is_file():
                print(f"Resuming from checkpoint {latest_path}")
                self.load_checkpoint(path=latest_path)

        dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseLowdimDataset):
            raise TypeError("Sim-Hand dataset must inherit BaseLowdimDataset")
        train_samples = int(cfg.training.steps_per_epoch) * int(cfg.dataloader.batch_size)
        train_sampler = EpochRandomSampler(
            dataset,
            num_samples=train_samples,
            seed=int(cfg.training.seed),
        )
        train_dataloader = DataLoader(dataset, sampler=train_sampler, **cfg.dataloader)
        val_dataset = dataset.get_validation_dataset()
        val_samples = (
            int(cfg.training.validation_steps)
            * int(cfg.val_dataloader.batch_size)
        )
        val_sampler = EvenlySpacedSampler(val_dataset, num_samples=val_samples)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            **cfg.val_dataloader,
        )
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

        env_runner: BaseLowdimRunner = hydra.utils.instantiate(
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
        wandb_run.config.update({"output_dir": str(output_dir)})

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = min(int(cfg.training.num_epochs), 2)
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        train_sampling_batch = None
        log_path = output_dir / "logs.json.txt"
        try:
            with JsonLogger(str(log_path)) as json_logger:
                while self.epoch < int(cfg.training.num_epochs):
                    local_epoch_idx = self.epoch
                    train_sampler.set_epoch(self.epoch)
                    step_log = self._train_epoch(
                        cfg,
                        train_dataloader,
                        device,
                        lr_scheduler,
                        ema,
                    )
                    if train_sampling_batch is None:
                        train_sampling_batch = next(iter(train_dataloader))
                        train_sampling_batch = dict_apply(
                            train_sampling_batch,
                            lambda value: value.to(device, non_blocking=True),
                        )

                    policy = self.ema_model or self.model
                    policy.eval()
                    if local_epoch_idx % int(cfg.training.rollout_every) == 0:
                        step_log.update(env_runner.run(policy))
                    if local_epoch_idx % int(cfg.training.val_every) == 0:
                        val_loss = self._validate_epoch(
                            cfg,
                            val_dataloader,
                            device,
                        )
                        if val_loss is not None:
                            step_log["val_loss"] = val_loss
                    if local_epoch_idx % int(cfg.training.sample_every) == 0:
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
                    step_log["epoch"] = self.epoch
                    step_log["global_step"] = self.global_step
                    self._save_step_checkpoints(cfg)
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    policy.train()
        finally:
            wandb_run.finish()

    def _train_epoch(
        self,
        cfg,
        train_dataloader,
        device,
        lr_scheduler,
        ema: EMAModel | None,
    ) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        max_steps = cfg.training.max_train_steps
        with tqdm.tqdm(
            train_dataloader,
            desc=f"Training epoch {self.epoch}",
            leave=False,
            mininterval=cfg.training.tqdm_interval_sec,
        ) as progress:
            for batch_idx, batch in enumerate(progress):
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                raw_loss = self.model.compute_loss(batch)
                (raw_loss / cfg.training.gradient_accumulate_every).backward()
                losses.append(raw_loss.item())

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
        return {
            "train_loss": float(np.mean(losses)),
            "lr": lr_scheduler.get_last_lr()[0],
        }

    def _validate_epoch(self, cfg, val_dataloader, device) -> float | None:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = dict_apply(
                    batch,
                    lambda value: value.to(device, non_blocking=True),
                )
                losses.append(self.model.compute_loss(batch))
                if (
                    cfg.training.max_val_steps is not None
                    and batch_idx + 1 >= int(cfg.training.max_val_steps)
                ):
                    break
        if not losses:
            return None
        return torch.stack(losses).mean().item()

    def _save_step_checkpoints(self, cfg) -> None:
        checkpoint_every = int(cfg.training.checkpoint_every)
        should_save_periodic = (
            self.global_step > 0
            and self.global_step % checkpoint_every == 0
        )
        is_final_epoch = self.epoch >= int(cfg.training.num_epochs)
        if should_save_periodic:
            self.save_checkpoint(
                path=pathlib.Path(self.output_dir)
                / "checkpoints"
                / f"step_{self.global_step:08d}.ckpt",
                use_thread=False,
            )
            if cfg.checkpoint.save_last_snapshot:
                self.save_snapshot(tag=f"step_{self.global_step:08d}")
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
