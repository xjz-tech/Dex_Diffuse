from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import hydra
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace


@dataclass(frozen=True)
class LoadedPolicy:
    checkpoint_path: Path
    cfg: Any
    workspace: BaseWorkspace
    policy: torch.nn.Module
    used_ema: bool


def load_workspace_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedPolicy:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(
        checkpoint_path.open("rb"),
        pickle_module=dill,
        map_location="cpu",
    )
    if "cfg" not in payload:
        raise ValueError(f"Checkpoint has no cfg: {checkpoint_path}")

    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    if not hasattr(workspace, "model"):
        raise ValueError(f"Workspace has no model: {checkpoint_path}")

    use_ema = bool(getattr(cfg.training, "use_ema", False))
    ema_model = getattr(workspace, "ema_model", None)
    policy = ema_model if use_ema and ema_model is not None else workspace.model
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    return LoadedPolicy(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        workspace=workspace,
        policy=policy,
        used_ema=(policy is ema_model),
    )
