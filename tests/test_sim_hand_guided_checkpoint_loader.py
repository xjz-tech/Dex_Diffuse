from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import dill
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from diffusion_policy.guidance.checkpoint_loader import LoadedPolicy, load_workspace_policy


@dataclass(frozen=True)
class FakeTemporal:
    usable_action_slice: slice


class FakeNormalizer(nn.Module):
    def __init__(self, sentinel: float):
        super().__init__()
        self.sentinel = nn.Parameter(
            torch.tensor(sentinel, dtype=torch.float32), requires_grad=False
        )


class FakePolicy(nn.Module):
    def __init__(self, weight: float):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(weight)
        self.num_inference_steps = 16
        self.temporal = FakeTemporal(usable_action_slice=slice(3, 8))
        self.normalizer = FakeNormalizer(sentinel=weight)


class FakeWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = FakePolicy(weight=1.0)
        self.ema_model = FakePolicy(weight=2.0)

    def load_payload(self, payload, exclude_keys=None, include_keys=None):
        for name, state in payload["state_dicts"].items():
            getattr(self, name).load_state_dict(state)
        for name in ("model", "ema_model"):
            meta_key = f"{name}_metadata"
            if meta_key in payload.get("pickles", {}):
                meta = dill.loads(payload["pickles"][meta_key])
                policy = getattr(self, name)
                policy.num_inference_steps = meta["num_inference_steps"]
                policy.temporal = meta["temporal"]
                policy.normalizer = meta["normalizer"]


class WorkspaceWithoutModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_payload(self, payload, exclude_keys=None, include_keys=None):
        return None


def _policy_metadata(policy: FakePolicy) -> dict:
    return {
        "num_inference_steps": policy.num_inference_steps,
        "temporal": policy.temporal,
        "normalizer": policy.normalizer,
    }


def _write_checkpoint(
    path: Path,
    *,
    use_ema: bool = False,
    include_cfg: bool = True,
    include_model: bool = True,
    model_weight: float = 1.0,
    ema_weight: float = 2.0,
    model_num_inference_steps: int = 42,
    ema_num_inference_steps: int = 43,
    model_temporal_slice: slice = slice(3, 8),
    ema_temporal_slice: slice = slice(4, 9),
    model_normalizer_sentinel: float = 11.0,
    ema_normalizer_sentinel: float = 22.0,
) -> None:
    cfg = OmegaConf.create(
        {
            "_target_": "tests.test_sim_hand_guided_checkpoint_loader.FakeWorkspace",
            "training": {"use_ema": use_ema},
        }
    )
    payload: dict = {"state_dicts": {}, "pickles": {}}
    if include_cfg:
        payload["cfg"] = cfg

    model = FakePolicy(weight=model_weight)
    ema_model = FakePolicy(weight=ema_weight)
    model.num_inference_steps = model_num_inference_steps
    ema_model.num_inference_steps = ema_num_inference_steps
    model.temporal = FakeTemporal(usable_action_slice=model_temporal_slice)
    ema_model.temporal = FakeTemporal(usable_action_slice=ema_temporal_slice)
    model.normalizer = FakeNormalizer(sentinel=model_normalizer_sentinel)
    ema_model.normalizer = FakeNormalizer(sentinel=ema_normalizer_sentinel)

    if include_model:
        payload["state_dicts"]["model"] = model.state_dict()
        payload["state_dicts"]["ema_model"] = ema_model.state_dict()
        payload["pickles"]["model_metadata"] = dill.dumps(_policy_metadata(model))
        payload["pickles"]["ema_model_metadata"] = dill.dumps(_policy_metadata(ema_model))

    torch.save(payload, path.open("wb"), pickle_module=dill)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def loader_patch():
    with patch(
        "diffusion_policy.guidance.checkpoint_loader.hydra.utils.get_class",
        return_value=FakeWorkspace,
    ):
        yield


def test_loader_selects_ema_when_cfg_requests_it(tmp_path, device, loader_patch):
    checkpoint = tmp_path / "ema.ckpt"
    _write_checkpoint(checkpoint, use_ema=True, ema_weight=7.0, model_weight=3.0)

    loaded = load_workspace_policy(checkpoint, device)

    assert isinstance(loaded, LoadedPolicy)
    assert loaded.used_ema is True
    assert loaded.policy.linear.weight.item() == pytest.approx(7.0)


def test_loader_uses_model_when_ema_is_disabled(tmp_path, device, loader_patch):
    checkpoint = tmp_path / "model.ckpt"
    _write_checkpoint(checkpoint, use_ema=False, ema_weight=7.0, model_weight=3.0)

    loaded = load_workspace_policy(checkpoint, device)

    assert loaded.used_ema is False
    assert loaded.policy.linear.weight.item() == pytest.approx(3.0)


def test_loader_moves_policy_to_device_eval_and_disables_grad(tmp_path, loader_patch):
    checkpoint = tmp_path / "device.ckpt"
    _write_checkpoint(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = load_workspace_policy(checkpoint, device)

    assert next(loaded.policy.parameters()).device.type == device.type
    assert loaded.policy.training is False
    assert all(not parameter.requires_grad for parameter in loaded.policy.parameters())


def test_loader_rejects_missing_checkpoint_or_cfg_or_model(tmp_path, device, loader_patch):
    missing = tmp_path / "missing.ckpt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_workspace_policy(missing, device)

    no_cfg = tmp_path / "no_cfg.ckpt"
    _write_checkpoint(no_cfg, include_cfg=False)
    with pytest.raises(ValueError, match="Checkpoint has no cfg"):
        load_workspace_policy(no_cfg, device)

    no_model = tmp_path / "no_model.ckpt"
    _write_checkpoint(no_model, include_model=False)
    with patch(
        "diffusion_policy.guidance.checkpoint_loader.hydra.utils.get_class",
        return_value=WorkspaceWithoutModel,
    ):
        with pytest.raises(ValueError, match="Workspace has no model"):
            load_workspace_policy(no_model, device)


def test_loader_does_not_override_policy_temporal_or_num_inference_steps(
    tmp_path, device, loader_patch
):
    checkpoint = tmp_path / "temporal.ckpt"
    _write_checkpoint(
        checkpoint,
        use_ema=True,
        ema_num_inference_steps=99,
        ema_temporal_slice=slice(5, 10),
        model_num_inference_steps=88,
        model_temporal_slice=slice(1, 2),
    )

    loaded = load_workspace_policy(checkpoint, device)

    assert loaded.policy.num_inference_steps == 99
    assert loaded.policy.temporal.usable_action_slice == slice(5, 10)


def test_loader_preserves_normalizer_state_in_selected_policy(
    tmp_path, device, loader_patch
):
    checkpoint = tmp_path / "normalizer.ckpt"
    _write_checkpoint(
        checkpoint,
        use_ema=True,
        ema_normalizer_sentinel=55.0,
        model_normalizer_sentinel=44.0,
    )

    loaded = load_workspace_policy(checkpoint, device)

    assert loaded.policy.normalizer.sentinel.item() == pytest.approx(55.0)


def test_guidance_package_import_has_no_hardware_side_effects():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import diffusion_policy.guidance.checkpoint_loader
assert "inference_dp" not in sys.modules
assert "direct_robot_env" not in sys.modules
assert "diffusion_policy.real_world" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=repo_root,
    )
