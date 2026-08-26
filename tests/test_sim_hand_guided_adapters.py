from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from diffusion_policy.guidance.checkpoint_loader import LoadedPolicy
from diffusion_policy.guidance.real_adapter import RealPolicyAdapter
from diffusion_policy.guidance.sim_adapter import SimPolicyAdapter
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)


HAND_DIM = 22
ACTION_DIM = 31
EE_DIM = 9


@dataclass(frozen=True)
class FakeTemporal:
    usable_action_slice: slice


class TrackingRealPolicy(ModuleAttrMixin):
    def __init__(
        self,
        *,
        n_obs_steps: int = 2,
        action_t: int = 50,
        action_dim: int = ACTION_DIM,
        batch: int = 1,
        ndim: int = 3,
        finite: bool = True,
        include_action_key: bool = True,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self._action_t = action_t
        self._action_dim = action_dim
        self._batch = batch
        self._ndim = ndim
        self._finite = finite
        self._include_action_key = include_action_key
        self.predict_calls = 0
        self.last_observation: dict[str, torch.Tensor] | None = None
        self.sentinel = nn.Parameter(torch.zeros(1), requires_grad=False)

    def predict_action(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.predict_calls += 1
        self.last_observation = observation
        if not self._include_action_key:
            return {}
        shape = [self._batch]
        if self._ndim >= 2:
            shape.append(self._action_t)
        if self._ndim >= 3:
            shape.append(self._action_dim)
        while len(shape) < self._ndim:
            shape.append(1)
        action = torch.ones(shape, dtype=self.dtype, device=self.device)
        if not self._finite:
            action[0, 0, 0] = torch.nan
        return {"action": action}


def _real_shape_meta() -> dict[str, Any]:
    return {
        "obs": {
            "front_image": {"shape": [3, 240, 320], "type": "rgb"},
            "wrist_image": {"shape": [3, 240, 320], "type": "rgb"},
            "ee_pose": {"shape": [9], "type": "low_dim"},
            "hand_joint": {"shape": [22], "type": "low_dim"},
        },
        "action": {"shape": [31]},
    }


def _make_loaded_real(
    policy: TrackingRealPolicy | None = None,
    *,
    n_obs_steps: int = 2,
) -> LoadedPolicy:
    if policy is None:
        policy = TrackingRealPolicy(n_obs_steps=n_obs_steps)
    cfg = OmegaConf.create({"shape_meta": _real_shape_meta(), "n_obs_steps": n_obs_steps})
    return LoadedPolicy(
        checkpoint_path=Path("/tmp/fake_real.ckpt"),
        cfg=cfg,
        workspace=MagicMock(),
        policy=policy,
        used_ema=False,
    )


class TrackingSimModel(nn.Module):
    def __init__(self, action_dim: int = HAND_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.bias = nn.Parameter(torch.tensor(0.25), requires_grad=False)

    def forward(self, sample, timestep, global_cond=None, local_cond=None):
        self.calls.append((sample, timestep, global_cond))
        return sample + self.bias


class TrackingSimPolicy(nn.Module):
    def __init__(
        self,
        *,
        n_obs_steps: int,
        horizon: int,
        n_pred_action_steps: int,
        usable_start: int,
        obs_dim: int = HAND_DIM,
        action_dim: int = HAND_DIM,
        obs_scale: float = 2.0,
        action_scale: float = 3.0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_pred_action_steps = n_pred_action_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.temporal = FakeTemporal(
            usable_action_slice=slice(usable_start, usable_start + n_pred_action_steps)
        )
        self.model = TrackingSimModel(action_dim=action_dim)
        self.normalizer = LinearNormalizer()
        self.normalizer["obs"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((obs_dim,), obs_scale, dtype=torch.float32),
            offset=torch.zeros(obs_dim, dtype=torch.float32),
            input_stats_dict={
                "min": torch.full((obs_dim,), -1.0),
                "max": torch.full((obs_dim,), 1.0),
                "mean": torch.zeros(obs_dim),
                "std": torch.ones(obs_dim),
            },
        )
        self.normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((action_dim,), action_scale, dtype=torch.float32),
            offset=torch.zeros(action_dim, dtype=torch.float32),
            input_stats_dict={
                "min": torch.full((action_dim,), -1.0),
                "max": torch.full((action_dim,), 1.0),
                "mean": torch.zeros(action_dim),
                "std": torch.ones(action_dim),
            },
        )
        self.predict_action_calls = 0

    def predict_action(self, obs_dict):
        self.predict_action_calls += 1
        raise AssertionError("SimPolicyAdapter must never call policy.predict_action")


def make_sim_policy(
    *,
    n_obs_steps: int,
    horizon: int,
    n_pred_action_steps: int,
    usable_start: int,
    obs_dim: int = HAND_DIM,
    action_dim: int = HAND_DIM,
    obs_scale: float = 2.0,
    action_scale: float = 3.0,
) -> LoadedPolicy:
    policy = TrackingSimPolicy(
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_pred_action_steps=n_pred_action_steps,
        usable_start=usable_start,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_scale=obs_scale,
        action_scale=action_scale,
    )
    return LoadedPolicy(
        checkpoint_path=Path("/tmp/fake_sim.ckpt"),
        cfg=OmegaConf.create({}),
        workspace=MagicMock(),
        policy=policy,
        used_ema=False,
    )


# ---------------------------------------------------------------------------
# Real adapter
# ---------------------------------------------------------------------------


def test_real_build_synthetic_observation_matches_shape_meta_and_n_obs_steps():
    loaded = _make_loaded_real(n_obs_steps=2)
    adapter = RealPolicyAdapter(loaded)

    obs = adapter.build_synthetic_observation()
    policy = loaded.policy

    assert obs["front_image"].shape == (1, policy.n_obs_steps, 3, 240, 320)
    assert obs["wrist_image"].shape == (1, policy.n_obs_steps, 3, 240, 320)
    assert obs["ee_pose"].shape == (1, policy.n_obs_steps, 9)
    assert obs["hand_joint"].shape == (1, policy.n_obs_steps, 22)
    for tensor in obs.values():
        assert tensor.dtype == policy.dtype
        assert tensor.device.type == policy.device.type
        assert torch.isfinite(tensor).all()


def test_real_predict_proposal_calls_policy_once_and_returns_finite_action():
    policy = TrackingRealPolicy(action_t=50)
    adapter = RealPolicyAdapter(_make_loaded_real(policy))

    proposal = adapter.predict_proposal()

    assert policy.predict_calls == 1
    assert proposal.shape == (1, 50, 31)
    assert torch.isfinite(proposal).all()
    assert policy.last_observation is not None
    assert set(policy.last_observation) == {
        "front_image",
        "wrist_image",
        "ee_pose",
        "hand_joint",
    }


def test_real_predict_proposal_accepts_non_fifty_temporal_length():
    policy = TrackingRealPolicy(action_t=17)
    adapter = RealPolicyAdapter(_make_loaded_real(policy))

    proposal = adapter.predict_proposal()

    assert proposal.shape == (1, 17, 31)
    assert torch.isfinite(proposal).all()


def test_real_predict_proposal_rejects_wrong_batch_rank_or_action_dim():
    for kwargs, match in (
        ({"batch": 2}, r"\(1,T,31\)"),
        ({"ndim": 2, "action_t": 31}, r"\(1,T,31\)"),
        ({"action_dim": 30}, r"\(1,T,31\)"),
    ):
        policy = TrackingRealPolicy(**kwargs)
        adapter = RealPolicyAdapter(_make_loaded_real(policy))
        with pytest.raises(ValueError, match=match):
            adapter.predict_proposal()


def test_real_predict_proposal_rejects_missing_action_or_non_finite():
    missing = TrackingRealPolicy(include_action_key=False)
    adapter = RealPolicyAdapter(_make_loaded_real(missing))
    with pytest.raises(KeyError, match="no 'action'"):
        adapter.predict_proposal()

    non_finite = TrackingRealPolicy(finite=False)
    adapter = RealPolicyAdapter(_make_loaded_real(non_finite))
    with pytest.raises(ValueError, match="non-finite"):
        adapter.predict_proposal()


def test_real_predict_proposal_uses_provided_observation_without_rebuild():
    policy = TrackingRealPolicy(action_t=11)
    adapter = RealPolicyAdapter(_make_loaded_real(policy))
    custom = {
        "front_image": torch.ones(1, 2, 3, 240, 320),
        "wrist_image": torch.ones(1, 2, 3, 240, 320),
        "ee_pose": torch.ones(1, 2, 9),
        "hand_joint": torch.ones(1, 2, 22),
    }

    proposal = adapter.predict_proposal(custom)

    assert policy.predict_calls == 1
    assert policy.last_observation is custom
    assert proposal.shape == (1, 11, 31)


# ---------------------------------------------------------------------------
# Sim adapter
# ---------------------------------------------------------------------------


def test_sim_adapter_reads_temporal_metadata_from_each_policy():
    current = make_sim_policy(
        n_obs_steps=4,
        horizon=12,
        n_pred_action_steps=9,
        usable_start=3,
    )
    alternate = make_sim_policy(
        n_obs_steps=2,
        horizon=8,
        n_pred_action_steps=7,
        usable_start=1,
    )

    current_adapter = SimPolicyAdapter(current)
    alternate_adapter = SimPolicyAdapter(alternate)

    assert current_adapter.horizon == 12
    assert current_adapter.n_obs_steps == 4
    assert current_adapter.n_pred_action_steps == 9
    assert current_adapter.action_dim == 22
    assert current_adapter.oa_start == 3
    assert current_adapter.guided_slice(5) == slice(3, 8)

    assert alternate_adapter.horizon == 8
    assert alternate_adapter.n_obs_steps == 2
    assert alternate_adapter.n_pred_action_steps == 7
    assert alternate_adapter.action_dim == 22
    assert alternate_adapter.oa_start == 1
    assert alternate_adapter.guided_slice(3) == slice(1, 4)


def test_sim_guided_slice_rejects_invalid_execution_steps():
    adapter = SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=4,
            horizon=12,
            n_pred_action_steps=9,
            usable_start=3,
        )
    )

    with pytest.raises(ValueError):
        adapter.guided_slice(0)
    with pytest.raises(ValueError):
        adapter.guided_slice(10)
    with pytest.raises(ValueError):
        # oa_start(3) + E(10) would exceed horizon even if pred allowed it
        SimPolicyAdapter(
            make_sim_policy(
                n_obs_steps=4,
                horizon=12,
                n_pred_action_steps=9,
                usable_start=4,
            )
        ).guided_slice(9)


def test_sim_adapter_rejects_non_22_dims():
    with pytest.raises(ValueError, match="22"):
        SimPolicyAdapter(
            make_sim_policy(
                n_obs_steps=4,
                horizon=12,
                n_pred_action_steps=9,
                usable_start=3,
                obs_dim=21,
                action_dim=21,
            )
        )


def test_sim_normalizers_are_independent_for_obs_and_action():
    adapter = SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=4,
            horizon=12,
            n_pred_action_steps=9,
            usable_start=3,
            obs_scale=2.0,
            action_scale=3.0,
        )
    )
    history = torch.ones(1, 4, HAND_DIM)
    reference = torch.ones(1, 5, HAND_DIM)

    history_norm = adapter.normalize_history(history)
    reference_norm = adapter.normalize_reference(reference)
    action_back = adapter.unnormalize_action(reference_norm)

    torch.testing.assert_close(history_norm, history * 2.0)
    torch.testing.assert_close(reference_norm, reference * 3.0)
    torch.testing.assert_close(action_back, reference)
    assert not torch.allclose(history_norm, reference_norm[:, :4, :])


def test_sim_global_condition_flattens_dynamic_n_obs_steps():
    current = SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=4,
            horizon=12,
            n_pred_action_steps=9,
            usable_start=3,
            obs_scale=2.0,
        )
    )
    alternate = SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=2,
            horizon=8,
            n_pred_action_steps=7,
            usable_start=1,
            obs_scale=2.0,
        )
    )
    history = torch.arange(1 * 6 * HAND_DIM, dtype=torch.float32).reshape(1, 6, HAND_DIM)

    current_cond = current.global_condition(history)
    alternate_cond = alternate.global_condition(history)

    assert current_cond.shape == (1, 4 * HAND_DIM)
    assert alternate_cond.shape == (1, 2 * HAND_DIM)
    torch.testing.assert_close(
        current_cond,
        (history[:, :4, :] * 2.0).reshape(1, -1),
    )
    torch.testing.assert_close(
        alternate_cond,
        (history[:, :2, :] * 2.0).reshape(1, -1),
    )


def test_sim_predict_epsilon_calls_model_directly_and_never_predict_action():
    loaded = make_sim_policy(
        n_obs_steps=4,
        horizon=12,
        n_pred_action_steps=9,
        usable_start=3,
    )
    adapter = SimPolicyAdapter(loaded)
    policy = loaded.policy
    sample = torch.zeros(1, 12, HAND_DIM)
    global_cond = torch.zeros(1, 4 * HAND_DIM)

    output = adapter.predict_epsilon(sample, timestep=3, global_cond=global_cond)

    assert policy.predict_action_calls == 0
    assert len(policy.model.calls) == 1
    called_sample, called_timestep, called_cond = policy.model.calls[0]
    assert called_sample is sample
    assert called_cond is global_cond
    assert int(called_timestep) == 3 or called_timestep.item() == 3
    torch.testing.assert_close(output, sample + 0.25)
    assert torch.isfinite(output).all()


def test_sim_rejects_non_finite_history_reference_or_epsilon():
    adapter = SimPolicyAdapter(
        make_sim_policy(
            n_obs_steps=4,
            horizon=12,
            n_pred_action_steps=9,
            usable_start=3,
        )
    )
    bad_history = torch.ones(1, 4, HAND_DIM)
    bad_history[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        adapter.normalize_history(bad_history)

    bad_reference = torch.ones(1, 5, HAND_DIM)
    bad_reference[0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="non-finite"):
        adapter.normalize_reference(bad_reference)

    class NanModel(nn.Module):
        def forward(self, sample, timestep, global_cond=None, local_cond=None):
            out = sample.clone()
            out[0, 0, 0] = torch.nan
            return out

    loaded = make_sim_policy(
        n_obs_steps=4,
        horizon=12,
        n_pred_action_steps=9,
        usable_start=3,
    )
    loaded.policy.model = NanModel()
    adapter = SimPolicyAdapter(loaded)
    with pytest.raises(ValueError, match="non-finite"):
        adapter.predict_epsilon(
            torch.zeros(1, 12, HAND_DIM),
            timestep=0,
            global_cond=torch.zeros(1, 4 * HAND_DIM),
        )
    assert loaded.policy.predict_action_calls == 0
