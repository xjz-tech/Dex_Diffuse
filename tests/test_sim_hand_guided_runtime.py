from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from diffusion_policy.guidance.checkpoint_loader import LoadedPolicy
from diffusion_policy.guidance.real_adapter import RealPolicyAdapter
from diffusion_policy.guidance.sim_adapter import SimPolicyAdapter
from diffusion_policy.guidance.sim_hand_guidance import (
    SimHandGuidance,
    SimHandGuidanceConfig,
)
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

HAND_DIM = 22
REAL_ACTION_DIM = 31
HAND_SLICE = slice(9, 31)
EXPECTED_CURRENT_TIMESTEPS = (88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)


@dataclass(frozen=True)
class FakeTemporal:
    usable_action_slice: slice


class TrackingRealPolicy(ModuleAttrMixin):
    def __init__(
        self,
        *,
        n_obs_steps: int = 2,
        action_t: int = 50,
        full_prediction_t: int | None = None,
        action_dim: int = REAL_ACTION_DIM,
        normalizer_sentinel: float = 1.0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self._action_t = action_t
        self._full_prediction_t = full_prediction_t
        self._action_dim = action_dim
        self.predict_calls = 0
        self.last_observation: dict[str, torch.Tensor] | None = None
        self.sentinel = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.normalizer = LinearNormalizer()
        self.normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((action_dim,), normalizer_sentinel, dtype=torch.float32),
            offset=torch.zeros(action_dim, dtype=torch.float32),
            input_stats_dict={
                "min": torch.full((action_dim,), -1.0),
                "max": torch.full((action_dim,), 1.0),
                "mean": torch.zeros(action_dim),
                "std": torch.ones(action_dim),
            },
        )

    def predict_action(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.predict_calls += 1
        self.last_observation = observation
        # The public proposal is an aligned prefix of a longer full prediction.
        action_start = self.n_obs_steps - 1
        aligned_prediction_t = (
            self._action_t + 4
            if self._full_prediction_t is None
            else self._full_prediction_t
        )
        prediction_t = action_start + aligned_prediction_t
        action_pred = torch.zeros(
            (1, prediction_t, self._action_dim),
            dtype=self.dtype,
            device=self.device,
        )
        action_pred[..., :9] = 1.0
        action_pred[..., 9:] = torch.linspace(
            0.1,
            0.9,
            prediction_t * HAND_DIM,
            dtype=self.dtype,
            device=self.device,
        ).reshape(prediction_t, HAND_DIM)
        action = action_pred[:, action_start : action_start + self._action_t]
        return {"action": action, "action_pred": action_pred}


class TrackingSimModel(nn.Module):
    def __init__(self, action_dim: int = HAND_DIM):
        super().__init__()
        self.action_dim = action_dim
        self.bias = nn.Parameter(torch.tensor(0.25), requires_grad=False)

    def forward(self, sample, timestep, global_cond=None, local_cond=None):
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
        num_train_timesteps: int = 100,
        normalizer_sentinel: float = 7.0,
    ):
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_pred_action_steps = n_pred_action_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.temporal = FakeTemporal(
            usable_action_slice=slice(
                usable_start, usable_start + n_pred_action_steps
            )
        )
        self.model = TrackingSimModel(action_dim=action_dim)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            variance_type="fixed_small",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.normalizer = LinearNormalizer()
        self.normalizer["obs"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((obs_dim,), obs_scale * normalizer_sentinel, dtype=torch.float32),
            offset=torch.zeros(obs_dim, dtype=torch.float32),
            input_stats_dict={
                "min": torch.full((obs_dim,), -1.0),
                "max": torch.full((obs_dim,), 1.0),
                "mean": torch.zeros(obs_dim),
                "std": torch.ones(obs_dim),
            },
        )
        self.normalizer["action"] = SingleFieldLinearNormalizer.create_manual(
            scale=torch.full((action_dim,), action_scale * normalizer_sentinel, dtype=torch.float32),
            offset=torch.zeros(action_dim, dtype=torch.float32),
            input_stats_dict={
                "min": torch.full((action_dim,), -1.0),
                "max": torch.full((action_dim,), 1.0),
                "mean": torch.zeros(action_dim),
                "std": torch.ones(action_dim),
            },
        )
        self.predict_action_calls = 0

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def predict_action(self, obs_dict):
        self.predict_action_calls += 1
        raise AssertionError("runtime must never call Sim policy.predict_action")


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


def make_real_loaded(
    *,
    action_t: int = 50,
    n_obs_steps: int = 2,
    full_prediction_t: int | None = None,
    normalizer_sentinel: float = 1.0,
) -> LoadedPolicy:
    policy = TrackingRealPolicy(
        n_obs_steps=n_obs_steps,
        action_t=action_t,
        full_prediction_t=full_prediction_t,
        normalizer_sentinel=normalizer_sentinel,
    )
    cfg = OmegaConf.create(
        {"shape_meta": _real_shape_meta(), "n_obs_steps": n_obs_steps}
    )
    return LoadedPolicy(
        checkpoint_path=Path("/tmp/fake_real.ckpt"),
        cfg=cfg,
        workspace=MagicMock(),
        policy=policy,
        used_ema=False,
    )


def make_sim_loaded(
    *,
    n_obs_steps: int = 4,
    horizon: int = 12,
    n_pred_action_steps: int = 9,
    usable_start: int = 3,
    num_train_timesteps: int = 100,
    normalizer_sentinel: float = 7.0,
) -> LoadedPolicy:
    policy = TrackingSimPolicy(
        n_obs_steps=n_obs_steps,
        horizon=horizon,
        n_pred_action_steps=n_pred_action_steps,
        usable_start=usable_start,
        num_train_timesteps=num_train_timesteps,
        normalizer_sentinel=normalizer_sentinel,
    )
    return LoadedPolicy(
        checkpoint_path=Path("/tmp/fake_sim.ckpt"),
        cfg=OmegaConf.create({}),
        workspace=MagicMock(),
        policy=policy,
        used_ema=False,
    )


class RecordingGuidance:
    """Wraps SimHandGuidance to record guide/verify call counts and args."""

    def __init__(self, inner: SimHandGuidance):
        self.inner = inner
        self.config = inner.config
        self.adapter = inner.adapter
        self.guidance_slice = inner.guidance_slice
        self.trajectory_shape = inner.trajectory_shape
        self.guide_calls: list[dict[str, Any]] = []
        self.verify_calls: list[dict[str, Any]] = []

    def guide_segment(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        self.guide_calls.append(
            {
                "history": hand_state_history.detach().clone(),
                "reference": hand_reference.detach().clone(),
                "generator": generator,
            }
        )
        return self.inner.guide_segment(
            hand_state_history,
            hand_reference,
            generator=generator,
        )

    def verify_zero_guidance(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
    ):
        self.verify_calls.append(
            {
                "history": hand_state_history.detach().clone(),
                "reference": hand_reference.detach().clone(),
                "initial_noise": initial_noise.detach().clone(),
            }
        )
        return self.inner.verify_zero_guidance(
            hand_state_history,
            hand_reference,
            initial_noise=initial_noise,
        )


def _build_loaded(
    *,
    real: LoadedPolicy | None = None,
    sim: LoadedPolicy | None = None,
    config: SimHandGuidanceConfig | None = None,
    record: bool = False,
):
    from diffusion_policy.guidance.runtime import LoadedGuidedPolicies

    real = real or make_real_loaded()
    sim = sim or make_sim_loaded()
    config = config or SimHandGuidanceConfig(
        execution_steps=5,
        guidance_scale=1.0,
        num_inference_steps=12,
        eta=0.0,
    )
    real_adapter = RealPolicyAdapter(real)
    sim_adapter = SimPolicyAdapter(sim)
    guidance: Any = SimHandGuidance(sim_adapter, config)
    if record:
        guidance = RecordingGuidance(guidance)
    return LoadedGuidedPolicies(
        real=real,
        sim=sim,
        real_adapter=real_adapter,
        sim_adapter=sim_adapter,
        guidance=guidance,
    )


def test_load_guided_policies_calls_loader_twice_without_overrides():
    from diffusion_policy.guidance.runtime import load_guided_policies

    real = make_real_loaded()
    sim = make_sim_loaded()
    device = torch.device("cpu")
    config = SimHandGuidanceConfig()
    calls: list[tuple[Any, ...]] = []

    def fake_loader(checkpoint_path, device_arg):
        calls.append((checkpoint_path, device_arg))
        path = Path(checkpoint_path)
        if "real" in path.name:
            return real
        return sim

    with patch(
        "diffusion_policy.guidance.runtime.load_workspace_policy",
        side_effect=fake_loader,
    ) as mocked:
        loaded = load_guided_policies(
            Path("/tmp/fake_real.ckpt"),
            Path("/tmp/fake_sim.ckpt"),
            device,
            config,
        )

    assert mocked.call_count == 2
    for call in mocked.call_args_list:
        assert call.args == (call.args[0], device)
        assert call.kwargs == {}
        # No DINO / temporal override kwargs were accepted or forwarded.
        assert "dino" not in str(call).lower()
        assert "temporal" not in str(call).lower()
        assert "override" not in str(call).lower()

    assert isinstance(loaded.real_adapter, RealPolicyAdapter)
    assert isinstance(loaded.sim_adapter, SimPolicyAdapter)
    assert isinstance(loaded.guidance, SimHandGuidance)
    assert loaded.sim_adapter.horizon == sim.policy.horizon
    assert loaded.sim_adapter.n_obs_steps == sim.policy.n_obs_steps
    assert loaded.sim_adapter.n_pred_action_steps == sim.policy.n_pred_action_steps
    assert loaded.real.policy.normalizer is not loaded.sim.policy.normalizer
    assert id(loaded.real.policy.normalizer) != id(loaded.sim.policy.normalizer)


def test_run_check_current_target_shapes_and_timesteps():
    from diffusion_policy.guidance.runtime import run_check

    loaded = _build_loaded(record=True)
    report = run_check(loaded, seed=0)

    assert loaded.real.policy.predict_calls == 1
    assert len(loaded.guidance.guide_calls) == 1
    assert len(loaded.guidance.verify_calls) == 1
    assert report.real_action_shape == (1, 50, 31)
    assert report.real_hand_reference_shape == (1, 54, 22)
    assert report.segment_count == 10
    assert report.guided_hand_shape == (1, 5, 22)
    assert report.sim_horizon == 12
    assert report.sim_obs_steps == 4
    assert report.sim_pred_action_steps == 9
    assert report.guidance_slice == (3, 12)
    assert report.execution_slice == (3, 8)
    assert report.real_hand_reference.shape == (1, 54, 22)
    assert loaded.guidance.guide_calls[0]["reference"].shape == (1, 9, 22)
    assert report.timesteps == EXPECTED_CURRENT_TIMESTEPS
    assert report.max_x0_error >= 0.0
    assert report.max_prev_error >= 0.0
    assert tuple(report.real_action_proposal.shape) == report.real_action_shape


def test_run_check_calls_guide_segment_even_when_scale_is_zero():
    from diffusion_policy.guidance.runtime import run_check

    loaded = _build_loaded(
        config=SimHandGuidanceConfig(
            execution_steps=5,
            guidance_scale=0.0,
            num_inference_steps=12,
            eta=0.0,
        ),
        record=True,
    )
    report = run_check(loaded, seed=3)

    assert len(loaded.guidance.guide_calls) == 1
    assert len(loaded.guidance.verify_calls) == 1
    assert report.guided_hand_shape == (1, 5, 22)
    assert report.timesteps == EXPECTED_CURRENT_TIMESTEPS


def test_run_check_rejects_real_sim_dtype_mismatch_without_casting():
    from diffusion_policy.guidance.runtime import run_check

    loaded = _build_loaded(record=True)
    loaded.real.policy.to(dtype=torch.float64)

    with pytest.raises(ValueError, match="Real proposal.*dtype"):
        run_check(loaded, seed=0)


def test_run_check_uses_seeded_synthetic_history_not_real_hand_joint():
    from diffusion_policy.guidance.runtime import run_check

    loaded = _build_loaded(record=True)
    report = run_check(loaded, seed=11)

    history = loaded.guidance.guide_calls[0]["history"]
    verify_history = loaded.guidance.verify_calls[0]["history"]
    n_obs = loaded.sim_adapter.n_obs_steps
    assert history.shape == (1, n_obs, HAND_DIM)
    assert verify_history.shape == (1, n_obs, HAND_DIM)
    torch.testing.assert_close(history, verify_history)
    # Synthetic Real hand_joint is zeros; history must be seeded noise, not that.
    assert not torch.allclose(history, torch.zeros_like(history))
    obs = loaded.real.policy.last_observation
    assert obs is not None
    assert not torch.equal(history.cpu(), obs["hand_joint"][:, :n_obs, :].cpu())

    noise = loaded.guidance.verify_calls[0]["initial_noise"]
    assert noise.shape == (
        1,
        loaded.sim_adapter.horizon,
        loaded.sim_adapter.action_dim,
    )
    # Oracle noise is independent of guide_segment's internal draw.
    assert noise.shape != history.shape or not torch.equal(noise, history)
    assert report.segment_count == report.real_action_shape[1] // 5


def test_run_check_alternate_fake_changes_dynamic_fields():
    from diffusion_policy.guidance.runtime import run_check

    loaded = _build_loaded(
        real=make_real_loaded(action_t=15),
        sim=make_sim_loaded(
            n_obs_steps=2,
            horizon=8,
            n_pred_action_steps=7,
            usable_start=1,
            num_train_timesteps=50,
        ),
        config=SimHandGuidanceConfig(
            execution_steps=5,
            guidance_scale=1.0,
            num_inference_steps=5,
            eta=0.0,
        ),
        record=True,
    )
    report = run_check(loaded, seed=0)

    assert report.real_action_shape == (1, 15, 31)
    assert report.segment_count == 3
    assert report.guided_hand_shape == (1, 5, 22)
    assert report.sim_horizon == 8
    assert report.sim_obs_steps == 2
    assert report.sim_pred_action_steps == 7
    assert report.guidance_slice == (1, 8)
    assert report.execution_slice == (1, 6)
    assert report.timesteps != EXPECTED_CURRENT_TIMESTEPS
    assert len(report.timesteps) == 5


def test_runtime_production_has_no_fixed_current_constants():
    source = (
        Path(__file__).resolve().parents[1]
        / "diffusion_policy"
        / "guidance"
        / "runtime.py"
    ).read_text()
    assert "EXPECTED_CURRENT_TIMESTEPS" not in source
    assert str(EXPECTED_CURRENT_TIMESTEPS) not in source
    # No production hardcodes of the current-target contract values.
    for forbidden in ("(1, 50, 31)", "segment_count = 10", "horizon == 12", "= 50"):
        assert forbidden not in source


def test_run_dry_run_reuses_proposal_and_completes_ten_segments():
    from diffusion_policy.guidance.runtime import run_dry_run

    loaded = _build_loaded(record=True)
    report = run_dry_run(loaded, seed=0)

    # Real inference once total across check + dry-run composition.
    assert loaded.real.policy.predict_calls == 1
    assert torch.equal(
        report.guided.proposal,
        report.check.real_action_proposal,
    )
    assert report.check.segment_count == 10
    assert len(report.guided.records) == 10
    assert report.guided.guided_action.shape == (1, 50, 31)
    # guide_segment: 1 from check + 10 from coordinator
    assert len(loaded.guidance.guide_calls) == 11
    assert len(loaded.guidance.verify_calls) == 1
    coordinator_calls = loaded.guidance.guide_calls[1:]
    assert all(call["reference"].shape == (1, 9, 22) for call in coordinator_calls)
    torch.testing.assert_close(
        coordinator_calls[-1]["reference"],
        report.check.real_hand_reference[:, 45:54],
    )


def test_run_dry_run_long_horizon_guides_once_and_executes_fifty_steps():
    from diffusion_policy.guidance.runtime import run_dry_run

    loaded = _build_loaded(
        real=make_real_loaded(
            action_t=50,
            n_obs_steps=1,
            full_prediction_t=68,
        ),
        sim=make_sim_loaded(
            n_obs_steps=4,
            horizon=68,
            n_pred_action_steps=65,
            usable_start=3,
        ),
        config=SimHandGuidanceConfig(
            execution_steps=50,
            guidance_scale=0.0,
            num_inference_steps=2,
            eta=0.0,
        ),
        record=True,
    )

    report = run_dry_run(loaded, seed=4)

    assert loaded.real.policy.predict_calls == 1
    assert loaded.sim.policy.predict_action_calls == 0
    assert report.check.real_action_shape == (1, 50, 31)
    assert report.check.real_hand_reference_shape == (1, 68, 22)
    assert report.check.sim_horizon == 68
    assert report.check.sim_pred_action_steps == 65
    assert report.check.guidance_slice == (3, 68)
    assert report.check.execution_slice == (3, 53)
    assert report.check.segment_count == 1
    assert len(report.guided.records) == 1
    assert report.guided.guided_action.shape == (1, 50, 31)
    coordinator_call = loaded.guidance.guide_calls[-1]
    assert coordinator_call["reference"].shape == (1, 65, 22)
    torch.testing.assert_close(
        coordinator_call["reference"],
        report.check.real_hand_reference[:, :65],
    )


def test_run_dry_run_alternate_segment_count_is_dynamic():
    from diffusion_policy.guidance.runtime import run_dry_run

    loaded = _build_loaded(
        real=make_real_loaded(action_t=15),
        sim=make_sim_loaded(
            n_obs_steps=2,
            horizon=8,
            n_pred_action_steps=7,
            usable_start=1,
            num_train_timesteps=50,
        ),
        config=SimHandGuidanceConfig(
            execution_steps=5,
            guidance_scale=0.0,
            num_inference_steps=5,
            eta=0.0,
        ),
        record=True,
    )
    report = run_dry_run(loaded, seed=2)

    assert report.check.segment_count == 3
    assert len(report.guided.records) == 3
    assert loaded.real.policy.predict_calls == 1
    # 1 check + 3 coordinator
    assert len(loaded.guidance.guide_calls) == 4


def test_run_dry_run_uses_seed_plus_one_generator_for_coordinator():
    from diffusion_policy.guidance.runtime import run_dry_run

    loaded = _build_loaded(
        config=SimHandGuidanceConfig(
            execution_steps=5,
            guidance_scale=0.0,
            num_inference_steps=2,
            eta=0.0,
        ),
        record=True,
    )
    seed = 9
    report = run_dry_run(loaded, seed=seed)

    # First call is check; remaining are coordinator segments.
    coord_gens = [
        call["generator"] for call in loaded.guidance.guide_calls[1:]
    ]
    assert len(coord_gens) == report.check.segment_count
    assert all(g is not None for g in coord_gens)
    # All coordinator segments share one generator object (seed+1).
    assert all(g is coord_gens[0] for g in coord_gens)


def test_runtime_import_has_no_hardware_side_effects():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
import diffusion_policy.guidance.runtime
assert "inference_dp" not in sys.modules
assert "direct_robot_env" not in sys.modules
assert "diffusion_policy.real_world" not in sys.modules
assert "multi_realsense" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=repo_root,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_run_check_cuda_smoke_with_cpu_seeded_generators():
    from diffusion_policy.guidance.runtime import run_check

    loaded = _build_loaded(
        config=SimHandGuidanceConfig(
            execution_steps=5,
            guidance_scale=0.0,
            num_inference_steps=2,
            eta=0.0,
        ),
        record=True,
    )
    loaded.real.policy.to("cuda")
    loaded.sim.policy.to("cuda")
    assert loaded.sim.policy.device.type == "cuda"

    report = run_check(loaded, seed=0)

    assert report.real_action_shape[0] == 1
    assert report.guided_hand_shape == (1, 5, HAND_DIM)
    assert len(report.timesteps) == 2
    assert len(loaded.guidance.guide_calls) == 1
    assert loaded.guidance.guide_calls[0]["history"].device.type == "cuda"
    assert loaded.guidance.guide_calls[0]["reference"].device.type == "cuda"
    assert loaded.guidance.guide_calls[0]["generator"].device.type == "cpu"
