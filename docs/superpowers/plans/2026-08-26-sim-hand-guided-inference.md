# Sim-Hand Guided Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a hardware-independent inference path that obtains one 50-action proposal from the target 64-horizon Bulb policy, refines only action dimensions 9:31 with the Sim-Hand policy in ten closed-loop five-action segments, and proves that custom DDIM is equivalent to Diffusers at zero guidance.

**Architecture:** Add a focused diffusion_policy.guidance package. Workspace loaders and Real/Sim adapters recover independent checkpoints and normalized representations; a custom pinned-DDIM module performs analytic x0 guidance; a coordinator owns full-action composition and fake closed-loop execution. A new root CLI exposes check and dry-run modes without importing the existing inference_dp.py or any robot package.

**Tech Stack:** Python 3.9+, PyTorch, Diffusers 0.11.1, Hydra/OmegaConf, dill, dataclasses, argparse, pytest, and the repository's existing Workspace, policy, normalizer, and ConditionalUnet1D code.

**Spec:** docs/superpowers/specs/2026-08-26-sim-hand-guided-inference-design.md

## Global Constraints

- The target Real checkpoint saved config is horizon=64, n_obs_steps=1, n_action_steps=50; its actual public action output must be finite (1,50,31).
- Real action [0:9] is relative-EE policy representation; [9:31] is the confirmed 22-D absolute hand target. Guidance never reads or modifies non-hand dimensions.
- The restored Sim checkpoint remains n_obs_steps=4, n_pred_action_steps=9, n_action_steps=4, horizon=12, obs_dim=action_dim=22. Do not mutate its public four-action slice.
- Guided execution length is 5 and its Sim trajectory slice is derived as slice(3,8). Real 50 actions form ten non-overlapping segments.
- Real policy inference runs once per check or dry-run. Every segment references the corresponding slice from that original proposal.
- Real and Sim normalizers remain independent. Reference distance is computed only in Sim normalized action space.
- Runtime Diffusers must be exactly 0.11.1. The accepted DDIM contract is epsilon prediction, eta=0.0, thresholding=False, set_alpha_to_one=True, steps_offset=0, and twelve timesteps [88,80,72,64,56,48,40,32,24,16,8,0].
- Official clipping produces x0_base. Analytic guidance updates x0_base[:,3:8] without a second clip and uses raw DDIM variance before eta scaling.
- guidance_scale=0 uses the same custom arithmetic path and must match official DDIM pred_original_sample and prev_sample at every step and through the full chain.
- The low-level sampler accepts scale zero for its oracle. SimHandGuidance check/dry-run configuration requires guidance_scale>0 so the configured chain is genuinely nonzero and separate from that oracle.
- No U-Net backpropagation is used. MSE gradient reduction is per sample over exactly 5*22 elements and never crosses the batch.
- Each segment consumes fresh initial noise. After five returned states, the next condition is the latest four states.
- Coordinator and executor support B=1 only. Partial execution terminates the run and never fabricates a state history.
- Check and dry-run must not import inference_dp.py, DirectRobotEnv, ViTacFormer, or any hardware/controller module.
- DDIM alphas_cumprod must exactly match the restored training DDPM schedule. Raw DDIM variance is verified independently from the official zero-scale oracle and must not be replaced by its square root.
- Remove the hard-coded Sim horizon modulo-four validator. Actual U-Net compatibility remains enforced by the existing no-gradient forward shape probe.
- Do not add a requirement that Real horizon 64, Sim horizon 12, or Sim prediction length 9 be divisible by execution length 5.
- Apply TDD for every production change and commit after each independently reviewable task.

---

## File Map

Create or modify the following focused units:

~~~text
diffusion_policy/common/sim_hand_temporal_util.py
    Remove only the modulo-four heuristic; retain semantic temporal validation.

diffusion_policy/guidance/__init__.py
    Package boundary; no import-time checkpoint, CUDA, or external-resource work.

diffusion_policy/guidance/checkpoint_loader.py
    Restore a Workspace payload, select EMA, preserve normalizer, freeze policy.

diffusion_policy/guidance/real_adapter.py
    Validate the target 64/1/50 Bulb contract and build a synthetic observation.

diffusion_policy/guidance/sim_adapter.py
    Validate the old Sim checkpoint and expose normalize/flatten/model calls.

diffusion_policy/guidance/guided_ddim.py
    Pinned scheduler factory, analytic gradient, custom step, full chain,
    and zero-guidance oracle verification.

diffusion_policy/guidance/sim_hand_guidance.py
    Hand-only five-step API, fresh noise, Sim normalization, and unnormalization.

diffusion_policy/guidance/executor.py
    SegmentExecutor protocol, ExecutionError, deterministic FakeSegmentExecutor.

diffusion_policy/guidance/coordinator.py
    Ten-segment Real slicing, hand replacement, execution, and latest-four state.

diffusion_policy/guidance/runtime.py
    Compose loaded policies and implement check and dry-run services.

inference_sim_hand_guided.py
    Hardware-free argparse entry point.

tests/test_sim_hand_temporal.py
tests/test_diffusion_unet_sim_hand_policy.py
tests/test_sim_hand_guided_checkpoint_loader.py
tests/test_sim_hand_guided_adapters.py
tests/test_sim_hand_guided_ddim.py
tests/test_sim_hand_guidance.py
tests/test_sim_hand_guided_coordinator.py
tests/test_sim_hand_guided_runtime.py
tests/test_inference_sim_hand_guided_cli.py
    Focused unit, regression, and integration coverage.
~~~

## Test Environment

The repository declares Diffusers 0.11.1, while the available IsaacLab
environment currently contains another version. Build a disposable overlay that
reuses its PyTorch installation but pins the oracle version:

~~~bash
uv venv \
  --python /home/carus/miniforge3/envs/isaaclab/bin/python3.11 \
  --system-site-packages \
  /tmp/dex-diffuse-guided-test

uv pip install \
  --python /tmp/dex-diffuse-guided-test/bin/python \
  'diffusers==0.11.1' \
  'huggingface-hub==0.11.1' \
  'accelerate==0.13.2' \
  'dill>=0.3.5,<0.4' \
  'zarr>=2.12,<3' \
  'numcodecs>=0.10,<0.16'

GUIDED_TEST_PYTHON=/tmp/dex-diffuse-guided-test/bin/python
$GUIDED_TEST_PYTHON -c \
  'import diffusers; assert diffusers.__version__ == "0.11.1"'
~~~

All commands below assume they run at the repository root with
GUIDED_TEST_PYTHON set as above.

### Task 1: Replace the Sim modulo heuristic with the real U-Net probe

**Files:**

- Modify: diffusion_policy/common/sim_hand_temporal_util.py:6-8,93-98
- Modify: tests/test_sim_hand_temporal.py:29-62
- Modify: tests/test_diffusion_unet_sim_hand_policy.py
- Modify: docs/superpowers/specs/2026-08-26-sim-hand-diffusion-policy-design.md
- Modify: docs/superpowers/plans/2026-08-26-sim-hand-diffusion-policy.md

**Interfaces:**

- Consumes: validate_sim_hand_temporal_config and the existing DiffusionUnetSimHandPolicy._validate_unet_temporal_shape.
- Produces: semantic temporal validation that accepts a derived horizon regardless of modulo; actual model construction remains the shape authority.

- [ ] **Step 1: Replace the old modulo-failure test with a semantic acceptance test**

In tests/test_sim_hand_temporal.py, remove the parameterized horizon=11
multiple-of-four failure and add:

~~~python
def test_derived_non_multiple_of_four_horizon_is_semantically_valid():
    config = validate_sim_hand_temporal_config(
        n_obs_steps=4,
        n_pred_action_steps=8,
        n_action_steps=3,
        horizon=11,
        obs_dim=22,
        action_dim=22,
    )

    assert config.horizon == 11
    assert config.usable_action_slice == slice(3, 11)
    assert config.execution_action_slice == slice(3, 6)
~~~

Change test_invalid_temporal_error_reports_all_config_values so that it still
uses horizon=11 but sets obs_dim=21; it then exercises error formatting without
depending on modulo behavior.

- [ ] **Step 2: Add a failing actual-network probe test**

In tests/test_diffusion_unet_sim_hand_policy.py, construct the existing small
ConditionalUnet1D with horizon=11 and require the policy-level shape error:

~~~python
def test_actual_unet_probe_rejects_horizon_when_output_length_changes():
    model = ConditionalUnet1D(
        input_dim=HAND_DIM,
        local_cond_dim=None,
        global_cond_dim=4 * HAND_DIM,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64, 128),
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=True,
    )

    with pytest.raises(ValueError, match="U-Net output temporal length"):
        DiffusionUnetSimHandPolicy(
            model=model,
            noise_scheduler=_scheduler(),
            horizon=11,
            obs_dim=HAND_DIM,
            action_dim=HAND_DIM,
            n_obs_steps=4,
            n_pred_action_steps=8,
            n_action_steps=3,
        )
~~~

- [ ] **Step 3: Run the two focused test files and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_temporal.py \
  tests/test_diffusion_unet_sim_hand_policy.py -q
~~~

Expected: the semantic horizon=11 test fails because the utility still raises
the modulo-four error; the policy test fails for the same early reason instead
of reaching the U-Net output-shape message.

- [ ] **Step 4: Remove only the modulo heuristic**

Delete TEMPORAL_DOWNSAMPLE_FACTOR and this branch from
validate_sim_hand_temporal_config:

~~~python
if config.horizon % TEMPORAL_DOWNSAMPLE_FACTOR != 0:
    raise _invalid(
        config,
        "Diffusion horizon must be a multiple of 4 for the current "
        "ConditionalUnet1D temporal down/up-sampling structure.",
    )
~~~

Do not change derived-horizon validation, dimensions, OA convention,
DiffusionUnetSimHandPolicy.compute_loss, n_action_steps, or checkpoint behavior.

- [ ] **Step 5: Update the earlier Sim design and plan**

Replace their modulo-four statements with the exact rule:

~~~text
Temporal utilities validate semantic lengths and derived horizon only.
DiffusionUnetSimHandPolicy performs a no-gradient dummy forward and requires
the configured U-Net output shape to equal (B,horizon,22). It does not pad,
crop, or repair an incompatible trajectory.
~~~

- [ ] **Step 6: Run focused tests and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_temporal.py \
  tests/test_diffusion_unet_sim_hand_policy.py -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/common/sim_hand_temporal_util.py \
  tests/test_sim_hand_temporal.py \
  tests/test_diffusion_unet_sim_hand_policy.py \
  docs/superpowers/specs/2026-08-26-sim-hand-diffusion-policy-design.md \
  docs/superpowers/plans/2026-08-26-sim-hand-diffusion-policy.md
git commit -m "refactor: probe sim-hand temporal shape at runtime"
~~~

### Task 2: Add a hardware-free Workspace checkpoint loader

**Files:**

- Create: diffusion_policy/guidance/__init__.py
- Create: diffusion_policy/guidance/checkpoint_loader.py
- Test: tests/test_sim_hand_guided_checkpoint_loader.py

**Interfaces:**

- Consumes: BaseWorkspace payloads with cfg, state_dicts, and pickles.
- Produces:

~~~python
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
) -> LoadedPolicy
~~~

- [ ] **Step 1: Write failing model/EMA restoration tests**

Use a test FakeWorkspace whose model and ema_model are small nn.Linear modules.
Save a payload with distinct model and EMA weights, then assert:

~~~python
class FakeWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = nn.Linear(1, 1, bias=False)
        self.ema_model = nn.Linear(1, 1, bias=False)
        self.model.n_action_steps = 4
        self.model.num_inference_steps = 100
        self.ema_model.n_action_steps = 4
        self.ema_model.num_inference_steps = 100

    def load_payload(self, payload, exclude_keys=None, include_keys=None):
        for name, state_dict in payload["state_dicts"].items():
            getattr(self, name).load_state_dict(state_dict)


def make_fake_workspace_payload(use_ema, model_weight, ema_weight):
    cfg = OmegaConf.create({
        "_target_": "tests.fake.FakeWorkspace",
        "training": {"use_ema": use_ema},
    })
    model = nn.Linear(1, 1, bias=False)
    ema_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(model_weight)
        ema_model.weight.fill_(ema_weight)
    return {
        "cfg": cfg,
        "state_dicts": {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
        },
        "pickles": {},
    }
~~~

The fake target string is never imported because every test monkeypatches
hydra.utils.get_class before calling the loader.

~~~python
def test_loader_selects_ema_freezes_parameters_and_enters_eval(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.ckpt"
    payload = make_fake_workspace_payload(
        use_ema=True,
        model_weight=1.0,
        ema_weight=2.0,
    )
    torch.save(payload, checkpoint, pickle_module=dill)
    monkeypatch.setattr(hydra.utils, "get_class", lambda _: FakeWorkspace)

    loaded = load_workspace_policy(checkpoint, torch.device("cpu"))

    assert loaded.used_ema is True
    assert loaded.policy is loaded.workspace.ema_model
    torch.testing.assert_close(
        loaded.policy.weight,
        torch.full_like(loaded.policy.weight, 2.0),
    )
    assert loaded.policy.training is False
    assert all(not parameter.requires_grad for parameter in loaded.policy.parameters())
~~~

Add these explicit tests in the same file:

~~~text
test_loader_uses_model_when_ema_is_disabled
test_loader_rejects_missing_checkpoint
test_loader_rejects_payload_without_cfg
test_loader_rejects_workspace_without_model
test_loader_does_not_override_policy_temporal_or_inference_fields
test_guidance_package_import_has_no_hardware_side_effects
~~~

The final import test runs a subprocess importing
diffusion_policy.guidance.checkpoint_loader and rejects any loaded module whose
name is inference_dp, direct_robot_env, starts with diffusion_policy.real_world,
contains vitacformer, or starts with pyrealsense2 or ur_rtde.

Also add an actual Sim Workspace checkpoint round-trip, not only a fake linear
module test. Compose train_diffusion_unet_sim_hand_workspace, set
training.use_ema=True and small model down_dims, construct
TrainDiffusionUnetSimHandWorkspace, then install distinct non-identity obs and
action statistics before saving synchronously:

~~~python
def make_small_sim_workspace_cfg():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = (
        Path(__file__).resolve().parents[1]
        / "diffusion_policy"
        / "config"
    )
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        cfg = compose(
            config_name="train_diffusion_unet_sim_hand_workspace"
        )
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.training.use_ema = True
        cfg.policy.model.down_dims = [32, 64, 128]
        cfg.policy.model.diffusion_step_embed_dim = 32
        cfg.policy.model.kernel_size = 3
    return cfg


cfg = make_small_sim_workspace_cfg()
workspace = TrainDiffusionUnetSimHandWorkspace(
    cfg,
    output_dir=str(tmp_path / "workspace"),
)
normalizer = LinearNormalizer()
normalizer.fit(
    {
        "obs": torch.stack((
            torch.full((22,), -2.0),
            torch.full((22,), 4.0),
        )),
        "action": torch.stack((
            torch.arange(22, dtype=torch.float32),
            torch.arange(22, dtype=torch.float32) + 20.0,
        )),
    },
    last_n_dims=1,
    mode="limits",
)
workspace.model.set_normalizer(normalizer)
workspace.ema_model.set_normalizer(normalizer)
checkpoint = workspace.save_checkpoint(
    path=tmp_path / "sim-normalizer.ckpt",
    use_thread=False,
)
expected = workspace.ema_model.normalizer.state_dict()

loaded = load_workspace_policy(Path(checkpoint), torch.device("cpu"))
actual = loaded.policy.normalizer.state_dict()

assert actual.keys() == expected.keys()
for key in expected:
    torch.testing.assert_close(actual[key], expected[key])
    assert actual[key].device == next(loaded.policy.parameters()).device

known_action = torch.linspace(1.0, 18.0, 22).reshape(1, 22)
restored_action_normalizer = loaded.policy.normalizer["action"]
torch.testing.assert_close(
    restored_action_normalizer.unnormalize(
        restored_action_normalizer.normalize(known_action)
    ),
    known_action,
    rtol=1e-5,
    atol=1e-6,
)
~~~

No dataset or workspace.run call is used.

- [ ] **Step 2: Run the loader test and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_checkpoint_loader.py -q
~~~

Expected: FAIL because diffusion_policy.guidance and load_workspace_policy do
not exist.

- [ ] **Step 3: Implement the minimal loader**

Use this loading sequence in checkpoint_loader.py:

~~~python
def load_workspace_policy(checkpoint_path, device):
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
    used_ema = policy is ema_model

    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    return LoadedPolicy(
        checkpoint_path=checkpoint_path,
        cfg=cfg,
        workspace=workspace,
        policy=policy,
        used_ema=used_ema,
    )
~~~

The package __init__.py contains no imports beyond stable public dataclasses and
functions; it must not load a checkpoint or initialize a device.

- [ ] **Step 4: Run loader tests and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_checkpoint_loader.py -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/__init__.py \
  diffusion_policy/guidance/checkpoint_loader.py \
  tests/test_sim_hand_guided_checkpoint_loader.py
git commit -m "feat: load guided inference checkpoints"
~~~

### Task 3: Add exact Real and Sim policy adapters

**Files:**

- Create: diffusion_policy/guidance/real_adapter.py
- Create: diffusion_policy/guidance/sim_adapter.py
- Test: tests/test_sim_hand_guided_adapters.py

**Interfaces:**

- Consumes: LoadedPolicy from Task 2.
- Produces:

~~~python
class RealPolicyAdapter:
    __init__(loaded: LoadedPolicy) -> None
    validate_target_contract() -> None
    build_synthetic_observation() -> dict[str, torch.Tensor]
    predict_proposal() -> torch.Tensor

class SimHandModelAdapter:
    __init__(loaded: LoadedPolicy) -> None
    validate_checkpoint_contract() -> None
    normalize_history(history: torch.Tensor) -> torch.Tensor
    normalize_reference(reference: torch.Tensor) -> torch.Tensor
    unnormalize_action(action: torch.Tensor) -> torch.Tensor
    global_condition(history: torch.Tensor) -> torch.Tensor
    predict_epsilon(
        self,
        sample: torch.Tensor,
        timestep: int | torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor
~~~

RealPolicyAdapter.predict_proposal returns exactly finite (1,50,31) in the
unnormalized mixed policy representation. SimHandModelAdapter never calls
policy.predict_action.

- [ ] **Step 1: Write failing target Real contract tests**

Define concrete fake loaded-policy helpers at the top of the test file:

~~~python
class FakeRealPolicy(nn.Module):
    def __init__(self, proposal):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_dim = 31
        self.proposal = proposal
        self.normalizer = object()

    def predict_action(self, observation):
        return {"action": self.proposal.clone()}


def make_loaded_real_policy(horizon, n_obs_steps, n_action_steps):
    cfg = OmegaConf.create({
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": n_action_steps,
        "shape_meta": {
            "obs": {
                "front_image": {"shape": [3, 240, 320], "type": "rgb"},
                "wrist_image": {"shape": [3, 240, 320], "type": "rgb"},
                "ee_pose": {"shape": [9], "type": "low_dim"},
                "hand_joint": {"shape": [22], "type": "low_dim"},
            },
            "action": {"shape": [31]},
        },
    })
    policy = FakeRealPolicy(torch.zeros(1, n_action_steps, 31))
    return LoadedPolicy(
        checkpoint_path=Path("fake-real.ckpt"),
        cfg=cfg,
        workspace=SimpleNamespace(model=policy),
        policy=policy,
        used_ema=False,
    )
~~~

Then add:
~~~python
def test_real_adapter_accepts_only_target_64_1_50_contract():
    adapter = RealPolicyAdapter(
        make_loaded_real_policy(horizon=64, n_obs_steps=1, n_action_steps=50)
    )
    adapter.validate_target_contract()

    with pytest.raises(ValueError, match="horizon.*64"):
        RealPolicyAdapter(
            make_loaded_real_policy(horizon=16, n_obs_steps=2, n_action_steps=8)
        ).validate_target_contract()
~~~

Add tests that build_synthetic_observation returns exactly:

~~~python
{
    "front_image": torch.zeros(1, 1, 3, 240, 320),
    "wrist_image": torch.zeros(1, 1, 3, 240, 320),
    "ee_pose": torch.zeros(1, 1, 9),
    "hand_joint": torch.zeros(1, 1, 22),
}
~~~

Also require predict_proposal to reject a missing action key, wrong
(1,50,31) shape, NaN/Inf, and any policy action_dim other than 31.

- [ ] **Step 2: Write failing old-Sim-checkpoint adapter tests**

Define a Sim helper that uses identity normalizers and fails if public
predict_action is touched:

~~~python
def make_training_scheduler():
    return DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )


class RecordingEpsilonModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, sample, timestep, global_cond=None):
        self.calls.append((sample.clone(), global_cond.clone()))
        return torch.zeros_like(sample)


class FakeSimPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RecordingEpsilonModel()
        self.noise_scheduler = make_training_scheduler()
        self.normalizer = LinearNormalizer()
        self.normalizer["obs"] = SingleFieldLinearNormalizer.create_identity()
        self.normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
        self.n_obs_steps = 4
        self.n_pred_action_steps = 9
        self.n_action_steps = 4
        self.horizon = 12
        self.obs_dim = 22
        self.action_dim = 22
        self.temporal = SimpleNamespace(
            execution_action_slice=slice(3, 7),
        )

    def predict_action(self, observation):
        raise AssertionError("Sim public predict_action must not be called")


def make_loaded_sim_policy(**overrides):
    policy = FakeSimPolicy()
    for name, value in overrides.items():
        setattr(policy, name, value)
    cfg = OmegaConf.create({"training": {"use_ema": False}})
    return LoadedPolicy(
        checkpoint_path=Path("fake-sim.ckpt"),
        cfg=cfg,
        workspace=SimpleNamespace(model=policy),
        policy=policy,
        used_ema=False,
    )
~~~

Then add:

~~~python
def test_sim_adapter_preserves_old_public_four_step_contract():
    loaded = make_loaded_sim_policy(
        n_obs_steps=4,
        n_pred_action_steps=9,
        n_action_steps=4,
        horizon=12,
        obs_dim=22,
        action_dim=22,
    )
    adapter = SimHandModelAdapter(loaded)
    adapter.validate_checkpoint_contract()

    assert loaded.policy.n_action_steps == 4
    assert loaded.policy.temporal.execution_action_slice == slice(3, 7)
~~~

Add these explicit cases:

~~~text
test_sim_adapter_requires_obs_and_action_normalizers
test_global_condition_is_normalized_first_four_states_flattened_to_1_88
test_reference_round_trip_uses_only_sim_action_normalizer
test_predict_epsilon_uses_full_1_12_22_model_input
test_predict_epsilon_never_calls_public_predict_action
test_sim_adapter_rejects_wrong_temporal_dimension_or_scheduler_contract
test_real_and_sim_normalizers_remain_independent
~~~

The scheduler contract test requires num_train_timesteps=100,
prediction_type=epsilon, thresholding=False, and action clip configuration from
the restored Sim scheduler.

- [ ] **Step 3: Run adapter tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_adapters.py -q
~~~

Expected: FAIL because both adapters are missing.

- [ ] **Step 4: Implement shared tensor validation locally in each adapter**

Both adapter constructors store loaded, loaded.cfg, and loaded.policy as
self.loaded, self.cfg, and self.policy. They do not copy either normalizer.

Use explicit expected shapes and no coercion:

~~~python
def _require_finite_tensor(name, value, expected_shape):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} expected shape {tuple(expected_shape)}, "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
~~~

Do not create a broad generic validation framework.

- [ ] **Step 5: Implement RealPolicyAdapter**

validate_target_contract checks saved cfg horizon=64, n_obs_steps=1,
n_action_steps=50, policy.action_dim=31, policy.normalizer presence, and the
exact Bulb observation schema
front_image, wrist_image, ee_pose, hand_joint in cfg.shape_meta.

build_synthetic_observation allocates zeros on the policy device and dtype using
cfg.shape_meta shapes. predict_proposal executes only:

~~~python
with torch.inference_mode():
    prediction = self.policy.predict_action(
        self.build_synthetic_observation()
    )
proposal = prediction["action"]
_require_finite_tensor("Real prediction['action']", proposal, (1, 50, 31))
parameter = next(self.policy.parameters())
if proposal.device != parameter.device:
    raise ValueError("Real proposal device differs from Real policy")
if proposal.dtype != parameter.dtype:
    raise ValueError("Real proposal dtype differs from Real policy")
return proposal
~~~

Do not import mixed_actions_to_absolute; [0:9] remains relative and [9:31]
already represents absolute hand targets.

- [ ] **Step 6: Implement SimHandModelAdapter**

validate_checkpoint_contract checks:

~~~python
expected = {
    "n_obs_steps": 4,
    "n_pred_action_steps": 9,
    "horizon": 12,
    "obs_dim": 22,
    "action_dim": 22,
}
~~~

It accepts the restored n_action_steps=4 and never changes it. It checks both
normalizer fields by indexing policy.normalizer["obs"] and
policy.normalizer["action"].

Implement condition/model access as:

~~~python
def global_condition(self, history):
    normalized = self.normalize_history(history)
    return normalized[:, :4, :].reshape(1, 88)

def predict_epsilon(self, sample, timestep, global_cond):
    _require_finite_tensor("sample", sample, (1, 12, 22))
    _require_finite_tensor("global_cond", global_cond, (1, 88))
    with torch.inference_mode():
        output = self.policy.model(
            sample,
            timestep,
            global_cond=global_cond,
        )
    _require_finite_tensor("epsilon prediction", output, (1, 12, 22))
    return output
~~~

normalize_history expects (1,4,22); normalize_reference and
unnormalize_action expect (1,5,22). Validate shape, device, dtype, and finiteness
before and after every normalizer call.

- [ ] **Step 7: Run adapter tests and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_adapters.py -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/real_adapter.py \
  diffusion_policy/guidance/sim_adapter.py \
  tests/test_sim_hand_guided_adapters.py
git commit -m "feat: add real and sim guidance adapters"
~~~

### Task 4: Add the pinned DDIM factory and analytic MSE gradient

**Files:**

- Create: diffusion_policy/guidance/guided_ddim.py
- Test: tests/test_sim_hand_guided_ddim.py

**Interfaces:**

- Consumes: a restored Sim DDPMScheduler config.
- Produces:

~~~python
EXPECTED_DIFFUSERS_VERSION = "0.11.1"
EXPECTED_TIMESTEPS = (88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)

assert_pinned_diffusers_version() -> None

create_compatible_ddim_scheduler(
    training_scheduler: DDPMScheduler,
) -> DDIMScheduler

mse_guidance_gradient(
    x0_base: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
) -> torch.Tensor
~~~

- [ ] **Step 1: Write failing scheduler factory tests**

Add this exact fixture and test:

~~~python
def make_training_scheduler(**overrides):
    values = {
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "variance_type": "fixed_small",
        "clip_sample": True,
        "prediction_type": "epsilon",
    }
    values.update(overrides)
    return DDPMScheduler(**values)
~~~

~~~python
def test_factory_preserves_training_schedule_and_pins_ddim_fields():
    training = make_training_scheduler()
    scheduler = create_compatible_ddim_scheduler(training)
    scheduler.set_timesteps(12)

    assert scheduler.config.num_train_timesteps == 100
    assert scheduler.config.beta_schedule == "squaredcos_cap_v2"
    assert scheduler.config.clip_sample is True
    assert scheduler.config.prediction_type == "epsilon"
    assert scheduler.config.set_alpha_to_one is True
    assert scheduler.config.steps_offset == 0
    assert tuple(int(t) for t in scheduler.timesteps) == EXPECTED_TIMESTEPS
    torch.testing.assert_close(
        scheduler.alphas_cumprod,
        training.alphas_cumprod,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        scheduler.final_alpha_cumprod,
        torch.tensor(1.0, dtype=scheduler.final_alpha_cumprod.dtype),
    )
~~~

Add failure tests for a monkeypatched Diffusers version, prediction_type other
than epsilon, thresholding=True, and a training scheduler whose
num_train_timesteps is not 100.

- [ ] **Step 2: Write failing per-sample gradient tests**

Add:

~~~python
def test_mse_gradient_is_slice_isolated_and_does_not_reduce_batch():
    x0_base = torch.zeros(2, 12, 22)
    reference = torch.zeros(2, 5, 22)
    reference[0] = 1.0

    gradient = mse_guidance_gradient(
        x0_base,
        reference,
        slice(3, 8),
    )

    expected = torch.zeros_like(x0_base)
    expected[0, 3:8] = -2.0 / 110.0
    torch.testing.assert_close(gradient, expected)
    assert torch.count_nonzero(gradient[1]) == 0
~~~

Also reject non-3-D x0, an invalid slice, a reference other than (B,5,22), a
different device/dtype, and non-finite inputs.

- [ ] **Step 3: Run focused tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_ddim.py \
  -k "factory or gradient or pinned" -q
~~~

Expected: FAIL because guided_ddim.py is missing.

- [ ] **Step 4: Implement the version gate and scheduler factory**

Use:

~~~python
def assert_pinned_diffusers_version():
    if diffusers.__version__ != EXPECTED_DIFFUSERS_VERSION:
        raise RuntimeError(
            "Guided DDIM requires diffusers=="
            f"{EXPECTED_DIFFUSERS_VERSION}, got {diffusers.__version__}"
        )

def create_compatible_ddim_scheduler(training_scheduler):
    assert_pinned_diffusers_version()
    config = training_scheduler.config
    if config.num_train_timesteps != 100:
        raise ValueError("Sim DDPM num_train_timesteps must be 100")
    if config.prediction_type != "epsilon":
        raise ValueError("Sim DDPM prediction_type must be epsilon")
    if bool(getattr(config, "thresholding", False)):
        raise ValueError("Dynamic thresholding is not supported")

    return DDIMScheduler.from_config(
        config,
        set_alpha_to_one=True,
        steps_offset=0,
    )
~~~

- [ ] **Step 5: Implement the analytic gradient**

Use:

~~~python
gradient = torch.zeros_like(x0_base)
difference = x0_base[:, guidance_slice, :] - reference
gradient[:, guidance_slice, :] = (
    2.0 / (reference.shape[1] * reference.shape[2])
) * difference
return gradient
~~~

Run all shape/device/dtype/finite checks before allocating the result.

- [ ] **Step 6: Run focused tests and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_ddim.py \
  -k "factory or gradient or pinned" -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/guided_ddim.py \
  tests/test_sim_hand_guided_ddim.py
git commit -m "feat: add pinned sim-hand DDIM primitives"
~~~

### Task 5: Implement one custom guided DDIM reverse step

**Files:**

- Modify: diffusion_policy/guidance/guided_ddim.py
- Modify: tests/test_sim_hand_guided_ddim.py

**Interfaces:**

- Consumes: factory and mse_guidance_gradient from Task 4.
- Produces:

~~~python
@dataclass(frozen=True)
class GuidedDDIMStepOutput:
    timestep: int
    prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    base_pred_original_sample: torch.Tensor
    raw_pred_original_sample: torch.Tensor
    alpha_bar_t: torch.Tensor
    alpha_bar_prev: torch.Tensor
    raw_variance: torch.Tensor
    direction_coefficient: torch.Tensor
    guidance_loss_before: torch.Tensor
    guidance_loss_after: torch.Tensor
    guidance_gradient_norm: torch.Tensor
    guided_out_of_range_ratio: torch.Tensor

def guided_ddim_step(
    scheduler: DDIMScheduler,
    model_output: torch.Tensor,
    timestep: int | torch.Tensor,
    sample: torch.Tensor,
    reference: torch.Tensor,
    guidance_scale: float,
    guidance_slice: slice,
    eta: float = 0.0,
) -> GuidedDDIMStepOutput
~~~

pred_original_sample is x0_guided, the clean estimate used by the reverse
update. At scale zero it equals official pred_original_sample.
base_pred_original_sample is x0_base; raw_pred_original_sample is x0_raw.

Use these exact test helpers in tests/test_sim_hand_guided_ddim.py:

~~~python
def make_ddim_scheduler():
    return create_compatible_ddim_scheduler(make_training_scheduler())


def run_controlled_guided_step(timestep, guidance_scale):
    scheduler = make_ddim_scheduler()
    scheduler.set_timesteps(12)
    sample = torch.zeros(1, 12, 22)
    epsilon = torch.zeros_like(sample)
    reference = torch.ones(1, 5, 22)
    return guided_ddim_step(
        scheduler=scheduler,
        model_output=epsilon,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=guidance_scale,
        guidance_slice=slice(3, 8),
        eta=0.0,
    )


def expected_ddim_terms(timestep):
    scheduler = make_ddim_scheduler()
    scheduler.set_timesteps(12)
    timestep = int(timestep)
    prev_timestep = timestep - (
        scheduler.config.num_train_timesteps
        // scheduler.num_inference_steps
    )
    alpha_t = scheduler.alphas_cumprod[timestep]
    alpha_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )
    variance = (
        (1.0 - alpha_prev)
        / (1.0 - alpha_t)
        * (1.0 - alpha_t / alpha_prev)
    )
    return alpha_t, alpha_prev, variance
~~~

- [ ] **Step 1: Write the twelve-timestep zero-guidance oracle test**

Use a fixture that forces clipping:

~~~python
@pytest.mark.parametrize("timestep", EXPECTED_TIMESTEPS)
def test_zero_scale_step_matches_official_with_clipping(timestep):
    official_scheduler = make_ddim_scheduler()
    custom_scheduler = make_ddim_scheduler()
    official_scheduler.set_timesteps(12)
    custom_scheduler.set_timesteps(12)

    sample = torch.full((2, 12, 22), 10.0)
    epsilon = torch.zeros_like(sample)
    reference = torch.zeros(2, 5, 22)

    official = official_scheduler.step(
        model_output=epsilon,
        timestep=timestep,
        sample=sample,
        eta=0.0,
    )
    custom = guided_ddim_step(
        scheduler=custom_scheduler,
        model_output=epsilon,
        timestep=timestep,
        sample=sample,
        reference=reference,
        guidance_scale=0.0,
        guidance_slice=slice(3, 8),
        eta=0.0,
    )

    assert custom.timestep == int(timestep)
    assert torch.any(custom.raw_pred_original_sample.abs() > 1.0)
    torch.testing.assert_close(
        custom.pred_original_sample,
        official.pred_original_sample,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        custom.prev_sample,
        official.prev_sample,
        rtol=1e-5,
        atol=1e-6,
    )
~~~

Monkeypatch custom_scheduler.step to raise if called; the custom function must
still pass, proving scale zero is not an official-step bypass.

- [ ] **Step 2: Write failing nonzero-guidance semantic tests**

Add:

~~~python
def test_nonzero_guidance_directly_changes_only_slice_3_8():
    result = run_controlled_guided_step(timestep=88, guidance_scale=1.0)

    torch.testing.assert_close(
        result.pred_original_sample[:, :3],
        result.base_pred_original_sample[:, :3],
    )
    torch.testing.assert_close(
        result.pred_original_sample[:, 8:],
        result.base_pred_original_sample[:, 8:],
    )
    assert not torch.equal(
        result.pred_original_sample[:, 3:8],
        result.base_pred_original_sample[:, 3:8],
    )

def test_final_step_has_zero_raw_variance_and_direct_guidance_is_noop():
    result = run_controlled_guided_step(timestep=0, guidance_scale=1.0)

    torch.testing.assert_close(
        result.raw_variance,
        torch.zeros_like(result.raw_variance),
    )
    torch.testing.assert_close(
        result.pred_original_sample,
        result.base_pred_original_sample,
    )

@pytest.mark.parametrize("timestep", EXPECTED_TIMESTEPS)
def test_step_reports_raw_ddim_variance_not_standard_deviation(timestep):
    result = run_controlled_guided_step(
        timestep=timestep,
        guidance_scale=0.0,
    )

    alpha_t, alpha_prev, expected_variance = expected_ddim_terms(timestep)
    torch.testing.assert_close(result.alpha_bar_t, alpha_t)
    torch.testing.assert_close(result.alpha_bar_prev, alpha_prev)
    torch.testing.assert_close(result.raw_variance, expected_variance)

def test_guided_x0_multiplies_gradient_by_raw_variance():
    scale = 1.0
    result = run_controlled_guided_step(
        timestep=88,
        guidance_scale=scale,
    )
    _, _, expected_variance = expected_ddim_terms(88)
    reference = torch.ones(1, 5, 22)
    expected_gradient = mse_guidance_gradient(
        result.base_pred_original_sample,
        reference,
        slice(3, 8),
    )
    expected_guided = (
        result.base_pred_original_sample
        - scale * expected_variance * expected_gradient
    )
    torch.testing.assert_close(
        result.pred_original_sample,
        expected_guided,
    )
~~~

Add a no-second-clipping test whose guidance deliberately moves one element
outside [-1,1], and assert guided_out_of_range_ratio is positive.
Add test_zero_distance_has_zero_gradient_and_unchanged_guided_x0 with
reference equal to x0_base[:,3:8].

For the loss-decrease test, derive a safe scale from the fixture:

~~~python
probe = run_controlled_guided_step(timestep=88, guidance_scale=0.0)
raw_variance = probe.raw_variance
scale = 55.0 / raw_variance.item()
result = run_controlled_guided_step(timestep=88, guidance_scale=scale)
assert 0.0 < scale * raw_variance.item() < 110.0
assert result.guidance_loss_after < result.guidance_loss_before
~~~

Also require eta!=0, negative guidance scale, wrong prediction type, wrong
shapes, and non-finite inputs to raise descriptive errors.

- [ ] **Step 3: Run the step tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_ddim.py \
  -k "step or guidance or variance or clipping" -q
~~~

Expected: FAIL because GuidedDDIMStepOutput and guided_ddim_step do not exist.

- [ ] **Step 4: Implement arithmetic in official 0.11.1 order**

Implement:

~~~python
timestep_int = int(timestep)
prev_timestep = timestep_int - (
    scheduler.config.num_train_timesteps // scheduler.num_inference_steps
)
alpha_t = scheduler.alphas_cumprod[timestep_int]
alpha_prev = (
    scheduler.alphas_cumprod[prev_timestep]
    if prev_timestep >= 0
    else scheduler.final_alpha_cumprod
)
beta_t = 1.0 - alpha_t

x0_raw = (
    sample - beta_t.sqrt() * model_output
) / alpha_t.sqrt()
x0_base = (
    x0_raw.clamp(-1.0, 1.0)
    if scheduler.config.clip_sample
    else x0_raw
)

raw_variance = scheduler._get_variance(timestep_int, prev_timestep)
gradient = mse_guidance_gradient(x0_base, reference, guidance_slice)
x0_guided = x0_base - guidance_scale * raw_variance * gradient

direction_coefficient = (1.0 - alpha_prev).sqrt()
prev_sample = (
    alpha_prev.sqrt() * x0_guided
    + direction_coefficient * model_output
)

loss_before = (
    (x0_base[:, guidance_slice, :] - reference).square()
    .mean(dim=(1, 2))
)
loss_after = (
    (x0_guided[:, guidance_slice, :] - reference).square()
    .mean(dim=(1, 2))
)
return GuidedDDIMStepOutput(
    timestep=timestep_int,
    prev_sample=prev_sample,
    pred_original_sample=x0_guided,
    base_pred_original_sample=x0_base,
    raw_pred_original_sample=x0_raw,
    alpha_bar_t=alpha_t,
    alpha_bar_prev=alpha_prev,
    raw_variance=raw_variance,
    direction_coefficient=direction_coefficient,
    guidance_loss_before=loss_before,
    guidance_loss_after=loss_after,
    guidance_gradient_norm=gradient.flatten(1).norm(dim=1),
    guided_out_of_range_ratio=(
        (x0_guided.abs() > 1.0).to(x0_guided.dtype).mean(dim=(1, 2))
    ),
)
~~~

Do not reconstruct epsilon from clipped or guided x0. Do not clip x0_guided a
second time. Compute per-sample loss before/after on guidance_slice and finite
diagnostics before returning. The test also compares the analytic gradient to
an autograd-only reference:

~~~python
x0_for_grad = x0_base.detach().clone().requires_grad_(True)
per_sample_loss = (
    (x0_for_grad[:, 3:8] - reference).square().mean(dim=(1, 2))
)
autograd_gradient = torch.autograd.grad(
    per_sample_loss.sum(),
    x0_for_grad,
)[0]
torch.testing.assert_close(analytic_gradient, autograd_gradient)
~~~

- [ ] **Step 5: Run the full DDIM test file and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_ddim.py -q
~~~

Expected: all tests written through Task 5 PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/guided_ddim.py \
  tests/test_sim_hand_guided_ddim.py
git commit -m "feat: add zero-equivalent guided DDIM step"
~~~

### Task 6: Add full-chain sampling and the zero-guidance regression report

**Files:**

- Modify: diffusion_policy/guidance/guided_ddim.py
- Modify: tests/test_sim_hand_guided_ddim.py

**Interfaces:**

- Consumes: guided_ddim_step.
- Produces:

~~~python
EpsilonModel = Callable[
    [torch.Tensor, int | torch.Tensor, torch.Tensor],
    torch.Tensor,
]

@dataclass(frozen=True)
class GuidedDDIMSampleOutput:
    trajectory: torch.Tensor
    steps: tuple[GuidedDDIMStepOutput, ...]

@dataclass(frozen=True)
class ZeroGuidanceReport:
    timesteps: tuple[int, ...]
    max_x0_error: float
    max_prev_error: float
    terminal_shape: tuple[int, ...]

sample_guided_trajectory(
    model: EpsilonModel,
    scheduler: DDIMScheduler,
    initial_noise: torch.Tensor,
    global_cond: torch.Tensor,
    reference: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_slice: slice,
    eta: float = 0.0,
) -> GuidedDDIMSampleOutput

verify_zero_guidance_equivalence(
    model: EpsilonModel,
    training_scheduler: DDPMScheduler,
    initial_noise: torch.Tensor,
    global_cond: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> ZeroGuidanceReport
~~~

- [ ] **Step 1: Write a failing independent full-chain oracle test**

Use a deterministic model:

~~~python
class DeterministicEpsilonModel:
    def __call__(self, sample, timestep, global_cond):
        bias = global_cond[:, :1].reshape(-1, 1, 1)
        return 0.15 * sample + bias
~~~

Run official and custom chains with separate scheduler instances and evolving
samples. For every step compare official pred_original_sample with custom
pred_original_sample, then official prev_sample with custom prev_sample. Finally
compare terminal trajectories. Use rtol=1e-5 and atol=1e-6.

The test must call the deterministic model independently on official_sample and
custom_sample; it must not feed one shared epsilon tensor to both chains after
the first step.

- [ ] **Step 2: Add failure and report tests**

Add:

~~~text
test_sample_chain_returns_twelve_diagnostic_steps
test_sample_chain_rejects_eta_other_than_zero
test_zero_guidance_report_contains_expected_timestep_sequence
test_zero_guidance_verifier_reports_first_mismatching_timestep
test_full_chain_does_not_create_an_autograd_graph
~~~

For test_sample_chain_returns_twelve_diagnostic_steps, assert the executed
sequence comes from the outputs themselves:

~~~python
assert tuple(step.timestep for step in result.steps) == EXPECTED_TIMESTEPS
~~~

For the mismatch test, wrap the custom model so that exactly the sixth custom
call adds 1e-2 to epsilon and assert the error names timestep 48.

- [ ] **Step 3: Run full-chain tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_ddim.py \
  -k "chain or report or verifier or autograd" -q
~~~

Expected: FAIL because the chain and report interfaces are missing.

- [ ] **Step 4: Implement the custom chain**

Use:

~~~python
scheduler.set_timesteps(
    num_inference_steps,
    device=initial_noise.device,
)
trajectory = initial_noise.clone()
outputs = []

with torch.no_grad():
    for timestep in scheduler.timesteps:
        model_output = model(
            trajectory,
            timestep,
            global_cond=global_cond,
        )
        output = guided_ddim_step(
            scheduler=scheduler,
            model_output=model_output,
            timestep=timestep,
            sample=trajectory,
            reference=reference,
            guidance_scale=guidance_scale,
            guidance_slice=guidance_slice,
            eta=eta,
        )
        trajectory = output.prev_sample
        outputs.append(output)

return GuidedDDIMSampleOutput(
    trajectory=trajectory,
    steps=tuple(outputs),
)
~~~

There is no condition mask because the Sim policy is global-condition-only.

- [ ] **Step 5: Implement the runtime oracle verifier**

The verifier creates independent official/custom schedulers from the same DDPM
config. It performs:

~~~text
Phase A: for every timestep, compare one custom scale-zero step with official
         step using the same fixed sample and model output.

Phase B: start both chains from cloned initial noise, call the model separately
         on each evolving sample, and compare x0 and prev_sample after every
         reverse step.
~~~

Use torch.testing.assert_close inside a try/except that raises AssertionError
with phase, timestep, max absolute x0 error, and max absolute prev error. Return
the maximum errors and terminal shape on success.

- [ ] **Step 6: Run the full DDIM suite and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_ddim.py -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/guided_ddim.py \
  tests/test_sim_hand_guided_ddim.py
git commit -m "feat: verify full guided DDIM chains"
~~~

### Task 7: Build the hand-only Sim guidance service

**Files:**

- Create: diffusion_policy/guidance/sim_hand_guidance.py
- Test: tests/test_sim_hand_guidance.py

**Interfaces:**

- Consumes: SimHandModelAdapter, DDIM factory and full-chain sampler.
- Produces:

~~~python
NoiseFactory = Callable[
    [tuple[int, ...], torch.device, torch.dtype, torch.Generator | None],
    torch.Tensor,
]

@dataclass(frozen=True)
class SimHandGuidanceConfig:
    execution_steps: int = 5
    oa_start: int = 3
    num_inference_steps: int = 12
    guidance_scale: float = 1.0
    eta: float = 0.0

    @property
    def guidance_slice(self) -> slice:
        return slice(self.oa_start, self.oa_start + self.execution_steps)

@dataclass(frozen=True)
class GuidedSegmentOutput:
    hand_action: torch.Tensor
    terminal_shape: tuple[int, ...]
    timesteps: tuple[int, ...]

class SimHandGuidance:
    __init__(
        adapter: SimHandModelAdapter,
        config: SimHandGuidanceConfig,
        noise_factory: NoiseFactory | None = None,
    ) -> None

    def guide_segment(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor

    def guide_segment_detailed(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> GuidedSegmentOutput

    def verify_zero_guidance(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
    ) -> ZeroGuidanceReport
~~~

guide_segment is the stable hand-only API and returns physical Sim-unnormalized
hand action with shape (1,5,22). It delegates to guide_segment_detailed and
returns only hand_action. The detailed form executes the same single chain and
additionally reports its terminal (1,12,22) shape and twelve timesteps for check
mode; it never samples twice. The class accepts an injectable noise_factory in
its constructor for deterministic tests; the default uses torch.randn.

- [ ] **Step 1: Write failing normalization and slice tests**

Define the recording adapter and sampler stub in the test file:

~~~python
class RecordingSimAdapter:
    def __init__(self):
        self.policy = SimpleNamespace(
            n_obs_steps=4,
            n_pred_action_steps=9,
            n_action_steps=4,
            horizon=12,
            temporal=SimpleNamespace(
                execution_action_slice=slice(3, 7),
            ),
            noise_scheduler=make_training_scheduler(),
        )
        self.normalized_history_calls = 0
        self.normalized_reference_calls = 0
        self.unnormalize_calls = 0
        self.last_unnormalized_input = None

    def normalize_history(self, history):
        self.normalized_history_calls += 1
        return history + 0.1

    def normalize_reference(self, reference):
        self.normalized_reference_calls += 1
        return reference + 0.2

    def global_condition(self, history):
        return self.normalize_history(history).reshape(1, 88)

    def predict_epsilon(self, sample, timestep, global_cond):
        return torch.zeros_like(sample)

    def unnormalize_action(self, action):
        self.unnormalize_calls += 1
        self.last_unnormalized_input = action.clone()
        return action + 10.0


def recording_noise_factory(shape, device, dtype, generator):
    return torch.zeros(shape, device=device, dtype=dtype)
~~~

Add this install helper for sim_hand_guidance.sample_guided_trajectory:

~~~python
def install_fake_sampler(monkeypatch):
    captured = {}

    def fake_sample_guided_trajectory(**kwargs):
        captured.update(kwargs)
        timeline = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1)
        trajectory = timeline.expand(1, 12, 22).clone()
        return GuidedDDIMSampleOutput(
            trajectory=trajectory,
            steps=tuple(
                SimpleNamespace(timestep=timestep)
                for timestep in EXPECTED_TIMESTEPS
            ),
        )

    monkeypatch.setattr(
        sim_hand_guidance,
        "sample_guided_trajectory",
        fake_sample_guided_trajectory,
    )
    return captured
~~~

Then add:

~~~python
def test_guide_segment_normalizes_both_inputs_and_returns_physical_1_5_22(monkeypatch):
    captured = install_fake_sampler(monkeypatch)
    adapter = RecordingSimAdapter()
    guidance = SimHandGuidance(
        adapter,
        SimHandGuidanceConfig(guidance_scale=0.5),
        noise_factory=recording_noise_factory,
    )

    output = guidance.guide_segment(
        torch.zeros(1, 4, 22),
        torch.ones(1, 5, 22),
    )

    assert output.shape == (1, 5, 22)
    assert adapter.normalized_history_calls == 1
    assert adapter.normalized_reference_calls == 1
    assert adapter.unnormalize_calls == 1
    assert captured["guidance_slice"] == slice(3, 8)
    torch.testing.assert_close(
        adapter.last_unnormalized_input,
        captured["initial_noise"].new_tensor(
            [3, 4, 5, 6, 7]
        ).reshape(1, 5, 1).expand(1, 5, 22),
    )
~~~

Require terminal trajectory [3:8], not public policy.action, to be passed to
Sim action unnormalization.

Add test_guide_segment_detailed_reports_the_actual_nonzero_full_chain. It calls
the detailed method and asserts hand_action.shape==(1,5,22),
terminal_shape==(1,12,22), timesteps==EXPECTED_TIMESTEPS, and exactly one
sampler call. Make the fake sampler return eleven steps once and a NaN terminal
once; both must raise instead of returning diagnostics. Also return twelve
steps with one duplicated/reordered timestep and require a descriptive error.

- [ ] **Step 2: Write failing fresh-noise and legacy-checkpoint tests**

Add:

~~~text
test_each_guide_segment_call_requests_a_fresh_1_12_22_noise_tensor
test_fixed_generator_reproduces_the_same_sequence_across_runs
test_successive_segments_do_not_reuse_the_same_noise_object
test_guidance_does_not_mutate_sim_policy_n_action_steps_or_temporal_slice
test_execution_five_requires_no_divisibility_of_sim_horizon_twelve_or_prediction_nine
test_guide_segment_rejects_wrong_shapes_devices_dtypes_and_nonfinite_values
test_verify_zero_guidance_delegates_to_the_full_oracle
test_guidance_service_rejects_zero_or_negative_configured_scale
~~~

For fresh noise, make noise_factory return tensors filled with an incrementing
call number and assert the sampler receives values 1 then 2 on successive
calls.

- [ ] **Step 3: Run the guidance tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guidance.py -q
~~~

Expected: FAIL because SimHandGuidance is missing.

- [ ] **Step 4: Implement configuration validation**

Validate:

~~~python
if config.execution_steps <= 0:
    raise ValueError("execution_steps must be positive")
if config.execution_steps > adapter.policy.n_pred_action_steps:
    raise ValueError("execution_steps exceeds Sim prediction length")
if config.oa_start + config.execution_steps > adapter.policy.horizon:
    raise ValueError("guided slice exceeds Sim horizon")
if config.num_inference_steps != 12:
    raise ValueError("first-version Sim inference steps must be 12")
if config.eta != 0.0:
    raise ValueError("first-version eta must be 0.0")
if config.guidance_scale <= 0.0:
    raise ValueError("guided check/dry-run scale must be positive")
~~~

For the target runtime, execution_steps is 5 and the derived slice is [3:8].
Set the default noise factory exactly once in __init__:

~~~python
def _default_noise_factory(shape, device, dtype, generator):
    return torch.randn(
        shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

self.noise_factory = noise_factory or _default_noise_factory
~~~

- [ ] **Step 5: Implement guide_segment and diagnostic logging**

The method performs:

~~~python
global_cond = self.adapter.global_condition(hand_state_history)
reference_norm = self.adapter.normalize_reference(hand_reference)
initial_noise = self.noise_factory(
    shape=(1, 12, 22),
    device=global_cond.device,
    dtype=global_cond.dtype,
    generator=generator,
)
scheduler = create_compatible_ddim_scheduler(
    self.adapter.policy.noise_scheduler
)
started_at = time.perf_counter()
sample = sample_guided_trajectory(
    model=self.adapter.predict_epsilon,
    scheduler=scheduler,
    initial_noise=initial_noise,
    global_cond=global_cond,
    reference=reference_norm,
    num_inference_steps=12,
    guidance_scale=self.config.guidance_scale,
    guidance_slice=self.config.guidance_slice,
    eta=0.0,
)
sim_inference_latency = time.perf_counter() - started_at
if sample.trajectory.shape != (1, 12, 22):
    raise ValueError("Sim DDIM terminal trajectory must have shape (1,12,22)")
if len(sample.steps) != self.config.num_inference_steps:
    raise ValueError("Sim DDIM chain did not execute exactly twelve steps")
if not torch.isfinite(sample.trajectory).all():
    raise ValueError("Sim DDIM terminal trajectory is non-finite")
executed_timesteps = tuple(step.timestep for step in sample.steps)
if executed_timesteps != EXPECTED_TIMESTEPS:
    raise ValueError("Sim DDIM chain executed an unexpected timestep sequence")

guided_norm = sample.trajectory[:, self.config.guidance_slice, :]
logger.debug(
    "sim_guidance latency_s=%.6f reference_error_norm=%.6f",
    sim_inference_latency,
    float((guided_norm - reference_norm).norm()),
)
hand_action = self.adapter.unnormalize_action(guided_norm)
if hand_action.shape != (1, 5, 22) or not torch.isfinite(hand_action).all():
    raise ValueError("Guided hand action must be finite with shape (1,5,22)")
return GuidedSegmentOutput(
    hand_action=hand_action,
    terminal_shape=tuple(sample.trajectory.shape),
    timesteps=executed_timesteps,
)
~~~

The code above is guide_segment_detailed. Keep guide_segment a one-line
non-resampling wrapper:

~~~python
return self.guide_segment_detailed(
    hand_state_history,
    hand_reference,
    generator=generator,
).hand_action
~~~

Record at debug level each step's timestep, alphas, raw variance, direction
coefficient, before/after loss, gradient norm, x0 ranges, out-of-range ratio,
and prev-sample range. Measure the whole sample call with time.perf_counter
and record sim_inference_latency plus the normalized guided-minus-reference
norm. Do not log complete production tensors by default.

Implement verify_zero_guidance with the same normalization boundary:

~~~python
def verify_zero_guidance(
    self,
    hand_state_history,
    hand_reference,
    *,
    initial_noise,
):
    global_cond = self.adapter.global_condition(hand_state_history)
    reference_norm = self.adapter.normalize_reference(hand_reference)
    return verify_zero_guidance_equivalence(
        model=self.adapter.predict_epsilon,
        training_scheduler=self.adapter.policy.noise_scheduler,
        initial_noise=initial_noise,
        global_cond=global_cond,
        reference=reference_norm,
        guidance_slice=self.config.guidance_slice,
        rtol=1e-5,
        atol=1e-6,
    )
~~~

- [ ] **Step 6: Run guidance tests and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guidance.py -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/sim_hand_guidance.py \
  tests/test_sim_hand_guidance.py
git commit -m "feat: add five-step sim-hand guidance service"
~~~

### Task 8: Add the fake executor and ten-segment coordinator

**Files:**

- Create: diffusion_policy/guidance/executor.py
- Create: diffusion_policy/guidance/coordinator.py
- Test: tests/test_sim_hand_guided_coordinator.py

**Interfaces:**

- Consumes: SimHandGuidance.guide_segment.
- Produces:

~~~python
ExecutionError(
    segment_id: int,
    executed_count: int,
    reason: str,
    partial_states: torch.Tensor | None = None,
) -> RuntimeError

class FakeSegmentExecutor:
    __init__(
        initial_hand_history: torch.Tensor,
        fail_segment_id: int | None = None,
        fail_after_actions: int | None = None,
    ) -> None
    reset() -> torch.Tensor
    execute(
        full_action_segment: torch.Tensor,
        segment_id: int,
    ) -> torch.Tensor

class SegmentExecutor(Protocol):
    reset() -> torch.Tensor
    execute(
        self,
        full_action_segment: torch.Tensor,
        *,
        segment_id: int,
    ) -> torch.Tensor

@dataclass(frozen=True)
class GuidedSegmentRecord:
    segment_id: int
    start: int
    end: int
    input_history: torch.Tensor
    real_hand_reference: torch.Tensor
    guided_hand: torch.Tensor
    post_states: torch.Tensor

@dataclass(frozen=True)
class GuidedRunResult:
    guided_full_actions: torch.Tensor
    records: tuple[GuidedSegmentRecord, ...]
    final_hand_history: torch.Tensor

class GuidedCoordinator:
    __init__(
        guidance: SimHandGuidance,
        real_execution_steps: int = 50,
        execution_steps: int = 5,
        hand_slice: slice = slice(9, 31),
    ) -> None

    def run(
        self,
        real_action_proposal: torch.Tensor,
        executor: SegmentExecutor,
        *,
        generator: torch.Generator | None = None,
    ) -> GuidedRunResult
~~~

- [ ] **Step 1: Write failing FakeSegmentExecutor contract tests**

Require reset to return finite (1,4,22), execute to accept only finite
(1,5,31), and successful execute to return finite (1,5,22). Add deterministic
same-seed behavior and a configured failure that raises:

~~~python
with pytest.raises(ExecutionError) as exc_info:
    executor.execute(segment, segment_id=3)

assert exc_info.value.segment_id == 3
assert exc_info.value.executed_count == 2
~~~

The fake executor records cloned action segments and derives post-states
deterministically from each segment's [9:31] values.

- [ ] **Step 2: Write failing coordinator partition and isolation tests**

Use a recording guidance spy and sentinel-valued Real proposal:

~~~python
def test_coordinator_partitions_50_and_only_replaces_9_31():
    proposal = torch.arange(
        1 * 50 * 31,
        dtype=torch.float32,
    ).reshape(1, 50, 31)
    result = coordinator.run(proposal, executor)

    assert len(result.records) == 10
    assert [(r.start, r.end) for r in result.records] == [
        (0,5), (5,10), (10,15), (15,20), (20,25),
        (25,30), (30,35), (35,40), (40,45), (45,50),
    ]
    torch.testing.assert_close(
        result.guided_full_actions[..., :9],
        proposal[..., :9],
        rtol=0.0,
        atol=0.0,
    )
~~~

Assert each spy reference equals proposal[:,5*i:5*i+5,9:31].

- [ ] **Step 3: Write failing closed-loop and failure-propagation tests**

Use sentinel states:

~~~text
reset:   [s0,s1,s2,s3]
segment: [s4,s5,s6,s7,s8]
next guidance history must be [s5,s6,s7,s8]
~~~

Add explicit tests that invalid proposal fails before reset, invalid guidance
output prevents executor.execute, malformed executor states stop immediately,
ExecutionError at segment 4 leaves exactly five executor calls, and B>1 is
rejected.

- [ ] **Step 4: Run coordinator tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_coordinator.py -q
~~~

Expected: FAIL because executor and coordinator modules are missing.

- [ ] **Step 5: Implement executor and pure-tensor coordinator**

Implement the exception and fake executor first:

~~~python
class ExecutionError(RuntimeError):
    def __init__(
        self,
        segment_id,
        executed_count,
        reason,
        partial_states=None,
    ):
        super().__init__(
            f"segment={segment_id} executed={executed_count}: {reason}"
        )
        self.segment_id = int(segment_id)
        self.executed_count = int(executed_count)
        self.partial_states = partial_states


class FakeSegmentExecutor:
    def __init__(
        self,
        initial_hand_history,
        fail_segment_id=None,
        fail_after_actions=None,
    ):
        self.initial_hand_history = initial_hand_history.clone()
        self.fail_segment_id = fail_segment_id
        self.fail_after_actions = fail_after_actions
        self.executed_segments = []

    def reset(self):
        return self.initial_hand_history.clone()

    def execute(self, full_action_segment, *, segment_id):
        _require_tensor(
            "fake executor action segment",
            full_action_segment,
            (1, 5, 31),
        )
        self.executed_segments.append(full_action_segment.clone())
        post_states = full_action_segment[..., 9:31].clone()
        if segment_id == self.fail_segment_id:
            count = int(self.fail_after_actions or 0)
            raise ExecutionError(
                segment_id=segment_id,
                executed_count=count,
                reason="configured fake execution failure",
                partial_states=post_states[:, :count].clone(),
            )
        return post_states
~~~

GuidedCoordinator constructor stores real_execution_steps=50,
execution_steps=5, and hand_slice=slice(9,31). Validate:

~~~python
def _require_tensor(name, value, expected_shape):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise ValueError(
            f"{name} expected shape {expected_shape}, "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or Inf")
    return value
~~~

~~~python
real_action_proposal = _require_tensor(
    "Real proposal",
    real_action_proposal,
    (1, 50, 31),
)
if 50 % 5 != 0:
    raise ValueError("Real execution length 50 is not divisible by 5")
if hand_slice.stop - hand_slice.start != 22:
    raise ValueError("Hand slice must contain exactly 22 dimensions")
~~~

Then:

~~~python
history = _require_tensor(
    "executor reset history",
    executor.reset(),
    (1, 4, 22),
)
segments = []
records = []

for start in range(0, 50, 5):
    end = start + 5
    real_segment = real_action_proposal[:, start:end, :]
    reference = real_segment[..., 9:31]
    guided_hand = self.guidance.guide_segment(
        history,
        reference,
        generator=generator,
    )
    guided_hand = _require_tensor(
        "guided hand",
        guided_hand,
        (1, 5, 22),
    )
    if guided_hand.device != real_segment.device:
        raise ValueError("guided hand device differs from Real segment")
    if guided_hand.dtype != real_segment.dtype:
        raise ValueError("guided hand dtype differs from Real segment")
    guided_segment = real_segment.clone()
    guided_segment[..., 9:31] = guided_hand

    post_states = executor.execute(
        guided_segment,
        segment_id=start // 5,
    )
    post_states = _require_tensor(
        "executor post states",
        post_states,
        (1, 5, 22),
    )
    records.append(
        GuidedSegmentRecord(
            segment_id=start // 5,
            start=start,
            end=end,
            input_history=history.clone(),
            real_hand_reference=reference.clone(),
            guided_hand=guided_hand.clone(),
            post_states=post_states.clone(),
        )
    )
    segments.append(guided_segment)
    history = post_states[:, -4:, :]

return GuidedRunResult(
    guided_full_actions=torch.cat(segments, dim=1),
    records=tuple(records),
    final_hand_history=history,
)
~~~

Do not catch ExecutionError except to add segment context with exception
chaining. Never append a record or update history before complete success.
At debug level log segment_id, start, end, guidance scale, reference/guided
distance, and executor outcome for every completed segment; on failure log the
same segment identifiers before re-raising.

- [ ] **Step 6: Run coordinator tests and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_coordinator.py \
  tests/test_sim_hand_guidance.py -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/executor.py \
  diffusion_policy/guidance/coordinator.py \
  tests/test_sim_hand_guided_coordinator.py
git commit -m "feat: add closed-loop guidance coordinator"
~~~

### Task 9: Compose check and dry-run runtime services

**Files:**

- Create: diffusion_policy/guidance/runtime.py
- Test: tests/test_sim_hand_guided_runtime.py

**Interfaces:**

- Consumes: both checkpoint adapters, SimHandGuidance, GuidedCoordinator, and FakeSegmentExecutor.
- Produces:

~~~python
@dataclass(frozen=True)
class LoadedGuidedPolicies:
    real: LoadedPolicy
    sim: LoadedPolicy
    real_adapter: RealPolicyAdapter
    sim_adapter: SimHandModelAdapter
    guidance: SimHandGuidance

@dataclass(frozen=True)
class CheckReport:
    real_action_proposal: torch.Tensor
    real_action_shape: tuple[int, ...]
    guided_hand_shape: tuple[int, ...]
    sim_terminal_shape: tuple[int, ...]
    sim_timesteps: tuple[int, ...]
    segment_count: int
    max_x0_error: float
    max_prev_error: float

@dataclass(frozen=True)
class DryRunReport:
    check: CheckReport
    guided: GuidedRunResult

def load_guided_policies(
    real_checkpoint: Path,
    sim_checkpoint: Path,
    device: torch.device,
    guidance_config: SimHandGuidanceConfig,
) -> LoadedGuidedPolicies

def run_check(
    loaded: LoadedGuidedPolicies,
    *,
    seed: int,
) -> CheckReport

def run_dry_run(
    loaded: LoadedGuidedPolicies,
    *,
    seed: int,
) -> DryRunReport
~~~

- [ ] **Step 1: Write failing composition and check tests**

Using fake LoadedPolicy objects, assert load_guided_policies calls the common
loader twice, keeps the policy/normalizer identities distinct, validates both
adapters, and constructs guidance without changing Sim n_action_steps=4.

Add:

~~~python
def test_run_check_calls_real_policy_once_and_returns_exact_contract():
    report = run_check(loaded, seed=7)

    assert real_policy.predict_calls == 1
    assert report.real_action_shape == (1, 50, 31)
    assert report.guided_hand_shape == (1, 5, 22)
    assert report.sim_terminal_shape == (1, 12, 22)
    assert report.sim_timesteps == EXPECTED_TIMESTEPS
    assert report.segment_count == 10
    assert report.max_x0_error <= 1e-5
    assert report.max_prev_error <= 1e-5
~~~

Assert guidance.guide_segment_detailed was called exactly once with the
configured nonzero guidance scale, the first five Real hand targets, and the
seeded (1,4,22) history. Its fake GuidedSegmentOutput contains hand shape
(1,5,22), terminal shape (1,12,22), and EXPECTED_TIMESTEPS. Make the fake
return a wrong terminal shape, eleven timesteps, and NaN hand action in separate
cases; each must fail. This proves check is not satisfied by the zero-guidance
oracle alone and that its full-chain fields come from the nonzero run.
Add test_run_check_rejects_zero_configured_guidance_scale_before_real_inference.

Also reject a wrong Diffusers version, a non-target Real cfg, malformed Real
prediction, a failed U-Net/DDIM equivalence report, and shared Real/Sim
normalizer identity.

- [ ] **Step 2: Write failing dry-run tests**

Add:

~~~python
def test_dry_run_reuses_the_single_real_proposal_for_all_ten_segments():
    report = run_dry_run(loaded, seed=11)

    assert real_policy.predict_calls == 1
    assert len(report.guided.records) == 10
    for index, record in enumerate(report.guided.records):
        torch.testing.assert_close(
            record.real_hand_reference,
            report.check.real_action_proposal[
                :, 5 * index:5 * index + 5, 9:31
            ],
        )
~~~

Add fixed-seed reproducibility, latest-four state, fresh segment noise, no
hardware import, guidance failure before command, and executor failure stopping
later segments.

- [ ] **Step 3: Run runtime tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_runtime.py -q
~~~

Expected: FAIL because runtime.py is missing.

- [ ] **Step 4: Implement composition and check**

load_guided_policies loads both checkpoints independently, creates adapters,
validates them, then creates SimHandGuidance.
It compares the selected Real and Sim policy parameter device/dtype and raises
a descriptive error if they differ.

~~~python
def load_guided_policies(
    real_checkpoint,
    sim_checkpoint,
    device,
    guidance_config,
):
    real = load_workspace_policy(real_checkpoint, device)
    sim = load_workspace_policy(sim_checkpoint, device)
    real_adapter = RealPolicyAdapter(real)
    sim_adapter = SimHandModelAdapter(sim)
    real_adapter.validate_target_contract()
    sim_adapter.validate_checkpoint_contract()

    real_parameter = next(real.policy.parameters())
    sim_parameter = next(sim.policy.parameters())
    if real_parameter.device != sim_parameter.device:
        raise ValueError("Real and Sim policy devices differ")
    if real_parameter.dtype != sim_parameter.dtype:
        raise ValueError("Real and Sim policy dtypes differ")
    if real.policy.normalizer is sim.policy.normalizer:
        raise ValueError("Real and Sim policies share one normalizer")

    guidance = SimHandGuidance(sim_adapter, guidance_config)
    return LoadedGuidedPolicies(
        real=real,
        sim=sim,
        real_adapter=real_adapter,
        sim_adapter=sim_adapter,
        guidance=guidance,
    )
~~~

Define the runtime-only deterministic history helper in runtime.py:

~~~python
def _seeded_initial_history(policy, seed):
    parameter = next(policy.parameters())
    generator = torch.Generator(device=parameter.device)
    generator.manual_seed(seed)
    return torch.randn(
        (1, 4, 22),
        device=parameter.device,
        dtype=parameter.dtype,
        generator=generator,
    )
~~~

run_check:

~~~python
assert_pinned_diffusers_version()
if loaded.guidance.config.guidance_scale <= 0.0:
    raise ValueError("check requires a positive configured guidance scale")
proposal = loaded.real_adapter.predict_proposal()

sim_parameter = next(loaded.sim.policy.parameters())
history = _seeded_initial_history(loaded.sim.policy, seed)
reference = proposal[:, :5, 9:31]

guided_generator = torch.Generator(device=sim_parameter.device)
guided_generator.manual_seed(seed + 1)
guided_result = loaded.guidance.guide_segment_detailed(
    history,
    reference,
    generator=guided_generator,
)
guided_hand = guided_result.hand_action
if guided_hand.shape != (1, 5, 22) or not torch.isfinite(guided_hand).all():
    raise ValueError("Configured Sim guidance returned invalid hand actions")
if guided_result.terminal_shape != (1, 12, 22):
    raise ValueError("Configured Sim guidance returned invalid terminal shape")
if guided_result.timesteps != EXPECTED_TIMESTEPS:
    raise ValueError("Configured Sim guidance returned invalid timesteps")

zero_generator = torch.Generator(device=sim_parameter.device)
zero_generator.manual_seed(seed + 2)
initial_noise = torch.randn(
    (1, 12, 22),
    device=sim_parameter.device,
    dtype=sim_parameter.dtype,
    generator=zero_generator,
)
zero_report = loaded.guidance.verify_zero_guidance(
    history,
    reference,
    initial_noise=initial_noise,
)
if zero_report.terminal_shape != guided_result.terminal_shape:
    raise ValueError("Nonzero and zero-guidance terminal shapes differ")
if zero_report.timesteps != guided_result.timesteps:
    raise ValueError("Nonzero and zero-guidance timesteps differ")

return CheckReport(
    real_action_proposal=proposal,
    real_action_shape=tuple(proposal.shape),
    guided_hand_shape=tuple(guided_hand.shape),
    sim_terminal_shape=guided_result.terminal_shape,
    sim_timesteps=guided_result.timesteps,
    segment_count=proposal.shape[1] // 5,
    max_x0_error=zero_report.max_x0_error,
    max_prev_error=zero_report.max_prev_error,
)
~~~

The proposal tensor is retained so dry-run does not invoke Real policy again.

- [ ] **Step 5: Implement dry-run using only FakeSegmentExecutor**

run_dry_run calls run_check exactly once, creates a deterministic
FakeSegmentExecutor with the same seed, creates a separate seeded generator for
segment noise, and passes report.real_action_proposal to GuidedCoordinator.run.
It never imports or constructs a robot environment.
The fake executor reset history is generated by _seeded_initial_history and the
segment-noise generator is seeded with seed+1, so reset sampling cannot consume
or shift the segment diffusion-noise sequence.

~~~python
def run_dry_run(loaded, *, seed):
    check = run_check(loaded, seed=seed)
    initial_history = _seeded_initial_history(loaded.sim.policy, seed)
    executor = FakeSegmentExecutor(initial_history)
    parameter = next(loaded.sim.policy.parameters())
    segment_generator = torch.Generator(device=parameter.device)
    segment_generator.manual_seed(seed + 1)
    coordinator = GuidedCoordinator(loaded.guidance)
    guided = coordinator.run(
        check.real_action_proposal,
        executor,
        generator=segment_generator,
    )
    return DryRunReport(check=check, guided=guided)
~~~

- [ ] **Step 6: Run runtime tests and commit**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_runtime.py \
  tests/test_sim_hand_guided_coordinator.py -q
~~~

Expected: PASS.

Commit:

~~~bash
git add \
  diffusion_policy/guidance/runtime.py \
  tests/test_sim_hand_guided_runtime.py
git commit -m "feat: add guided inference check and dry-run"
~~~

### Task 10: Add the independent CLI and run the complete regression gate

**Files:**

- Create: inference_sim_hand_guided.py
- Test: tests/test_inference_sim_hand_guided_cli.py

**Interfaces:**

- Consumes: load_guided_policies, run_check, and run_dry_run.
- Produces:

~~~python
parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace
main(argv: Sequence[str] | None = None) -> int
~~~

The module returns 0 on success and 1 for a caught runtime/contract failure.
argparse usage errors retain argparse's normal nonzero exit.

- [ ] **Step 1: Write failing CLI dispatch and validation tests**

Create real temporary files so path validation is exercised:

~~~python
@pytest.fixture
def checkpoint_paths(tmp_path):
    real = tmp_path / "real.ckpt"
    sim = tmp_path / "sim.ckpt"
    real.write_bytes(b"test-real")
    sim.write_bytes(b"test-sim")
    return SimpleNamespace(real=real, sim=sim)
~~~

Monkeypatch load_guided_policies and run_check to return a CheckReport with
the exact target shapes/timesteps, then add:

~~~python
def test_cli_check_dispatches_and_prints_contract(capsys, checkpoint_paths):
    exit_code = main([
        "--mode", "check",
        "--real-checkpoint", str(checkpoint_paths.real),
        "--sim-checkpoint", str(checkpoint_paths.sim),
        "--device", "cpu",
        "--guidance-scale", "1.0",
        "--execution-steps", "5",
        "--sim-inference-steps", "12",
        "--seed", "7",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "real_action_shape=(1, 50, 31)" in output
    assert "guided_hand_shape=(1, 5, 22)" in output
    assert "segment_count=10" in output
    assert "timesteps=(88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)" in output
~~~

Add:

~~~text
test_cli_dry_run_prints_ten_completed_segments
test_cli_rejects_missing_checkpoint_before_loading
test_cli_rejects_negative_guidance_scale
test_cli_rejects_zero_guidance_scale
test_cli_rejects_execution_steps_other_than_five
test_cli_rejects_sim_inference_steps_other_than_twelve
test_cli_runtime_failure_returns_one_and_writes_context_to_stderr
test_cli_import_does_not_import_inference_dp_or_robot_packages
test_cli_help_does_not_import_hardware_or_construct_a_workspace
~~~

Both independence tests run fresh subprocesses (one imports the module, one
invokes --help) and examine sys.modules. Reject inference_dp,
direct_robot_env, every diffusion_policy.real_world prefix, any module name
containing vitacformer, and pyrealsense2/ur_rtde prefixes.

- [ ] **Step 2: Run CLI tests and verify the red state**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_inference_sim_hand_guided_cli.py -q
~~~

Expected: FAIL because inference_sim_hand_guided.py is missing.

- [ ] **Step 3: Implement argparse and dispatch**

Arguments:

~~~python
parser.add_argument("--mode", choices=("check", "dry-run"), required=True)
parser.add_argument("--real-checkpoint", type=Path, required=True)
parser.add_argument("--sim-checkpoint", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--guidance-scale", type=float, default=1.0)
parser.add_argument("--execution-steps", type=int, default=5)
parser.add_argument("--sim-inference-steps", type=int, default=12)
parser.add_argument("--seed", type=int, default=0)
~~~

Validate both checkpoint files, CUDA availability, strictly positive guidance scale,
execution_steps==5, and sim_inference_steps==12 before loading. Construct
SimHandGuidanceConfig, call load_guided_policies once, then dispatch:

~~~python
def print_check_report(report):
    print(f"real_action_shape={report.real_action_shape}")
    print(f"guided_hand_shape={report.guided_hand_shape}")
    print(f"sim_terminal_shape={report.sim_terminal_shape}")
    print(f"segment_count={report.segment_count}")
    print(f"timesteps={report.sim_timesteps}")
    print(f"max_x0_error={report.max_x0_error:.8g}")
    print(f"max_prev_error={report.max_prev_error:.8g}")


def print_dry_run_report(report):
    print_check_report(report.check)
    print(f"completed_segments={len(report.guided.records)}")
~~~

Dispatch with:

~~~python
try:
    if args.mode == "check":
        report = run_check(loaded, seed=args.seed)
        print_check_report(report)
    else:
        report = run_dry_run(loaded, seed=args.seed)
        print_dry_run_report(report)
    return 0
except (FileNotFoundError, ValueError, RuntimeError, AssertionError) as exc:
    print(f"[guided-inference:error] {exc}", file=sys.stderr)
    return 1
~~~

The module-level executable block is:

~~~python
if __name__ == "__main__":
    raise SystemExit(main())
~~~

Do not copy logic from inference_dp.py and do not import it.

- [ ] **Step 4: Run CLI and feature tests**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_guided_checkpoint_loader.py \
  tests/test_sim_hand_guided_adapters.py \
  tests/test_sim_hand_guided_ddim.py \
  tests/test_sim_hand_guidance.py \
  tests/test_sim_hand_guided_coordinator.py \
  tests/test_sim_hand_guided_runtime.py \
  tests/test_inference_sim_hand_guided_cli.py -q
~~~

Expected: PASS.

- [ ] **Step 5: Run existing Sim and standalone regression tests**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m pytest \
  tests/test_sim_hand_temporal.py \
  tests/test_diffusion_unet_sim_hand_policy.py \
  tests/test_sim_hand_lowdim_dataset.py \
  tests/test_null_lowdim_runner.py \
  tests/test_sim_hand_train_smoke.py -q
~~~

Expected: PASS. This proves the temporal validator update did not alter the Sim
training loss, public four-action inference output, data contract, or
checkpoint lifecycle.

- [ ] **Step 6: Run static and repository checks**

Run:

~~~bash
$GUIDED_TEST_PYTHON -m py_compile \
  diffusion_policy/guidance/checkpoint_loader.py \
  diffusion_policy/guidance/real_adapter.py \
  diffusion_policy/guidance/sim_adapter.py \
  diffusion_policy/guidance/guided_ddim.py \
  diffusion_policy/guidance/sim_hand_guidance.py \
  diffusion_policy/guidance/executor.py \
  diffusion_policy/guidance/coordinator.py \
  diffusion_policy/guidance/runtime.py \
  inference_sim_hand_guided.py \
  inference_dp.py \
  train.py

git diff --check
git status --short
~~~

Expected: py_compile exits 0, git diff --check prints nothing, and status lists
only the intended feature files.

- [ ] **Step 7: Run check and dry-run against actual target checkpoints**

The repository contains no checkpoint files, so set these two variables to the
target Real 64/1/50 checkpoint and the trained Sim 4/9/4/12 checkpoint before
the gate:

~~~bash
test -n "$REAL_CKPT_PATH"
test -f "$REAL_CKPT_PATH"
test -n "$SIM_CKPT_PATH"
test -f "$SIM_CKPT_PATH"

$GUIDED_TEST_PYTHON inference_sim_hand_guided.py \
  --mode check \
  --real-checkpoint "$REAL_CKPT_PATH" \
  --sim-checkpoint "$SIM_CKPT_PATH" \
  --device cuda:0 \
  --guidance-scale 1.0 \
  --execution-steps 5 \
  --sim-inference-steps 12 \
  --seed 7

$GUIDED_TEST_PYTHON inference_sim_hand_guided.py \
  --mode dry-run \
  --real-checkpoint "$REAL_CKPT_PATH" \
  --sim-checkpoint "$SIM_CKPT_PATH" \
  --device cuda:0 \
  --guidance-scale 1.0 \
  --execution-steps 5 \
  --sim-inference-steps 12 \
  --seed 7
~~~

Expected check output includes Real shape (1,50,31), a configured nonzero
guided hand result (1,5,22), Sim zero-oracle terminal shape (1,12,22), the
twelve pinned timesteps, ten segments, and zero-guidance errors within
rtol=1e-5/atol=1e-6. Expected dry-run output reports ten completed segments and
no hardware import or connection.

- [ ] **Step 8: Commit the CLI and final integration gate**

Commit:

~~~bash
git add \
  inference_sim_hand_guided.py \
  tests/test_inference_sim_hand_guided_cli.py
git commit -m "feat: add sim-hand guided inference CLI"
~~~
