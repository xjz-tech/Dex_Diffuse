# Sim-Hand Guided Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hardware-free guidance path that preserves the existing Real
and Sim model/loss systems, refines the complete nine-step coupled 22-D hand
trajectory with a custom zero-equivalent DDIM sampler, and executes its first
five steps in closed-loop segments.

**Architecture:** Restore each checkpoint with the repository's existing
Workspace payload sequence. Thin adapters expose policy-derived shape and
normalization contracts; a focused DDIM module owns dynamic scheduling and
reverse-step-mean guidance; a coordinator composes hand guidance with the untouched Real
proposal. The Sim training default changes only `n_action_steps` from 4 to 5;
model architecture, loss, temporal validation, DINO, and hardware code remain
unchanged.

**Tech Stack:** Python 3.9+, PyTorch, Diffusers 0.11.1, Hydra/OmegaConf, dill,
dataclasses, argparse, pytest, and the repository's existing Workspace,
normalizer, Real policy, and Sim-Hand policy.

**Spec:** `docs/superpowers/specs/2026-08-26-sim-hand-guided-inference-design.md`

**Open checks (rulings / 待核验):**
`docs/superpowers/plans/2026-08-26-sim-hand-guided-inference-open-checks.md`

## Global Constraints

- Do not modify `inference_dp.py`, either training workspace, either policy,
  `sim_hand_temporal_util.py`, old temporal tests, DINO loading, or checkpoint
  format.
- The new entry point does not import `inference_dp.py`, `DirectRobotEnv`,
  ViTacFormer, or robot/controller modules.
- Real inference calls `policy.predict_action(obs)` once; public `action` is
  executed and aligned `action_pred` supplies full guidance references.
- Real temporal length and segment count come from the actual proposal.
- Sim horizon, observation steps, prediction steps, dimensions, and
  `oa_start` come from the restored policy and
  `policy.temporal.usable_action_slice`.
- Guidance configuration contains only `execution_steps`,
  `guidance_scale`, `num_inference_steps`, and `eta`.
- The current target produces `(1,50,31)`, ten five-step segments, Sim
  `4/12/9/5`, guidance slice `[3:12]`, and execution slice `[3:8]`; these are
  integration-test expectations, not duplicated production model configuration.
- Production DDIM never requires 100 train steps and never imports a fixed
  timestep array. `scheduler.set_timesteps(num_inference_steps)` is the
  authority.
- DDIM uses epsilon prediction, `eta=0`, no dynamic thresholding, official
  x0 clipping before the reverse update, original epsilon in the direction
  term, and guidance on the resulting reverse-step mean without clipping it.
- `guidance_scale >= 0`; scale zero follows the same custom arithmetic and is
  compared with official Diffusers step-by-step and through a full chain.
- Real and Sim normalizers remain independent. Guidance loss is computed only
  in Sim normalized action space.
- Only Real `[9:31]` is replaced; `[0:9]` is unchanged.
- Each segment gets fresh initial noise. Coordinator/executor support `B=1`
  and stop on partial execution.
- Apply TDD: every production change must first have a focused failing test.

---

## File Map

Only create the following files:

~~~text
diffusion_policy/guidance/__init__.py
    Side-effect-free package boundary.

diffusion_policy/guidance/checkpoint_loader.py
    Thin Workspace payload restoration and EMA/model selection.

diffusion_policy/guidance/real_adapter.py
    Synthetic Real observation and actual proposal validation.

diffusion_policy/guidance/sim_adapter.py
    Policy-derived Sim temporal/normalizer/model interface.

diffusion_policy/guidance/guided_ddim.py
    Dynamic DDIM factory, analytic gradient, custom step, sampler, zero oracle.

diffusion_policy/guidance/sim_hand_guidance.py
    Five-step public hand-guidance service using policy-derived shapes.

diffusion_policy/guidance/executor.py
    Executor protocol, ExecutionError, deterministic fake executor.

diffusion_policy/guidance/coordinator.py
    Dynamic Real segmentation, hand replacement, closed-loop history update.

diffusion_policy/guidance/runtime.py
    Check/dry-run composition.

inference_sim_hand_guided.py
    Independent hardware-free CLI.

tests/test_sim_hand_guided_checkpoint_loader.py
tests/test_sim_hand_guided_adapters.py
tests/test_sim_hand_guided_ddim.py
tests/test_sim_hand_guidance.py
tests/test_sim_hand_guided_coordinator.py
tests/test_sim_hand_guided_runtime.py
tests/test_inference_sim_hand_guided_cli.py
~~~

## Test Environment

The repository environments pin Diffusers 0.11.1. If the active Python does not
provide it, use the disposable overlay:

~~~bash
uv venv +  --python /home/carus/miniforge3/envs/isaaclab/bin/python3.11 +  --system-site-packages +  /tmp/dex-diffuse-guided-test

uv pip install +  --python /tmp/dex-diffuse-guided-test/bin/python +  'diffusers==0.11.1' +  'huggingface-hub==0.11.1' +  'accelerate==0.13.2' +  'dill>=0.3.5,<0.4' +  'zarr>=2.12,<3' +  'numcodecs>=0.10,<0.16'

GUIDED_TEST_PYTHON=/tmp/dex-diffuse-guided-test/bin/python
$GUIDED_TEST_PYTHON -c +  'import diffusers; assert diffusers.__version__ == "0.11.1"'
~~~

All commands below run from the repository root.

### Task 1: Add the thin Workspace checkpoint loader

**Files:**

- Create: `diffusion_policy/guidance/__init__.py`
- Create: `diffusion_policy/guidance/checkpoint_loader.py`
- Test: `tests/test_sim_hand_guided_checkpoint_loader.py`

**Interfaces:**

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

- [ ] **Step 1: Write failing loader tests**

Use a fake workspace with distinct model/EMA weights:

~~~python
class FakeWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = nn.Linear(1, 1, bias=False)
        self.ema_model = nn.Linear(1, 1, bias=False)

    def load_payload(self, payload, exclude_keys=None, include_keys=None):
        for name, state in payload["state_dicts"].items():
            getattr(self, name).load_state_dict(state)
~~~

Test:

~~~text
test_loader_selects_ema_when_cfg_requests_it
test_loader_uses_model_when_ema_is_disabled
test_loader_moves_policy_to_device_eval_and_disables_grad
test_loader_rejects_missing_checkpoint_or_cfg_or_model
test_loader_does_not_override_policy_temporal_or_num_inference_steps
test_loader_preserves_normalizer_state_in_selected_policy
test_guidance_package_import_has_no_hardware_side_effects
~~~

The temporal preservation test sets sentinel values on both fake policies,
loads them, and asserts the same values remain. The import test starts a fresh
subprocess, imports only `diffusion_policy.guidance.checkpoint_loader`, and
asserts `inference_dp`, `direct_robot_env`, and
`diffusion_policy.real_world` are absent from `sys.modules`.

- [ ] **Step 2: Run the loader tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_checkpoint_loader.py -q
~~~

Expected: FAIL because `diffusion_policy.guidance` does not exist.

- [ ] **Step 3: Implement the same thin sequence as the existing Real loader**

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
~~~

Do not set `policy.num_inference_steps`, temporal fields, DINO fields, or
normalizer values.

- [ ] **Step 4: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_checkpoint_loader.py -q

git add +  diffusion_policy/guidance/__init__.py +  diffusion_policy/guidance/checkpoint_loader.py +  tests/test_sim_hand_guided_checkpoint_loader.py
git commit -m "feat: add thin guidance checkpoint loader"
~~~

### Task 2: Add policy-derived Real and Sim adapters

**Files:**

- Create: `diffusion_policy/guidance/real_adapter.py`
- Create: `diffusion_policy/guidance/sim_adapter.py`
- Test: `tests/test_sim_hand_guided_adapters.py`

**Interfaces:**

~~~python
class RealPolicyAdapter:
    def __init__(self, loaded: LoadedPolicy) -> None
    def build_synthetic_observation(self) -> dict[str, torch.Tensor]
    def predict_proposal(
        self,
        observation: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor
    def predict_for_guidance(
        self,
        observation: dict[str, torch.Tensor] | None = None,
    ) -> RealGuidancePrediction

class SimPolicyAdapter:
    def __init__(self, loaded: LoadedPolicy) -> None
    @property
    def horizon(self) -> int
    @property
    def n_obs_steps(self) -> int
    @property
    def n_pred_action_steps(self) -> int
    @property
    def action_dim(self) -> int
    @property
    def oa_start(self) -> int
    @property
    def guidance_slice(self) -> slice
    def execution_slice(self, execution_steps: int) -> slice
    def normalize_history(self, value: torch.Tensor) -> torch.Tensor
    def normalize_reference(self, value: torch.Tensor) -> torch.Tensor
    def unnormalize_action(self, value: torch.Tensor) -> torch.Tensor
    def global_condition(self, history: torch.Tensor) -> torch.Tensor
    def predict_epsilon(
        self,
        sample: torch.Tensor,
        timestep: int | torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor
~~~

- [ ] **Step 1: Write failing adapter tests**

Real tests use shape_meta with front/wrist RGB, 9-D EE, 22-D hand, and 31-D
action. Assert synthetic tensors have:

~~~python
assert obs["front_image"].shape == (1, policy.n_obs_steps, 3, 240, 320)
assert obs["ee_pose"].shape == (1, policy.n_obs_steps, 9)
assert obs["hand_joint"].shape == (1, policy.n_obs_steps, 22)
~~~

`predict_proposal` must call the original policy once, return finite
`(1,T,31)`, and reject wrong batch/rank/action dimension without constraining
`T` to 50 in production. `predict_for_guidance` additionally retains
`action_pred`, aligns it from `n_obs_steps-1`, verifies that its prefix equals
the public proposal, and exposes its 22-D hand slice.

Sim tests create two fake policies:

~~~python
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
~~~

Assert values and slices come from each policy:

~~~python
assert SimPolicyAdapter(current).guidance_slice == slice(3, 12)
assert SimPolicyAdapter(current).execution_slice(5) == slice(3, 8)
assert SimPolicyAdapter(alternate).guidance_slice == slice(1, 8)
assert SimPolicyAdapter(alternate).execution_slice(3) == slice(1, 4)
~~~

Also test independent obs/action normalization, dynamic global-cond flattening,
direct `policy.model` epsilon calls, non-finite rejection, and that
`policy.predict_action` is never called.

- [ ] **Step 2: Run adapter tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_adapters.py -q
~~~

Expected: FAIL because the adapter modules do not exist.

- [ ] **Step 3: Implement Real adapter from actual policy metadata**

`build_synthetic_observation` reads `loaded.cfg.shape_meta.obs` and
`policy.n_obs_steps`, then allocates finite float tensors on the policy
device/dtype. `predict_proposal` calls:

~~~python
result = self.policy.predict_action(observation)
if "action" not in result:
    raise KeyError("Real policy prediction has no 'action'")
proposal = result["action"]
if proposal.ndim != 3 or proposal.shape[0] != 1 or proposal.shape[-1] != 31:
    raise ValueError(
        f"Real proposal must have shape (1,T,31), got {tuple(proposal.shape)}"
    )
if not torch.isfinite(proposal).all():
    raise ValueError("Real proposal is non-finite")
return proposal
~~~

The guidance path requires `result["action_pred"]`; missing, misaligned, or
too-short full predictions fail loudly instead of padding or repeating actions.

- [ ] **Step 4: Implement Sim adapter without copied temporal constants**

Use:

~~~python
self.horizon = int(policy.horizon)
self.n_obs_steps = int(policy.n_obs_steps)
self.n_pred_action_steps = int(policy.n_pred_action_steps)
self.action_dim = int(policy.action_dim)
self.obs_dim = int(policy.obs_dim)
self.oa_start = int(policy.temporal.usable_action_slice.start)
~~~

Require `obs_dim == action_dim == 22`. `guidance_slice` covers all
`n_pred_action_steps`; `execution_slice(E)` checks `E>0`,
`E<=n_pred_action_steps`, and `oa_start+E<=horizon`. Global condition is:

~~~python
normalized = self.normalize_history(history)
return normalized[:, :self.n_obs_steps, :].reshape(history.shape[0], -1)
~~~

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_adapters.py -q

git add +  diffusion_policy/guidance/real_adapter.py +  diffusion_policy/guidance/sim_adapter.py +  tests/test_sim_hand_guided_adapters.py
git commit -m "feat: add policy-derived guidance adapters"
~~~

### Task 3: Add the dynamic DDIM factory and analytic gradient

**Files:**

- Create: `diffusion_policy/guidance/guided_ddim.py`
- Test: `tests/test_sim_hand_guided_ddim.py`

**Interfaces:**

~~~python
EXPECTED_DIFFUSERS_VERSION = "0.11.1"

def assert_pinned_diffusers_version() -> None
def create_ddim_scheduler(
    training_scheduler: DDPMScheduler,
) -> DDIMScheduler
def mse_guidance_gradient(
    base_sample: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
) -> torch.Tensor
~~~

`EXPECTED_CURRENT_TIMESTEPS` may exist only in the test file.

- [ ] **Step 1: Write failing dynamic scheduler tests**

Create 100-step and 60-step DDPM fixtures. For each:

~~~python
ddim = create_ddim_scheduler(training)
ddim.set_timesteps(inference_steps)
torch.testing.assert_close(
    ddim.alphas_cumprod,
    training.alphas_cumprod,
    rtol=0.0,
    atol=0.0,
)
assert len(ddim.timesteps) == inference_steps
assert int(ddim.timesteps.max()) < training.config.num_train_timesteps
~~~

Only the current 100/12 fixture asserts:

~~~python
EXPECTED_CURRENT_TIMESTEPS = (88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0)
assert tuple(map(int, ddim.timesteps)) == EXPECTED_CURRENT_TIMESTEPS
~~~

Add a static test that reads `guided_ddim.py` and asserts neither
`EXPECTED_CURRENT_TIMESTEPS` nor the literal tuple appears in production.
Reject wrong Diffusers version, non-epsilon prediction, and
`thresholding=True`. Do not reject a scheduler merely because its train-step
count is not 100.

- [ ] **Step 2: Write failing dynamic gradient tests**

Use shapes `(2,12,22)` with slice `3:8` and `(2,8,22)` with slice
`1:4`. Assert:

~~~python
expected[:, guidance_slice] = (
    2.0 / (reference.shape[1] * reference.shape[2])
) * (base_sample[:, guidance_slice] - reference)
~~~

Reject mismatched batch, action dim, slice length, dtype/device, and non-finite
inputs.

- [ ] **Step 3: Run focused tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_ddim.py +  -k "factory or gradient or version" -q
~~~

Expected: FAIL because `guided_ddim.py` is missing.

- [ ] **Step 4: Implement factory and gradient**

~~~python
def create_ddim_scheduler(training_scheduler):
    assert_pinned_diffusers_version()
    config = training_scheduler.config
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

The factory does not call `set_timesteps` and contains no train-step count
check. The caller supplies `num_inference_steps`.

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_ddim.py +  -k "factory or gradient or version" -q

git add +  diffusion_policy/guidance/guided_ddim.py +  tests/test_sim_hand_guided_ddim.py
git commit -m "feat: add dynamic DDIM guidance primitives"
~~~

### Task 4: Implement one zero-equivalent guided DDIM step

**Files:**

- Modify: `diffusion_policy/guidance/guided_ddim.py`
- Modify: `tests/test_sim_hand_guided_ddim.py`

**Interfaces:**

~~~python
@dataclass(frozen=True)
class GuidedDDIMStepOutput:
    timestep: int
    prev_sample: torch.Tensor
    base_prev_sample: torch.Tensor
    pred_original_sample: torch.Tensor
    raw_pred_original_sample: torch.Tensor
    alpha_bar_t: torch.Tensor
    alpha_bar_prev: torch.Tensor
    raw_variance: torch.Tensor
    direction_coefficient: torch.Tensor
    guidance_loss_before: torch.Tensor
    guidance_loss_after: torch.Tensor

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

- [ ] **Step 1: Write failing official-oracle tests**

For every timestep produced by a 100/12 fixture and a 60/10 fixture, create
independent official/custom DDIM schedulers, use the same clipping-triggering
sample/model output, and compare at `guidance_scale=0`:

~~~python
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

Monkeypatch `custom_scheduler.step` to raise, proving zero scale is not an
official-step bypass.

- [ ] **Step 2: Write failing guidance semantic tests**

Test:

~~~text
test_guidance_updates_reverse_mean_without_modifying_predicted_x0
test_raw_variance_matches_independent_formula_for_each_dynamic_timestep
test_guided_reverse_mean_uses_variance_not_standard_deviation
test_final_timestep_zero_variance_makes_reverse_mean_guidance_a_noop
test_guided_reverse_mean_is_not_clipped
test_zero_reference_distance_has_zero_update
test_guidance_losses_are_measured_on_reverse_mean
test_negative_scale_and_nonzero_eta_are_rejected
test_step_rejects_shape_dtype_device_and_finite_violations
~~~

Build expected variance directly from the scheduler's alpha table and dynamic
previous timestep, not from output diagnostics.

- [ ] **Step 3: Run step tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_ddim.py +  -k "step or variance or clipping or official" -q
~~~

Expected: FAIL because `guided_ddim_step` is missing.

- [ ] **Step 4: Implement official 0.11.1 arithmetic**

~~~python
t = int(timestep)
step_ratio = (
    scheduler.config.num_train_timesteps
    // scheduler.num_inference_steps
)
prev_t = t - step_ratio
alpha_t = scheduler.alphas_cumprod[t]
alpha_prev = (
    scheduler.alphas_cumprod[prev_t]
    if prev_t >= 0
    else scheduler.final_alpha_cumprod
)
beta_t = 1.0 - alpha_t
x0_raw = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
x0_base = (
    x0_raw.clamp(-1.0, 1.0)
    if scheduler.config.clip_sample
    else x0_raw
)
raw_variance = scheduler._get_variance(t, prev_t)
direction = (1.0 - alpha_prev).sqrt()
base_prev_sample = (
    alpha_prev.sqrt() * x0_base + direction * model_output
)
gradient = mse_guidance_gradient(
    base_prev_sample, reference, guidance_slice
)
prev_sample = (
    base_prev_sample - guidance_scale * raw_variance * gradient
)
~~~

Validate `scheduler.num_inference_steps` is set, `eta==0`, scale is
non-negative, and inputs are compatible. Do not rederive epsilon, do not
modify `x0_base`, and do not clip the guided reverse-step mean. The raw
variance is retained as the guidance scale even though `eta=0` removes the
stochastic noise term.

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_ddim.py -q

git add +  diffusion_policy/guidance/guided_ddim.py +  tests/test_sim_hand_guided_ddim.py
git commit -m "feat: add zero-equivalent guided DDIM step"
~~~

### Task 5: Add the dynamic full-chain sampler and zero oracle

**Files:**

- Modify: `diffusion_policy/guidance/guided_ddim.py`
- Modify: `tests/test_sim_hand_guided_ddim.py`

**Interfaces:**

~~~python
@dataclass(frozen=True)
class GuidedDDIMSampleOutput:
    trajectory: torch.Tensor
    steps: tuple[GuidedDDIMStepOutput, ...]

@dataclass(frozen=True)
class ZeroGuidanceReport:
    timesteps: tuple[int, ...]
    terminal_shape: tuple[int, ...]
    max_x0_error: float
    max_prev_error: float

def sample_guided_trajectory(
    model,
    scheduler: DDIMScheduler,
    initial_noise: torch.Tensor,
    global_cond: torch.Tensor,
    reference: torch.Tensor,
    num_inference_steps: int,
    guidance_scale: float,
    guidance_slice: slice,
    eta: float = 0.0,
) -> GuidedDDIMSampleOutput

def verify_zero_guidance_equivalence(
    model,
    training_scheduler: DDPMScheduler,
    initial_noise: torch.Tensor,
    global_cond: torch.Tensor,
    reference: torch.Tensor,
    guidance_slice: slice,
    num_inference_steps: int,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> ZeroGuidanceReport
~~~

- [ ] **Step 1: Write failing dynamic-chain tests**

Use deterministic epsilon models with 100/12 and 60/10 schedulers. Assert:

~~~python
assert len(result.steps) == num_inference_steps
assert tuple(step.timestep for step in result.steps) == tuple(
    map(int, scheduler.timesteps)
)
assert result.trajectory.shape == initial_noise.shape
assert all(not step.prev_sample.requires_grad for step in result.steps)
~~~

Add `num_inference_steps<=0`, non-finite model output, and wrong output-shape
failures.

- [ ] **Step 2: Write failing full official/custom oracle**

Start official/custom samples from cloned initial noise. At every scheduler-
generated timestep call the model independently on each evolving sample and
compare `pred_original_sample` and `prev_sample`. Assert report timesteps
equal the executed custom step outputs, not a fixed constant.

- [ ] **Step 3: Run chain tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_ddim.py +  -k "chain or sampler or report" -q
~~~

Expected: FAIL because the chain interfaces are missing.

- [ ] **Step 4: Implement dynamic sampling**

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
        outputs.append(output)
        trajectory = output.prev_sample
return GuidedDDIMSampleOutput(trajectory, tuple(outputs))
~~~

The zero verifier constructs independent official/custom DDIM instances from
the same training config and uses the requested inference-step count.

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_ddim.py -q

git add +  diffusion_policy/guidance/guided_ddim.py +  tests/test_sim_hand_guided_ddim.py
git commit -m "feat: add dynamic guided DDIM sampling"
~~~

### Task 6: Add the policy-derived Sim hand guidance service

**Files:**

- Create: `diffusion_policy/guidance/sim_hand_guidance.py`
- Test: `tests/test_sim_hand_guidance.py`

**Interfaces:**

~~~python
@dataclass(frozen=True)
class SimHandGuidanceConfig:
    execution_steps: int = 5
    guidance_scale: float = 1.0
    num_inference_steps: int = 12
    eta: float = 0.0

NoiseFactory = Callable[
    [tuple[int, ...], torch.device, torch.dtype, torch.Generator | None],
    torch.Tensor,
]

class SimHandGuidance:
    def __init__(
        self,
        adapter: SimPolicyAdapter,
        config: SimHandGuidanceConfig,
        noise_factory=None,
    ) -> None
    def guide_segment(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor
    def verify_zero_guidance(
        self,
        hand_state_history: torch.Tensor,
        hand_reference: torch.Tensor,
        *,
        initial_noise: torch.Tensor,
    ) -> ZeroGuidanceReport
~~~

- [ ] **Step 1: Write failing configuration and derivation tests**

With the current fake adapter assert:

~~~python
assert guidance.guidance_slice == slice(3, 12)
assert guidance.execution_slice == slice(3, 8)
assert guidance.trajectory_shape == (1, 12, 22)
~~~

With the alternate adapter assert the derived slice/shape change without
changing config. Reject execution beyond usable prediction, slice beyond
horizon, negative scale, non-positive inference steps, and nonzero eta.
Explicitly assert scale zero is accepted.

- [ ] **Step 2: Write failing normalize/sample/unnormalize tests**

Use recording adapter and sampler stubs. Assert each call:

- normalizes history and reference once;
- requests fresh `(1,adapter.horizon,adapter.action_dim)` noise;
- passes adapter-derived slice and configured inference-step count;
- guides terminal `[:,guidance_slice,:]` against all nine reference steps;
- takes only terminal `[:,execution_slice,:]` for execution;
- unnormalizes exactly that tensor;
- returns finite `(1,execution_steps,22)`;
- never calls `policy.predict_action`;
- does not mutate any policy temporal field.

Call twice with one generator and assert two distinct noise draws; repeat with a
fresh generator of the same seed and assert the sequence reproduces.

- [ ] **Step 3: Run service tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guidance.py -q
~~~

Expected: FAIL because the service module is missing.

- [ ] **Step 4: Implement the service**

In `__init__`:

~~~python
self.guidance_slice = adapter.guidance_slice
self.execution_slice = adapter.execution_slice(config.execution_steps)
self.trajectory_shape = (1, adapter.horizon, adapter.action_dim)
~~~

`guide_segment`:

~~~python
global_cond = adapter.global_condition(hand_state_history)
reference_norm = adapter.normalize_reference(hand_reference)
noise = noise_factory(
    self.trajectory_shape,
    global_cond.device,
    global_cond.dtype,
    generator,
)
scheduler = create_ddim_scheduler(adapter.policy.noise_scheduler)
sample = sample_guided_trajectory(
    model=adapter.predict_epsilon,
    scheduler=scheduler,
    initial_noise=noise,
    global_cond=global_cond,
    reference=reference_norm,
    num_inference_steps=config.num_inference_steps,
    guidance_scale=config.guidance_scale,
    guidance_slice=self.guidance_slice,
    eta=config.eta,
)
guided_norm = sample.trajectory[:, self.execution_slice, :]
return adapter.unnormalize_action(guided_norm)
~~~

`verify_zero_guidance` uses the same normalized condition/reference and calls
the zero verifier with the configured inference-step count.

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guidance.py +  tests/test_sim_hand_guided_ddim.py -q

git add +  diffusion_policy/guidance/sim_hand_guidance.py +  tests/test_sim_hand_guidance.py
git commit -m "feat: add policy-derived hand guidance"
~~~

### Task 7: Add the fake executor and dynamic segment coordinator

**Files:**

- Create: `diffusion_policy/guidance/executor.py`
- Create: `diffusion_policy/guidance/coordinator.py`
- Test: `tests/test_sim_hand_guided_coordinator.py`

**Interfaces:**

~~~python
class ExecutionError(RuntimeError):
    segment_id: int
    executed_count: int
    partial_states: torch.Tensor | None

class SegmentExecutor(Protocol):
    def reset(self) -> torch.Tensor
    def execute(
        self,
        segment: torch.Tensor,
        *,
        segment_id: int,
    ) -> torch.Tensor

class FakeSegmentExecutor:
    def __init__(self, initial_history: torch.Tensor) -> None

@dataclass(frozen=True)
class SegmentRecord:
    segment_id: int
    start: int
    stop: int
    real_hand_reference: torch.Tensor
    guided_hand: torch.Tensor
    executed_action: torch.Tensor
    post_states: torch.Tensor
    history_before: torch.Tensor
    history_after: torch.Tensor

@dataclass(frozen=True)
class GuidedRunResult:
    proposal: torch.Tensor
    guided_action: torch.Tensor
    records: tuple[SegmentRecord, ...]
    final_history: torch.Tensor

class GuidedCoordinator:
    def __init__(
        self,
        guidance: SimHandGuidance,
        hand_slice: slice = slice(9, 31),
    ) -> None
    def run(
        self,
        proposal: torch.Tensor,
        executor: SegmentExecutor,
        *,
        hand_reference: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> GuidedRunResult
~~~

- [ ] **Step 1: Write failing dynamic segmentation tests**

Use a recording guidance stub. For `(1,50,31)`, `P=9`, and `E=5` assert ten
exact execution boundaries plus overlapping reference windows `[5i:5i+9]`.
The hand reference therefore covers at least 54 steps. Also use `(1,15,31)`
and assert three boundaries, proving no production constant 50/10.

For every segment:

~~~python
torch.testing.assert_close(
    record.real_hand_reference,
    hand_reference[:, record.start:record.start + 9],
)
torch.testing.assert_close(
    record.executed_action[:, :, :9],
    proposal[:, record.start:record.stop, :9],
)
~~~

Reject wrong rank/batch/action dim, non-finite input, proposal length not
divisible by `execution_steps`, and a full hand reference shorter than
`T-E+P`.

- [ ] **Step 2: Write failing closed-loop/executor tests**

Assert:

~~~python
expected_history = torch.cat(
    (previous_history, post_states),
    dim=1,
)[:, -guidance.adapter.n_obs_steps:, :]
~~~

Test initial reset shape, exact command length, fresh guidance calls, fixed-seed
reproducibility, malformed executor output, `B>1`, and partial
`ExecutionError` stopping all later segments.

- [ ] **Step 3: Run coordinator tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_coordinator.py -q
~~~

Expected: FAIL because executor/coordinator modules are missing.

- [ ] **Step 4: Implement fake executor and coordinator**

Coordinator derives:

~~~python
execution_steps = guidance.config.execution_steps
total_steps = proposal.shape[1]
if total_steps % execution_steps != 0:
    raise ValueError(
        f"Real proposal length {total_steps} is not divisible by "
        f"execution_steps {execution_steps}"
    )
segment_count = total_steps // execution_steps
guidance_steps = guidance.adapter.n_pred_action_steps
required_reference_steps = total_steps - execution_steps + guidance_steps
~~~

Clone proposal once, iterate dynamic boundaries, replace only the hand slice,
execute, validate post states, and update the latest observation window with
`torch.cat(...)[-n_obs_steps:]`. On `ExecutionError`, re-raise with segment
context and do not continue.

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_coordinator.py +  tests/test_sim_hand_guidance.py -q

git add +  diffusion_policy/guidance/executor.py +  diffusion_policy/guidance/coordinator.py +  tests/test_sim_hand_guided_coordinator.py
git commit -m "feat: add dynamic closed-loop coordinator"
~~~

### Task 8: Compose check and fake dry-run services

**Files:**

- Create: `diffusion_policy/guidance/runtime.py`
- Test: `tests/test_sim_hand_guided_runtime.py`

**Interfaces:**

~~~python
@dataclass(frozen=True)
class LoadedGuidedPolicies:
    real: LoadedPolicy
    sim: LoadedPolicy
    real_adapter: RealPolicyAdapter
    sim_adapter: SimPolicyAdapter
    guidance: SimHandGuidance

@dataclass(frozen=True)
class CheckReport:
    real_action_proposal: torch.Tensor
    real_hand_reference: torch.Tensor
    real_action_shape: tuple[int, ...]
    real_hand_reference_shape: tuple[int, ...]
    guided_hand_shape: tuple[int, ...]
    segment_count: int
    sim_horizon: int
    sim_obs_steps: int
    sim_pred_action_steps: int
    guidance_slice: tuple[int, int]
    execution_slice: tuple[int, int]
    timesteps: tuple[int, ...]
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

def run_check(loaded: LoadedGuidedPolicies, *, seed: int) -> CheckReport
def run_dry_run(loaded: LoadedGuidedPolicies, *, seed: int) -> DryRunReport
~~~

- [ ] **Step 1: Write failing composition/check tests**

Assert loader is called twice with no DINO/temporal overrides, adapters derive
their values from policies, and the two normalizer identities differ.

`run_check` must:

1. call Real `predict_action` once;
2. calculate `segment_count = proposal.shape[1] // execution_steps`;
3. call configured `guide_segment` once, including when scale is zero;
4. independently run `verify_zero_guidance`;
5. report actual zero-oracle timesteps.

The current target fake asserts:

~~~python
assert report.real_action_shape == (1, 50, 31)
assert report.segment_count == 10
assert report.guided_hand_shape == (1, 5, 22)
assert report.sim_horizon == 12
assert report.sim_obs_steps == 4
assert report.sim_pred_action_steps == 9
assert report.guidance_slice == (3, 12)
assert report.execution_slice == (3, 8)
assert report.timesteps == EXPECTED_CURRENT_TIMESTEPS
~~~

An alternate fake asserts dynamic fields change and no fixed-current assertion
exists in production.

- [ ] **Step 2: Write failing dry-run tests**

Assert Real inference occurs once total, the retained proposal feeds the
coordinator, the seeded fake executor completes the dynamic segment count, and
the current target completes ten. Verify no hardware module is imported.

- [ ] **Step 3: Run runtime tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_runtime.py -q
~~~

Expected: FAIL because `runtime.py` is missing.

- [ ] **Step 4: Implement runtime composition**

`load_guided_policies` only calls the thin loader and constructors:

~~~python
real = load_workspace_policy(real_checkpoint, device)
sim = load_workspace_policy(sim_checkpoint, device)
real_adapter = RealPolicyAdapter(real)
sim_adapter = SimPolicyAdapter(sim)
guidance = SimHandGuidance(sim_adapter, guidance_config)
~~~

`run_check` retains the public Real proposal and aligned full hand reference,
uses a policy-derived seeded `(1,n_obs_steps,22)` history, runs the first
nine-step reference window, creates a separate seeded full-trajectory noise
tensor for the zero oracle, and returns actual policy/scheduler values.

`run_dry_run` reuses both report tensors, creates `FakeSegmentExecutor`, and
runs `GuidedCoordinator` with a separate `seed+1` generator.

- [ ] **Step 5: Verify GREEN and commit**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_runtime.py +  tests/test_sim_hand_guided_coordinator.py -q

git add +  diffusion_policy/guidance/runtime.py +  tests/test_sim_hand_guided_runtime.py
git commit -m "feat: add guided check and fake dry-run"
~~~

### Task 9: Add the independent CLI and complete regression gate

**Files:**

- Create: `inference_sim_hand_guided.py`
- Test: `tests/test_inference_sim_hand_guided_cli.py`

**Interfaces:**

~~~python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace
def main(argv: Sequence[str] | None = None) -> int
~~~

- [ ] **Step 1: Write failing CLI tests**

Arguments:

~~~text
--mode check|dry-run
--real-checkpoint PATH
--sim-checkpoint PATH
--device DEVICE
--execution-steps INT          default 5
--guidance-scale FLOAT         default 1.0, zero allowed
--num-inference-steps INT      default 8
--eta FLOAT                    default 0.0
--seed INT                     default 0
~~~

Test check/dry-run dispatch, missing checkpoint, negative scale, non-positive
execution/inference steps, nonzero eta, unavailable CUDA, runtime failure exit
1, and import/`--help` without hardware modules. Do not test or expose DINO
environment options.

- [ ] **Step 2: Run CLI tests and verify RED**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_inference_sim_hand_guided_cli.py -q
~~~

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement thin argparse dispatch**

Construct only `SimHandGuidanceConfig`, call `load_guided_policies` once,
then `run_check` or `run_dry_run`. Print actual report values:

~~~python
print(f"real_action_shape={report.real_action_shape}")
print(f"real_hand_reference_shape={report.real_hand_reference_shape}")
print(f"segment_count={report.segment_count}")
print(f"sim_horizon={report.sim_horizon}")
print(f"sim_obs_steps={report.sim_obs_steps}")
print(f"sim_pred_action_steps={report.sim_pred_action_steps}")
print(f"guidance_slice={report.guidance_slice}")
print(f"execution_slice={report.execution_slice}")
print(f"timesteps={report.timesteps}")
~~~

Catch `FileNotFoundError`, `ValueError`, `RuntimeError`, and
`AssertionError`, print one contextual stderr line, and return 1. Do not
import or copy robot observation/execution logic.

- [ ] **Step 4: Run all new feature tests**

~~~bash
$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_guided_checkpoint_loader.py +  tests/test_sim_hand_guided_adapters.py +  tests/test_sim_hand_guided_ddim.py +  tests/test_sim_hand_guidance.py +  tests/test_sim_hand_guided_coordinator.py +  tests/test_sim_hand_guided_runtime.py +  tests/test_inference_sim_hand_guided_cli.py -q
~~~

Expected: PASS.

- [ ] **Step 5: Prove existing systems were untouched**

~~~bash
git diff --name-only 2b40ee6..HEAD -- +  inference_dp.py +  train.py +  diffusion_policy/common/sim_hand_temporal_util.py +  diffusion_policy/policy/diffusion_unet_sim_hand_policy.py +  diffusion_policy/workspace/train_diffusion_unet_sim_hand_workspace.py +  diffusion_policy/config

$GUIDED_TEST_PYTHON -m pytest +  tests/test_sim_hand_temporal.py +  tests/test_diffusion_unet_sim_hand_policy.py +  tests/test_sim_hand_train_smoke.py -q
~~~

Expected: the diff command prints nothing and all existing tests pass.

- [ ] **Step 6: Run static checks**

~~~bash
$GUIDED_TEST_PYTHON -m py_compile +  diffusion_policy/guidance/checkpoint_loader.py +  diffusion_policy/guidance/real_adapter.py +  diffusion_policy/guidance/sim_adapter.py +  diffusion_policy/guidance/guided_ddim.py +  diffusion_policy/guidance/sim_hand_guidance.py +  diffusion_policy/guidance/executor.py +  diffusion_policy/guidance/coordinator.py +  diffusion_policy/guidance/runtime.py +  inference_sim_hand_guided.py

git diff --check
git status --short
~~~

- [ ] **Step 7: Run actual current-checkpoint gates when paths are available**

~~~bash
test -f "$REAL_CKPT_PATH"
test -f "$SIM_CKPT_PATH"

$GUIDED_TEST_PYTHON inference_sim_hand_guided.py +  --mode check +  --real-checkpoint "$REAL_CKPT_PATH" +  --sim-checkpoint "$SIM_CKPT_PATH" +  --device cuda:0 +  --execution-steps 5 +  --guidance-scale 0.0 +  --num-inference-steps 8 +  --eta 0.0 +  --seed 7

$GUIDED_TEST_PYTHON inference_sim_hand_guided.py +  --mode dry-run +  --real-checkpoint "$REAL_CKPT_PATH" +  --sim-checkpoint "$SIM_CKPT_PATH" +  --device cuda:0 +  --execution-steps 5 +  --guidance-scale 1.0 +  --num-inference-steps 8 +  --eta 0.0 +  --seed 7
~~~

Expected current-checkpoint output includes Real `(1,50,31)`, a full hand
reference covering at least 54 steps, ten segments, Sim `4/12/9`, guidance
slice `(3,12)`, execution slice `(3,8)`, the current eight generated
timesteps, zero-oracle errors within tolerance, and ten completed fake segments.

- [ ] **Step 8: Commit the CLI**

~~~bash
git add +  inference_sim_hand_guided.py +  tests/test_inference_sim_hand_guided_cli.py
git commit -m "feat: add minimal sim-hand guided inference"
~~~
