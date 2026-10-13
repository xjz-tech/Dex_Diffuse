#!/usr/bin/env python3
"""Run Sim-Hand Diffusion Policy continuously in the bulb2 Isaac Gym task."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


# Isaac Gym must be imported before torch in its Python 3.8 environment.
try:
    import isaacgym  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "Isaac Gym is unavailable. Run this entrypoint through eval/eval.sh, "
        "which adds dex-controller/third_party/isaacgym/python to PYTHONPATH."
    ) from exc

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from termcolor import cprint  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
LOCAL_MANIPTRANS_ROOT = PROJECT_ROOT / "maniptrans_envs"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from episode_stats import (  # noqa: E402
    EpisodeRecorder,
    censored_timeout_records,
    format_hold_summary,
    summarize_episode_records,
)
from ipc import connect_unix, recv_message, send_message  # noqa: E402
from policy_observation import (  # noqa: E402
    compose_policy_observation,
    observation_mode_from_dim,
    validate_policy_spec,
)
from recording import (  # noqa: E402
    RecordingConfig,
    RecordingConstructionHooks,
    RecordingRuntime,
    parse_vec3,
)


HAND_DIM = 22
OBS_STEPS = 4
SHARPA_URDF_NAME = "v3right_sharpa_wave-forhammer5.urdf"


def _add_reset_arguments(parser):
    # 注意（给后续维护者/模型）：严格默认值用于 RL 训练，请保留。
    # 当前 diffusion policy / guidance 的 hold 评估不用这套默认值。
    # 调用方应显式传入 eval/xjz_test.sh 对应的评估配置：
    # --failure-obj-pos-thres-m 0.05 --failure-tip-pos-thres-m 0.1
    # --failure-tolerance-scale 10000.0 --fixed-tolerance-steps 20000
    # 并对齐轨迹范围等其他评估参数，避免直接调用时误用 RL 阈值。
    group = parser.add_argument_group("episode reset conditions")
    group.add_argument("--failure-obj-pos-thres-m", type=float, default=0.012)
    group.add_argument("--failure-tip-pos-thres-m", type=float, default=0.036)
    group.add_argument("--failure-obj-rot-thres-deg", type=float, default=180.0)
    group.add_argument("--invalid-obj-pos-thres-m", type=float, default=0.15)
    group.add_argument("--failure-tolerance-scale", type=float, default=1.0)
    group.add_argument("--fixed-tolerance-steps", type=int, default=200)
    group.add_argument("--traj-steps-limit", type=int, default=12000)
    group.add_argument(
        "--reset-on-reach-goal",
        type=int,
        choices=(0, 1),
        default=0,
        help="1 resets an environment after every stable reach_final_goal.",
    )
    group.add_argument(
        "--cross-trajectory-goal-prob",
        type=float,
        default=None,
        help=(
            "Probability of selecting a target from another demonstration "
            "after reach_final_goal. Omit to preserve the task config."
        ),
    )


def _validate_reset_args(args):
    positive_float_args = (
        "failure_obj_pos_thres_m",
        "failure_tip_pos_thres_m",
        "failure_obj_rot_thres_deg",
        "invalid_obj_pos_thres_m",
        "failure_tolerance_scale",
    )
    for name in positive_float_args:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("--%s must be finite and positive" % name.replace("_", "-"))
    if args.fixed_tolerance_steps <= 0:
        raise ValueError("--fixed-tolerance-steps must be positive")
    if args.traj_steps_limit <= 0:
        raise ValueError("--traj-steps-limit must be positive")
    if args.failure_obj_rot_thres_deg > 180.0:
        raise ValueError("--failure-obj-rot-thres-deg must be <= 180")
    if (
        args.cross_trajectory_goal_prob is not None
        and (
            not np.isfinite(args.cross_trajectory_goal_prob)
            or not 0.0 <= args.cross_trajectory_goal_prob <= 1.0
        )
    ):
        raise ValueError("--cross-trajectory-goal-prob must be in [0, 1]")
    if args.invalid_obj_pos_thres_m < args.failure_obj_pos_thres_m:
        raise ValueError(
            "--invalid-obj-pos-thres-m must be >= --failure-obj-pos-thres-m"
        )
    mass = args.object_mass_kg
    friction_min = args.object_friction_min
    friction_max = args.object_friction_max
    if mass is not None and (not np.isfinite(mass) or mass <= 0.0):
        raise ValueError("--object-mass-kg must be finite and positive")
    if (friction_min is None) != (friction_max is None):
        raise ValueError(
            "--object-friction-min and --object-friction-max must be set together"
        )
    if friction_min is not None and not (
        np.isfinite(friction_min)
        and np.isfinite(friction_max)
        and 0.0 <= friction_min < friction_max
    ):
        raise ValueError("--object-friction range must be finite and increasing")
    if args.rotation_preview_raw and not args.recording:
        raise ValueError("--rotation-preview-raw requires --recording")
    if args.rotation_preview_stride <= 0 or args.rotation_preview_frames <= 0:
        raise ValueError("rotation preview stride and frame count must be positive")


def _reset_overrides_from_args(args):
    tip_threshold = float(args.failure_tip_pos_thres_m)
    overrides = {
        "failureObjPosThres": float(args.failure_obj_pos_thres_m),
        "failureThumbTipPosThres": tip_threshold,
        "failureIndexTipPosThres": tip_threshold,
        "failureMiddleTipPosThres": tip_threshold,
        "failurePinkyTipPosThres": tip_threshold,
        "failureRingTipPosThres": tip_threshold,
        "failureObjRotThres": float(args.failure_obj_rot_thres_deg),
        "invalidObjPosThres": float(args.invalid_obj_pos_thres_m),
        "FailureToleranceScale": float(args.failure_tolerance_scale),
        "fixedToleranceSteps": int(args.fixed_tolerance_steps),
        "trajStepsLimit": int(args.traj_steps_limit),
        "resetOnReachGoal": bool(args.reset_on_reach_goal),
    }
    if args.cross_trajectory_goal_prob is not None:
        # Match dex-controller's reach-goal branch while allowing evaluation
        # to set its random cross-demo selection probability to zero.
        overrides["enableCrossTrajectoryReset"] = True
        overrides["crossTrajectoryGoalProb"] = float(
            args.cross_trajectory_goal_prob
        )
    return overrides


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Continuously evaluate a Sim-Hand DP checkpoint in bulb2."
    )
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--record-env",
        type=int,
        default=0,
        help="Zero-based environment index shown in the viewer and MP4.",
    )
    parser.add_argument("--data-indices", default="000,001")
    parser.add_argument("--nokov3-data-dir", required=True)
    parser.add_argument("--nokov3-retarget-dir", required=True)
    parser.add_argument("--sharpa-asset-dir", required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 means run forever (the default).",
    )
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--max-failure-episodes",
        type=int,
        default=0,
        help="Stop after this many terminal failures. 0 means never stop for this.",
    )
    parser.add_argument(
        "--episode-log",
        default="",
        help="Optional JSONL path for per-episode env/length/reason records.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Label stored in --episode-log records.",
    )
    parser.add_argument(
        "--initial-state-dump",
        default="",
        help="Optional compressed NPZ snapshot written after the initial reset.",
    )
    parser.add_argument(
        "--first-episode-only",
        action="store_true",
        help="Count only the first episode of each env (ignore post-reset trials).",
    )
    parser.add_argument(
        "--censor-unfinished-at-cap",
        action="store_true",
        help="At --max-steps, record still-running first episodes as timeout at the cap.",
    )
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--record-dir", default=str(EVAL_DIR / "record"))
    parser.add_argument("--record-width", type=int, default=1280)
    parser.add_argument("--record-height", type=int, default=720)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument(
        "--record-camera-position",
        default="-0.10,0.55,0.10",
        help="Fixed camera xyz in the environment frame.",
    )
    parser.add_argument(
        "--record-camera-target",
        default="-0.10,0.00,-0.14",
        help="Fixed camera look-at xyz in the environment frame.",
    )
    parser.add_argument("--record-camera-fov", type=float, default=60.0)
    parser.add_argument("--record-axis-length", type=float, default=0.20)
    parser.add_argument("--record-axis-thickness", type=float, default=0.008)
    parser.add_argument(
        "--object-mass-kg",
        type=float,
        default=None,
        help="Fix the manipulated object mass. Disables object mass scaling.",
    )
    parser.add_argument("--object-friction-min", type=float, default=None)
    parser.add_argument("--object-friction-max", type=float, default=None)
    parser.add_argument(
        "--rotation-preview-raw",
        default="",
        help="Optional raw RGB file of sim-time frames from the recorded env.",
    )
    parser.add_argument("--rotation-preview-stride", type=int, default=3)
    parser.add_argument("--rotation-preview-frames", type=int, default=300)
    _add_reset_arguments(parser)
    parser.set_defaults(randomize_demo_on_failure=True)
    parser.add_argument(
        "--no-randomize-demo-on-failure",
        action="store_false",
        dest="randomize_demo_on_failure",
    )
    return parser.parse_args()


def _expand_data_indices(spec):
    values = []
    seen = set()
    for raw_token in spec.replace(" ", ",").split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token.startswith("v3:bulb2@"):
            token = token.rsplit("@", 1)[1]
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError("invalid DATA_INDICES range: %r" % token)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError("descending DATA_INDICES range: %r" % token)
            width = max(len(start_text), len(end_text), 3)
            expanded = [str(value).zfill(width) for value in range(start, end + 1)]
        else:
            if not token.isdigit():
                raise ValueError("invalid DATA_INDICES item: %r" % token)
            expanded = [token.zfill(3)]
        for value in expanded:
            full = "v3:bulb2@%s" % value
            if full not in seen:
                seen.add(full)
                values.append(full)
    if not values:
        raise ValueError("DATA_INDICES must select at least one bulb2 trajectory")
    return values


def _validate_inputs(args, data_indices):
    controller_root = Path(args.controller_root).expanduser().resolve()
    # Keep repository-local symlink paths visible instead of resolving them
    # back to dex-controller.  The paths still need to be absolute because the
    # simulator changes cwd to controller_root below.
    config_path = Path(os.path.abspath(os.path.expanduser(args.sim_config)))
    data_root = Path(os.path.abspath(os.path.expanduser(args.nokov3_data_dir)))
    retarget_root = Path(
        os.path.abspath(os.path.expanduser(args.nokov3_retarget_dir))
    )
    if not controller_root.is_dir():
        raise FileNotFoundError("dex-controller root not found: %s" % controller_root)
    if not config_path.is_file():
        raise FileNotFoundError("simulation config not found: %s" % config_path)
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.record_env < 0 or args.record_env >= args.num_envs:
        raise ValueError(
            "--record-env must be in [0, --num-envs), got %d for %d envs"
            % (args.record_env, args.num_envs)
        )
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if args.max_failure_episodes < 0:
        raise ValueError("--max-failure-episodes cannot be negative")
    if args.print_every < 0:
        raise ValueError("--print-every cannot be negative")
    if args.recording and args.headless:
        print(
            "[sim] headless camera recording; no viewer window",
            flush=True,
        )

    for full_index in data_indices:
        index = full_index.rsplit("@", 1)[1]
        source = data_root / "data" / "bulb2" / (index + ".h5")
        retarget = (
            retarget_root
            / "mano2sharpa_rh"
            / "bulb2"
            / (index + ".pkl")
        )
        if not source.is_file():
            raise FileNotFoundError("bulb2 source trajectory not found: %s" % source)
        if not retarget.is_file():
            raise FileNotFoundError("bulb2 retarget trajectory not found: %s" % retarget)
    return controller_root, config_path, data_root, retarget_root


def _resolve_sharpa_urdf(asset_dir):
    asset_dir = Path(os.path.abspath(os.path.expanduser(asset_dir)))
    urdf_path = asset_dir / SHARPA_URDF_NAME
    if not urdf_path.is_file():
        raise FileNotFoundError("SharpA URDF not found: %s" % urdf_path)
    if not (asset_dir / "meshes").is_dir():
        raise FileNotFoundError("SharpA mesh directory not found: %s" % asset_dir)
    return urdf_path


def _import_local_maniptrans(controller_root):
    """Load the evaluator's private maniptrans_envs copy."""
    expected_lib = (LOCAL_MANIPTRANS_ROOT / "lib" / "__init__.py").resolve()
    if not expected_lib.is_file():
        raise FileNotFoundError(
            "local maniptrans_envs copy not found: %s" % LOCAL_MANIPTRANS_ROOT
        )

    # Keep dex-controller available for legacy top-level dependencies, while
    # resolving the maniptrans_envs namespace from Dex_Diffuse first.
    controller_path = str(controller_root)
    project_path = str(PROJECT_ROOT)
    for source_path in (controller_path, project_path):
        while source_path in sys.path:
            sys.path.remove(source_path)
    sys.path.insert(0, controller_path)
    sys.path.insert(0, project_path)

    import maniptrans_envs.lib as maniptrans_envs_lib

    loaded_lib = Path(maniptrans_envs_lib.__file__).resolve()
    if loaded_lib != expected_lib:
        raise RuntimeError(
            "wrong maniptrans_envs source loaded: %s (expected %s)"
            % (loaded_lib, expected_lib)
        )
    return maniptrans_envs_lib


def _install_sharpa_asset_override(urdf_path):
    """Point this process's SharpA factory entry at the evaluator asset path."""
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

    registry_key = "sharpa_rh"
    original_class = DexHandFactory._registry[registry_key]
    resolved_urdf = os.path.abspath(os.path.expanduser(str(urdf_path)))

    class EvalSharpaRH(original_class):
        def __init__(self):
            super().__init__()
            # DexHand.urdf_path preserves an absolute _urdf_path unchanged.
            self._urdf_path = resolved_urdf

    DexHandFactory._registry[registry_key] = EvalSharpaRH
    return DexHandFactory, registry_key, original_class


def _restore_sharpa_asset_override(override):
    factory, registry_key, original_class = override
    factory._registry[registry_key] = original_class


def _load_manifest(config_path):
    manifest_path = config_path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError("simulation-data manifest not found: %s" % manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("unsupported sim-data manifest schema")
    if int(manifest.get("dof", -1)) != HAND_DIM:
        raise ValueError("sim-data manifest does not describe a 22-DoF hand")
    return manifest


def _make_task_config(
    config_path,
    num_envs,
    data_indices,
    data_root,
    retarget_root,
    reset_overrides=None,
):
    full_cfg = OmegaConf.load(str(config_path))
    OmegaConf.set_struct(full_cfg, False)
    task_cfg = full_cfg.task
    env_cfg = task_cfg.env
    env_cfg.numEnvs = int(num_envs)
    env_cfg.dataIndices = list(data_indices)
    env_cfg.nokov3DataDir = str(data_root)
    env_cfg.nokov3RetargetDir = str(retarget_root)
    env_cfg.training = False
    # Keep collection-time forces/domain randomization so evaluation matches the
    # distribution used to create the Diffusion Policy training set.
    env_cfg.simDataCollection = True
    env_cfg.randomStateInit = True
    # DP predicts robot/target_after in absolute radians.  The task's "abs"
    # path accepts those targets after joint-limit normalization.
    env_cfg.actStyle = "abs"
    if reset_overrides is not None:
        for key, value in reset_overrides.items():
            env_cfg[key] = value
    return task_cfg


def _apply_sim2real_randomization(task_cfg, args):
    """Eval-only object mass/friction. Does not write the training config."""
    if args.object_mass_kg is None and args.object_friction_min is None:
        return
    obj = task_cfg.task.randomization_params.actor_params.manip_obj
    if args.object_friction_min is not None:
        obj.rigid_shape_properties.friction.range = [
            float(args.object_friction_min),
            float(args.object_friction_max),
        ]
    if args.object_mass_kg is not None:
        # Scaling by 1 keeps the mass written into the randomization baseline.
        obj.rigid_body_properties.mass.operation = "scaling"
        obj.rigid_body_properties.mass.range = [1.0, 1.0]
    print(
        "[sim] sim2real domain | object_mass_kg=%s object_friction=[%s, %s]"
        % (args.object_mass_kg, args.object_friction_min, args.object_friction_max),
        flush=True,
    )


def _scale_inertia(inertia, ratio):
    for row in "xyz":
        axis = getattr(inertia, row)
        for column in "xyz":
            setattr(axis, column, float(getattr(axis, column)) * ratio)


def _lock_object_mass(env, mass_kg):
    """Set every bulb to mass_kg and make later mass randomization keep it."""
    originals = env.original_props.get("rigid_body_properties", {})
    for index, env_ptr in enumerate(env.envs):
        actor = env.gym.find_actor_handle(env_ptr, "manip_obj")
        props = env.gym.get_actor_rigid_body_properties(env_ptr, actor)
        body = props[0]
        old_mass = float(body.mass)
        if old_mass <= 0.0:
            raise RuntimeError("object mass must be positive before the sim2real lock")
        _scale_inertia(body.inertia, float(mass_kg) / old_mass)
        body.mass = float(mass_kg)
        env.gym.set_actor_rigid_body_properties(env_ptr, actor, props)
        cached = originals.get("%d_%d" % (index, actor))
        if cached is None:
            raise RuntimeError(
                "missing randomization baseline for env %d actor %d" % (index, actor)
            )
        for record in cached:
            if "mass" in record:
                record["mass"] = float(mass_kg)
    env._update_object_mass()


def _read_object_domain(env):
    masses = []
    frictions = []
    for env_ptr in env.envs:
        actor = env.gym.find_actor_handle(env_ptr, "manip_obj")
        masses.append(float(env.gym.get_actor_rigid_body_properties(env_ptr, actor)[0].mass))
        frictions.append(
            [
                float(shape.friction)
                for shape in env.gym.get_actor_rigid_shape_properties(env_ptr, actor)
            ]
        )
    return np.asarray(masses), np.asarray(frictions)


class _RotationPreview:
    def __init__(self, args):
        self.path = Path(args.rotation_preview_raw) if args.rotation_preview_raw else None
        self.stride = int(args.rotation_preview_stride)
        self.limit = int(args.rotation_preview_frames)
        self.count = 0
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("wb")

    def write(self, rgb):
        if self._handle is None or self.count >= self.limit:
            return
        frame = np.ascontiguousarray(rgb)
        self._handle.write(frame.tobytes())
        self.count += 1
        if self.count == self.limit:
            self._handle.flush()
            print(
                "[sim] rotation preview raw ready | frames=%d path=%s"
                % (self.count, self.path),
                flush=True,
            )

    def close(self):
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


def _write_rotation_preview(preview, recording_runtime, step):
    if preview.path is None or preview.count >= preview.limit:
        return
    if step % preview.stride != 0:
        return
    rgb = getattr(recording_runtime, "last_rgb", None)
    if rgb is not None:
        preview.write(rgb)


def _check_sim2real_domain(env, args):
    masses, frictions = _read_object_domain(env)
    print(
        "[sim] sim2real actual | mass min/max=%.6f/%.6f kg | "
        "friction min/max=%.4f/%.4f | envs=%d"
        % (masses.min(), masses.max(), frictions.min(), frictions.max(), masses.size),
        flush=True,
    )
    if args.object_mass_kg is not None and not np.allclose(
        masses, args.object_mass_kg, atol=1e-5
    ):
        raise RuntimeError("object mass was not locked to the sim2real value")
    if args.object_friction_min is not None and (
        frictions.min() < args.object_friction_min - 1e-4
        or frictions.max() > args.object_friction_max + 1e-4
    ):
        raise RuntimeError("object friction left the sim2real interval")


def _make_recording_config(args):
    config = RecordingConfig(
        output_dir=Path(args.record_dir).expanduser().resolve(),
        width=args.record_width,
        height=args.record_height,
        fps=args.record_fps,
        camera_position=parse_vec3(args.record_camera_position),
        camera_target=parse_vec3(args.record_camera_target),
        camera_horizontal_fov=args.record_camera_fov,
        axis_length=args.record_axis_length,
        axis_thickness=args.record_axis_thickness,
        environment_index=args.record_env,
    )
    config.validate()
    return config


class PolicyClient:
    def __init__(self, socket_path, timeout):
        self.socket = connect_unix(socket_path, timeout_seconds=timeout)
        self.socket.settimeout(float(timeout))
        self.request_id = 0
        self.info = self._hello()

    def _next_id(self):
        self.request_id += 1
        return self.request_id

    def _request(self, message_type, array=None):
        request_id = self._next_id()
        send_message(
            self.socket,
            {"type": message_type, "request_id": request_id},
            array,
        )
        response, result = recv_message(self.socket)
        if response.get("request_id") != request_id:
            raise RuntimeError("inference response id mismatch")
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "inference server error"))
        return response, result

    def _hello(self):
        response, _ = self._request("hello")
        try:
            spec = validate_policy_spec(response.get("spec"))
        except ValueError as exc:
            raise RuntimeError(
                "checkpoint temporal spec mismatch: %s" % exc
            ) from exc
        self.obs_dim = spec["obs_dim"]
        self.n_action_steps = spec["n_action_steps"]
        self.observation_mode = observation_mode_from_dim(self.obs_dim)
        return response

    def predict(self, history):
        response, action = self._request(
            "predict",
            np.asarray(history, dtype=np.float32),
        )
        if action is None or action.shape != (
            history.shape[0],
            self.n_action_steps,
            HAND_DIM,
        ):
            raise RuntimeError(
                "invalid action chunk from inference server: %r"
                % (None if action is None else action.shape,)
            )
        return action, float(response.get("inference_seconds", 0.0))

    def close(self):
        if self.socket is None:
            return
        try:
            self._request("shutdown")
        except Exception:
            pass
        try:
            self.socket.close()
        finally:
            self.socket = None


def _validate_environment(env, manifest):
    dof_names = list(env.dexhand.dof_names)
    expected_names = manifest.get("metadata", {}).get("dof_names")
    if expected_names is not None and dof_names != list(expected_names):
        raise ValueError(
            "SharpA DOF order differs from the training manifest:\n"
            "env=%r\nmanifest=%r" % (dof_names, expected_names)
        )
    if int(env.dexhand.n_dofs) != HAND_DIM:
        raise ValueError("Isaac Gym task does not expose 22 SharpA DOFs")
    if env.act_style != "abs":
        raise ValueError("evaluator requires task actStyle=abs")
    return (
        env.dexhand_dof_lower_limits.detach().to(env.device),
        env.dexhand_dof_upper_limits.detach().to(env.device),
    )


def _absolute_targets_to_env_action(targets, lower, upper, device):
    target_tensor = torch.as_tensor(targets, device=device, dtype=torch.float32)
    target_tensor = torch.maximum(torch.minimum(target_tensor, upper), lower)
    action = 2.0 * (target_tensor - lower) / (upper - lower) - 1.0
    return torch.clamp(action, -1.0, 1.0)


def _destroy_environment(env):
    if env is None:
        return
    gym = getattr(env, "gym", None)
    if gym is None:
        return
    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        try:
            gym.destroy_viewer(viewer)
        except Exception as exc:
            print("[sim] viewer cleanup warning: %s" % exc, flush=True)
        env.viewer = None
    sim = getattr(env, "sim", None)
    if sim is not None:
        try:
            gym.destroy_sim(sim)
        except Exception as exc:
            print("[sim] simulator cleanup warning: %s" % exc, flush=True)
        env.sim = None


def _current_policy_observation(env, mode):
    qpos = env._q.detach().cpu().numpy().astype(np.float32, copy=True)
    target_before = (
        env.curr_targets.detach().cpu().numpy().astype(np.float32, copy=True)
    )
    return compose_policy_observation(qpos, target_before, mode)


def _tensor_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _inertia_matrix(body_property):
    inertia = body_property.inertia
    return [
        [float(getattr(getattr(inertia, row), column)) for column in "xyz"]
        for row in "xyz"
    ]


def _dump_initial_state(env, observation, destination):
    """Archive the post-reset state and actual Isaac Gym actor properties."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    object_mass = []
    object_scale = []
    object_inertia = []
    object_com = []
    object_friction = []
    object_rolling_friction = []
    object_torsion_friction = []
    object_restitution = []
    hand_mass = []
    hand_friction = []
    hand_rolling_friction = []
    hand_torsion_friction = []
    hand_restitution = []
    hand_dof_stiffness = []
    hand_dof_damping = []

    for env_ptr in env.envs:
        hand_handle = env.gym.find_actor_handle(env_ptr, "dexhand")
        object_handle = env.gym.find_actor_handle(env_ptr, "manip_obj")
        object_body = env.gym.get_actor_rigid_body_properties(
            env_ptr, object_handle
        )[0]
        object_shapes = env.gym.get_actor_rigid_shape_properties(
            env_ptr, object_handle
        )
        hand_bodies = env.gym.get_actor_rigid_body_properties(
            env_ptr, hand_handle
        )
        hand_shapes = env.gym.get_actor_rigid_shape_properties(
            env_ptr, hand_handle
        )
        hand_dofs = env.gym.get_actor_dof_properties(env_ptr, hand_handle)

        object_mass.append(float(object_body.mass))
        object_scale.append(float(env.gym.get_actor_scale(env_ptr, object_handle)))
        object_inertia.append(_inertia_matrix(object_body))
        object_com.append(
            [
                float(object_body.com.x),
                float(object_body.com.y),
                float(object_body.com.z),
            ]
        )
        object_friction.append([float(item.friction) for item in object_shapes])
        object_rolling_friction.append(
            [float(item.rolling_friction) for item in object_shapes]
        )
        object_torsion_friction.append(
            [float(item.torsion_friction) for item in object_shapes]
        )
        object_restitution.append(
            [float(item.restitution) for item in object_shapes]
        )
        hand_mass.append([float(item.mass) for item in hand_bodies])
        hand_friction.append([float(item.friction) for item in hand_shapes])
        hand_rolling_friction.append(
            [float(item.rolling_friction) for item in hand_shapes]
        )
        hand_torsion_friction.append(
            [float(item.torsion_friction) for item in hand_shapes]
        )
        hand_restitution.append(
            [float(item.restitution) for item in hand_shapes]
        )
        hand_dof_stiffness.append(np.asarray(hand_dofs["stiffness"]).copy())
        hand_dof_damping.append(np.asarray(hand_dofs["damping"]).copy())

    arrays = {
        "observation": np.asarray(observation).copy(),
        "q": _tensor_numpy(env._q),
        "qd": _tensor_numpy(env._qd),
        "wrist": _tensor_numpy(env._base_state),
        "object": _tensor_numpy(env._manip_obj_root_state),
        "target": _tensor_numpy(env.curr_targets),
        "demo": _tensor_numpy(env.envidx_to_demoidx),
        "frame": _tensor_numpy(env.global_cur_idx),
        "progress": _tensor_numpy(env.progress_buf),
        "torch_cuda_rng": _tensor_numpy(torch.cuda.get_rng_state()),
        "random_force_probability": _tensor_numpy(env.random_force_prob),
        "cached_object_mass": _tensor_numpy(env.manip_obj_mass),
        "object_mass": np.asarray(object_mass),
        "object_scale": np.asarray(object_scale),
        "object_inertia": np.asarray(object_inertia),
        "object_com": np.asarray(object_com),
        "object_friction": np.asarray(object_friction),
        "object_rolling_friction": np.asarray(object_rolling_friction),
        "object_torsion_friction": np.asarray(object_torsion_friction),
        "object_restitution": np.asarray(object_restitution),
        "hand_mass": np.asarray(hand_mass),
        "hand_friction": np.asarray(hand_friction),
        "hand_rolling_friction": np.asarray(hand_rolling_friction),
        "hand_torsion_friction": np.asarray(hand_torsion_friction),
        "hand_restitution": np.asarray(hand_restitution),
        "hand_dof_stiffness": np.asarray(hand_dof_stiffness),
        "hand_dof_damping": np.asarray(hand_dof_damping),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(str(temporary), str(destination))
    print(
        "[sim] initial state archived | path=%s fields=%d envs=%d"
        % (destination, len(arrays), len(env.envs)),
        flush=True,
    )


def _print_policy_info(client):
    checkpoint = client.info["checkpoint"]
    print(
        "[sim] policy connected | weights=%s step=%s epoch=%s obs_dim=%d "
        "mode=%s exec=%d pred=%s"
        % (
            checkpoint["weight_source"],
            checkpoint["global_step"],
            checkpoint["epoch"],
            client.obs_dim,
            client.observation_mode,
            client.n_action_steps,
            client.info["spec"]["n_pred_action_steps"],
        ),
        flush=True,
    )
    if checkpoint.get("salvaged"):
        print(
            "[sim] WARNING: checkpoint EMA is incomplete; using its strictly "
            "recovered base model.",
            flush=True,
        )


def _print_hold_summary(recorder):
    print(
        format_hold_summary("[sim] hold summary", summarize_episode_records(recorder.records)),
        flush=True,
    )


def run(args):
    _validate_reset_args(args)
    data_indices = _expand_data_indices(args.data_indices)
    controller_root, config_path, data_root, retarget_root = _validate_inputs(
        args,
        data_indices,
    )
    manifest = _load_manifest(config_path)
    sharpa_urdf = _resolve_sharpa_urdf(args.sharpa_asset_dir)
    recording_config = _make_recording_config(args) if args.recording else None

    # Controller assets and several legacy relative paths assume this cwd.
    os.chdir(str(controller_root))
    maniptrans_envs_lib = _import_local_maniptrans(controller_root)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    task_cfg = _make_task_config(
        config_path,
        args.num_envs,
        data_indices,
        data_root,
        retarget_root,
        _reset_overrides_from_args(args),
    )
    _apply_sim2real_randomization(task_cfg, args)
    print(
        "[sim] creating bulb2 Isaac Gym task | envs=%d trajectories=%d "
        "headless=%s record_env=%d"
        % (args.num_envs, len(data_indices), args.headless, args.record_env),
        flush=True,
    )
    env = None
    client = None
    recording_runtime = None
    preview = None
    try:
        construction_hooks = None
        sharpa_asset_override = _install_sharpa_asset_override(sharpa_urdf)
        try:
            if recording_config is not None:
                task_class = maniptrans_envs_lib.TASK_MAP[str(task_cfg.name)]
                construction_hooks = RecordingConstructionHooks(recording_config)
                construction_hooks.install(task_class)
            env = maniptrans_envs_lib.make(
                sim_device=args.sim_device,
                rl_device=args.rl_device,
                graphics_device_id=args.graphics_device_id,
                multi_gpu=False,
                cfg=task_cfg,
                display=False,
                record=args.recording,
                has_headless_arg=True,
                headless=args.headless,
            )
        finally:
            if construction_hooks is not None:
                construction_hooks.restore()
            _restore_sharpa_asset_override(sharpa_asset_override)
        if args.object_mass_kg is not None:
            _lock_object_mass(env, args.object_mass_kg)
        env.compute_observations()
        env.reset()
        if args.object_mass_kg is not None or args.object_friction_min is not None:
            _check_sim2real_domain(env, args)
        lower, upper = _validate_environment(env, manifest)
        if recording_config is not None:
            recording_runtime = RecordingRuntime(env, recording_config)

        client = PolicyClient(args.socket_path, args.request_timeout)
        _print_policy_info(client)
        recorder = EpisodeRecorder(
            args.episode_log or None,
            args.max_failure_episodes,
        )
        if args.max_failure_episodes:
            print(
                "[sim] stopping after %d failure episodes"
                % args.max_failure_episodes,
                flush=True,
            )

        observation = _current_policy_observation(env, client.observation_mode)
        if args.initial_state_dump:
            _dump_initial_state(env, observation, args.initial_state_dump)
        action_steps = client.n_action_steps
        history = np.repeat(observation[:, None, :], OBS_STEPS, axis=1)
        action_plan = np.zeros(
            (args.num_envs, action_steps, HAND_DIM),
            dtype=np.float32,
        )
        plan_position = np.full(args.num_envs, action_steps, dtype=np.int64)
        episode_steps = np.zeros(args.num_envs, dtype=np.int64)
        episode_number = np.zeros(args.num_envs, dtype=np.int64)
        first_episode_recorded = np.zeros(args.num_envs, dtype=bool)
        if args.first_episode_only:
            print("[sim] counting only the first episode of each env", flush=True)
        if args.censor_unfinished_at_cap:
            if args.max_steps <= 0:
                raise ValueError("--censor-unfinished-at-cap requires --max-steps > 0")
            print(
                "[sim] unfinished first episodes will be recorded as timeout at %d"
                % args.max_steps,
                flush=True,
            )
        total_inference_time = 0.0
        inference_calls = 0
        object_drop_count = 0
        object_pose_reset_count = 0
        started = time.monotonic()

        print(
            "[sim] running continuously; press Ctrl-C or close/ESC the viewer to stop",
            flush=True,
        )
        step = 0
        preview = _RotationPreview(args)
        with torch.no_grad():
            while args.max_steps == 0 or step < args.max_steps:
                if (
                    recording_runtime is not None
                    and recording_runtime.poll_viewer_events()
                ):
                    break
                needs_plan = np.flatnonzero(plan_position >= action_steps)
                if needs_plan.size:
                    predicted, inference_seconds = client.predict(history[needs_plan])
                    action_plan[needs_plan] = predicted
                    plan_position[needs_plan] = 0
                    total_inference_time += inference_seconds
                    inference_calls += 1

                selected_targets = action_plan[
                    np.arange(args.num_envs),
                    plan_position,
                ]
                env_action = _absolute_targets_to_env_action(
                    selected_targets,
                    lower,
                    upper,
                    env.device,
                )
                _obs, rewards, dones, infos = env.step(env_action)
                plan_position += 1
                episode_steps += 1
                step += 1

                # Q/Esc/window close is recorded by the viewer callback during
                # env.step(). Leave the loop normally so MP4 finalization does
                # not depend on propagating SystemExit through Isaac Gym.
                if (
                    recording_runtime is not None
                    and recording_runtime.poll_viewer_events()
                ):
                    break

                done_mask = dones.to(env.device).bool()
                failures = env.failure_buf.detach().clone().bool()
                successes = env.success_buf.detach().clone().bool()
                reach_goal = env.reach_final_goal.detach().clone().bool()

                if torch.any(reach_goal):
                    reached = reach_goal.nonzero(as_tuple=False).flatten().tolist()
                    cprint(
                        "[sim] reached target | step=%d env_ids=%s" % (step, reached),
                        "green",
                        flush=True,
                    )

                if recording_runtime is not None:
                    # Capture the terminal pose before reset_done replaces it.
                    recording_runtime.update_axes()
                    recording_runtime.capture_if_active()
                    _write_rotation_preview(preview, recording_runtime, step)

                reset_ids_np = np.empty(0, dtype=np.int64)
                if torch.any(done_mask):
                    done_ids = done_mask.nonzero(as_tuple=False).flatten()
                    # The task has no dedicated object-dropped signal.  Its
                    # terminal failure flag is the closest available measure:
                    # it covers severe object/target divergence and sustained
                    # pose or fingertip tracking failure.
                    terminal_failures = done_mask & failures
                    object_drop_count += int(terminal_failures.sum().item())
                    timeout_value = infos.get(
                        "time_outs",
                        torch.zeros_like(done_mask),
                    )
                    if not isinstance(timeout_value, torch.Tensor):
                        timeout_value = torch.as_tensor(
                            timeout_value,
                            device=env.device,
                        )
                    timeouts = timeout_value.to(env.device).bool()
                    for env_id in done_ids.tolist():
                        if bool(failures[env_id]):
                            reason = "failure"
                        elif bool(successes[env_id]):
                            reason = "success"
                        elif bool(timeouts[env_id]):
                            reason = "timeout"
                        else:
                            reason = "done"
                        if args.first_episode_only and int(episode_number[env_id]) != 0:
                            continue
                        print(
                            "[sim] episode end | env=%d episode=%d length=%d "
                            "reason=%s reward=%.5f"
                            % (
                                env_id,
                                episode_number[env_id],
                                episode_steps[env_id],
                                reason,
                                float(rewards[env_id].item()),
                            ),
                            flush=True,
                        )
                        recorder.record(
                            {
                                "run": args.run_name,
                                "env": int(env_id),
                                "episode": int(episode_number[env_id]),
                                "length": int(episode_steps[env_id]),
                                "reason": reason,
                                "reward": float(rewards[env_id].item()),
                            }
                        )
                        first_episode_recorded[env_id] = True

                    failed_ids = failures.nonzero(as_tuple=False).flatten()
                    if (
                        args.randomize_demo_on_failure
                        and failed_ids.numel() > 0
                        and len(data_indices) > 1
                    ):
                        env.envidx_to_demoidx[failed_ids] = torch.randint(
                            low=0,
                            high=len(data_indices),
                            size=failed_ids.shape,
                            device=env.device,
                        )
                    _obs, reset_ids = env.reset_done()
                    reset_ids_np = (
                        reset_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                    )
                    # Count per reset environment, since each reset_idx call
                    # restores that environment's object pose.
                    object_pose_reset_count += int(reset_ids_np.size)
                    for env_id in reset_ids_np.tolist():
                        demo_id = int(env.envidx_to_demoidx[env_id].item())
                        init_frame = int(env.global_cur_idx[env_id].item())
                        print(
                            "[sim] random reset | env=%d demo=%s init_frame=%d"
                            % (env_id, data_indices[demo_id], init_frame),
                            flush=True,
                        )
                    episode_number[reset_ids_np] += 1
                    episode_steps[reset_ids_np] = 0
                    plan_position[reset_ids_np] = action_steps
                    if recording_runtime is not None:
                        recording_runtime.update_axes()
                    if recorder.should_stop():
                        print(
                            "[sim] reached max failure episodes: %d"
                            % recorder.failure_count,
                            flush=True,
                        )
                        break

                observation = _current_policy_observation(
                    env, client.observation_mode
                )
                history[:, :-1] = history[:, 1:]
                history[:, -1] = observation
                if reset_ids_np.size:
                    # This matches the dataset's edge-padding convention at a
                    # fresh episode: repeat the new random initial observation.
                    history[reset_ids_np] = np.repeat(
                        observation[reset_ids_np, None, :],
                        OBS_STEPS,
                        axis=1,
                    )

                if args.print_every and step % args.print_every == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    mean_inference = (
                        total_inference_time / inference_calls
                        if inference_calls
                        else 0.0
                    )
                    cprint(
                        "[sim] progress | steps=%d control_hz=%.2f "
                        "inference_calls=%d mean_inference=%.3fs "
                        "object_drops=%d object_pose_resets=%d"
                        % (
                            step,
                            step / elapsed,
                            inference_calls,
                            mean_inference,
                            object_drop_count,
                            object_pose_reset_count,
                        ),
                        "yellow",
                        flush=True,
                    )
        if args.censor_unfinished_at_cap:
            open_envs = np.flatnonzero(~first_episode_recorded).tolist()
            for record in censored_timeout_records(
                open_envs,
                args.max_steps,
                run_name=args.run_name,
            ):
                print(
                    "[sim] episode end | env=%d episode=%d length=%d "
                    "reason=timeout reward=0.00000"
                    % (record["env"], record["episode"], record["length"]),
                    flush=True,
                )
                recorder.record(record)
            print(
                "[sim] censored %d unfinished first episodes at cap=%d"
                % (len(open_envs), args.max_steps),
                flush=True,
            )
        _print_hold_summary(recorder)
    finally:
        if preview is not None:
            preview.close()
        if recording_runtime is not None:
            recording_runtime.close()
        if client is not None:
            client.close()
        _destroy_environment(env)
    return 0


def main():
    args = _parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[sim] Ctrl-C received, shutting down", flush=True)
        return 130


if __name__ == "__main__":
    sys.exit(main())
