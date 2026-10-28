"""Portable JSONL snapshots for real/simulation comparisons (no hardware I/O)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np

from hardware import POLICY_SHARPA_DOF_NAMES, REAL_SHARPA_DOF_NAMES, real_to_policy


def snapshot(value):
    """Freeze data before control code reuses it; serialize arrays in the writer."""
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        if str(tensor.dtype) == "torch.bfloat16":
            tensor = tensor.float()
        return tensor.numpy().copy()
    if hasattr(value, "get_state") and hasattr(value, "device"):
        return dict(device=str(value.device), state=snapshot(value.get_state()))
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns)


class DebugRecorder:
    """Bounded, lossless during normal operation; disk failures are reported."""
    def __init__(self, path, stage):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8", buffering=1)
        self.stage = stage
        self.pending = queue.Queue(maxsize=128)
        self.error = None
        self.closed = False
        self.worker = threading.Thread(target=self._write, name="real-debug-writer", daemon=True)
        self.worker.start()

    def _write(self):
        while True:
            row = self.pending.get()
            try:
                if row is None:
                    return
                if self.error is None:
                    self.stream.write(json.dumps(json_value(row), allow_nan=False,
                                                 separators=(",", ":")) + "\n")
            except Exception as exc:
                self.error = exc
                print(f"[debug] recording failed: {exc}; subsequent records unavailable", file=sys.stderr, flush=True)
            finally:
                self.pending.task_done()

    def emit(self, event, **fields):
        if self.closed or self.error is not None:
            return
        row = dict(schema_version=1, event=event, stage=self.stage, pid=os.getpid(),
                   monotonic_ns=time.monotonic_ns(), wall_time_ns=time.time_ns())
        row.update(fields)
        # Backpressure bounds memory; no silent dropping of observation/command pairs.
        self.pending.put(snapshot(row))

    def close(self):
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.worker.join()
            self.stream.close()


def recorder_from_env(stage, args):
    path = os.environ.get("REAL_DEBUG_LOG")
    if not path:
        return None
    recorder = DebugRecorder(path, stage)
    recorder.emit(
        "session", args=vars(args), python=sys.version, numpy=np.__version__,
        policy_joint_names=POLICY_SHARPA_DOF_NAMES, real_joint_names=REAL_SHARPA_DOF_NAMES,
        units=dict(position="rad", velocity="rad/s", time="ns"),
        clock="host monotonic_ns for intervals; wall_time_ns for external alignment",
        observation_layout="[qpos(22), previous_policy_target(22), target-qpos(22)] for obs66",
        velocity_available=False,
        launcher={name: os.environ.get(name) for name in (
            "RUN_LOG", "REAL_DEBUG_LOG", "CUDA_VISIBLE_DEVICES", "INIT_POSE",
            "MOVE_TO_INITIAL_POSE", "INITIAL_POSE_STEPS", "INITIAL_POSE_SECONDS")},
    )
    print(f"[debug] structured record: {path}", flush=True)
    return recorder


def record_model(recorder, policy, args, device):
    import importlib.metadata
    import torch

    versions = {}
    for distribution in ("torch", "diffusers", "torch-tensorrt", "tensorrt", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    components = {"prior": getattr(policy, "prior", policy)}
    checkpoints = {"prior": checkpoint_identity(args.checkpoint)}
    if getattr(args, "guide_checkpoint", None):
        components["guide"] = policy.guide
        checkpoints["guide"] = checkpoint_identity(args.guide_checkpoint)
    descriptions = {}
    for name, component in components.items():
        scheduler = getattr(component, "noise_scheduler", None)
        normalizer = getattr(component, "normalizer", None)
        descriptions[name] = dict(
            spec={field: getattr(component, field, None) for field in (
                "horizon", "n_obs_steps", "obs_dim", "action_dim", "n_action_steps",
                "num_inference_steps", "oa_step_convention")},
            weights=getattr(component, "_loaded_checkpoint_metadata", None),
            sampling_dtype=str(getattr(component, "dtype", "unknown")),
            scheduler_class=None if scheduler is None else type(scheduler).__name__,
            scheduler_config=None if scheduler is None else dict(scheduler.config),
            scheduler_step_kwargs=getattr(component, "scheduler_step_kwargs", {}),
            normalizer_state=None if not hasattr(normalizer, "state_dict") else normalizer.state_dict(),
        )
    source_root = Path(__file__).resolve().parent
    repo_root = source_root.parent.parent
    sources = {str(path.relative_to(repo_root)): checkpoint_identity(path)["sha256"]
               for path in (source_root / "inference_real.py", source_root / "sim_hand_policy.py",
                            source_root / "guided_policy.py", source_root / "hardware.py",
                            source_root / "policy_loader.py", source_root / "policy_observation.py",
                            source_root.parent / "trt_unet.py",
                            repo_root / "diffusion_policy/guidance/guided_ddim.py")}
    recorder.emit("model", checkpoints=checkpoints, components=descriptions, versions=versions,
                  source_sha256=sources, device=str(device), cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


class RecordedHand:
    """Wrap existing calls without introducing new reads, sends, or interpolation."""
    def __init__(self, hand, recorder, kind="initial_pose"):
        self.hand = hand
        self.recorder = recorder
        self.context = dict(kind=kind)
        self.last_qpos = None
        self.last_read_ns = None
        self.last_target = None
        self.command_id = 0

    def get_state(self):
        started = time.monotonic_ns()
        try:
            state = self.hand.get_state()
        except BaseException as exc:
            self.recorder.emit("state_error", read_start_ns=started,
                               read_end_ns=time.monotonic_ns(), error=repr(exc), **self.context)
            raise
        ended = time.monotonic_ns()
        qpos = real_to_policy(state)
        self.last_qpos = qpos.copy()
        self.last_read_ns = ended
        self.recorder.emit(
            "state", read_start_ns=started, read_end_ns=ended, qpos_policy_rad=qpos,
            qpos_real_rad=np.asarray(state), velocity_policy_rad_s=None,
            last_command_id=self.command_id or None, target_policy_rad=self.last_target,
            tracking_error_rad=None if self.last_target is None else self.last_target - qpos,
            **self.context,
        )
        return state

    def set_action(self, angles, interpolate=False):
        target = real_to_policy(angles)
        started = time.monotonic_ns()
        try:
            result = self.hand.set_action(angles, interpolate=interpolate)
        except BaseException as exc:
            self.recorder.emit("command_error", send_start_ns=started,
                               send_end_ns=time.monotonic_ns(), target_policy_rad=target,
                               target_real_rad=np.asarray(angles), status="acknowledgement_unknown",
                               error=repr(exc), **self.context)
            raise
        ended = time.monotonic_ns()
        self.command_id += 1
        self.recorder.emit(
            "command", command_id=self.command_id, status="acknowledged",
            send_start_ns=started, send_end_ns=ended, target_policy_rad=target,
            target_real_rad=np.asarray(angles), previous_target_policy_rad=self.last_target,
            latest_qpos_policy_rad=self.last_qpos, latest_state_read_end_ns=self.last_read_ns,
            tracking_before_rad=None if self.last_qpos is None else target - self.last_qpos,
            interpolate=bool(interpolate), **self.context,
        )
        self.last_target = target.copy()
        return result

    def close(self):
        return self.hand.close()


def set_hand_context(hand, **context):
    if isinstance(hand, RecordedHand):
        hand.context = context
