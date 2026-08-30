#!/usr/bin/env python3
"""ZeroMQ evaluator for Sim-Hand Diffusion Policy.

The simulator and model run in separate processes connected by ZeroMQ.  After
five model actions have executed, the simulator sends its latest four-state
history.  It can either keep stepping with the final action target (WAIT=0) or
freeze physics/rendering until the next action chunk arrives (WAIT=1).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Isaac Gym must be imported before torch in its Python 3.8 environment.
try:
    import isaacgym  # noqa: F401
    from isaacgym import gymapi
except ImportError as exc:
    raise RuntimeError(
        "Isaac Gym is unavailable. Run this entrypoint through "
        "eval/eval_para.sh."
    ) from exc

import torch  # noqa: E402
from termcolor import cprint  # noqa: E402
import zmq  # noqa: E402


EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from ipc_para import endpoint_for_path, recv_dealer, send_dealer  # noqa: E402
from recording import RecordingConstructionHooks, RecordingRuntime  # noqa: E402
from sim_eval import (  # noqa: E402
    ACTION_STEPS,
    HAND_DIM,
    OBS_STEPS,
    _add_reset_arguments,
    _absolute_targets_to_env_action,
    _expand_data_indices,
    _import_local_maniptrans,
    _install_sharpa_asset_override,
    _load_manifest,
    _make_recording_config,
    _make_task_config,
    _reset_overrides_from_args,
    _resolve_sharpa_urdf,
    _restore_sharpa_asset_override,
    _validate_environment,
    _validate_inputs,
    _validate_reset_args,
)


EXPECTED_POLICY_SPEC = {
    "obs_dim": HAND_DIM,
    "action_dim": HAND_DIM,
    "n_obs_steps": OBS_STEPS,
    "n_pred_action_steps": 9,
    "n_action_steps": ACTION_STEPS,
    "horizon": 12,
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Sim-Hand DP with ZeroMQ inference and selectable "
            "hold-last-action or wait-at-chunk-boundary simulation."
        )
    )
    parser.add_argument("--controller-root", required=True)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--record-env", type=int, default=0)
    parser.add_argument("--data-indices", default="000,001")
    parser.add_argument("--nokov3-data-dir", required=True)
    parser.add_argument("--nokov3-retarget-dir", required=True)
    parser.add_argument("--sharpa-asset-dir", required=True)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--wait",
        type=int,
        choices=(0, 1),
        default=0,
        dest="wait_for_policy",
        help=(
            "0: keep stepping/rendering with the last target during inference; "
            "1: pause physics/rendering at chunk boundaries until a reply arrives."
        ),
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=30.0,
        help="Target wall-clock PhysX/action-step frequency.",
    )
    parser.add_argument(
        "--render-hz",
        type=float,
        default=30.0,
        help="Target viewer refresh frequency, independent of policy replies.",
    )
    parser.add_argument(
        "--deadline-tolerance-ms",
        type=float,
        default=2.0,
        help="Lateness threshold counted as a control deadline miss.",
    )
    parser.add_argument("--recording", action="store_true")
    parser.add_argument("--record-dir", default=str(EVAL_DIR / "record"))
    parser.add_argument("--record-width", type=int, default=1280)
    parser.add_argument("--record-height", type=int, default=720)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--record-camera-position", default="-0.10,0.55,0.10")
    parser.add_argument("--record-camera-target", default="-0.10,0.00,-0.14")
    parser.add_argument("--record-camera-fov", type=float, default=60.0)
    parser.add_argument("--record-axis-length", type=float, default=0.20)
    parser.add_argument("--record-axis-thickness", type=float, default=0.008)
    _add_reset_arguments(parser)

    parser.set_defaults(randomize_demo_on_failure=True)
    parser.add_argument(
        "--no-randomize-demo-on-failure",
        action="store_false",
        dest="randomize_demo_on_failure",
    )
    return parser.parse_args()


def _validate_parallel_args(args):
    _validate_reset_args(args)
    if not np.isfinite(args.control_hz) or args.control_hz <= 0.0:
        raise ValueError("--control-hz must be finite and positive")
    if not np.isfinite(args.render_hz) or args.render_hz <= 0.0:
        raise ValueError("--render-hz must be finite and positive")
    if (
        not np.isfinite(args.deadline_tolerance_ms)
        or args.deadline_tolerance_ms < 0.0
    ):
        raise ValueError("--deadline-tolerance-ms must be finite and non-negative")
    if not np.isfinite(args.request_timeout) or args.request_timeout <= 0.0:
        raise ValueError("--request-timeout must be finite and positive")


@dataclass
class PendingRequest:
    request_id: int
    generation: int
    state_step: int
    sent_at: float


class ZmqPolicyClient:
    """One-in-flight asynchronous DEALER client owned by the sim thread."""

    def __init__(self, socket_path, timeout_seconds):
        self.timeout_seconds = float(timeout_seconds)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.setsockopt(zmq.RCVHWM, 2)
        self.socket.setsockopt(
            zmq.IDENTITY,
            ("dex-diffuse-sim-%d" % os.getpid()).encode("ascii"),
        )
        self.socket.connect(endpoint_for_path(socket_path))
        self.request_id = 0
        self.pending = None
        self.info = self._hello()

    def _next_id(self):
        self.request_id += 1
        return self.request_id

    def _hello(self):
        request_id = self._next_id()
        send_dealer(
            self.socket,
            {"type": "hello", "request_id": request_id},
        )
        timeout_ms = max(1, int(round(self.timeout_seconds * 1000.0)))
        if not self.socket.poll(timeout_ms, zmq.POLLIN):
            raise TimeoutError("timed out waiting for policy hello")
        response, array = recv_dealer(self.socket)
        if array is not None:
            raise RuntimeError("policy hello unexpectedly contained an array")
        if response.get("request_id") != request_id:
            raise RuntimeError("policy hello response id mismatch")
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "policy hello failed"))
        if response.get("spec") != EXPECTED_POLICY_SPEC:
            raise RuntimeError(
                "checkpoint temporal spec mismatch: %r" % response.get("spec")
            )
        return response

    def submit(self, history, generation, state_step):
        if self.pending is not None:
            raise RuntimeError("a policy request is already in flight")
        history = np.asarray(history, dtype=np.float32, order="C")
        expected = (history.shape[0], OBS_STEPS, HAND_DIM)
        if history.shape != expected:
            raise ValueError("invalid observation history shape: %r" % (history.shape,))
        if not np.isfinite(history).all():
            raise ValueError("observation history contains NaN or Inf")
        request_id = self._next_id()
        send_dealer(
            self.socket,
            {
                "type": "state",
                "request_id": request_id,
                "generation": int(generation),
                "state_step": int(state_step),
            },
            history,
            flags=zmq.NOBLOCK,
        )
        self.pending = PendingRequest(
            request_id=request_id,
            generation=int(generation),
            state_step=int(state_step),
            sent_at=time.monotonic(),
        )
        return self.pending

    def poll_action(self, num_envs):
        pending = self.pending
        if pending is None:
            return None
        if not self.socket.poll(0, zmq.POLLIN):
            age = time.monotonic() - pending.sent_at
            if age > self.timeout_seconds:
                raise TimeoutError(
                    "policy request %d timed out after %.3f s"
                    % (pending.request_id, age)
                )
            return None

        response, action = recv_dealer(self.socket, flags=zmq.NOBLOCK)
        if response.get("request_id") != pending.request_id:
            raise RuntimeError(
                "policy response id mismatch: expected %d, got %r"
                % (pending.request_id, response.get("request_id"))
            )
        self.pending = None
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "policy inference failed"))
        if response.get("type") != "action":
            raise RuntimeError("unexpected policy response: %r" % response.get("type"))
        if int(response.get("generation", -1)) != pending.generation:
            raise RuntimeError("policy response generation was not echoed correctly")
        if int(response.get("state_step", -1)) != pending.state_step:
            raise RuntimeError("policy response state_step was not echoed correctly")
        expected = (int(num_envs), ACTION_STEPS, HAND_DIM)
        if action is None or action.shape != expected:
            raise RuntimeError(
                "invalid action chunk from policy: %r, expected %r"
                % (None if action is None else action.shape, expected)
            )
        if not np.isfinite(action).all():
            raise RuntimeError("policy action contains NaN or Inf")
        return {
            "action": action,
            "pending": pending,
            "inference_seconds": float(response.get("inference_seconds", 0.0)),
            "roundtrip_seconds": time.monotonic() - pending.sent_at,
        }

    def close(self):
        if self.socket is None:
            return
        had_pending_request = self.pending is not None
        shutdown_id = self._next_id()
        try:
            send_dealer(
                self.socket,
                {"type": "shutdown", "request_id": shutdown_id},
                flags=zmq.NOBLOCK,
            )
        except (zmq.Again, zmq.ZMQError):
            pass
        else:
            # When the model is idle, allow a short graceful handshake. If an
            # inference is active, never make simulator shutdown wait for it;
            # eval_para.sh owns bounded process cleanup in that case.
            if not had_pending_request and self.socket.poll(250, zmq.POLLIN):
                try:
                    response, _array = recv_dealer(self.socket, flags=zmq.NOBLOCK)
                    if response.get("request_id") != shutdown_id:
                        print(
                            "[sim-para] ignored non-shutdown reply during close",
                            flush=True,
                        )
                except (zmq.Again, zmq.ZMQError):
                    pass
        self.socket.close(linger=100)
        self.socket = None
        self.context.term()
        self.context = None


class ParallelRenderer:
    """Keep all Isaac Gym viewer/camera calls on the simulator main thread."""

    def __init__(self, env, recording_runtime, render_hz):
        self.env = env
        self.recording_runtime = recording_runtime
        self.period = 1.0 / float(render_hz)
        self.next_draw = None
        self.quit_requested = False
        self.quit_reason = None
        self.draw_count = 0
        self.total_draw_seconds = 0.0
        self.max_draw_seconds = 0.0

        if env.viewer is not None and recording_runtime is None:
            env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "QUIT")

        # VecTask calls render() before each physics substep. Those callbacks
        # only service events; this scheduler owns actual viewer draws.
        env.render = lambda mode="rgb_array": self._step_render_hook(mode)

    @property
    def available(self):
        return self.env.viewer is not None

    def _step_render_hook(self, mode="rgb_array"):
        del mode
        self.poll_events()

    def poll_events(self):
        if self.quit_requested or self.env.viewer is None:
            return self.quit_requested
        if self.recording_runtime is not None:
            self.quit_requested = self.recording_runtime.poll_viewer_events()
            self.quit_reason = self.recording_runtime.quit_reason
            return self.quit_requested

        if self.env.gym.query_viewer_has_closed(self.env.viewer):
            self.quit_requested = True
            self.quit_reason = "viewer closed"
        else:
            for event in self.env.gym.query_viewer_action_events(self.env.viewer):
                if event.action == "QUIT" and event.value > 0:
                    self.quit_requested = True
                    self.quit_reason = "Q/Esc"
                    break
                if event.action == "toggle_viewer_sync" and event.value > 0:
                    self.env.enable_viewer_sync = not self.env.enable_viewer_sync
        if self.quit_requested:
            print(
                "[sim-para] %s received; shutting down" % self.quit_reason,
                flush=True,
            )
        return self.quit_requested

    def seconds_until_due(self):
        if not self.available or self.next_draw is None:
            return float("inf")
        return max(0.0, self.next_draw - time.monotonic())

    def draw_if_due(self, graphics_ready=False, force=False):
        if not self.available or self.poll_events():
            return not self.quit_requested, False
        now = time.monotonic()
        if not force and self.next_draw is not None and now < self.next_draw:
            return True, False

        deadline = now if self.next_draw is None else self.next_draw
        started = time.monotonic()
        did_draw = False
        if not self.env.enable_viewer_sync:
            self.env.gym.poll_viewer_events(self.env.viewer)
        else:
            if not graphics_ready:
                if self.env.device != "cpu":
                    self.env.gym.fetch_results(self.env.sim, True)
                self.env.gym.step_graphics(self.env.sim)
            if self.recording_runtime is not None:
                self.recording_runtime._draw_viewer_axes(self.env.control_steps)
            self.env.gym.draw_viewer(self.env.viewer, self.env.sim, True)
            self.draw_count += 1
            did_draw = True
        elapsed = time.monotonic() - started
        if did_draw:
            self.total_draw_seconds += elapsed
            self.max_draw_seconds = max(self.max_draw_seconds, elapsed)

        next_draw = deadline + self.period
        finished = time.monotonic()
        if next_draw <= finished:
            skipped = int((finished - next_draw) // self.period) + 1
            next_draw += skipped * self.period
        self.next_draw = next_draw
        return not self.quit_requested, did_draw


class FixedControlPacer:
    """Pace every PhysX/action start without catch-up bursts."""

    def __init__(self, frequency_hz, tolerance_ms):
        self.period = 1.0 / float(frequency_hz)
        self.tolerance = float(tolerance_ms) / 1000.0
        self.last_start = None
        self.interval_count = 0
        self.interval_seconds = 0.0
        self.deadline_misses = 0
        self.max_lateness = 0.0

    def wait_for_tick(self, renderer):
        if self.last_start is None:
            started = time.monotonic()
            self.last_start = started
            return started

        deadline = self.last_start + self.period
        while True:
            if renderer.poll_events():
                return None
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0.0:
                break

            until_render = renderer.seconds_until_due()
            if until_render <= 0.0 and remaining > 0.0005:
                renderer.draw_if_due()
                continue
            sleep_for = min(remaining, until_render, 0.01)
            if not np.isfinite(sleep_for):
                sleep_for = min(remaining, 0.01)
            if sleep_for > 0.0:
                time.sleep(sleep_for)

        started = time.monotonic()
        interval = started - self.last_start
        self.interval_seconds += interval
        self.interval_count += 1
        lateness = max(0.0, started - deadline)
        self.max_lateness = max(self.max_lateness, lateness)
        if lateness > self.tolerance:
            self.deadline_misses += 1
        self.last_start = started
        return started

    def restart_after_pause(self):
        """Drop only a deadline that expired during an intentional pause."""
        if (
            self.last_start is not None
            and time.monotonic() >= self.last_start + self.period
        ):
            self.last_start = None

    @property
    def measured_hz(self):
        if self.interval_count == 0 or self.interval_seconds <= 0.0:
            return 0.0
        return self.interval_count / self.interval_seconds


def _print_policy_info(client):
    checkpoint = client.info["checkpoint"]
    print(
        "[sim-para] policy connected via ZeroMQ | weights=%s step=%s epoch=%s"
        % (
            checkpoint["weight_source"],
            checkpoint["global_step"],
            checkpoint["epoch"],
        ),
        flush=True,
    )
    if checkpoint.get("salvaged"):
        print(
            "[sim-para] WARNING: checkpoint EMA is incomplete; using its "
            "strictly recovered base model.",
            flush=True,
        )


def run(args):
    _validate_parallel_args(args)
    data_indices = _expand_data_indices(args.data_indices)
    controller_root, config_path, data_root, retarget_root = _validate_inputs(
        args,
        data_indices,
    )
    manifest = _load_manifest(config_path)
    sharpa_urdf = _resolve_sharpa_urdf(args.sharpa_asset_dir)
    recording_config = _make_recording_config(args) if args.recording else None

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
    print(
        "[sim-para] creating bulb2 task | envs=%d trajectories=%d "
        "headless=%s record_env=%d control_hz=%.2f render_hz=%.2f"
        % (
            args.num_envs,
            len(data_indices),
            args.headless,
            args.record_env,
            args.control_hz,
            args.render_hz,
        ),
        flush=True,
    )

    env = None
    client = None
    recording_runtime = None
    renderer = None
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

        env.compute_observations()
        env.reset()
        lower, upper = _validate_environment(env, manifest)
        if recording_config is not None:
            recording_runtime = RecordingRuntime(env, recording_config)
        renderer = ParallelRenderer(env, recording_runtime, args.render_hz)

        client = ZmqPolicyClient(args.socket_path, args.request_timeout)
        _print_policy_info(client)

        qpos = env._q.detach().cpu().numpy().astype(np.float32, copy=True)
        history = np.repeat(qpos[:, None, :], OBS_STEPS, axis=1)
        hold_targets = qpos.copy()
        active_chunk = None
        chunk_cursor = 0
        generation = 0
        replan_needed = False
        episode_steps = np.zeros(args.num_envs, dtype=np.int64)
        episode_number = np.zeros(args.num_envs, dtype=np.int64)

        requests_sent = 0
        responses_received = 0
        chunks_received = 0
        chunks_completed = 0
        chunks_aborted = 0
        stale_replies = 0
        action_steps = 0
        hold_steps = 0
        hold_events = 0
        current_hold_steps = 0
        max_hold_steps = 0
        total_inference_time = 0.0
        total_roundtrip_time = 0.0
        policy_waits = 0
        total_policy_wait_time = 0.0
        total_env_step_time = 0.0
        max_env_step_time = 0.0
        capture_calls = 0
        total_capture_time = 0.0
        max_capture_time = 0.0
        object_drop_count = 0
        object_pose_reset_count = 0

        step = 0
        client.submit(history, generation, step)
        requests_sent += 1
        pacer = FixedControlPacer(args.control_hz, args.deadline_tolerance_ms)
        started = time.monotonic()

        def consume_policy_reply():
            nonlocal active_chunk
            nonlocal chunk_cursor
            nonlocal chunks_received
            nonlocal replan_needed
            nonlocal responses_received
            nonlocal stale_replies
            nonlocal total_inference_time
            nonlocal total_roundtrip_time

            received = client.poll_action(args.num_envs)
            if received is None:
                return False

            responses_received += 1
            total_inference_time += received["inference_seconds"]
            total_roundtrip_time += received["roundtrip_seconds"]
            pending = received["pending"]
            if pending.generation != generation:
                stale_replies += 1
                replan_needed = True
                print(
                    "[sim-para] discarded stale action chunk | "
                    "request=%d request_generation=%d current_generation=%d"
                    % (
                        pending.request_id,
                        pending.generation,
                        generation,
                    ),
                    flush=True,
                )
            else:
                if active_chunk is not None:
                    raise RuntimeError(
                        "received a policy chunk while another is active"
                    )
                active_chunk = received["action"]
                chunk_cursor = 0
                chunks_received += 1
                replan_needed = False
            return True

        def submit_latest_state_if_needed():
            nonlocal replan_needed
            nonlocal requests_sent

            if args.max_steps and step >= args.max_steps:
                return
            if (
                replan_needed
                and client.pending is None
                and active_chunk is None
            ):
                client.submit(history, generation, step)
                requests_sent += 1
                replan_needed = False

        sim_control_hz = 1.0 / (float(env.dt) * int(env.control_freq_inv))
        if not np.isclose(sim_control_hz, args.control_hz, rtol=0.0, atol=1e-3):
            print(
                "[sim-para] WARNING: task simulation-time control rate is "
                "%.3f Hz, wall-clock target is %.3f Hz"
                % (sim_control_hz, args.control_hz),
                flush=True,
            )

        if args.wait_for_policy:
            print(
                "[sim-para] chunk_boundary_wait | execute 5 actions, send "
                "latest state, then pause PhysX/rendering until the next chunk; "
                "Q/Esc events are still polled",
                flush=True,
            )
        else:
            print(
                "[sim-para] chunk_boundary_hold | execute 5 actions, send "
                "latest state, then keep stepping with action[4] until the "
                "next chunk; press Q/Esc, close viewer, or Ctrl-C to stop",
                flush=True,
            )

        with torch.no_grad():
            while args.max_steps == 0 or step < args.max_steps:
                if args.wait_for_policy and active_chunk is None:
                    # WAIT=1 deliberately freezes simulation time and drawing at
                    # a chunk boundary. Only IPC and viewer quit events are
                    # serviced until a current-generation chunk is available.
                    if client.pending is None and not replan_needed:
                        replan_needed = True
                    submit_latest_state_if_needed()

                    wait_started = time.monotonic()
                    policy_waits += 1
                    while active_chunk is None:
                        # Give a simultaneous Q/viewer-close event priority over
                        # a newly ready policy reply so no extra step is run.
                        if renderer.poll_events():
                            break
                        consume_policy_reply()
                        submit_latest_state_if_needed()
                        if active_chunk is not None:
                            break
                        time.sleep(0.002)
                    total_policy_wait_time += time.monotonic() - wait_started
                    # A long intentional wait is not a missed deadline. An
                    # unusually early reply still observes the remaining part
                    # of the current control period.
                    pacer.restart_after_pause()
                    if renderer.poll_events():
                        break

                tick_started = pacer.wait_for_tick(renderer)
                if tick_started is None or renderer.quit_requested:
                    break

                consume_policy_reply()
                submit_latest_state_if_needed()

                executing_chunk = active_chunk is not None
                if executing_chunk:
                    selected_targets = active_chunk[:, chunk_cursor]
                    hold_targets = selected_targets.copy()
                    action_steps += 1
                    current_hold_steps = 0
                else:
                    selected_targets = hold_targets
                    hold_steps += 1
                    if current_hold_steps == 0:
                        hold_events += 1
                    current_hold_steps += 1
                    max_hold_steps = max(max_hold_steps, current_hold_steps)

                env_action = _absolute_targets_to_env_action(
                    selected_targets,
                    lower,
                    upper,
                    env.device,
                )
                env_step_started = time.monotonic()
                _obs, rewards, dones, infos = env.step(env_action)
                env_step_seconds = time.monotonic() - env_step_started
                total_env_step_time += env_step_seconds
                max_env_step_time = max(max_env_step_time, env_step_seconds)
                episode_steps += 1
                step += 1

                if executing_chunk:
                    chunk_cursor += 1
                    if chunk_cursor >= ACTION_STEPS:
                        active_chunk = None
                        chunk_cursor = 0
                        chunks_completed += 1
                        # Submit only after reset handling and the post-step
                        # observation history have been updated below.
                        replan_needed = True

                done_mask = dones.to(env.device).bool()
                failures = env.failure_buf.detach().clone().bool()
                successes = env.success_buf.detach().clone().bool()
                reach_goal = env.reach_final_goal.detach().clone().bool()

                if torch.any(reach_goal):
                    reached = reach_goal.nonzero(as_tuple=False).flatten().tolist()
                    cprint(
                        "[sim-para] reached target | step=%d env_ids=%s"
                        % (step, reached),
                        "green",
                        flush=True,
                    )

                # Capture/draw the terminal pose before reset_done changes it.
                if recording_runtime is not None:
                    capture_started = time.monotonic()
                    recording_runtime.update_axes()
                    # Match the original serial recorder: register Isaac Gym
                    # Lines API geometry before render_all_camera_sensors().
                    # This build then renders the native lines into both the
                    # viewer and the recording camera; no 2D axis compositing.
                    recording_runtime._draw_viewer_axes(env.control_steps)
                    recording_runtime.capture_if_active()
                    capture_seconds = time.monotonic() - capture_started
                    capture_calls += 1
                    total_capture_time += capture_seconds
                    max_capture_time = max(max_capture_time, capture_seconds)
                renderer.draw_if_due(
                    graphics_ready=recording_runtime is not None,
                )
                if renderer.quit_requested:
                    break

                reset_ids_np = np.empty(0, dtype=np.int64)
                if torch.any(done_mask):
                    done_ids = done_mask.nonzero(as_tuple=False).flatten()
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
                        print(
                            "[sim-para] episode end | env=%d episode=%d length=%d "
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
                    object_pose_reset_count += int(reset_ids_np.size)
                    for env_id in reset_ids_np.tolist():
                        demo_id = int(env.envidx_to_demoidx[env_id].item())
                        init_frame = int(env.global_cur_idx[env_id].item())
                        print(
                            "[sim-para] random reset | env=%d demo=%s init_frame=%d"
                            % (env_id, data_indices[demo_id], init_frame),
                            flush=True,
                        )
                    episode_number[reset_ids_np] += 1
                    episode_steps[reset_ids_np] = 0
                    if active_chunk is not None:
                        chunks_aborted += 1
                    active_chunk = None
                    chunk_cursor = 0
                    generation += 1
                    replan_needed = True
                    if recording_runtime is not None:
                        recording_runtime.update_axes()

                qpos = env._q.detach().cpu().numpy().astype(np.float32, copy=True)
                history[:, :-1] = history[:, 1:]
                history[:, -1] = qpos
                if reset_ids_np.size:
                    history[reset_ids_np] = np.repeat(
                        qpos[reset_ids_np, None, :],
                        OBS_STEPS,
                        axis=1,
                    )
                    # A reset env must hold its new initial configuration, not
                    # an absolute target from the previous episode.
                    hold_targets[reset_ids_np] = qpos[reset_ids_np]

                submit_latest_state_if_needed()

                if args.print_every and step % args.print_every == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    mean_inference = (
                        total_inference_time / responses_received
                        if responses_received
                        else 0.0
                    )
                    mean_roundtrip = (
                        total_roundtrip_time / responses_received
                        if responses_received
                        else 0.0
                    )
                    mean_env_step_ms = 1000.0 * total_env_step_time / step
                    mean_draw_ms = (
                        1000.0
                        * renderer.total_draw_seconds
                        / renderer.draw_count
                        if renderer.draw_count
                        else 0.0
                    )
                    mean_capture_ms = (
                        1000.0 * total_capture_time / capture_calls
                        if capture_calls
                        else 0.0
                    )
                    hold_ratio = hold_steps / float(step)
                    if active_chunk is not None:
                        next_mode = "action"
                    elif args.wait_for_policy:
                        next_mode = "wait"
                    else:
                        next_mode = "hold"
                    cprint(
                        "[sim-para] progress | steps=%d target_hz=%.2f "
                        "control_hz=%.2f wall_hz=%.2f target_render_hz=%.2f "
                        "render_hz=%.2f deadline_misses=%d max_lag_ms=%.2f "
                        "mean_env_step_ms=%.2f max_env_step_ms=%.2f "
                        "mean_draw_ms=%.2f max_draw_ms=%.2f "
                        "mean_capture_ms=%.2f max_capture_ms=%.2f requests=%d "
                        "responses=%d chunks=%d chunks_completed=%d "
                        "chunks_aborted=%d mean_inference=%.3fs "
                        "mean_roundtrip=%.3fs wait_mode=%d policy_waits=%d "
                        "policy_wait=%.2fs action_steps=%d hold_steps=%d "
                        "hold_ratio=%.3f hold_events=%d max_hold_steps=%d "
                        "stale_replies=%d pending=%d next_mode=%s object_drops=%d "
                        "object_pose_resets=%d"
                        % (
                            step,
                            args.control_hz,
                            pacer.measured_hz,
                            step / elapsed,
                            args.render_hz,
                            renderer.draw_count / elapsed,
                            pacer.deadline_misses,
                            1000.0 * pacer.max_lateness,
                            mean_env_step_ms,
                            1000.0 * max_env_step_time,
                            mean_draw_ms,
                            1000.0 * renderer.max_draw_seconds,
                            mean_capture_ms,
                            1000.0 * max_capture_time,
                            requests_sent,
                            responses_received,
                            chunks_received,
                            chunks_completed,
                            chunks_aborted,
                            mean_inference,
                            mean_roundtrip,
                            args.wait_for_policy,
                            policy_waits,
                            total_policy_wait_time,
                            action_steps,
                            hold_steps,
                            hold_ratio,
                            hold_events,
                            max_hold_steps,
                            stale_replies,
                            int(client.pending is not None),
                            next_mode,
                            object_drop_count,
                            object_pose_reset_count,
                        ),
                        "yellow",
                        flush=True,
                    )
    finally:
        if recording_runtime is not None:
            recording_runtime.close()
        if client is not None:
            client.close()
        # Explicit destruction is unstable in this legacy Isaac Gym build after
        # a GPU camera tensor has been used. The dedicated process exits below.
        global _GYM_KEEPALIVE
        _GYM_KEEPALIVE = (env, renderer, recording_runtime)
    return 0


def main():
    args = _parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[sim-para] Ctrl-C received, shutting down", flush=True)
        return 130


if __name__ == "__main__":
    try:
        exit_status = main()
    except SystemExit as exc:
        exit_status = int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException:
        traceback.print_exc()
        exit_status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_status)
