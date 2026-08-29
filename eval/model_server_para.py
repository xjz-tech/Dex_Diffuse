#!/usr/bin/env python3
"""ZeroMQ policy process for the non-blocking parallel evaluator.

The simulator sends a four-state observation only after an action chunk has
finished (plus startup/reset replans). This process performs the blocking CUDA
inference independently and returns the checkpoint's normal five-action chunk.
The simulator never waits for the reply.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import zmq


EVAL_DIR = Path(__file__).resolve().parent
DEX_ROOT = EVAL_DIR.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
if str(DEX_ROOT) not in sys.path:
    sys.path.insert(0, str(DEX_ROOT))

import model_server  # noqa: E402
from ipc_para import endpoint_for_path, recv_router, send_router  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Serve five-step Sim-Hand action chunks over ZeroMQ IPC."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--socket", required=True, dest="socket_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--no-salvage", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def _checkpoint_message(checkpoint_info):
    return {
        "weight_source": checkpoint_info.weight_source,
        "global_step": checkpoint_info.global_step,
        "epoch": checkpoint_info.epoch,
        "salvaged": checkpoint_info.salvaged,
    }


def _serve(args, policy, device, checkpoint_info, spec, normalizer_summary):
    socket_path = Path(args.socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists() or socket_path.is_symlink():
        socket_path.unlink()

    context = zmq.Context()
    server = context.socket(zmq.ROUTER)
    server.setsockopt(zmq.LINGER, 0)
    server.setsockopt(zmq.SNDHWM, 2)
    server.setsockopt(zmq.RCVHWM, 2)
    endpoint = endpoint_for_path(socket_path)
    try:
        server.bind(endpoint)
        try:
            os.chmod(str(socket_path), 0o600)
        except FileNotFoundError:
            # Some libzmq builds create the IPC node immediately after bind.
            pass
        print("[model-para] ready: %s" % endpoint, flush=True)

        while True:
            identity, message, array = recv_router(server)
            message_type = message.get("type")
            request_id = message.get("request_id")

            if message_type == "hello":
                send_router(
                    server,
                    identity,
                    {
                        "type": "hello_ack",
                        "request_id": request_id,
                        "ok": True,
                        "spec": spec,
                        "checkpoint": _checkpoint_message(checkpoint_info),
                        "normalizer": normalizer_summary,
                    },
                )
                continue

            if message_type == "shutdown":
                send_router(
                    server,
                    identity,
                    {
                        "type": "shutdown_ack",
                        "request_id": request_id,
                        "ok": True,
                    },
                )
                break

            if message_type != "state":
                send_router(
                    server,
                    identity,
                    {
                        "type": "error",
                        "request_id": request_id,
                        "ok": False,
                        "error": "unsupported request type: %r" % message_type,
                    },
                )
                continue

            generation = int(message.get("generation", -1))
            state_step = int(message.get("state_step", -1))
            try:
                action, elapsed = model_server._predict(policy, array, device)
                send_router(
                    server,
                    identity,
                    {
                        "type": "action",
                        "request_id": request_id,
                        "generation": generation,
                        "state_step": state_step,
                        "ok": True,
                        "inference_seconds": elapsed,
                    },
                    action,
                )
            except Exception as exc:
                traceback.print_exc()
                send_router(
                    server,
                    identity,
                    {
                        "type": "error",
                        "request_id": request_id,
                        "generation": generation,
                        "state_step": state_step,
                        "ok": False,
                        "error": "%s: %s" % (type(exc).__name__, exc),
                    },
                )
    finally:
        server.close(linger=0)
        context.term()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main():
    args = _parse_args()
    device = model_server._device_or_raise(args.device)
    print("[model-para] loading checkpoint: %s" % args.checkpoint, flush=True)
    loaded = model_server.load_checkpoint(
        args.checkpoint,
        allow_salvage=not args.no_salvage,
    )
    policy, spec = model_server.build_policy(loaded)
    if args.inference_steps is not None:
        if args.inference_steps <= 0:
            raise ValueError("--inference-steps must be positive")
        policy.num_inference_steps = int(args.inference_steps)
    policy = policy.to(device)
    policy.eval()
    summary = model_server._normalizer_summary(policy)

    print(
        "[model-para] loaded %s | step=%s epoch=%s | DDPM steps=%d"
        % (
            loaded.weight_source,
            loaded.global_step,
            loaded.epoch,
            policy.num_inference_steps,
        ),
        flush=True,
    )
    if loaded.salvaged:
        print(
            "[model-para] WARNING: EMA is damaged; using the complete "
            "non-EMA model from the checkpoint.",
            flush=True,
        )
    if not args.no_warmup:
        model_server._warm_up(policy, device, args.seed, summary["obs"]["mean"])
    _serve(args, policy, device, loaded, spec, summary)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[model-para] stopped", flush=True)
