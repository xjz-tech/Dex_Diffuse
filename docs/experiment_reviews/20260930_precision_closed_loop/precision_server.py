"""Persistent, matched FP32 / TensorRT FP16 server for the 10B bulb-turn cases."""

import argparse
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OBJECT = ROOT / "docs/experiment_reviews/20260924_object_state_data"
SPEED = ROOT / "docs/experiment_reviews/20260928_suedit_guidance_speed"
sys.path[:0] = [str(SPEED), str(OBJECT), str(ROOT / "eval"), str(ROOT)]

import benchmark_trt_fused as fastmod  # noqa: E402
from ipc import recv_message, send_message  # noqa: E402
from reference_resampling import interpolate_large_jumps  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--method", choices=("edit015", "guidance4"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--sessions", type=int, required=True)
    args = parser.parse_args()
    assert args.sessions > 0 and not args.socket.exists()

    torch.set_num_threads(2)
    fastmod.METHODS = {"edit015": (0, 0.0), "guidance4": (4, 50.0)}
    editor = fastmod.ReferenceActionEditor(args.checkpoint, .15, steps=4, execution_steps=2)
    sampler = fastmod.FastSampler(editor)
    if args.precision == "fp16":
        compiled = fastmod.compile_unet(
            sampler.p.model, horizon=12, action_dim=22, global_cond_dim=264,
            device=torch.device("cuda:0"), fp16=True, max_batch=1,
        )
        sampler.p.model = fastmod.TensorRTUnetAdapter(compiled)

    source = np.load(args.reference, allow_pickle=False)["hand_target_rad"]
    reference, _ = interpolate_large_jumps(source, .1)
    guide_steps = 9 if args.method == "edit015" else 4
    scale = 0.0 if args.method == "edit015" else 50.0
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(args.socket))
        server.listen(1)
        print("READY", args.precision, args.method, args.reference, flush=True)
        for session in range(args.sessions):
            client, _ = server.accept()
            records = []
            output_dir = None
            try:
                with client:
                    while True:
                        msg, history = recv_message(client)
                        if msg["type"] == "hello":
                            output_dir = Path(msg["output_dir"])
                            send_message(client, dict(
                                ok=True, prior=str(args.checkpoint), spec=editor.spec,
                                ddim=4, execution_steps=2, guidance_steps=guide_steps,
                                guidance_scale=scale, reference=str(args.reference),
                                reference_repeat=1, reference_interpolation=0,
                                reference_interpolation_threshold=.1,
                                reference_interpolation_equal_jump=None,
                                reference_mode="expanded", precision=args.precision,
                                editor=editor.metadata if args.method == "edit015" else None,
                            ))
                            continue
                        if msg["type"] == "shutdown":
                            break
                        assert msg["type"] == "predict"
                        assert msg["guidance_scale"] == scale
                        j = int(msg["reference_index"])
                        ids = msg["reference_ids"]
                        assert len(ids) == len(msg["seeds"]) == len(history) == 1
                        indices = np.minimum(np.arange(j, j + 9), reference.shape[1] - 1)
                        future = np.stack([reference[int(ri), indices] for ri in ids])
                        editor.controller.set_fixed_noise_from_seeds(msg["seeds"])
                        noise = editor.controller._fixed_noise.clone()
                        start = time.monotonic()
                        action = sampler.predict(history, future, noise, args.method).cpu().numpy()
                        row = dict(reference_index=j, seeds=msg["seeds"],
                                   guidance_scale=scale,
                                   inference_seconds=time.monotonic() - start,
                                   output_vs_reference_rmse_rad=float(np.sqrt(
                                       ((action - future[:, :2]) ** 2).mean())),
                                   precision=args.precision)
                        records.append(row)
                        send_message(client, dict(ok=True, **row), action.astype(np.float32))
            finally:
                if output_dir is not None:
                    (output_dir / "predictions.json").write_text(
                        json.dumps(records, indent=2) + "\n")
            print("SESSION", session + 1, "/", args.sessions, output_dir, len(records), flush=True)
    finally:
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
