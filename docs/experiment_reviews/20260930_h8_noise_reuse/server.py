"""Experimental IPC adapter. Uses the unchanged SDEdit and native replay loop."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '20260924_object_state_data'
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from reference_action_editor import ReferenceActionEditor
from reference_resampling import interpolate_large_jumps
from ipc import recv_message, send_message
from noise_stream import AlignedNoiseStream


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--baseline-trace', type=Path, required=True)
    p.add_argument('--rho', type=float, required=True, choices=(0., .5, .8, 1.))
    p.add_argument('--seed', type=int, required=True)
    args = p.parse_args()
    archived_actions = [r for r in json.loads(args.baseline_trace.read_text()) if r['phase'] == 'action']
    torch.set_num_threads(2)
    editor = ReferenceActionEditor(args.checkpoint, .15, steps=4, execution_steps=2)
    assert editor.spec['horizon'] == 8 and editor.future_steps == 5
    assert editor.timesteps == [8, 5, 3, 0]
    assert editor.metadata['weight_source'] == 'EMA model'
    stream = AlignedNoiseStream(args.seed, args.rho, editor.controller.device, editor.policy.dtype)
    editor.metadata.update(noise_protocol='time_aligned_raw_gaussian_v1', rho=args.rho,
                           history_noise='fresh', seed=args.seed)
    ref, progress = interpolate_large_jumps(np.load(args.reference)['hand_target_rad'], .1)
    path = Path(args.socket)
    if path.exists():
        raise FileExistsError(path)
    records = []
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(path))
        server.listen(1)
        print('READY', json.dumps(editor.metadata), flush=True)
        client, _ = server.accept()
        with client:
            while True:
                msg, history = recv_message(client)
                if msg['type'] == 'shutdown':
                    break
                if msg['type'] == 'hello':
                    send_message(client, dict(ok=True, prior=args.checkpoint, spec=editor.spec,
                        ddim=4, execution_steps=2, guidance_steps=5, guidance_scale=0.,
                        reference=str(args.reference), reference_repeat=1, reference_interpolation=0,
                        reference_interpolation_threshold=.1, reference_interpolation_equal_jump=None,
                        reference_mode='expanded', editor=editor.metadata))
                    continue
                assert msg['type'] == 'predict' and msg['guidance_scale'] == 0
                assert msg['seeds'] == [args.seed] and msg['reference_ids'] == [0]
                j = int(msg['reference_index'])
                assert j < ref.shape[1]
                indices = np.minimum(np.arange(j, j + 5), ref.shape[1] - 1)
                future = ref[:, indices]
                noise, xi = stream.draw(j)
                start = time.monotonic()
                plan, stats = editor.predict(history, future, msg['seeds'],
                                             return_full_plan=True, initial_noise=noise)
                if j == 0:
                    # No rollout may proceed if the shared first draw already
                    # disagrees with the archived checkpoint/initial-state pair.
                    np.testing.assert_allclose(plan[0, :2],
                        np.asarray([r['command'] for r in archived_actions[:2]]), rtol=0, atol=1e-6)
                row = dict(reference_index=j, guidance_scale=0., noise_ratio=.15,
                    seeds=msg['seeds'], rho=args.rho, inference_seconds=time.monotonic() - start,
                    valid_future_steps=min(5, ref.shape[1] - j),
                    full_plan_rad=plan[0].tolist(), reference_rad=future[0].tolist(),
                    observation=history[0].tolist(), raw_noise=noise[0].cpu().tolist(),
                    innovation=xi[0].cpu().tolist(), **stats)
                records.append(row)
                # IPC payload remains exactly the two commands the old server returned.
                send_message(client, dict(ok=True, reference_index=j, guidance_scale=0.),
                             plan[:, :2].astype(np.float32))
    finally:
        server.close()
        path.unlink(missing_ok=True)
        args.log.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
