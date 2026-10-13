#!/usr/bin/env python3
"""Use reference editing up to a replan boundary, then run the same prior freely."""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'eval'))
from ipc import recv_message, send_message
from reference_action_editor import ReferenceActionEditor
from reference_resampling import interpolate_large_jumps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', default='/home/carus/data_usb/10B_obs_4-66.ckpt')
    p.add_argument('--noise-ratio', type=float, default=.15)
    p.add_argument('--ddim-steps', type=int, default=4)
    p.add_argument('--execution-steps', type=int, default=2)
    p.add_argument('--reference-interpolation-threshold', type=float, default=.1)
    p.add_argument('--switch-action-index', type=int, default=102,
                   help='First zero-based action index generated without future reference')
    args = p.parse_args()
    if args.switch_action_index % args.execution_steps:
        p.error('switch index must be an execution-step replan boundary')

    torch.set_num_threads(2)
    editor = ReferenceActionEditor(args.checkpoint, args.noise_ratio,
                                   steps=args.ddim_steps,
                                   execution_steps=args.execution_steps)
    controller = editor.controller
    controller.guidance_scale = 0.
    controller.set_guidance_horizon(9)
    reference = np.load(args.reference)['hand_target_rad']
    reference, _ = interpolate_large_jumps(
        reference, args.reference_interpolation_threshold)

    path = Path(args.socket)
    if path.exists():
        raise FileExistsError(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    records = []
    independence_written = False
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
                    send_message(client, dict(
                        ok=True, prior=args.checkpoint, spec=editor.spec,
                        ddim=args.ddim_steps, execution_steps=args.execution_steps,
                        guidance_steps=9, guidance_scale=0.,
                        reference=str(args.reference), reference_repeat=1,
                        reference_interpolation=0,
                        reference_interpolation_threshold=args.reference_interpolation_threshold,
                        reference_interpolation_equal_jump=None,
                        reference_mode='expanded', editor=editor.metadata,
                        switch_action_index=args.switch_action_index,
                        after_switch='autonomous_prior_no_future_reference'))
                    continue
                assert msg['type'] == 'predict' and msg['guidance_scale'] == 0
                j = int(msg['reference_index'])
                seeds = msg['seeds']
                start = time.monotonic()
                if j < args.switch_action_index:
                    indices = np.minimum(np.arange(j, j + 9), reference.shape[1] - 1)
                    future = np.stack([reference[int(ri), indices]
                                       for ri in msg['reference_ids']])
                    action, stats = editor.predict(history, future, seeds)
                    mode = 'reference_edit'
                else:
                    # At scale zero the target is not used by GuidedDDIMController.
                    # The observation contains the last four states; action slots
                    # 22:44 therefore preserve the last three actually issued edit
                    # targets at the first autonomous call.
                    target = np.zeros((len(history), 9, 22), dtype=np.float32)
                    controller.set_fixed_noise_from_seeds(seeds)
                    action, prior_stats = controller.predict(history, target)
                    stats = dict(
                        guidance_mse_before=float(prior_stats.mse_before),
                        guidance_mse_after=float(prior_stats.mse_after))
                    mode = 'autonomous_prior'
                    if not independence_written:
                        controller.set_fixed_noise_from_seeds(seeds)
                        check, _ = controller.predict(history, np.ones_like(target))
                        max_change = float(np.max(np.abs(action - check)))
                        assert np.array_equal(action, check)
                        args.log.with_name('prior_independence.json').write_text(
                            json.dumps(dict(
                                first_autonomous_action_index=j,
                                reference_actions_used=False,
                                changed_dummy_reference_outputs_exact=True,
                                max_abs_output_change_rad=max_change), indent=2) + '\n')
                        independence_written = True
                row = dict(reference_index=j, mode=mode, guidance_scale=0.,
                           seeds=seeds, inference_seconds=time.monotonic() - start,
                           **stats)
                records.append(row)
                send_message(client, dict(ok=True, **row), action.astype(np.float32))
                print(json.dumps(row), flush=True)
    finally:
        server.close()
        path.unlink(missing_ok=True)
        args.log.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
