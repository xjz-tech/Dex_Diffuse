#!/usr/bin/env python3
"""Serve the selected guided controller with a recorded action reference."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'eval'))
from inference_dp_controller import GuidedDDIMController
from ipc import recv_message,send_message
from reference_resampling import interpolate_actions,interpolate_large_jumps,interpolate_equal_jumps

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('/home/carus/data_usb/obs_4-66.ckpt'))
    p.add_argument('--reference',type=Path,default=Path('/home/carus/Data/bulb_right_turn_reference_260909/id_37_f325_400/reference.npz'))
    p.add_argument('--guidance-steps',type=int,default=2)
    p.add_argument('--ddim-steps',type=int,default=4)
    p.add_argument('--guidance-scale',type=float,default=25.)
    p.add_argument('--execution-steps',type=int,default=2)
    p.add_argument('--reference-repeat',type=int,default=1,
                   help='Repeat each recorded action this many control steps before forming guidance windows.')
    p.add_argument('--reference-interpolation',type=int,default=0,
                   help='Insert this many linearly interpolated targets between original actions.')
    p.add_argument('--reference-interpolation-threshold',type=float,default=None,
                   help='Insert one midpoint only when any adjacent joint jump exceeds this many radians.')
    p.add_argument('--reference-interpolation-equal-jump',type=float,default=None,
                   help='Insert one midpoint only when the maximum adjacent joint jump equals this value (absolute tolerance 1e-6 rad).')
    p.add_argument('--reference-mode',choices=['expanded','tile_per_call'],default='expanded')
    args=p.parse_args()
    torch.set_num_threads(2)
    if args.ddim_steps <= 0:raise ValueError('ddim-steps must be positive')
    controller=GuidedDDIMController(args.checkpoint,torch.device('cuda:0'),inference_steps=args.ddim_steps,
        execution_steps=args.execution_steps,guidance_scale=args.guidance_scale,eta=0.,fixed_noise=True,seed=42,allow_salvage=True)
    controller.set_guidance_horizon(args.guidance_steps)
    if args.reference_repeat < 1:raise ValueError('reference-repeat must be positive')
    if args.reference_interpolation < 0 or (args.reference_interpolation and
       (args.reference_repeat != 1 or args.reference_mode != 'expanded')):
        raise ValueError('interpolation requires expanded mode and repeat=1')
    if args.reference_interpolation_threshold is not None and (
        args.reference_interpolation != 0 or args.reference_repeat != 1 or
        args.reference_mode != 'expanded'):
        raise ValueError('threshold interpolation requires expanded mode, repeat=1, no uniform insertion')
    if args.reference_interpolation_equal_jump is not None and (
        args.reference_interpolation_threshold is not None or
        args.reference_interpolation != 0 or args.reference_repeat != 1 or
        args.reference_mode != 'expanded'):
        raise ValueError('equal-jump interpolation requires expanded mode, repeat=1, no other insertion')
    source_reference=np.load(args.reference,allow_pickle=False)['hand_target_rad']
    if args.reference_interpolation_equal_jump is not None:
        reference,_=interpolate_equal_jumps(source_reference,args.reference_interpolation_equal_jump)
    elif args.reference_interpolation_threshold is None:
        reference=interpolate_actions(np.repeat(source_reference,args.reference_repeat,axis=1),
                                      args.reference_interpolation)
    else:
        reference,_=interpolate_large_jumps(source_reference,args.reference_interpolation_threshold)
    path=Path(args.socket)
    if path.exists():raise FileExistsError('Refusing to replace an existing socket: '+str(path))
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    records=[]
    try:
        server.bind(str(path));server.listen(1)
        print('[reference-server] ready',path,flush=True)
        client,_=server.accept()
        with client:
            while True:
                msg,history=recv_message(client)
                if msg['type']=='shutdown':break
                if msg['type']=='hello':
                    send_message(client,dict(ok=True,prior=str(args.checkpoint),spec=controller.spec,
                        ddim=args.ddim_steps,execution_steps=args.execution_steps,guidance_steps=args.guidance_steps,
                        guidance_scale=args.guidance_scale,reference=str(args.reference),
                        reference_repeat=args.reference_repeat,
                        reference_interpolation=args.reference_interpolation,
                        reference_interpolation_threshold=args.reference_interpolation_threshold,
                        reference_interpolation_equal_jump=args.reference_interpolation_equal_jump,
                        reference_mode=args.reference_mode))
                    continue
                if msg['type']!='predict':raise ValueError(msg['type'])
                j=int(msg['reference_index'])
                requested_scale=float(msg.get('guidance_scale',args.guidance_scale))
                if not np.isfinite(requested_scale) or requested_scale < 0:
                    raise ValueError('guidance_scale must be finite and nonnegative')
                controller.guidance_scale=requested_scale
                if args.reference_mode=='expanded':assert 0<=j<reference.shape[1]
                else:assert 0<=j//args.execution_steps<source_reference.shape[1]
                batch=len(history);seeds=msg.get('seeds',[42+i for i in range(batch)])
                assert len(seeds)==batch
                if args.reference_mode=='expanded':
                    indices=np.minimum(np.arange(j,j+args.guidance_steps),reference.shape[1]-1)
                    target=np.stack([reference[int(ri),indices] for ri in msg['reference_ids']])
                else:
                    target=np.stack([np.repeat(source_reference[int(ri),j//args.execution_steps][None],
                                               args.guidance_steps,axis=0)
                                     for ri in msg['reference_ids']])
                start=time.monotonic()
                parts=[];befores=[];afters=[]
                for lo in range(0,batch,9):
                    controller.set_fixed_noise_from_seeds(seeds[lo:lo+9])
                    part,stats=controller.predict(history[lo:lo+9],target[lo:lo+9]);parts.append(part);befores.append(stats.mse_before);afters.append(stats.mse_after)
                result=np.concatenate(parts)
                elapsed=time.monotonic()-start
                record=dict(reference_index=j,guidance_scale=requested_scale,inference_seconds=elapsed,
                    guidance_mse_before=float(np.mean(befores)),guidance_mse_after=float(np.mean(afters)),
                    output_vs_reference_rmse_rad=np.sqrt(((result-target[:,:result.shape[1]])**2).mean((1,2))).tolist())
                records.append(record)
                send_message(client,dict(ok=True,**record),result.astype(np.float32))
                print('[reference-server]',json.dumps(record),flush=True)
    finally:
        server.close()
        if path.exists():path.unlink()
        args.log.write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
