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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'eval'))
from inference_dp_controller import GuidedDDIMController
from ipc import recv_message,send_message

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket',required=True)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('/home/carus/data_usb/obs_4-66.ckpt'))
    p.add_argument('--reference',type=Path,default=Path('/home/carus/Data/bulb_right_turn_reference_260909/id_37_f325_400/reference.npz'))
    args=p.parse_args()
    torch.set_num_threads(2)
    controller=GuidedDDIMController(args.checkpoint,torch.device('cuda:0'),inference_steps=4,
        execution_steps=2,guidance_scale=25.,eta=0.,fixed_noise=True,seed=42,allow_salvage=True)
    controller.set_guidance_horizon(2)
    reference=np.load(args.reference,allow_pickle=False)['hand_target_rad']
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
                        ddim=4,execution_steps=2,guidance_steps=2,guidance_scale=25,reference=str(args.reference)))
                    continue
                if msg['type']!='predict':raise ValueError(msg['type'])
                j=int(msg['reference_index']);assert 0<=j<len(reference)
                batch=len(history);seeds=msg.get('noise_seeds',[42+i for i in range(batch)])
                assert len(seeds)==batch
                if j==0:controller.set_fixed_noise_from_seeds(seeds)
                indices=np.minimum(np.arange(j,j+2),len(reference)-1)
                target=np.repeat(reference[indices][None],batch,axis=0)
                start=time.monotonic()
                result,stats=controller.predict(history,target)
                elapsed=time.monotonic()-start
                record=dict(reference_index=j,inference_seconds=elapsed,
                    guidance_mse_before=stats.mse_before,guidance_mse_after=stats.mse_after,
                    output_vs_reference_rmse_rad=np.sqrt(((result-target)**2).mean((1,2))).tolist())
                records.append(record)
                send_message(client,dict(ok=True,**record),result.astype(np.float32))
                print('[reference-server]',json.dumps(record),flush=True)
    finally:
        server.close()
        if path.exists():path.unlink()
        args.log.write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
