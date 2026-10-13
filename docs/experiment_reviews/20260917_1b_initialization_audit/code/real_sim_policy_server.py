"""Offline-only model server: archived real sampler, with recorded diffusion noise."""
import argparse
import json
import socket
import sys
from pathlib import Path
from types import MethodType
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'real'))
from policy_loader import load_policy
from trt_unet import _cached_fused_coeffs, fused_ddim_update
from ipc import send_message, recv_message


def sample(policy, condition_data, condition_mask, local_cond=None, global_cond=None, **kwargs):
    trajectory=policy.recorded_noise.clone()
    assert trajectory.shape==condition_data.shape and not condition_mask.any()
    coeffs=_cached_fused_coeffs(policy,condition_data)
    for i,timestep in enumerate(coeffs.timesteps):
        epsilon=policy.model(trajectory,timestep,local_cond=local_cond,global_cond=global_cond)
        trajectory=fused_ddim_update(trajectory,epsilon,i,coeffs)
    return trajectory


def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True)
    p.add_argument('--checkpoint',default='/home/carus/data_usb/obs_4-66.ckpt')
    p.add_argument('--out',required=True);a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    loaded,policy,spec=load_policy(a.checkpoint,torch.device('cuda:0'),'ddim',4)
    policy.conditional_sample=MethodType(sample,policy)
    del loaded
    def predict(history,noise):
        policy.recorded_noise=torch.as_tensor(noise,dtype=torch.float32,device='cuda:0')
        with torch.inference_mode():
            return policy.predict_action({'obs':torch.as_tensor(history,device='cuda:0',dtype=torch.float32)})['action'].cpu().numpy()
    root=HERE.parent
    checks=[]
    for path in [root/'reports/hold_comparison_20260917/pair1_seed50_hold0.jsonl',root/'reports/bulb_new_hold_comparison_20260917_1640/bulb_new_hold_164011.jsonl']:
        rows=[json.loads(l) for l in path.open()]
        ins={r['chunk']:r for r in rows if r['event']=='inference_input' and not r.get('synthetic')}
        outs={r['chunk']:r for r in rows if r['event']=='inference_output' and not r.get('synthetic')}
        for k in [0,1,len(outs)//2,len(outs)-1]:
            got=predict(ins[k]['observation'],outs[k]['details']['sampling']['initial_noise'])
            expected=np.array(outs[k]['returned_actions_rad'])
            error=float(np.max(np.abs(got[0]-expected)))
            checks.append(dict(source=str(path.relative_to(root)),chunk=k,max_error_rad=error,max_error_deg=float(np.rad2deg(error))))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'model_validation.json').write_text(json.dumps(dict(backend='PyTorch FP32, archived fused DDIM math; recorded noise',checks=checks),indent=2))
    print('MODEL_VALIDATION',checks,flush=True)
    if max(x['max_error_deg'] for x in checks)>.05:
        raise RuntimeError('Model replay differs from archived TensorRT by >0.05 degrees')
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.bind(a.socket);sock.listen(1)
    print('READY',flush=True)
    conn,_=sock.accept()
    try:
        while True:
            msg,obs=recv_message(conn)
            if msg['type']=='stop':break
            action=predict(obs,msg['noise'])
            send_message(conn,dict(ok=True),action)
    finally:
        conn.close();sock.close();Path(a.socket).unlink(missing_ok=True)


if __name__=='__main__':main()
