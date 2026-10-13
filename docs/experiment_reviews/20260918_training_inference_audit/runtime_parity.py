"""Exercise the real model server over IPC and the standalone real-policy loader.

No simulator or hardware interfaces are imported or started.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'eval'),str(ROOT)]
from checkpoint_loader import load_checkpoint,build_policy,configure_policy_sampler
from ipc import send_message,recv_message
torch.set_num_threads(4)
ckpt='/home/carus/data_usb/obs_4-66.ckpt'
loaded=load_checkpoint(ckpt,allow_salvage=False)
policy,spec=build_policy(loaded)
configure_policy_sampler(policy,'ddim',4)
policy.n_action_steps=2
policy.cuda().eval()
obs=np.load(OUT/'samples.npz')['obs'][:8,:4].copy()
with torch.inference_mode():
    torch.manual_seed(42)
    direct=policy.predict_action({'obs':torch.tensor(obs,device='cuda')})['action'].cpu().numpy()
result={}
with tempfile.TemporaryDirectory(prefix='dp_audit_') as temp:
    sockpath=str(Path(temp)/'model.sock')
    log=open(OUT/'server.log','w')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    proc=subprocess.Popen([sys.executable,str(ROOT/'eval/model_server.py'),'--checkpoint',ckpt,
         '--socket',sockpath,'--sampler','ddim','--inference-steps','4','--n-action-steps','2',
         '--seed','42','--no-salvage'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    sock=None
    try:
        until=time.monotonic()+45
        while not Path(sockpath).exists():
            if proc.poll() is not None: raise RuntimeError('model server exited; see server.log')
            if time.monotonic()>until: raise TimeoutError('model server did not start')
            time.sleep(.1)
        sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.settimeout(30);sock.connect(sockpath)
        send_message(sock,{'type':'hello','request_id':1})
        info,_=recv_message(sock)
        if not info.get('ok'): raise RuntimeError(info)
        send_message(sock,{'type':'predict','request_id':2},obs)
        response,pred=recv_message(sock)
        if not response['ok']: raise RuntimeError(response)
        result['server_spec']=info['spec']
        result['ipc_server_vs_direct_max_abs_error']=float(np.max(abs(pred-direct)))
        send_message(sock,{'type':'shutdown','request_id':3});recv_message(sock)
        proc.wait(timeout=10)
        result['server_exit_code']=proc.returncode
    finally:
        if sock is not None: sock.close()
        if proc.poll() is None: proc.terminate();proc.wait(timeout=10)
        log.close()

# The standalone real policy is a separate implementation; exercise it offline.
sys.path.insert(0,str(ROOT/'eval/real'))
from policy_loader import build_policy as build_real_policy
real,_=build_real_policy(loaded)
configure_policy_sampler(real,'ddim',4)
real.n_action_steps=2
real.cuda().eval()
with torch.inference_mode():
    torch.manual_seed(42)
    real_pred=real.predict_action({'obs':torch.tensor(obs,device='cuda')})['action'].cpu().numpy()
result['standalone_real_policy_vs_eval_max_abs_error']=float(np.max(abs(real_pred-direct)))
(OUT/'runtime_parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
