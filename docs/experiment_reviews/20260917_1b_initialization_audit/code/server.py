import sys, socket, json, os
from pathlib import Path
from types import MethodType
import numpy as np
import torch

BASE=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/sim-real-holding-comparison')
sys.path[:0]=[str(BASE/'eval/real'),str(BASE/'eval')]
from policy_loader import load_policy
from real_sim_policy_server import sample
from ipc import send_message, recv_message

out=Path(sys.argv[1]); torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
loaded,policy,spec=load_policy('/home/carus/data_usb/obs_4-66.ckpt',torch.device('cuda:0'),'ddim',4)
policy.conditional_sample=MethodType(sample,policy)
del loaded
def predict(obs,noise):
    policy.recorded_noise=torch.as_tensor(noise,device='cuda:0',dtype=torch.float32)
    with torch.inference_mode():
        return policy.predict_action({'obs':torch.as_tensor(obs,device='cuda:0',dtype=torch.float32)})['action'].cpu().numpy()
rows=[json.loads(x) for x in (BASE/'reports/hold_comparison_20260917/pair1_seed50_hold0.jsonl').open()]
inp=next(x for x in rows if x['event']=='inference_input')
res=next(x for x in rows if x['event']=='inference_output')
got=predict(inp['observation'],res['details']['sampling']['initial_noise'])
err=float(np.max(abs(got[0]-np.array(res['returned_actions_rad']))))
assert err<1e-5,err
(out/'model_validation.json').write_text(json.dumps({'checkpoint':'/home/carus/data_usb/obs_4-66.ckpt','reference_error_rad':err,'spec':spec},default=str,indent=2))
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.bind(os.environ['AUDIT_SOCKET']);s.listen(1)
print('READY',flush=True)
c,_=s.accept()
try:
    while True:
        m,obs=recv_message(c)
        if m['type']=='stop':break
        send_message(c,{'ok':True},predict(obs,m['noise']))
finally:
    c.close();s.close();Path(os.environ['AUDIT_SOCKET']).unlink()
