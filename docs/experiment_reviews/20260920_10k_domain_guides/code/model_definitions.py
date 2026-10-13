"""Three DDIM arms, with explicitly paired fresh prior noise per episode."""
import sys,os,json,socket,hashlib
from pathlib import Path
from types import MethodType
import numpy as np
import torch
from omegaconf import OmegaConf
# The new checkpoints retain the same arithmetic interpolations as training.
OmegaConf.register_new_resolver('eval', eval, replace=True)
PROJECT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
BASE=PROJECT/'.worktrees/sim-real-holding-comparison'
sys.path[:0]=[str(BASE/'eval/real'),str(BASE/'eval'),str(PROJECT)]
from policy_loader import load_policy
from real_sim_policy_server import sample
from trt_unet import _cached_fused_coeffs,fused_ddim_update,fused_guided_ddim_update
from ipc import send_message,recv_message
from diffusion_policy.guidance.guided_ddim import sample_guided_trajectory
out=Path(sys.argv[1]);torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
priorpath=Path('/home/carus/data_usb/obs_4-66.ckpt')
guidepath=Path(os.environ['GUIDE_CHECKPOINT'])
loaded,prior,spec=load_policy(priorpath,torch.device('cuda:0'),'ddim',4);del loaded
loaded,guide,gspec=load_policy(guidepath,torch.device('cuda:0'),'ddim',4);del loaded
assert spec==gspec
guide.conditional_sample=MethodType(sample,guide)
for p in [prior,guide]:
    for par in p.parameters():par.requires_grad_(False)

def predict(obs,noise,gnoise,arms):
    hist=torch.as_tensor(obs,device='cuda:0',dtype=torch.float32)
    xt=torch.as_tensor(noise,device='cuda:0',dtype=torch.float32).clone()
    arms=torch.as_tensor(arms,device='cuda:0',dtype=torch.long)
    cond=prior.normalizer['obs'].normalize(hist).reshape(len(hist),-1)
    refs=torch.zeros((len(hist),2,22),device='cuda:0')
    guided=arms==2
    with torch.no_grad():
        if guided.any():
            guide.recorded_noise=torch.as_tensor(gnoise,device='cuda:0',dtype=torch.float32)[guided]
            prediction=guide.predict_action({'obs':hist[guided]})['action_pred'][:,3:5]
            refs[guided]=prior.normalizer['action'].normalize(prediction)
        coeffs=_cached_fused_coeffs(prior,xt)
        for i,t in enumerate(coeffs.timesteps):
            eps=prior.model(xt,t,local_cond=None,global_cond=cond)
            nxt=torch.empty_like(xt)
            for arm in [0,1,2]:
                mask=arms==arm
                if not mask.any():continue
                if arm==0:nxt[mask]=fused_ddim_update(xt[mask],eps[mask],i,coeffs)
                else:nxt[mask]=fused_guided_ddim_update(xt[mask],eps[mask],i,coeffs,refs[mask],0. if arm==1 else 25.,slice(3,5))
            xt=nxt
        action=prior.normalizer['action'].unnormalize(xt[:,3:5])
    return action.cpu().numpy(),refs,cond,xt
