"""Three DDIM arms, with explicitly paired fresh prior noise per episode."""
import sys,os,json,socket,hashlib
from pathlib import Path
from types import MethodType
import numpy as np
import torch
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
guidepath=PROJECT/'runs/sim_hand_10k_seed42/checkpoints/latest.ckpt'
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

old=PROJECT/'docs/experiment_reviews/20260917_1b_initialization_audit'
q=np.load(old/'initial_state.npz')['q'][32:48].astype(np.float32)
obs=np.repeat(np.concatenate([q,q,np.zeros_like(q)],axis=1)[:,None],4,axis=1)
noise=np.stack([np.random.default_rng(i).standard_normal((12,22)).astype(np.float32) for i in range(50,66)])
gnoise=np.stack([np.random.default_rng(i+100000).standard_normal((12,22)).astype(np.float32) for i in range(50,66)])
checks={};preds=[]
for arm in [0,1,2]:
    got,refs,cond,traj=predict(obs,noise,gnoise,[arm]*16);preds.append(got)
    if arm==0:
        expected=np.load(old/'initial_predictions.npz')['action'][32:48,:2]
    else:
        ref=sample_guided_trajectory(model=lambda x,t,global_cond:prior.model(x,t,local_cond=None,global_cond=global_cond),scheduler=prior.noise_scheduler,initial_noise=torch.as_tensor(noise,device='cuda:0'),global_cond=cond,reference=refs,num_inference_steps=4,guidance_scale=0. if arm==1 else 25.,guidance_slice=slice(3,5))
        expected=prior.normalizer['action'].unnormalize(ref.trajectory[:,3:5]).detach().cpu().numpy()
    err=float(np.max(abs(got-expected)));assert err<1e-5,(arm,err)
    checks[str(arm)]={'max_reference_error_rad':err}
checks['sampler_only_max_action_difference_rad']=float(np.max(abs(preds[0]-preds[1])))
checks['guide_max_action_difference_rad']=float(np.max(abs(preds[2]-preds[1])))
np.savez_compressed(out/'initial_predictions.npz',ordinary=preds[0],scale0=preds[1],scale25=preds[2])
(out/'model_validation.json').write_text(json.dumps(checks,indent=2))
(out/'model_manifest.json').write_text(json.dumps({'prior':str(priorpath),'guide':str(guidepath),'prior_sha256':hashlib.sha256(priorpath.read_bytes()).hexdigest(),'guide_sha256':hashlib.sha256(guidepath.read_bytes()).hexdigest(),'spec':spec,'guide_scale':25,'guide_steps':2,'ddim_steps':4,'guide_seed_offset':100000,'backend':'PyTorch FP32, fused analytic update verified against production autograd guidance'},indent=2))
s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.bind(os.environ['AUDIT_SOCKET']);s.listen(1);print('READY',checks,flush=True)
c,_=s.accept()
try:
    while True:
        msg,hist=recv_message(c)
        if msg['type']=='stop':break
        got,_,_,_=predict(hist,msg['noise'],msg['guide_noise'],msg['arms'])
        send_message(c,{'ok':True},got)
finally:
    c.close();s.close();Path(os.environ['AUDIT_SOCKET']).unlink()
