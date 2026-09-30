"""Cross-check 10B-prior / 1B-guide fused sampling against production autograd."""
from pathlib import Path
from types import MethodType
import sys,json
import torch,numpy as np
from omegaconf import OmegaConf
R=Path(__file__).resolve().parents[1];ROOT=R.parents[2];BASE=ROOT/'.worktrees/sim-real-holding-comparison/eval'
sys.path[:0]=[str(BASE/'real'),str(BASE),str(ROOT)]
from policy_loader import load_policy
from real_sim_policy_server import sample
from trt_unet import _cached_fused_coeffs,fused_guided_ddim_update
from diffusion_policy.guidance.guided_ddim import sample_guided_trajectory
OmegaConf.register_new_resolver('eval',eval,replace=True);torch.set_num_threads(4)
_,prior,ps=load_policy('/home/carus/data_usb/10B_obs_4-66.ckpt',torch.device('cpu'),'ddim',4)
_,guide,gs=load_policy('/home/carus/data_usb/obs_4-66.ckpt',torch.device('cpu'),'ddim',4);assert ps==gs
q=np.load(R.parent/'20260920_10k_domain_guides/evaluation/ordinary_1b/initial_state.npz')['q'][[0,12,24,36]]
hist=torch.as_tensor(np.repeat(np.concatenate([q,q,np.zeros_like(q)],axis=-1)[:,None],4,axis=1));noise=torch.as_tensor(np.tile(np.random.default_rng(8).standard_normal((1,12,22)).astype(np.float32),(4,1,1)));gnoise=torch.as_tensor(np.tile(np.random.default_rng(100008).standard_normal((1,12,22)).astype(np.float32),(4,1,1)))
guide.conditional_sample=MethodType(sample,guide);guide.recorded_noise=gnoise
with torch.no_grad():
 ref=prior.normalizer['action'].normalize(guide.predict_action({'obs':hist})['action_pred'][:,3:5]);cond=prior.normalizer['obs'].normalize(hist).reshape(4,-1);coeffs=_cached_fused_coeffs(prior,noise);fused=noise.clone()
 for i,t in enumerate(coeffs.timesteps):
  eps=prior.model(fused,t,local_cond=None,global_cond=cond)
  fused=fused_guided_ddim_update(fused,eps,i,coeffs,ref,25.,slice(3,5))
production=sample_guided_trajectory(prior.model,prior.noise_scheduler,noise.clone(),cond,ref,4,25.,slice(3,5)).trajectory
maxerr=float((fused-production).abs().max());raderr=float((prior.normalizer['action'].unnormalize(fused[:,3:5])-prior.normalizer['action'].unnormalize(production[:,3:5])).abs().max());assert maxerr<2e-5 and raderr<2e-5,(maxerr,raderr)
out={'passed':True,'prior':'10B_obs_4-66.ckpt','guide':'obs_4-66.ckpt','ddim_timesteps':[int(t) for t in coeffs.timesteps],'scale':25,'guided_executable_action_slice':[3,5],'max_normalized_trajectory_error':maxerr,'max_executed_target_error_rad':raderr}
(R/'guided_sampler_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
