"""Edit four reference actions while completing five free future actions.

The free tail first descends from near-terminal noise to the requested editing
level, conditioned on three known commands and four reference actions. At that
level the reference prefix is released for editing. No later references enter
this class. Warmup costs four extra network evaluations per planning call.
"""
import numpy as np
import torch
from reference_action_editor import ReferenceActionEditor,ddim_transition

class Reference4Free5Editor(ReferenceActionEditor):
 def __init__(self,checkpoint,noise_ratio,steps=4,execution_steps=2):
  super().__init__(checkpoint,noise_ratio,steps,execution_steps)
  self.warmup_timesteps=[99,75,50,25]
  assert self.metadata['num_train_timesteps']==100
  assert execution_steps<=4
  if self.timesteps:assert self.timesteps[0]<self.warmup_timesteps[-1]
  self.metadata.update(algorithm='reference4_free5_high_noise_completion_then_local_edit',future_reference_steps=4,generated_future_steps=9,free_tail_steps=5,tail_completion_timesteps=self.warmup_timesteps,tail_completion_network_calls=4,total_network_calls=4+steps,reference_prefix_hard_only_during_tail_completion=True,reference_prefix_editable_during_local_denoising=True,free_tail_initialization='standard Gaussian at t99 then conditional reverse DDIM to editing t0',reference_after_four_used=False)

 @torch.no_grad()
 def predict(self,history,future,seeds):
  future=np.asarray(future,dtype=np.float32)
  assert future.shape==(len(history),4,22), 'only four future reference actions may be supplied'
  if self.noise_ratio==0:
   return future[:,:self.execution_steps].copy(),dict(zero_edit_exact=True,edit_rmse_rad=0.,edit_max_abs_rad=0.,history_mask_max_error=0.,tail_completion_mask_max_error=0.,network_calls=0)
  c=self.controller;device=c.device
  h=torch.as_tensor(history,device=device,dtype=self.policy.dtype)
  assert h.shape[1:]==(4,66)
  clean_rad=torch.cat([h[:,1:,22:44],torch.as_tensor(future,device=device,dtype=h.dtype)],dim=1)
  clean=self.policy.normalizer['action'].normalize(clean_rad)
  cond=self.policy.normalizer['obs'].normalize(h).reshape(len(h),-1)
  c.set_fixed_noise_from_seeds(seeds);noise=c._noise(len(h),h.dtype)
  alphas=c.scheduler.alphas_cumprod.to(device=device,dtype=h.dtype)
  clip=bool(c.scheduler.config.clip_sample);sample=noise.clone();warmup_error=0.
  for i,t in enumerate(self.warmup_timesteps):
   at=alphas[t];known_at=at.sqrt()*clean+(1-at).sqrt()*noise[:,:7]
   sample[:,:7]=known_at
   warmup_error=max(warmup_error,float((sample[:,:7]-known_at).abs().max()))
   eps=c._predict_epsilon(sample,torch.tensor(t,device=device),global_cond=cond)
   nxt=self.warmup_timesteps[i+1] if i+1<len(self.warmup_timesteps) else self.timesteps[0]
   sample,_=ddim_transition(sample,eps,at,alphas[nxt],clip)
  # All twelve positions now correspond to the same editing noise level.
  a=alphas[self.timesteps[0]]
  prefix_at=a.sqrt()*clean+(1-a).sqrt()*noise[:,:7]
  sample[:,:7]=prefix_at
  injected=self.policy.normalizer['action'].unnormalize(sample[:,3:7])
  injected_rmse=float((injected-clean_rad[:,3:7]).square().mean().sqrt())
  mask_error=0.;clipped=0;count=0
  for i,t in enumerate(self.timesteps):
   at=alphas[t];history_at=at.sqrt()*clean[:,:3]+(1-at).sqrt()*noise[:,:3]
   sample[:,:3]=history_at
   mask_error=max(mask_error,float((sample[:,:3]-history_at).abs().max()))
   eps=c._predict_epsilon(sample,torch.tensor(t,device=device),global_cond=cond)
   nxt=self.timesteps[i+1] if i+1<len(self.timesteps) else -1
   an=alphas[nxt] if nxt>=0 else torch.ones_like(at)
   sample,x0=ddim_transition(sample,eps,at,an,clip)
   clipped+=int((x0[:,3:].abs()>1).sum());count+=x0[:,3:].numel()
  sample[:,:3]=clean[:,:3]
  edited=self.policy.normalizer['action'].unnormalize(sample)[:,3:]
  assert torch.isfinite(edited).all()
  delta=edited[:,:4]-clean_rad[:,3:7]
  stats=dict(edit_rmse_rad=float(delta.square().mean().sqrt()),edit_max_abs_rad=float(delta.abs().max()),executed_prefix_edit_rmse_rad=float(delta[:,:self.execution_steps].square().mean().sqrt()),injected_rmse_rad=injected_rmse,history_mask_max_error=mask_error,tail_completion_mask_max_error=warmup_error,predicted_x0_clip_fraction=clipped/count,network_calls=4+len(self.timesteps),future_reference_steps_used=4,free_tail_steps=5)
  return edited[:,:self.execution_steps].cpu().numpy(),stats
