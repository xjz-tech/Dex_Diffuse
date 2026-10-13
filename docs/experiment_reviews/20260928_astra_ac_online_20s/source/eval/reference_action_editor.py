"""Reference-initialized DDIM editing with a correctly noised known-action prefix."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'eval'))
from inference_dp_controller import GuidedDDIMController


def ddim_transition(sample, epsilon, alpha_t, alpha_next, clip_sample=True):
    """Deterministic epsilon DDIM for an explicit (possibly nonuniform) t->s."""
    x0=(sample-(1-alpha_t).sqrt()*epsilon)/alpha_t.sqrt()
    clean=x0.clamp(-1,1) if clip_sample else x0
    # Match existing controller's use_clipped_model_output=True convention.
    corrected=(sample-alpha_t.sqrt()*clean)/(1-alpha_t).sqrt()
    return alpha_next.sqrt()*clean+(1-alpha_next).sqrt()*corrected, x0


class ReferenceActionEditor:
    def __init__(self,checkpoint,noise_ratio,steps=4,execution_steps=2):
        self.controller=GuidedDDIMController(Path(checkpoint),torch.device('cuda:0'),inference_steps=steps,execution_steps=execution_steps,guidance_scale=0.,eta=0.,fixed_noise=True,seed=42,allow_salvage=True)
        self.steps=steps;self.execution_steps=execution_steps;self.noise_ratio=float(noise_ratio)
        self.policy=self.controller.policy;self.spec=self.controller.spec
        assert self.spec['horizon']==12 and self.spec['n_obs_steps']==4 and self.spec['obs_dim']==66
        alphas=self.controller.scheduler.alphas_cumprod
        if self.noise_ratio==0:
            self.timesteps=[];self.actual_noise_ratio=0.
        else:
            ratios=((1-alphas)/alphas).sqrt()
            candidates=torch.arange(steps-1,len(alphas))
            t0=int(candidates[(ratios[candidates]-noise_ratio).abs().argmin()])
            self.timesteps=np.rint(np.linspace(t0,0,steps)).astype(int).tolist()
            assert len(set(self.timesteps))==steps and all(a>b for a,b in zip(self.timesteps,self.timesteps[1:]))
            self.actual_noise_ratio=float(ratios[t0])
        self.metadata=dict(algorithm='reference_initialized_ddim',requested_noise_ratio=noise_ratio,actual_noise_ratio=self.actual_noise_ratio,timesteps=self.timesteps,num_train_timesteps=len(alphas),inference_steps=steps,execution_steps=execution_steps,future_reference_steps=9,known_history_steps=3,eta=0.,guidance_scale=0.,clip_sample=bool(self.controller.scheduler.config.clip_sample),history_source='target_before from observation frames 1:4; equals actually issued previous three commands',weight_source=self.controller.checkpoint_info.weight_source)

    @torch.no_grad()
    def predict(self,history,future,seeds):
        future=np.asarray(future,dtype=np.float32)
        assert future.shape==(len(history),9,22)
        # Explicit identity baseline, without normalization/rounding/clipping.
        if self.noise_ratio==0:
            return future[:,:self.execution_steps].copy(),dict(edit_rmse_rad=0.,edit_max_abs_rad=0.,history_mask_max_error=0.,zero_edit_exact=True)
        c=self.controller;device=c.device
        h=torch.as_tensor(history,device=device,dtype=self.policy.dtype)
        assert h.shape[1:]==(4,66)
        known=h[:,1:,22:44]
        clean_rad=torch.cat([known,torch.as_tensor(future,device=device,dtype=self.policy.dtype)],dim=1)
        clean=self.policy.normalizer['action'].normalize(clean_rad)
        cond=self.policy.normalizer['obs'].normalize(h).reshape(h.shape[0],-1)
        if seeds is not None: c.set_fixed_noise_from_seeds(seeds)
        noise=c._noise(len(h),h.dtype)
        alphas=c.scheduler.alphas_cumprod.to(device=device,dtype=h.dtype)
        a=alphas[self.timesteps[0]];sample=a.sqrt()*clean+(1-a).sqrt()*noise
        noisy_rad=self.policy.normalizer['action'].unnormalize(sample)
        injected_rmse=float((noisy_rad[:,3:]-clean_rad[:,3:]).square().mean().sqrt())
        mask_error=0.;clipped=0.;evaluated=0
        for i,t in enumerate(self.timesteps):
            at=alphas[t]
            known_at=at.sqrt()*clean[:,:3]+(1-at).sqrt()*noise[:,:3]
            sample[:,:3]=known_at
            mask_error=max(mask_error,float((sample[:,:3]-known_at).abs().max()))
            epsilon=c._predict_epsilon(sample,torch.tensor(t,device=device),global_cond=cond)
            next_t=self.timesteps[i+1] if i+1<len(self.timesteps) else -1
            an=alphas[next_t] if next_t>=0 else torch.ones_like(at)
            sample,x0=ddim_transition(sample,epsilon,at,an,bool(c.scheduler.config.clip_sample))
            clipped+=int((x0[:,3:].abs()>1).sum());evaluated+=x0[:,3:].numel()
        sample[:,:3]=clean[:,:3]
        edited=self.policy.normalizer['action'].unnormalize(sample)[:,3:]
        assert torch.isfinite(edited).all()
        delta=edited-clean_rad[:,3:]
        stats=dict(edit_rmse_rad=float(delta.square().mean().sqrt()),edit_max_abs_rad=float(delta.abs().max()),executed_prefix_edit_rmse_rad=float(delta[:,:self.execution_steps].square().mean().sqrt()),injected_rmse_rad=injected_rmse,history_mask_max_error=mask_error,predicted_x0_clip_fraction=clipped/evaluated)
        return edited[:,:self.execution_steps].cpu().numpy(),stats
