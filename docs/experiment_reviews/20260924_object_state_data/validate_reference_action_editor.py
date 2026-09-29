import json,sys
from pathlib import Path
import numpy as np,torch
from reference_action_editor import ReferenceActionEditor,ddim_transition
from diffusers import DDIMScheduler
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode54_reference_edit_20260928';O.mkdir(exist_ok=True)
torch.set_num_threads(2)
editor=ReferenceActionEditor('/home/carus/data_usb/10B_obs_4-66.ckpt',.1)
# Verify explicit-transition code against the official scheduler at equal spacing.
sched=DDIMScheduler.from_config(editor.controller.scheduler.config,set_alpha_to_one=True,steps_offset=0);sched.set_timesteps(4,device='cuda:0')
g=torch.Generator(device='cuda:0').manual_seed(123);maxerr=0.
for i,t in enumerate(sched.timesteps):
 x=torch.randn((2,12,22),device='cuda:0',generator=g);eps=torch.randn(x.shape,device=x.device,generator=g);at=sched.alphas_cumprod[t].to(x.device);nxt=sched.timesteps[i+1] if i+1<4 else -1;an=sched.alphas_cumprod[nxt].to(x.device) if nxt>=0 else torch.ones_like(at)
 a,_=ddim_transition(x,eps,at,an,bool(sched.config.clip_sample));b=sched.step(eps,t,x,eta=0.,use_clipped_model_output=True).prev_sample;torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6);maxerr=max(maxerr,float((a-b).abs().max()))
# Oracle epsilon must exactly recover a known clean trajectory on nonuniform times.
clean=torch.rand((2,12,22),device='cuda:0')-.5;noise=torch.randn(clean.shape,device=clean.device,generator=g);alphas=editor.controller.scheduler.alphas_cumprod.to('cuda:0');ts=editor.timesteps;x=alphas[ts[0]].sqrt()*clean+(1-alphas[ts[0]]).sqrt()*noise
for i,t in enumerate(ts):
 at=alphas[t];an=alphas[ts[i+1]] if i+1<len(ts) else torch.ones_like(at);epsilon=(x-at.sqrt()*clean)/(1-at).sqrt();x,_=ddim_transition(x,epsilon,at,an,False)
torch.testing.assert_close(x,clean,rtol=1e-5,atol=1e-6)
trace=json.loads((R/'qualified_comparison/episode_54/direct_m044_mu11/trace.json').read_text());ref=np.load(R/'qualified_comparison/episode_54/reference_full.npz')['hand_target_rad'][0]
def observation(row):
 q=np.asarray(row['q'],dtype=np.float32);target=np.asarray(row['command'],dtype=np.float32);return np.concatenate([q,target,target-q])
records=[]
# Reuse one loaded network while independently setting the declared schedule.
for ratio in [0.,.1,.2,.35]:
 editor.noise_ratio=ratio
 if ratio==0:editor.timesteps=[];actual=0.
 else:
  ratios=((1-alphas)/alphas).sqrt();candidates=torch.arange(3,len(alphas),device='cuda:0');t0=int(candidates[(ratios[candidates]-ratio).abs().argmin()]);editor.timesteps=np.rint(np.linspace(t0,0,4)).astype(int).tolist();actual=float(ratios[t0]);assert len(set(editor.timesteps))==4
 for j in [0,20,40,80]:
  hist=np.repeat(observation(trace[59])[None],4,axis=0) if j==0 else np.stack([observation(x) for x in trace[60+j-4:60+j]])
  future=ref[np.minimum(np.arange(j,j+9),len(ref)-1)][None];action,stats=editor.predict(hist[None],future,[44])
  if ratio==0:assert np.array_equal(action,future[:,:2])
  records.append(dict(requested_noise_ratio=ratio,actual_noise_ratio=actual,timesteps=editor.timesteps,reference_index=j,**stats))
result=dict(official_scheduler_max_abs_error=maxerr,oracle_reconstruction_max_abs_error=float((x-clean).abs().max()),zero_edit_exact=True,model=editor.metadata,offline_same_observation_comparisons=records)
(O/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
