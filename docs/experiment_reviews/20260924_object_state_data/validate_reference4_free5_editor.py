"""Check noisy masks, actual model calls and absence of future-reference leakage."""
import json
import numpy as np,torch
from reference4_free5_editor import Reference4Free5Editor
from reference_action_editor import ddim_transition
from server_reference4_free5_editor import reference_window4
from reference_resampling import interpolate_large_jumps
from run_four_reference4_free5_sweep import O,R
O.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
case=R/'qualified_comparison/episode_54'
ref,progress=interpolate_large_jumps(np.load(case/'reference_full.npz')['hand_target_rad'],.1)
a=reference_window4(ref,[0],0);changed=ref.copy();changed[:,4:]=12345.;b=reference_window4(changed,[0],0);assert np.array_equal(a,b)
t=json.loads((case/'direct_m044_mu11/trace.json').read_text());r=t[59];q=np.asarray(r['q'],np.float32);target=np.asarray(r['command'],np.float32);h=np.repeat(np.concatenate([q,target,target-q])[None,None],4,axis=1)
records=[]
for ratio,steps in [(.1,4),(.1,6),(.15,4),(.15,6),(.2,4),(.2,6)]:
 editor=Reference4Free5Editor('/home/carus/data_usb/10B_obs_4-66.ckpt',ratio,steps,2)
 c=editor.controller;orig=c._predict_epsilon;calls=[];maxmask=0.
 clean=torch.as_tensor(np.concatenate([h[:,1:,22:44],a],axis=1),device=c.device,dtype=editor.policy.dtype);clean=editor.policy.normalizer['action'].normalize(clean)
 c.set_fixed_noise_from_seeds([44]);noise=c._noise(1,editor.policy.dtype)
 def inspect(sample,timestep,global_cond):
  global maxmask
  ti=int(timestep);k=len(calls);calls.append(ti);m=7 if k<=4 else 3
  at=c.scheduler.alphas_cumprod[ti].to(device=c.device,dtype=sample.dtype)
  expected=at.sqrt()*clean[:,:m]+(1-at).sqrt()*noise[:,:m]
  maxmask=max(maxmask,float((sample[:,:m]-expected).abs().max()))
  assert torch.isfinite(sample).all()
  return orig(sample,timestep,global_cond=global_cond)
 c._predict_epsilon=inspect
 output,stats=editor.predict(h,a,[44]);assert calls==editor.warmup_timesteps+editor.timesteps and maxmask==0.
 assert output.shape==(1,2,22) and np.isfinite(output).all() and np.max(np.abs(output-a[:,:2]))>1e-5
 c._predict_epsilon=orig
 repeat,_=editor.predict(h,b,[44]);assert np.array_equal(output,repeat)
 try:editor.predict(h,ref[:,:9],[44])
 except AssertionError:pass
 else:raise AssertionError('engine accepted extra reference actions')
 # Oracle checks the full (warmup + local) nonuniform DDIM sequence.
 alpha=c.scheduler.alphas_cumprod.to(c.device);clean_full=torch.zeros((1,12,22),device=c.device);clean_full[:,7:]=.25
 ts=editor.warmup_timesteps+editor.timesteps;x=alpha[ts[0]].sqrt()*clean_full+(1-alpha[ts[0]]).sqrt()*noise
 for i,ti in enumerate(ts):
  at=alpha[ti];an=alpha[ts[i+1]] if i+1<len(ts) else torch.ones_like(at);eps=(x-at.sqrt()*clean_full)/(1-at).sqrt();x,_=ddim_transition(x,eps,at,an,False)
 err=float((x-clean_full).abs().max());assert err<1e-6
 records.append(dict(ratio=ratio,steps=steps,model_calls=calls,mask_error=maxmask,oracle_error=err,deterministic=True,extra_reference_actions_rejected=True,**stats))
 print('VALIDATED',ratio,steps,calls,stats['executed_prefix_edit_rmse_rad'],flush=True)
 del editor,c
 torch.cuda.empty_cache()
result=dict(passed=True,future_reference_leakage_check=True,records=records)
(O/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
