"""Small single-finger unload/reset/recontact experiments, kept apart from no-gait controls."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import run as batch
from astra_kinematic_step import propose,point_jacobian
from astra_bridge import atomic_json
BASE=batch.HERE

class Gait:
 def __init__(self,release_mm=1.,push_deg=4.,manual=False):
  self.release_mm=release_mm;self.push_deg=push_deg;self.manual=manual
  self.anchor=None;self.home=None;self.events=[];self.previous=None
 def plan(self,r,pure=False):
  if pure:return ORIGINAL(r,True)
  s=r['state'];names=r['joint_names'];start=np.array(s['target_before']);step=r['step']
  if self.home is None:self.home=np.array(s['qpos'])
  if step<8:
   actions=np.repeat(start[None,:],16,axis=0);phase='initial';finger=None;detail={}
  else:
   # 24-step push; then unload8, reset8, recontact16; repeat with next finger.
   cycle=(step-8)//56;offset=(step-8)%56;finger=['index','middle','ring'][cycle%3]
   phase='push' if offset<24 else 'unload' if offset<32 else 'reset' if offset<40 else 'recontact'
   detail={}
   if phase=='push':
    actions,diag=propose(r,-self.push_deg,16,0,max_delta=.06)
    end=actions[-1].copy()
    floors={'right_thumb_IP':.3,'right_thumb_MCP_FE':.55}
    for f in ['index','middle','ring']:floors.update({f'right_{f}_DIP':.65,f'right_{f}_PIP':.4})
    for name,floor in floors.items():end[names.index(name)]=max(end[names.index(name)],floor)
    end=np.clip(end,r['joint_lower'],r['joint_upper'])
    actions=np.array([start+(end-start)*(k+1)/16 for k in range(16)])
   else:
    if offset==24:
     self.anchor=start.copy()
     self.unloaded=self.anchor.copy()
     dq=np.zeros(len(names));dq[names.index(f'right_{finger}_PIP')]=-.04;dq[names.index(f'right_{finger}_DIP')]=-.025
     body=next(b for b in s['contact_body_names'] if '_'+finger+'_' in b)
     tip=body.replace('_elastomer','_fingertip');pad=(np.array(s['body_pose_world'][body][:3])+np.array(s['body_pose_world'][tip][:3]))*.5
     _,jac=point_jacobian(s,names,body,pad)
     est=np.linalg.norm(jac@dq);dq*=min(1.,self.release_mm/1000/max(est,1e-9))
     self.unloaded=np.clip(self.anchor+dq,r['joint_lower'],r['joint_upper'])
     self.reset=self.unloaded.copy()
     aa=names.index(f'right_{finger}_MCP_AA')
     self.reset[aa]+=np.clip(self.home[aa]-self.anchor[aa],-.06,.06)
     self.closed=self.reset.copy()
     for suffix in ['PIP','DIP']:self.closed[names.index(f'right_{finger}_{suffix}')]=self.anchor[names.index(f'right_{finger}_{suffix}')]
     self.release_info=dict(joint_delta=dq.tolist(),linearized_pad_displacement_mm=float(np.linalg.norm(jac@dq)*1000),release_bound_mm=self.release_mm,support_anchor_step=step)
    end=self.unloaded if phase=='unload' else self.reset if phase=='reset' else self.closed
    following=self.reset if phase=='unload' else self.closed
    # Explicit 8-step phase endpoint, then next-phase lookahead, not tail padding.
    actions=np.array([start+(end-start)*(k+1)/8 for k in range(8)]+[end+(following-end)*(k+1)/8 for k in range(8)])
    detail=dict(self.release_info,support_fingers=[f for f in ['thumb','index','middle','ring','little'] if f!=finger])
   actions=np.clip(actions,r['joint_lower'],r['joint_upper'])
  event=dict(step=step,phase=phase,finger=finger,twist_deg=r['measured_twist_degrees'],forces={k:float(np.linalg.norm(v)) for k,v in s['contact_force_world'].items()},image=r.get('image'),details=detail)
  self.events.append(event);atomic_json(Path(r['session_dir'])/'gait_events.json',self.events) if 'session_dir' in r else None
  self.previous=phase
  return dict(session_id=r['session_id'],request_id=r['request_id'],joint_names=names,units='absolute_joint_radians',actions=actions.tolist(),rationale='Astra small-lift trial: retain thumb and other finger reference anchors while one finger unloads slightly, resets its side angle, and recloses. Feedback sweep resumes between single-finger cycles.',helper='Astra-authored phased analytic controller; no per-window LLM calls',gait=event)

ORIGINAL=batch.plan
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--version',default='v1');p.add_argument('--method',choices=['astra_direct','guided25','guided38','guided50'],default='astra_direct');p.add_argument('--seed',type=int,default=50);p.add_argument('--noise',type=int,default=0);p.add_argument('--release-mm',type=float,default=1);p.add_argument('--push-deg',type=float,default=4);a=p.parse_args()
 root=BASE/'gait_trials'/a.version;root.mkdir(parents=True,exist_ok=True);(root/'runs').mkdir(exist_ok=True)
 for seed in [50,19,25]:
  q=root/'runs'/f'screen_e{seed}'
  if not q.exists():q.symlink_to(BASE/'runs'/q.name,target_is_directory=True)
 if not (root/'cache').exists():(root/'cache').symlink_to(BASE/'cache',target_is_directory=True)
 proto=dict(batch.PROTOCOL,controller='Small-lift finger gait:24 push/8 unload/8 reset/16 recontact; index,middle,ring alternate, thumb never lifted',gait_parameters=vars(a));atomic_json(root/'protocol.json',proto)
 batch.HERE=root;batch.PROTOCOL=proto
 controller=Gait(a.release_mm,a.push_deg)
 def submit(r,pure=False):
  response=controller.plan(r,pure)
  atomic_json(root/'runs'/f'{a.method}_e{a.seed}_n{a.noise}'/'gait_events.json',controller.events)
  return response
 batch.plan=submit
 result=batch.run('formal',a.seed,a.noise,a.method)
 atomic_json(root/'last_result.json',result)
