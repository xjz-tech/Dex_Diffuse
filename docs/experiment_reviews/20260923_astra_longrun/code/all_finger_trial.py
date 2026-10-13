"""User-requested simultaneous five-finger lift / left reset / close / right turn."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
import run as batch
from astra_bridge import atomic_json
from astra_kinematic_step import point_jacobian,propose
BASE=batch.HERE

class AllFingerGait:
 def __init__(self):self.events=[];self.ends={}
 def endpoint(self,r,phase):
  s=r['state'];names=r['joint_names'];q=np.clip(np.array(s['qpos']),r['joint_lower'],r['joint_upper']);delta=np.zeros(len(q));diags=[]
  center=np.array(s['object_position']);axis=np.array(r['axis_world']);rot=Rotation.from_rotvec(axis*np.deg2rad(4))
  for body in s['contact_body_names']:
   tip=body.replace('_elastomer','_fingertip');pad=(np.array(s['body_pose_world'][body][:3])+np.array(s['body_pose_world'][tip][:3]))/2
   _,jac=point_jacobian(s,names,body,pad);offset=pad-center;radial=offset-axis*np.dot(offset,axis);radial/=max(np.linalg.norm(radial),1e-8)
   desired= np.array([0.,0.,.001])+.0005*radial if phase=='lift_all' else rot.apply(offset)-offset if phase=='reset_left_all' else np.array([0.,0.,-.001])-.00075*radial
   active=np.flatnonzero(np.linalg.norm(jac,axis=0)>1e-10);lo=np.maximum(np.array(r['joint_lower'])[active]-q[active],-.06);hi=np.minimum(np.array(r['joint_upper'])[active]-q[active],.06)
   sol=lsq_linear(np.vstack([jac[:,active],.004*np.eye(len(active))]),np.r_[desired,np.zeros(len(active))],bounds=(lo,hi));dq=np.zeros(len(q));dq[active]=sol.x;delta+=dq
   diags.append(dict(body=body,pad_world=pad.tolist(),desired_mm=(desired*1000).tolist(),linearized_mm=(jac@dq*1000).tolist()))
  return np.clip(q+delta,r['joint_lower'],r['joint_upper']),diags
 def plan(self,r,pure=False):
  if pure:return batch_original(r,True)
  step=r['step'];start=np.array(r['state']['target_before']);detail=[]
  if step<8:phase='initial';end=start;duration=16
  else:
   offset=(step-8)%56;phase='push_right' if offset<24 else 'lift_all' if offset<32 else 'reset_left_all' if offset<40 else 'close_all'
   if phase=='push_right':
    actions,detail=propose(r,-4,16,max_delta=.06);end=actions[-1];duration=16
   else:
    # One measured-pose endpoint per phase. Closure lasts 16 executed steps.
    if offset in [24,32,40]:self.ends[phase],detail=self.endpoint(r,phase)
    end=self.ends[phase];duration=16 if offset==40 else 8
  actions=np.array([start+(end-start)*min((k+1)/duration,1) for k in range(16)])
  event=dict(step=step,phase=phase,finger='all_five',twist_deg=r['measured_twist_degrees'],forces={k:float(np.linalg.norm(v)) for k,v in r['state']['contact_force_world'].items()},details=detail)
  self.events.append(event)
  return dict(session_id=r['session_id'],request_id=r['request_id'],joint_names=r['joint_names'],units='absolute_joint_radians',actions=actions.tolist(),rationale='User-requested all five fingers together: right turn, small world-up lift with slight outward unloading, left circumferential reset, close downward/inward, repeat.',helper='Astra-authored analytic feedback controller; no per-window LLM calls',gait=event)
batch_original=batch.plan
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=50);p.add_argument('--noise',type=int,default=0);p.add_argument('--method',choices=['astra_direct','prior','guided25','guided38','guided50'],default='astra_direct');a=p.parse_args()
 root=BASE/'gait_trials/v3_all_fingers';root.mkdir(exist_ok=True,parents=True);(root/'runs').mkdir(exist_ok=True)
 for seed in [50,19,25]:
  q=root/'runs'/f'screen_e{seed}'
  if not q.exists():q.symlink_to(BASE/'runs'/q.name,target_is_directory=True)
 for name in ['cache','screening.json']:
  if not (root/name).exists():(root/name).symlink_to(BASE/name)
 proto=dict(batch.PROTOCOL,scope='User requested simultaneous all-five-finger gait, separate from earlier single-finger trials',controller='24 right push / 8 all lift / 8 all left reset / 16 all close, repeating',controller_parameters=dict(lift_world_up_mm=1,outward_unload_mm=.5,left_reset_degrees=4,close_world_down_mm=1,close_inward_mm=.75,push_degrees=-4,ik_increment_limit_rad=.06,phase_steps=[24,8,8,16],moving_fingers=['thumb','index','middle','ring','pinky']),reference_limitation='Nominal Cartesian increments are linearized targets, not guaranteed actual motion or successful contact release. All five fingers move, with no thumb support exemption. Analytic controller, no per-window LLM calls.')
 atomic_json(root/'protocol.json',proto);atomic_json(root/f'{a.method}_e{a.seed}_n{a.noise}_config.json',dict(arguments=vars(a),protocol=proto));batch.HERE=root;batch.PROTOCOL=proto;controller=AllFingerGait()
 def submit(r,pure=False):
  response=controller.plan(r,pure);atomic_json(root/'runs'/f'{a.method}_e{a.seed}_n{a.noise}'/'gait_events.json',controller.events);return response
 batch.plan=submit;atomic_json(root/'last_result.json',batch.run('formal',a.seed,a.noise,a.method))
