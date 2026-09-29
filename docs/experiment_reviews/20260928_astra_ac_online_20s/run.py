"""A/C style online Astra plans, six 20s rollouts. No archived action replay."""
import argparse,json,os,sys,time,subprocess,signal
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent; ROOT=P.parents[2]; SOURCE=P/'source'
sys.path.insert(0,str(SOURCE/'eval'))
from astra_bridge import atomic_json,validate_response
from astra_kinematic_step import propose
from left_planner import Sequence
RIGHT=json.loads((P/'right_original_parameters.json').read_text())

class OnlinePlanner:
 def __init__(self,folder,direction):
  self.folder=folder;self.direction=direction;self.left=Sequence(folder,15.,0.,.15,100.)
  self.previous_reference=None
 def right(self,r):
  q=np.asarray(r['state']['target_before']);step=r['step']
  template=next(x for x in reversed(RIGHT) if x['step']<=step)
  params=template['parameters'];goals=template['joint_goals'];diag=[]
  if step==0:actions=np.repeat(q[None],16,axis=0);phase='initialize'
  elif params:
   actions,diag=propose(r,params['degrees'],16,params['inward'],max_delta=params['max_delta'],fingers=params['fingers'],measured_base=params['measured_base'],axial=params['axial'])
   phase='right_push' if params['degrees']<0 else 'contact_recovery'
  else:
   target=q.copy()
   for name,v in goals.items():target[r['joint_names'].index(name)]=v
   target=np.clip(target,r['joint_lower'],r['joint_upper'])
   actions=q[None]+(target-q)[None]*np.arange(1,17)[:,None]/16
   phase='curl_recovery'
  return dict(session_id=r['session_id'],request_id=r['request_id'],joint_names=r['joint_names'],units='absolute_joint_radians',actions=actions.tolist(),phase=phase,parameters=params,joint_goals=goals,source_parameter_step=template['step'],diagnostics=diag,
   rationale='Astra C-style right sweep / contact recovery; original parameter stages on original control-step schedule, targets recomputed from this run current state. Continue final negative sweep after original archive ends.',helper='Astra-authored analytical URDF Jacobian feedback rule, no external LLM call or numeric action replay')
 def plan(self,r):
  response=self.left.plan(r) if self.direction=='left' else self.right(r)
  raw=np.asarray(response['actions'],dtype=np.float64);prev=self.previous_reference
  if prev is None:prev=np.asarray(r['state']['target_before'])
  expanded=[];origins=[];jumps=[]
  for i,q in enumerate(raw):
   jump=float(np.max(np.abs(q-prev)));jumps.append(jump)
   if jump>.09:
    expanded.append((prev+q)/2);origins.append({'raw_index':i,'midpoint':True})
   expanded.append(q);origins.append({'raw_index':i,'midpoint':False});prev=q
  actions=np.asarray(expanded[:16]);self.previous_reference=actions[7].copy()
  response.update(raw_actions=raw.tolist(),actions=actions.tolist(),interpolation=dict(threshold_rad=.09,rule='one midpoint per raw adjacent jump, including last executed reference to new plan',raw_max_jump_rad=max(jumps),generated_midpoints=sum(x['midpoint'] for x in origins),executed_prefix_midpoints=sum(x['midpoint'] for x in origins[:8]),plan_origins=origins[:16]),online_feedback_step=r['step'])
  return response

def run(direction,method):
 folder=P/'runs'/f'{direction}_{method}'
 if (folder/'run_complete.json').exists():print('already complete',folder,flush=True);return
 if folder.exists():raise FileExistsError(folder)
 planner=OnlinePlanner(folder,direction)
 scale=5 if direction=='left' else 25
 env={k:v for k,v in os.environ.items() if not k.startswith('ASTRA_')}
 env.update(ASTRA_SESSION=str(folder),ASTRA_MODE='direct' if method=='direct' else 'guided',
  GUIDANCE_SCALE=str(scale if method=='guidance' else 0), ASTRA_DDIM_STEPS='4',SEED='42',MAX_STEPS='600',
  CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',ASTRA_TARGET_TWIST_DEG='540' if direction=='left' else '-180',
  ASTRA_INSTRUCTION='Astra按A的半圈推进和暂停方式持续左转20秒' if direction=='left' else 'Astra按C的负向推进和接触恢复方式持续右转20秒',
  ASTRA_MODEL_NOISE_SEED='42',ASTRA_FIXED_NOISE='0',ASTRA_PHYSICAL_REVIEW='1',ASTRA_CONTINUOUS_20S='1',
  ASTRA_DEMO_CACHE=str(ROOT/'.worktrees/Astra-controller/outputs/astra_noise_direction/demo_cache'),
  MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python',CUDA_VISIBLE_DEVICES='0',
  PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',ASTRA_RESOURCE_LOCK='/tmp/codex-astra-ac-online-20s.lock',
  ASTRA_VIDEO_LABEL=direction+' '+method+' A/C online DDIM4 native physics')
 if method=='edit':env['ASTRA_EDIT_NOISE_RATIO']='.15'
 if method=='guidance' and direction=='left':env['ASTRA_GUIDANCE_SCHEDULE_FILE']=str(folder/'guidance_schedule.json')
 config={k:v for k,v in env.items() if k.startswith('ASTRA_') or k in ['GUIDANCE_SCALE','SEED','MAX_STEPS','CKPT_PATH']}
 atomic_json(P/(direction+'_'+method+'_config.json'),config)
 with folder.with_suffix('.log').open('w') as log:
  proc=subprocess.Popen(['bash','eval/astra_halfturn_16.sh'],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  deadline=time.monotonic()+3600
  try:
   while proc.poll() is None:
    if time.monotonic()>deadline:raise TimeoutError(str(folder))
    pending=folder/'pending.json'
    if pending.exists():
     handoff=json.loads(pending.read_text());dest=Path(handoff['response'])
     if not dest.exists():
      req=json.loads(Path(handoff['request']).read_text());response=planner.plan(req)
      validate_response(response,req);atomic_json(dest,response)
      if req['step']%80==0:print(direction,method,'step',req['step'],'angle',round(req['measured_twist_degrees'],2),'phase',response['phase'],flush=True)
    time.sleep(.025)
   if proc.returncode:raise RuntimeError(f'rollout exit {proc.returncode}: {folder.with_suffix(".log")}')
  except BaseException:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30);raise
 rows=[json.loads(x) for x in (folder/'trajectory.jsonl').read_text().splitlines()]
 assert len(rows)==600
 with np.load(folder/'initial_state.npz') as z,np.load(ROOT/'.worktrees/Astra-controller/outputs/astra_left_sequence/v2_s5_pause100/initial_state.npz') as old:
  assert set(z.files)==set(old.files)
  for k in z.files:np.testing.assert_array_equal(z[k],old[k],err_msg=k)
 atomic_json(folder/'run_complete.json',dict(steps=len(rows),first_native_failure=next((x['step'] for x in rows if x['failure']),None),final_twist=rows[-1]['twist_degrees'],original_initial_state_all_fields_equal=True))
 print('COMPLETE',folder.name,rows[-1]['twist_degrees'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('direction',choices=['left','right','all']);p.add_argument('method',choices=['direct','guidance','edit','all']);a=p.parse_args()
 for d in (['left','right'] if a.direction=='all' else [a.direction]):
  for m in (['guidance','direct','edit'] if a.method=='all' else [a.method]):run(d,m)
