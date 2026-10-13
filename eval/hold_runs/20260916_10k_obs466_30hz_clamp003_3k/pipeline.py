import os,json,time,subprocess,importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'20260916_10k_obs466_frequency_3k'
PREDECESSOR=ROOT.parent/'20260916_10k_obs466_20_25hz_3k'/'state.json'
spec=importlib.util.spec_from_file_location('baseline_pipeline',BASE/'pipeline.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def state(status,**kwargs):
 data=dict(status=status,updated_at=time.time(),**kwargs)
 tmp=ROOT/'state.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(ROOT/'state.json')
 print(json.dumps(data),flush=True)

def validate(folder,n,cap):
 rows,timing=base.verify(folder,30,n,cap)
 c=json.loads((folder/'clamp_statistics.json').read_text())
 assert c['max_hand_step']==.03 and c['max_sent_target_step']<=.030001
 assert c['env_action_samples']==sum(r['length'] for r in rows)
 assert c['joint_samples']==22*c['env_action_samples']
 assert 0<c['step_clipped_joints']<=c['joint_samples']
 assert sum(c['per_joint_step_clips'])==c['step_clipped_joints']
 base.compare_initial(folder,BASE/('check_hz30' if n==4 else 'hz30'))
 return rows,timing,c

def run(name,n,cap,smoke=False):
 folder=ROOT/name
 if folder.exists():raise RuntimeError('Refusing overwrite: '+str(folder))
 while not base.idle():
  state('waiting_gpu',phase=name);time.sleep(30)
 folder.mkdir()
 env=dict(os.environ,ARM_DIR=str(folder),CONTROL_HZ='30',NUM_ENV=str(n),MAX_STEPS=str(cap),DATA_INDICES='000-002' if smoke else '000-149',PYTHONDONTWRITEBYTECODE='1')
 with (folder/'console.log').open('w') as log:
  child=subprocess.Popen(['bash',str(ROOT/'run_arm.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
  state('running',phase=name,pid=child.pid,log=str(folder/'console.log'))
  rc=child.wait()
 if rc not in (0,139):raise RuntimeError(f'{name} exited {rc}')
 validate(folder,n,cap)
 state('arm_complete',phase=name,exit_code=rc)

def summarize():
 clamped,_,stats=validate(ROOT/'clamp003',3000,12000)
 baseline,_=base.verify(BASE/'hz30',30,3000,12000)
 results={}
 for key,rows in [('unclamped',baseline),('clamp003',clamped)]:
  lengths=np.array([r['length'] for r in rows])/30
  survivors=sum(r['reason']=='timeout' and r['length']==12000 for r in rows)
  results[key]=dict(environments=len(rows),survivors=survivors,completion_percent=100*survivors/len(rows),mean_capped_hold_seconds=float(lengths.mean()),median_capped_hold_seconds=float(np.median(lengths)),survival_after_seconds={str(t):sum(not(r['reason']=='failure' and r['length']<=t*30) for r in rows)/len(rows) for t in (30,60,120,240,400)})
 data=dict(results=results,clamp_statistics=stats,initial_states_equal=True,baseline=str(BASE/'hz30'),caveat='Single seed, 3000 first episodes, both30Hz/180Hz outer physics/2substeps/WAIT1. Only additional target-to-target step clamp0.03rad. Absolute limits precede clamp; observations contain the applied clipped target. No hardware-delay emulation; failure is existing simulator predicate. Telemetry covers first episodes only and includes their terminal action.')
 (ROOT/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
 lines=['# 30 Hz: no step clamp vs 0.03 rad step clamp','', '| Setting | 400-s survivors | Survival | Mean capped hold(s) | Median capped hold(s) |','|---|---:|---:|---:|---:|']
 for key,v in results.items():lines.append(f"| {key} | {v['survivors']}/3000 | {v['completion_percent']:.2f}% | {v['mean_capped_hold_seconds']:.3f} | {v['median_capped_hold_seconds']:.3f} |")
 lines+=['',f"Step clamp active on {100*stats['step_clipped_joint_fraction']:.2f}% of joint commands, {100*stats['step_clipped_env_action_fraction']:.2f}% of environment-action samples. Mean raw-to-sent absolute change: {stats['mean_abs_raw_minus_sent_rad']:.6f} rad. Mean measured tracking residual in clamp arm: {stats['mean_abs_tracking_residual_rad']:.6f} rad (no corresponding baseline telemetry was saved).",'',data['caveat']]
 (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
 return results

if __name__=='__main__':
 try:
  if (ROOT/'state.json').exists():raise RuntimeError('Existing state; refusing duplicate')
  while True:
   p=json.loads(PREDECESSOR.read_text())
   if p['status']=='complete':break
   if p['status']=='failed':raise RuntimeError('Predecessor failed; leaving queued evaluation unstarted')
   state('waiting_predecessor',predecessor=str(PREDECESSOR));time.sleep(45)
  run('smoke',4,60,True)
  run('clamp003',3000,12000)
  results=summarize()
  state('complete',comparison=str(ROOT/'comparison.md'),results=results)
 except Exception as exc:
  state('failed',error=repr(exc));raise
