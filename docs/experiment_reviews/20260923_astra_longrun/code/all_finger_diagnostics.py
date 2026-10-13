from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
h=Path(__file__).resolve().parents[1]/'gait_trials/v3_all_fingers';out={}
for p in sorted((h/'runs').glob('*/gait_events.json')):
 if p.parent.name.startswith('prior'):continue
 rows={}
 for line in (p.parent/'trajectory.jsonl').open():
  r=json.loads(line)
  if r['step'] in [32,40,48,64]:rows[r['step']]=r
 d={}
 for a,b in [(32,40),(40,48),(48,64)]:
  if a not in rows or b not in rows:continue
  s0=rows[a]['state'];s1=rows[b]['state'];w0=np.array(s0['wrist_state']);w1=np.array(s1['wrist_state']);r0=Rotation.from_quat(w0[3:7]);r1=Rotation.from_quat(w1[3:7]);f=[]
  for body in s0['contact_body_names']:
   tip=body.replace('_elastomer','_fingertip')
   p0=(np.array(s0['body_pose_world'][body][:3])+np.array(s0['body_pose_world'][tip][:3]))/2
   p1=(np.array(s1['body_pose_world'][body][:3])+np.array(s1['body_pose_world'][tip][:3]))/2
   local0=r0.inv().apply(p0-w0[:3]);local1=r1.inv().apply(p1-w1[:3]);comp=r0.apply(local1-local0)
   f.append(dict(finger=body,world_delta_mm=((p1-p0)*1000).tolist(),wrist_motion_compensated_delta_mm=(comp*1000).tolist(),start_force_N=float(np.linalg.norm(s0['contact_force_world'][body])),end_force_N=float(np.linalg.norm(s1['contact_force_world'][body]))))
  d[f'{a}-{b}']=f
 out[p.parent.name]=d
(h/'measured_phase_diagnostics.json').write_text(json.dumps(out,indent=2))
for k,v in out.items():
 print(k)
 for phase,vals in v.items():print(phase,[(f['finger'].split('_')[1],np.round(f['wrist_motion_compensated_delta_mm'],2).tolist(),round(f['end_force_N'],2)) for f in vals])
