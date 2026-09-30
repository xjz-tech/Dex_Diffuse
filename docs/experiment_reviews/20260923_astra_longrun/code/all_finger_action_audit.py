from pathlib import Path
import json
import numpy as np
v=Path(__file__).resolve().parents[1]/'gait_trials/v3_all_fingers';out=[]
for run in sorted((v/'runs').glob('*')):
 if run.name.startswith(('prior','screen')) or not (run/'summary.json').exists():continue
 checked=0;phases=set();all_changed=[]
 for p in sorted(run.glob('request_*.json')):
  r=json.loads(p.read_text());response=json.loads((run/p.name.replace('request_','response_')).read_text());a=np.array(response['actions']);names=r['joint_names'];assert a.shape==(16,len(names));assert np.all(a>=np.array(r['joint_lower'])-1e-7) and np.all(a<=np.array(r['joint_upper'])+1e-7)
  phase=response['gait']['phase'];phases.add(phase);step=r['step']
  if step>=8:
   off=(step-8)%56;expected='push_right' if off<24 else 'lift_all' if off<32 else 'reset_left_all' if off<40 else 'close_all';assert phase==expected
   if off in [24,32,40]:
    changed={f:float(np.linalg.norm((a[-1]-np.array(r['state']['target_before']))[[i for i,n in enumerate(names) if '_'+f+'_' in n]])) for f in ['thumb','index','middle','ring','pinky']}
    assert all(x>1e-9 for x in changed.values());assert len(response['gait']['details'])==5;all_changed.append(dict(step=step,phase=phase,per_finger_target_delta_norm_rad=changed))
  checked+=1
 out.append(dict(run=run.name,checked_references=checked,phases=sorted(phases),all_five_actuated=all_changed))
(v/'all_finger_action_audit.json').write_text(json.dumps(out,indent=2));print('Validated',len(out),'completed runs')
