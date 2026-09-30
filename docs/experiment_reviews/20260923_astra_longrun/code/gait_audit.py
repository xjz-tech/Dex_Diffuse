"""Verify single-finger changes and deliberately anchored support in actual references."""
import json,sys
from pathlib import Path
import numpy as np
h=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'gait_trials/v1'
results=[]
for run in (h/'runs').iterdir():
 if not (run/'summary.json').exists() or not (run/'gait_events.json').exists():continue
 if json.loads((run/'summary.json').read_text())['method']=='prior':continue
 requests={}
 for p in run.glob('request_*.json'):
  r=json.loads(p.read_text());requests[r['step']]=r
 count=0
 for p in run.glob('response_*.json'):
  a=json.loads(p.read_text());e=a['gait'];phase=e['phase']
  if phase not in ['unload','reset','recontact']:continue
  r=requests[e['step']];anchor=requests[e['details']['support_anchor_step']];names=r['joint_names'];finger=e['finger']
  support=[i for i,n in enumerate(names) if '_'+finger+'_' not in n];plan=np.array(a['actions'])
  np.testing.assert_allclose(plan[7:,support],np.repeat(np.array(anchor['state']['target_before'])[support][None,:],9,axis=0),atol=1e-7,rtol=0)
  assert e['details']['linearized_pad_displacement_mm']<=1+1e-9
  if phase=='reset':
   i=names.index(f'right_{finger}_MCP_AA');base=anchor['state']['qpos'] if e['details'].get('unload_from_measured') else anchor['state']['target_before'];reference_base=np.clip(base[i],r['joint_lower'][i],r['joint_upper'][i]);assert abs(plan[7,i]-reference_base)<=.06000001
  assert finger!='thumb';count+=1
 results.append(dict(label=run.name,phased_references_checked=count,support_reference_endpoints_anchored=True,linearized_unload_bound_mm=1.,thumb_never_scheduled_to_lift=True))
(h/'gait_audit.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
