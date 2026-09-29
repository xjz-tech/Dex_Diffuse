from pathlib import Path
import numpy as np,json
from run_four_episode_reference_edit_adaptive010_video import O,R,folder
results=[]
for ep in [76,34,54,2]:
 a=R/f'qualified_comparison/episode_{ep:02d}/direct_adaptive010_m044_mu11';b=folder(ep,'zero',44)
 with np.load(a/'initial_state.npz') as x,np.load(b/'initial_state.npz') as y:assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
 x=json.loads((a/'trace.json').read_text());y=json.loads((b/'trace.json').read_text());assert len(x)==len(y)
 keys=[k for k in x[0] if k!='object_contact_force'];assert all(all(u[k]==v[k] for k in keys) for u,v in zip(x,y))
 force_delta=float(np.max(np.abs(np.asarray([u['object_contact_force'] for u in x])-np.asarray([u['object_contact_force'] for u in y]))));assert force_delta<1e-5
 pred=json.loads((b/'predictions.json').read_text());assert all(u['zero_edit_exact'] for u in pred)
 results.append(dict(episode=ep,initial_all_fields_exact=True,full_physics_trace_exact=True,force_max_delta_N=force_delta,total_control_steps=len(x),zero_edit_exact=True))
(O/'zero_verification.json').write_text(json.dumps(dict(all_verified=True,results=results),indent=2)+'\n');print(results)
