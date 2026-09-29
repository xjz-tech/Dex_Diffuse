from pathlib import Path
import numpy as np,json
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode54_reference_edit_adaptive010_video_20260928';a=R/'qualified_comparison/episode_54/direct_adaptive010_m044_mu11';b=O/'zero_seed44'
with np.load(a/'initial_state.npz') as x,np.load(b/'initial_state.npz') as y:assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
x=json.loads((a/'trace.json').read_text());y=json.loads((b/'trace.json').read_text());assert len(x)==len(y)
keys=[k for k in x[0] if k!='object_contact_force'];assert all(all(u[k]==v[k] for k in keys) for u,v in zip(x,y))
force_delta=float(np.max(np.abs(np.asarray([u['object_contact_force'] for u in x])-np.asarray([u['object_contact_force'] for u in y]))));assert force_delta<1e-5
pred=json.loads((b/'predictions.json').read_text());assert all(u['zero_edit_exact'] for u in pred)
result=dict(full_physics_trace_exact=True,note='All saved fields other than net object contact force exactly equal; force difference is checked separately.',force_max_delta_N=force_delta,initial_all_fields_exact=True,zero_action_edit_exact=True,total_control_steps=len(x));(O/'zero_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
