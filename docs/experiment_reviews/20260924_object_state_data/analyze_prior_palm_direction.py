import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import concurrent.futures,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from random4_geometry import metrics
from compare_corrected_rollouts import PROTOCOL
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode2_prior_palm_direction_20260928';BASE=R/'qualified_comparison/episode_02/guide4exec2_adaptive010_ddim4_scale0_m044_mu11_video'
JOBS=[(n,s) for n in ['original','toward_palm_20'] for s in [44,45,46]]
bt=json.loads((BASE/'trace.json').read_text());u=Rotation.from_quat(bt[59]['object_pose'][3:]).apply([0,1,0]);axis=np.cross(u,[0,0,1]);axis/=np.linalg.norm(axis)
def analyze(job):
 name,seed=job;f=O/name/f'prior_seed{seed}'
 while not (f/'summary.json').exists():time.sleep(2)
 s=json.loads((f/'summary.json').read_text());t=json.loads((f/'trace.json').read_text())
 assert s['steps']==dict(settle=60,action=180,hold=60);assert s['prior_noise_seed']==seed;assert s['native_protocol']==PROTOCOL;assert s['guidance_scale']==0 and s['execution_steps']==2 and s['prior']['ddim']==4
 with np.load(BASE/'initial_state.npz') as a,np.load(f/'initial_state.npz') as b:
  differences=[k for k in a.files if not np.array_equal(a[k],b[k])];assert not differences if name=='original' else differences==['object'],differences
 if name=='original' and seed==44:
  assert all(all(x[k]==y[k] for k in ['command','q','object_pose']) for x,y in zip(t[:240],bt[:240]))
 geo=metrics(f,all_frames=True)['frames'];flags=[]
 for row,g in zip(t,geo):
  force=float(np.linalg.norm(row['object_contact_force']));g['force_N']=force;flags.append((g['mesh_vertex_gap_m']>.005 and force<.05) or g['mesh_vertex_gap_m']>.02)
 loss=next((i for i in range(len(flags)-2) if all(flags[i:i+3])),None);best=0;current=0
 for i,(row,g) in enumerate(zip(t,geo)):
  good=(loss is None or i<loss) and row['phase']=='action' and row['vertical_error_deg']<=30 and g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and g['mesh_vertex_gap_m']<.008 and g['force_N']>.1
  current=current+1 if good else 0;best=max(best,current)
 rot=Rotation.from_quat(np.asarray([x['object_pose'][3:] for x in t[59:240]]));increments=(rot[1:]*rot[:-1].inv()).as_rotvec();signed=np.degrees(increments@axis);cum=np.cumsum(signed)
 eligible=[x for i,x in enumerate(t) if x['phase']=='action' and (loss is None or i<loss)]
 result=dict(condition=name,seed=seed,initial_different_fields=differences,first_separation=(None if loss is None else dict(phase=t[loss]['phase'],control_step=t[loss]['index']+1,trace_index=loss)),stable_turn=best>=30,longest_vertical_contact_steps=best,min_vertical_error_before_separation=min(x['vertical_error_deg'] for x in eligible) if eligible else None,rotation_axis_world=axis.tolist(),first15_signed_rotation_deg=float(cum[14]),first30_signed_rotation_deg=float(cum[29]),angle_at_start=t[59]['vertical_error_deg'],angle_at_step15=t[74]['vertical_error_deg'],angle_at_step30=t[89]['vertical_error_deg'],first_native_failure=s['first_native_failure'])
 (f/'direction_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(analyze,JOBS))
 pairs=[]
 for seed in [44,45,46]:
  a=json.loads((O/'original'/f'prior_seed{seed}'/'trace.json').read_text());b=json.loads((O/'toward_palm_20'/f'prior_seed{seed}'/'trace.json').read_text())
  qa=np.asarray([x['command'] for x in a[60:75]])-np.asarray(a[59]['command']);qb=np.asarray([x['command'] for x in b[60:75]])-np.asarray(b[59]['command']);cos=float(np.sum(qa*qb)/np.linalg.norm(qa)/np.linalg.norm(qb))
  pairs.append(dict(seed=seed,first15_command_displacement_cosine=cos,first15_command_pair_rmse_rad=float(np.sqrt(np.mean((qa-qb)**2))),post_settle_q_pair_rmse_rad=float(np.sqrt(np.mean((np.asarray(a[59]['q'])-b[59]['q'])**2)))))
 (O/'RESULTS.json').write_text(json.dumps(dict(results=results,pairs=pairs),indent=2)+'\n');print('PAIRS',pairs,flush=True)
