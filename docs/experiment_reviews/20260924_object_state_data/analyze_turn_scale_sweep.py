from pathlib import Path
import json,time,sys,concurrent.futures
import numpy as np
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';C=R/'qualified_comparison';OUT=R/'scale45_55'
import random4_analyze as analyzer
from compare_corrected_rollouts import PROTOCOL
analyzer.OUT=C
JOBS=[(ep,exe,sc) for ep in [76,34,54,2] for exe in [2,1] for sc in [45,55]]
def analyze(job):
 ep,exe,sc=job;mode=f'guide{exe}_scale{sc}';folder=C/f'episode_{ep:02d}'/mode;deadline=time.monotonic()+2400
 while not (folder/'summary.json').exists():
  if time.monotonic()>deadline:raise TimeoutError(str(job))
  time.sleep(2)
 s=json.load(open(folder/'summary.json'));assert s['guidance_scale']==sc and s['execution_steps']==exe and s['reference_interpolation']==1 and s['prior']['ddim']==4 and s['prior']['guidance_steps']==2
 assert s['native_protocol']==PROTOCOL and not s['stop_on_native_failure'];assert s['steps']==s['intended_steps'];assert s['action_limit'] is None
 prior=json.load(open(folder/'predictions.json'));assert [p['reference_index'] for p in prior]==list(range(0,s['steps']['action'],exe))
 original=C/f'episode_{ep:02d}'/'direct';a=np.load(original/'initial_state.npz');b=np.load(folder/'initial_state.npz');assert all(np.array_equal(a[k],b[k]) for k in a.files)
 old=json.load(open(original/'trace.json'))[:60];tr=json.load(open(folder/'trace.json'));keys=[k for k in old[0] if k!='object_contact_force'];assert all(all(x[k]==y[k] for k in keys) for x,y in zip(old,tr[:60]))
 cf=float(np.max(np.abs(np.array([x['object_contact_force'] for x in old])-np.array([x['object_contact_force'] for x in tr[:60]]))));assert cf<1e-5
 ret=json.load(open(folder/'retention.json')) if (folder/'retention.json').exists() else analyzer.analyze(ep,mode)
 ret['initial_consistency_checked']=True;(folder/'retention.json').write_text(json.dumps(ret,indent=2))
 best=[];cur=[];limit=ret['first_separation']['trace_index'] if ret['first_separation'] else len(tr)
 for i,(row,g) in enumerate(zip(tr,ret['frames'])):
  good=row['phase']=='action' and i<limit and row['vertical_error_deg']<=30 and g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and g['mesh_vertex_gap_m']<.008 and g['object_contact_force_norm_N']>.1
  if good:
   cur.append(row['index']+1)
   if len(cur)>len(best):best=cur.copy()
  else:cur=[]
 sensitivity={}
 for gap in [.003,.005,.008,.01]:
  flags=[(g['mesh_vertex_gap_m']>gap and g['object_contact_force_norm_N']<.05) or g['mesh_vertex_gap_m']>.02 for g in ret['frames']]
  idx=next((i for i in range(len(flags)-2) if all(flags[i:i+3])),None)
  sensitivity[str(gap)]=1+tr[idx]['index']/2 if idx is not None and tr[idx]['phase']=='action' else None
 result=dict(episode=ep,execution_steps=exe,scale=sc,first_separation=ret['first_separation'],completed_turn=len(best)>=30,longest_vertical_contact_control_steps=len(best),vertical_contact_reference_interval=[1+(best[0]-1)/2,1+(best[-1]-1)/2] if best else None,first_native_failure=s['first_native_failure'],sensitivity_reference_action_number=sensitivity,validation=dict(all27_initial_fields_exact=True,all60_settle_fields_except_object_contact_force_exact=True,object_contact_force_max_float_difference_N=cf,native_protocol_exact=True,reference_advance_exact=True,full_tail_and_hold=True),video_recorded=s['video_recorded'])
 (folder/'scale_result.json').write_text(json.dumps(result,indent=2));print('ANALYZED',ep,exe,sc,'loss',ret['first_separation']['reference_action_number'] if ret['first_separation'] else None,'turn',len(best)>=30,flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 results=list(ex.map(analyze,JOBS))
(OUT/'new_results.json').write_text(json.dumps(results,indent=2));print('ALL16 ANALYZED',flush=True)
