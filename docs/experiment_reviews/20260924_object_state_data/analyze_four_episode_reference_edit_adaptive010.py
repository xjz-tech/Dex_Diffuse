import concurrent.futures,json,time
from pathlib import Path
import numpy as np
from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps
from compare_corrected_rollouts import PROTOCOL
from run_four_episode_reference_edit_adaptive010_video import O,R,folder
JOBS=[(ep,m,44) for ep in [76,34,54,2] for m in ['edit010','edit020','edit035']]
def analyze(job):
 ep,method,seed=job;f=folder(ep,method,seed);BASE=R/f'qualified_comparison/episode_{ep:02d}/direct_m044_mu11'
 REF,PROGRESS=interpolate_large_jumps(np.load(BASE.parent/'reference_full.npz')['hand_target_rad'],.1);REF=REF[0]
 for _ in range(1800):
  if (f/'summary.json').exists() and (f/'predictions.json').exists():break
  time.sleep(2)
 else:raise TimeoutError(f)
 if (f/'analysis.json').exists():return dict(episode=ep,**json.loads((f/'analysis.json').read_text()))
 s=json.loads((f/'summary.json').read_text());t=json.loads((f/'trace.json').read_text());bt=json.loads((BASE/'trace.json').read_text());pred=json.loads((f/'predictions.json').read_text())
 assert s['steps']==s['intended_steps']==dict(settle=60,action=len(REF),hold=60)
 assert s['mass_kg']==.044 and s['friction']==1.1 and s['object_size_multiplier']==1 and s['settle_target_source']=='qpos'
 assert s['native_protocol']==PROTOCOL and s['reference_interpolation']==0 and s['reference_interpolation_threshold']==.1
 assert s['reference_interpolation_equal_jump'] is None and not s['stop_on_native_failure'] and s['action_limit'] is None
 assert s['execution_steps']==2 and s['prior_noise_seed']==seed and s['prior']['ddim']==4
 assert [x['reference_index'] for x in pred]==list(range(0,len(REF),2))
 assert all(x['guidance_scale']==(50 if method=='standard_g4_s50' else 0) for x in pred)
 if method!='standard_g4_s50':
  assert all(x['history_mask_max_error']==0 for x in pred)
  assert s['prior']['editor']['known_history_steps']==3 and s['prior']['editor']['future_reference_steps']==9
 with np.load(BASE/'initial_state.npz') as a,np.load(f/'initial_state.npz') as b:
  assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
 keys=[k for k in bt[0] if k!='object_contact_force'];assert all(all(x[k]==y[k] for k in keys) for x,y in zip(bt[:60],t[:60]))
 fd=float(np.max(np.abs(np.asarray([x['object_contact_force'] for x in t[:60]])-np.asarray([x['object_contact_force'] for x in bt[:60]]))))
 assert fd<1e-5
 geo=metrics(f,all_frames=True)['frames'];flags=[]
 for row,g in zip(t,geo):
  g['force_N']=float(np.linalg.norm(row['object_contact_force']));flags.append((g['mesh_vertex_gap_m']>.005 and g['force_N']<.05) or g['mesh_vertex_gap_m']>.02)
 loss=next((i for i in range(len(flags)-2) if all(flags[i:i+3])),None);best=[];current=[];first_stable_start=None
 for i,(row,g) in enumerate(zip(t,geo)):
  good=(loss is None or i<loss) and row['phase']=='action' and row['vertical_error_deg']<=30 and g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and g['mesh_vertex_gap_m']<.008 and g['force_N']>.1
  if good:
   current.append(row['index'])
   if len(current)>len(best):best=current.copy()
   if len(current)==30 and first_stable_start is None:first_stable_start=current[0]+1
  else:current=[]
 actions=[x for x in t if x['phase']=='action'];cmd=np.asarray([x['command'] for x in actions]);cutoff=min(len(REF),max(0,(loss if loss is not None else 60+len(REF))-60));rmses=np.sqrt(np.mean((cmd-REF)**2,axis=1))
 result=dict(method=method,seed=seed,first_separation=None if loss is None else dict(phase=t[loss]['phase'],control_step=t[loss]['index']+1,trace_index=loss,original_reference_progress=float(PROGRESS[min(max(loss-60,0),len(PROGRESS)-1)])),stable_turn=len(best)>=30,longest_vertical_contact_steps=len(best),first_stable_turn_start_step=first_stable_start,first_stable_turn_start_time_s=None if first_stable_start is None else first_stable_start/30,min_vertical_error_before_separation=min((x['vertical_error_deg'] for x in actions[:cutoff]),default=None),command_reference_rmse_before_separation_rad=float(np.sqrt(np.mean((cmd[:cutoff]-REF[:cutoff])**2))) if cutoff else None,command_max_abs_edit_before_separation_rad=float(np.max(np.abs(cmd[:cutoff]-REF[:cutoff]))) if cutoff else None,mean_inference_seconds=float(np.mean([x['inference_seconds'] for x in pred])),first_native_failure=s['first_native_failure'],editor=s['prior'].get('editor'),validation=dict(initial_all_fields_exact=True,settle_state_exact=True,settle_contact_force_max_delta_N=fd,full_tail=True,reference_advances_by_two=True,native_protocol_exact=True))
 (f/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print('ANALYZED',ep,method,seed,result['first_separation'],result['stable_turn'],len(best),flush=True);return dict(episode=ep,**result)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(analyze,JOBS))
 (O/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n');print('ALL ANALYZED',flush=True)
