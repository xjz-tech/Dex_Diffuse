from pathlib import Path
import sys,json
import numpy as np
from random4_geometry import OUT,metrics

def analyze(ep,mode):
 folder=OUT/f'episode_{ep:02d}'/mode
 data=json.load(open(folder/'grasp_metrics.json')) if (folder/'grasp_metrics.json').exists() else metrics(folder,all_frames=True)
 tr=json.load(open(folder/'trace.json'));s=json.load(open(folder/'summary.json'))
 evidence=data['frames'];flags=[]
 for row,geo in zip(tr,evidence):
  force=float(np.linalg.norm(row['object_contact_force']))
  flag=(geo['mesh_vertex_gap_m']>.005 and force<.05) or geo['mesh_vertex_gap_m']>.02
  flags.append(flag);geo['object_contact_force_norm_N']=force;geo['separated']=bool(flag)
 loss=None
 for i in range(len(flags)-2):
  if all(flags[i:i+3]):
   row=tr[i];j=row['index'];phase=row['phase'];factor=1+s['reference_interpolation']
   progress=1+j/factor if phase=='action' else (0 if phase=='settle' else s['source_action_frames'][1]-s['source_start_frame']+1)
   loss=dict(zero_based_control_index=j,control_step=j+1,reference_action_number=progress,source_action_frame=s['source_start_frame']+progress-1,trace_index=i,confirmation_control_steps=3,**evidence[i]);break
 result=dict(episode=ep,method=mode,first_separation=loss,first_native_failure=s['first_native_failure'],action_count=s['steps']['action'],original_action_count=s['source_action_frames'][1]-s['source_start_frame']+1,initial_consistency_checked=False,drop_definition='first of 3 consecutive recorded control steps with min all-hand/bulb collision vertex gap >5mm AND object net force <.05N, OR gap >2cm; video checked separately. This is an analysis marker, does not change native failure or rollout.',frames=evidence)
 (folder/'retention.json').write_text(json.dumps(result,indent=2))
 print(ep,mode,loss,flush=True)
 return result
if __name__=='__main__':analyze(int(sys.argv[1]),sys.argv[2])
