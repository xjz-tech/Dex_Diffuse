"""Independent grasp analysis for counterfactual command replays."""
import json
from pathlib import Path
import numpy as np
from random4_geometry import metrics
from run_mass_scale_diagnostic import OUT

def analyze(folder):
    trace=json.loads((folder/'trace.json').read_text())
    gp=folder/'grasp_metrics.json'
    geo=json.loads(gp.read_text())['frames'] if gp.exists() else metrics(folder,all_frames=True)['frames']
    force=np.linalg.norm([x['object_contact_force'] for x in trace],axis=1)
    sepflags=[(g['mesh_vertex_gap_m']>.005 and f<.05) or g['mesh_vertex_gap_m']>.02 for g,f in zip(geo,force)]
    sep=next((i for i in range(len(trace)-2) if all(sepflags[i:i+3])),len(trace))
    best=0;count=0;bestend=None
    for i,(t,g,f) in enumerate(zip(trace,geo,force)):
        good=i<sep and t['phase']=='action' and t['vertical_error_deg']<=30 and g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and g['mesh_vertex_gap_m']<.008 and f>.1
        count=count+1 if good else 0
        if count>best:best=count;bestend=t['index']+1
    result=dict(name=folder.name,longest_vertical_steps=best,completed_turn=best>=30,best_interval=[bestend-best+1,bestend] if bestend else None,separation_phase=trace[sep]['phase'] if sep<len(trace) else None,separation_control_step=trace[sep]['index']+1 if sep<len(trace) else None)
    (folder/'grasp_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    results=[analyze(OUT/name) for name in ('replay_m044_commands_in_m044','replay_m044_commands_in_m170','replay_m170_commands_in_m044')]
    (OUT/'replay_grasp_results.json').write_text(json.dumps(results,indent=2)+'\n')
