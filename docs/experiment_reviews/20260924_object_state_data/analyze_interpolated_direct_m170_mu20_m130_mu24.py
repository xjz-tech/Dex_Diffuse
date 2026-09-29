"""Audit matched interpolated direct runs for both physics settings."""
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps
from run_interpolated_direct_m170_mu20_m130_mu24 import R,C,EPISODES,PHYSICS,folder


def analyze(job):
    tag,mass,friction,output,ep=job
    f=folder(output,ep);raw=output/f'episode_{ep:02d}'/'raw'
    source=np.load(C/f'episode_{ep:02d}/reference_full.npz')['hand_target_rad']
    expanded,progress=interpolate_large_jumps(source,.1);ref=expanded[0]
    for _ in range(1800):
        if (f/'summary.json').exists() and (raw/'summary.json').exists():break
        time.sleep(2)
    else:raise TimeoutError(f)
    s=json.loads((f/'summary.json').read_text());t=json.loads((f/'trace.json').read_text());bt=json.loads((raw/'trace.json').read_text())
    assert s['mode']=='direct' and s['steps']==s['intended_steps']==dict(settle=60,action=len(ref),hold=60)
    assert s['mass_kg']==mass and s['friction']==friction and s['object_size_multiplier']==1
    assert s['reference_interpolation_threshold']==.1 and s['reference_interpolation']==0
    assert s['reference_interpolation_equal_jump'] is None and s['native_protocol']==PROTOCOL
    assert s['settle_target_source']=='qpos' and not s['stop_on_native_failure'] and s['action_limit'] is None
    assert np.array_equal(np.load(f/'reference_progress.npy'),progress)
    cmd=np.asarray([x['command'] for x in t if x['phase']=='action'])
    assert np.array_equal(cmd.astype(np.float32),ref.astype(np.float32))
    with np.load(raw/'initial_state.npz') as a,np.load(f/'initial_state.npz') as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    keys=[k for k in bt[0] if k!='object_contact_force']
    assert all(all(a[k]==b[k] for k in keys) for a,b in zip(bt[:60],t[:60]))
    fd=float(np.max(np.abs(np.asarray([x['object_contact_force'] for x in bt[:60]])-
                           np.asarray([x['object_contact_force'] for x in t[:60]]))))
    assert fd<1e-5
    geo=metrics(f,all_frames=True)['frames'];flags=[]
    for row,g in zip(t,geo):
        force=float(np.linalg.norm(row['object_contact_force']))
        flags.append((g['mesh_vertex_gap_m']>.005 and force<.05) or g['mesh_vertex_gap_m']>.02)
    loss=next((i for i in range(len(flags)-2) if all(flags[i:i+3])),None)
    best=current=0;valid_steps=0
    for i,(row,g) in enumerate(zip(t,geo)):
        good=(loss is None or i<loss) and row['phase']=='action' and row['vertical_error_deg']<=30 and \
            g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and \
            g['mesh_vertex_gap_m']<.008 and np.linalg.norm(row['object_contact_force'])>.1
        valid_steps+=int(good);current=current+1 if good else 0;best=max(best,current)
    if loss is None:
        sep=float(progress[-1]);actual=None;right=True;phase=None
    else:
        phase=t[loss]['phase'];right=False
        if phase=='action':sep=float(progress[t[loss]['index']]);actual=t[loss]['index']+1
        elif phase=='settle':sep=0.;actual=None
        else:sep=float(progress[-1]);actual=None
    result=dict(episode=ep,method='direct_interp',physics=tag,
        first_separation_reference_progress=sep,first_separation_actual_action_step=actual,
        first_separation_phase=phase,no_separation_right_censored=right,
        turn_success=valid_steps>0,stable_turn=best>=30,
        valid_vertical_contact_steps=valid_steps,longest_vertical_contact_steps=best,
        original_action_count=source.shape[1],executed_action_steps=len(ref),
        first_native_failure=s['first_native_failure'],validation=dict(initial_all_fields_exact=True,
        settle_state_exact=True,settle_force_max_delta_N=fd,reference_progress_exact=True,
        command_sequence_exact=True,native_protocol_exact=True,full_tail_and_hold=True))
    (f/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ANALYZED',tag,ep,sep,result['turn_success'],result['stable_turn'],best,flush=True)
    return result


def main():
    jobs=[(*physics,ep) for physics in PHYSICS for ep in EPISODES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(analyze,jobs))
    for tag,mass,friction,output in PHYSICS:
        selected=[x for x in results if x['physics']==tag]
        (output/'DIRECT_INTERP_RESULTS.json').write_text(json.dumps(selected,indent=2)+'\n')
    print('ALL DIRECT RUNS ANALYZED',flush=True)


if __name__=='__main__':main()
