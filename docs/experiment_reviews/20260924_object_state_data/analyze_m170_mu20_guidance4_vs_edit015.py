"""Audit and compare the 170 g, friction 2.0 runs."""
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps
from run_m170_mu20_guidance4_vs_edit015 import O, R, C, EPISODES, METHODS, MASS, FRICTION, THRESHOLD, SEED, folder


def analyze(job):
    ep, method = job
    f = folder(ep, method)
    raw = folder(ep, 'raw')
    source = np.load(C/f'episode_{ep:02d}/reference_full.npz')['hand_target_rad']
    if method == 'raw':
        ref = source[0]
        progress = np.arange(1, len(ref)+1, dtype=np.float64)
    else:
        expanded, progress = interpolate_large_jumps(source, THRESHOLD)
        ref = expanded[0]
    for _ in range(1800):
        required = (f/'summary.json').exists() and (method == 'raw' or (f/'predictions.json').exists())
        if required and (raw/'summary.json').exists():
            break
        time.sleep(2)
    else:
        raise TimeoutError(f)
    saved = f/'analysis.json'
    if saved.exists():
        return json.loads(saved.read_text())
    s = json.loads((f/'summary.json').read_text())
    t = json.loads((f/'trace.json').read_text())
    bt = json.loads((raw/'trace.json').read_text())
    assert s['steps'] == s['intended_steps'] == dict(settle=60, action=len(ref), hold=60)
    assert len(t) == len(ref)+120
    assert s['mass_kg'] == MASS and s['friction'] == FRICTION and s['object_size_multiplier'] == 1
    assert s['settle_target_source'] == 'qpos' and s['native_protocol'] == PROTOCOL
    assert not s['stop_on_native_failure'] and s['action_limit'] is None
    assert all(s[k] is None for k in ('vertical_scale_switch_angle_deg','vertical_scale_after','vertical_scale_trigger'))
    if method == 'raw':
        assert s['mode'] == 'direct' and s['reference_interpolation_threshold'] is None
        commands = np.asarray([x['command'] for x in t if x['phase']=='action'])
        assert np.array_equal(commands.astype(np.float32), ref.astype(np.float32))
    else:
        assert s['mode'] == 'guided' and s['reference_interpolation_threshold'] == THRESHOLD
        assert np.array_equal(np.load(f/'reference_progress.npy'), progress)
        assert s['execution_steps'] == s['prior']['execution_steps'] == 2
        assert s['prior']['ddim'] == 4 and s['prior_noise_seed'] == SEED
        pred = json.loads((f/'predictions.json').read_text())
        assert [x['reference_index'] for x in pred] == list(range(0,len(ref),2))
        if method == 'guidance4':
            assert s['guidance_scale'] == s['prior']['guidance_scale'] == 50
            assert s['prior']['guidance_steps'] == 4
            assert all(x['guidance_scale'] == 50 for x in pred)
        else:
            assert s['guidance_scale'] == s['prior']['guidance_scale'] == 0
            e = s['prior']['editor']
            assert e['algorithm'] == 'reference_initialized_ddim'
            assert e['requested_noise_ratio'] == .15 and abs(e['actual_noise_ratio']-.1533970386)<1e-8
            assert e['timesteps'] == [8,5,3,0] and e['future_reference_steps'] == 9
            assert all(x['guidance_scale'] == 0 and x['history_mask_max_error'] == 0 and x['seeds'] == [SEED] for x in pred)
    with np.load(raw/'initial_state.npz') as a, np.load(f/'initial_state.npz') as b:
        assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    keys=[k for k in bt[0] if k!='object_contact_force']
    assert all(all(a[k]==b[k] for k in keys) for a,b in zip(bt[:60],t[:60]))
    fd=float(np.max(np.abs(np.asarray([x['object_contact_force'] for x in bt[:60]])-
                           np.asarray([x['object_contact_force'] for x in t[:60]]))))
    assert fd<1e-5
    geo=metrics(f,all_frames=True)['frames']
    flags=[]
    for row,g in zip(t,geo):
        force=float(np.linalg.norm(row['object_contact_force']))
        flags.append((g['mesh_vertex_gap_m']>.005 and force<.05) or g['mesh_vertex_gap_m']>.02)
    loss=next((i for i in range(len(flags)-2) if all(flags[i:i+3])),None)
    best=current=0
    for i,(row,g) in enumerate(zip(t,geo)):
        good=(loss is None or i<loss) and row['phase']=='action' and row['vertical_error_deg']<=30 and \
            g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and \
            g['mesh_vertex_gap_m']<.008 and np.linalg.norm(row['object_contact_force'])>.1
        current=current+1 if good else 0
        best=max(best,current)
    if loss is None:
        sep=float(progress[-1]);actual=None;phase=None;right=True
    else:
        phase=t[loss]['phase'];right=False
        if phase=='action':
            j=t[loss]['index'];sep=float(progress[j]);actual=j+1
        elif phase=='settle':
            sep=0.;actual=None
        else:
            sep=float(progress[-1]);actual=None
    actions=[x for x in t if x['phase']=='action'];cmd=np.asarray([x['command'] for x in actions])
    cutoff=len(ref) if loss is None or phase=='hold' else (0 if phase=='settle' else t[loss]['index'])
    result=dict(episode=ep,method=method,first_separation_reference_progress=sep,
        first_separation_actual_action_step=actual,first_separation_phase=phase,
        no_separation_right_censored=right,stable_turn=best>=30,
        longest_vertical_contact_steps=best,
        command_reference_rmse_before_separation_rad=(float(np.sqrt(np.mean((cmd[:cutoff]-ref[:cutoff])**2))) if cutoff else None),
        first_native_failure=s['first_native_failure'],original_action_count=source.shape[1],
        executed_action_steps=len(ref),validation=dict(initial_all_fields_exact=True,
        settle_state_exact=True,settle_force_max_delta_N=fd,full_tail_and_hold=True,
        native_protocol_exact=True,reference_progress_exact=True))
    saved.write_text(json.dumps(result,indent=2)+'\n')
    print('ANALYZED',ep,method,sep,result['stable_turn'],best,flush=True)
    return result


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(analyze,[(ep,m) for ep in EPISODES for m in METHODS]))
    (O/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ALL ANALYZED',flush=True)


if __name__=='__main__':
    main()
