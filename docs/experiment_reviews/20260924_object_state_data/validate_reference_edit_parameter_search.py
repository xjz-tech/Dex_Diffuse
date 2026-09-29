"""Paired-seed validation for seed-44 candidates reaching the preset target."""
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps
from run_reference_edit_parameter_search import O, P, R, EPISODES, folder, run_one
from analyze_reference_edit_parameter_search import audit


def guidance_folder(ep, seed):
    return O/f'validation_guidance_seed{seed}'/f'episode_{ep:02d}'


def run_guidance(ep, seed):
    f = guidance_folder(ep, seed)
    f.mkdir(parents=True, exist_ok=True)
    if (f/'summary.json').exists() and (f/'predictions.json').exists():
        return dict(episode=ep, seed=seed, folder=str(f), status='existing')
    case = R/f'qualified_comparison/episode_{ep:02d}'
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],
        PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    serverenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    sock = Path(f'/tmp/editsearch_valguide_{ep}_{seed}_{os.getpid()}.sock')
    server = None
    try:
        command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P/'server.py'),
            '--socket', str(sock), '--log', str(f/'predictions.json'), '--checkpoint',
            '/home/carus/data_usb/10B_obs_4-66.ckpt', '--reference', str(case/'reference_full.npz'),
            '--ddim-steps', '4', '--guidance-steps', '4', '--guidance-scale', '50',
            '--execution-steps', '2', '--reference-interpolation-threshold', '.1']
        with (f/'server.log').open('w') as log:
            server = subprocess.Popen(command, env=serverenv, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if sock.exists():
                break
            if server.poll() is not None:
                raise RuntimeError(f'server failed: {f}')
            time.sleep(.2)
        else:
            raise TimeoutError(f)
        command = ['/home/carus/miniforge3/envs/decv2/bin/python', str(P/'compare_reference_edit_rollouts.py'),
            '--mode', 'guided', '--out', str(f), '--reference', str(case/'reference_full.npz'),
            '--reference-id', '0', '--source-episode', str(ep), '--front-only',
            '--audit-recording', '--grasp-evidence', '--no-video',
            '--reference-interpolation-threshold', '.1', '--object-mass-kg', '.044',
            '--friction', '1.1', '--guidance-steps', '4', '--guidance-scale', '50',
            '--execution-steps', '2', '--prior-noise-seed', str(seed), '--socket', str(sock)]
        with (f/'sim.log').open('w') as log:
            subprocess.run(command, env=simenv, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert server.wait(timeout=30) == 0
        print('GUIDANCE DONE',ep,seed,flush=True)
        return dict(episode=ep,seed=seed,folder=str(f),status='complete')
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait()
        sock.unlink(missing_ok=True)


def audit_guidance(ep, seed):
    f = guidance_folder(ep, seed)
    case = R/f'qualified_comparison/episode_{ep:02d}'
    raw = case/'direct_m044_mu11'
    ref, progress = interpolate_large_jumps(np.load(case/'reference_full.npz')['hand_target_rad'], .1)
    ref = ref[0]
    s = json.loads((f/'summary.json').read_text())
    t = json.loads((f/'trace.json').read_text())
    bt = json.loads((raw/'trace.json').read_text())
    pred = json.loads((f/'predictions.json').read_text())
    assert s['steps'] == s['intended_steps'] == dict(settle=60, action=len(ref), hold=60)
    assert np.array_equal(np.load(f/'reference_progress.npy'), progress)
    assert len(t) == len(ref)+120 and s['mass_kg'] == .044 and s['friction'] == 1.1
    assert s['object_size_multiplier'] == 1 and s['settle_target_source'] == 'qpos'
    assert s['native_protocol'] == PROTOCOL and not s['stop_on_native_failure']
    assert s['action_limit'] is None and s['prior_noise_seed'] == seed
    assert s['reference_interpolation_threshold'] == .1
    assert s['execution_steps'] == s['prior']['execution_steps'] == 2
    assert s['guidance_scale'] == s['prior']['guidance_scale'] == 50
    assert s['prior']['ddim'] == 4 and s['prior']['guidance_steps'] == 4
    assert [x['reference_index'] for x in pred] == list(range(0,len(ref),2))
    with np.load(raw/'initial_state.npz') as a, np.load(f/'initial_state.npz') as b:
        assert set(a.files) == set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    keys = [k for k in bt[0] if k != 'object_contact_force']
    assert all(all(a[k] == b[k] for k in keys) for a,b in zip(bt[:60], t[:60]))
    fd = float(np.max(np.abs(np.asarray([x['object_contact_force'] for x in bt[:60]])-
                                  np.asarray([x['object_contact_force'] for x in t[:60]]))))
    assert fd < 1e-5
    geo = metrics(f, all_frames=True)['frames']
    flags=[]
    for row,g in zip(t,geo):
        force = float(np.linalg.norm(row['object_contact_force']))
        flags.append((g['mesh_vertex_gap_m']>.005 and force<.05) or g['mesh_vertex_gap_m']>.02)
    loss = next((i for i in range(len(flags)-2) if all(flags[i:i+3])), None)
    best = current = 0
    for i,(row,g) in enumerate(zip(t,geo)):
        good = (loss is None or i<loss) and row['phase']=='action' and row['vertical_error_deg']<=30 and \
            g['mesh_table_clearance_m']>.08 and g['near_contact_link_count']>=2 and \
            g['mesh_vertex_gap_m']<.008 and np.linalg.norm(row['object_contact_force'])>.1
        current = current+1 if good else 0
        best = max(best,current)
    if loss is not None and t[loss]['phase']=='action':
        sep=float(progress[t[loss]['index']]);actual=t[loss]['index']+1
    else:
        sep=float(progress[-1]);actual=None
    result=dict(episode=ep,seed=seed,first_separation_reference_progress=sep,
        first_separation_actual_action_step=actual,stable_turn=best>=30,
        longest_vertical_contact_steps=best,first_native_failure=s['first_native_failure'],
        validation=dict(initial_all_fields_exact=True,settle_state_exact=True,
                        settle_force_max_delta_N=fd,reference_progress_exact=True,
                        native_protocol_exact=True,full_tail_and_hold=True))
    (f/'search_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def archived_guidance44():
    results=[]
    for ep in EPISODES:
        f=R/f'qualified_comparison/episode_{ep:02d}/guide4exec2_adaptive010_ddim4_scale50_m044_mu11'
        a=json.loads((f/'m044_mu11_result.json').read_text())
        results.append(dict(episode=ep,seed=44,
            first_separation_reference_progress=a['first_separation']['reference_action_number'],
            first_separation_actual_action_step=a['first_separation']['control_step'],
            stable_turn=a['completed_turn'],
            longest_vertical_contact_steps=a['longest_vertical_contact_control_steps'],
            archived=True))
    assert [x['first_separation_reference_progress'] for x in results]==[112.,244.5,154.,253.]
    return results


def main():
    stage3=json.loads((O/'stage3_scores.json').read_text())
    eligible=[x for x in stage3['cumulative_ranking'] if x['stable_count']==4 and x['average_original_progress']>=190.875]
    selected=eligible[:2]
    plan=dict(selected=[x['config'] for x in selected],seeds=(44,45,46),
        threshold=190.875,condition='seed44 4/4 stable turn and average original progress >= threshold')
    (O/'validation_manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
    if not selected:
        (O/'validation_results.json').write_text(json.dumps(dict(status='no_seed44_candidate',plan=plan),indent=2)+'\n')
        print('NO ELIGIBLE CANDIDATE',flush=True)
        return
    candidates=[x['config'] for x in selected]
    jobs=[('edit',c,ep,seed) for c in candidates for seed in (45,46) for ep in EPISODES]
    jobs += [('guidance',None,ep,seed) for seed in (45,46) for ep in EPISODES]
    def job_run(job):
        method,c,ep,seed=job
        if method=='edit':
            cfg=dict(c,seed=seed)
            run_one((cfg,ep))
            return dict(method=method,**audit(cfg,ep))
        run_guidance(ep,seed)
        return dict(method=method,**audit_guidance(ep,seed))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        new=list(pool.map(job_run,jobs))
    guidance=archived_guidance44()+[x for x in new if x['method']=='guidance']
    summary=[]
    for c in candidates:
        byseed={}
        for seed in (44,45,46):
            if seed==44:
                match=next(x for x in stage3['cumulative_ranking'] if x['config']['t']==c['t'] and x['config']['ddim']==c['ddim'] and x['config']['exec']==c['exec'])
                group=match['episode_results']
            else:
                group=[x for x in new if x['method']=='edit' and x['config']['t']==c['t'] and x['config']['ddim']==c['ddim'] and x['config']['exec']==c['exec'] and x['config']['seed']==seed]
            g=[x for x in guidance if x['seed']==seed]
            assert len(group)==len(g)==4
            byseed[str(seed)]=dict(editor_average=float(np.mean([x['first_separation_reference_progress'] for x in group])),
                guidance_average=float(np.mean([x['first_separation_reference_progress'] for x in g])),
                editor_stable=sum(x['stable_turn'] for x in group),guidance_stable=sum(x['stable_turn'] for x in g),
                editor_episodes=group,guidance_episodes=g)
        summary.append(dict(config=c,byseed=byseed,
            three_seed_editor_average=float(np.mean([x['editor_average'] for x in byseed.values()])),
            three_seed_guidance_average=float(np.mean([x['guidance_average'] for x in byseed.values()])),
            all_three_seed_four_of_four=all(x['editor_stable']==4 for x in byseed.values())))
    (O/'validation_results.json').write_text(json.dumps(dict(status='complete',plan=plan,summary=summary),indent=2)+'\n')
    for x in summary:
        print('VALIDATED',x['config'],x['three_seed_editor_average'],x['three_seed_guidance_average'],x['all_three_seed_four_of_four'],flush=True)


if __name__=='__main__':
    main()
