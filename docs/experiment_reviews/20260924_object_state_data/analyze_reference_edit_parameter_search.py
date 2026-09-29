"""Verify and score each parameter-search rollout with the archived geometry rule."""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import numpy as np
from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps
from run_reference_edit_parameter_search import O, R, EPISODES, existing_folder, folder, config_key


def audit(c, ep):
    archived = existing_folder(c, ep)
    f = archived if archived is not None else folder(c, ep)
    raw = R/f'qualified_comparison/episode_{ep:02d}/direct_m044_mu11'
    source = np.load(raw.parent/'reference_full.npz')['hand_target_rad']
    ref, progress = interpolate_large_jumps(source, .1)
    ref = ref[0]
    for _ in range(1800):
        if (f/'summary.json').exists() and (f/'predictions.json').exists():
            break
        time.sleep(2)
    else:
        raise TimeoutError(f)
    s = json.loads((f/'summary.json').read_text())
    t = json.loads((f/'trace.json').read_text())
    bt = json.loads((raw/'trace.json').read_text())
    pred = json.loads((f/'predictions.json').read_text())
    assert s['steps'] == s['intended_steps'] == dict(settle=60, action=len(ref), hold=60)
    assert len(t) == len(ref)+120
    assert np.array_equal(np.load(f/'reference_progress.npy'), progress)
    assert s['mass_kg'] == .044 and s['friction'] == 1.1 and s['object_size_multiplier'] == 1
    assert s['settle_target_source'] == 'qpos' and s['native_protocol'] == PROTOCOL
    assert s['reference_interpolation'] == 0 and s['reference_interpolation_threshold'] == .1
    assert s['reference_interpolation_equal_jump'] is None and not s['stop_on_native_failure']
    assert s['action_limit'] is None and s['prior_noise_seed'] == c['seed']
    assert s['execution_steps'] == s['prior']['execution_steps'] == c['exec']
    assert s['prior']['ddim'] == c['ddim'] and s['prior']['guidance_steps'] == 9
    assert s['prior']['guidance_scale'] == 0
    editor = s['prior']['editor']
    assert editor['algorithm'] == 'reference_initialized_ddim'
    assert editor['future_reference_steps'] == 9 and editor['known_history_steps'] == 3
    assert editor['inference_steps'] == c['ddim'] and editor['execution_steps'] == c['exec']
    assert editor['timesteps'][0] == c['t'] and len(editor['timesteps']) == c['ddim']
    assert abs(editor['actual_noise_ratio']-c['ratio']) < 1e-6
    assert [x['reference_index'] for x in pred] == list(range(0, len(ref), c['exec']))
    assert all(x['guidance_scale'] == 0 and x['history_mask_max_error'] == 0 and x['seeds'] == [c['seed']] for x in pred)
    with np.load(raw/'initial_state.npz') as a, np.load(f/'initial_state.npz') as b:
        assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    keys = [k for k in bt[0] if k != 'object_contact_force']
    assert all(all(a[k] == b[k] for k in keys) for a,b in zip(bt[:60],t[:60]))
    force_delta = float(np.max(np.abs(np.asarray([x['object_contact_force'] for x in bt[:60]])-
                                      np.asarray([x['object_contact_force'] for x in t[:60]]))))
    assert force_delta < 1e-5
    if archived is not None:
        geo = json.loads((f/'grasp_metrics.json').read_text())['frames']
    else:
        geo = metrics(f, all_frames=True)['frames']
    assert len(geo) == len(t)
    flags = []
    for row, g in zip(t, geo):
        assert row['phase'] == g['phase'] and row['index'] == g['index']
        force = float(np.linalg.norm(row['object_contact_force']))
        flags.append((g['mesh_vertex_gap_m'] > .005 and force < .05) or g['mesh_vertex_gap_m'] > .02)
    loss = next((i for i in range(len(flags)-2) if all(flags[i:i+3])), None)
    best = current = 0
    for i,(row,g) in enumerate(zip(t,geo)):
        good = (loss is None or i < loss) and row['phase'] == 'action' and row['vertical_error_deg'] <= 30 and \
            g['mesh_table_clearance_m'] > .08 and g['near_contact_link_count'] >= 2 and \
            g['mesh_vertex_gap_m'] < .008 and np.linalg.norm(row['object_contact_force']) > .1
        current = current+1 if good else 0
        best = max(best,current)
    actions = [x for x in t if x['phase'] == 'action']
    cmd = np.asarray([x['command'] for x in actions])
    cutoff = len(ref) if loss is None else max(0, min(t[loss]['index'], len(ref)))
    if loss is None:
        sep_progress = float(source.shape[1])
        sep_step = None
    else:
        sep_progress = float(progress[cutoff]) if t[loss]['phase'] == 'action' else float(source.shape[1])
        sep_step = t[loss]['index']+1 if t[loss]['phase'] == 'action' else None
    output = dict(episode=ep, config=c, folder=str(f), archived=archived is not None,
        first_separation_reference_progress=sep_progress,
        first_separation_actual_action_step=sep_step,
        no_separation_right_censored=loss is None,
        stable_turn=best >= 30, longest_vertical_contact_steps=best,
        command_reference_rmse_before_separation_rad=float(np.sqrt(np.mean((cmd[:cutoff]-ref[:cutoff])**2))) if cutoff else None,
        first_native_failure=s['first_native_failure'],
        validation=dict(initial_all_fields_exact=True, settle_state_exact=True,
            settle_force_max_delta_N=force_delta, reference_progress_exact=True,
            predictions_advance_by_exec=True, native_protocol_exact=True,
            full_tail_and_hold=True))
    if archived is not None:
        a = json.loads((f/'analysis.json').read_text())
        assert a['stable_turn'] == output['stable_turn'] and a['longest_vertical_contact_steps'] == best
        assert a['first_separation']['original_reference_progress'] == sep_progress
    else:
        (f/'search_analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print('AUDITED', ep, config_key(c), sep_progress, output['stable_turn'], best, flush=True)
    return output


def score(cfg, results):
    group = [x for x in results if config_key(x['config']) == config_key(cfg)]
    assert len(group) == 4 and [x['episode'] for x in group] == list(EPISODES)
    return dict(config=cfg, stable_count=sum(x['stable_turn'] for x in group),
        average_original_progress=float(np.mean([x['first_separation_reference_progress'] for x in group])),
        total_original_progress=float(sum(x['first_separation_reference_progress'] for x in group)),
        episode_results=group)


def ranking(rows):
    return sorted(rows, key=lambda x:(-x['stable_count'], -x['average_original_progress'],
                                      x['config']['t'], x['config']['ddim'], x['config']['exec']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=(1,2,3))
    args = parser.parse_args()
    stage = args.stage
    manifest = json.loads((O/f'stage{stage}_manifest.json').read_text())
    configs = manifest['configs']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda job:audit(*job), [(c,ep) for c in configs for ep in EPISODES]))
    (O/f'stage{stage}_results.json').write_text(json.dumps(results, indent=2)+'\n')
    all_scores = []
    for earlier in range(1,stage):
        all_scores += json.loads((O/f'stage{earlier}_scores.json').read_text())['new_scores']
    new_scores = [score(c,results) for c in configs]
    all_scores += new_scores
    ranked = ranking(all_scores)
    selected = ranked[:2]
    payload = dict(stage=stage, target_average_original_progress=190.875,
        new_scores=new_scores, cumulative_ranking=ranked,
        selected_for_next_stage=selected)
    (O/f'stage{stage}_scores.json').write_text(json.dumps(payload, indent=2)+'\n')
    for x in ranked:
        print('SCORE',config_key(x['config']),x['average_original_progress'],x['stable_count'],flush=True)
    print('STAGE SCORED',stage,flush=True)


if __name__ == '__main__':
    main()
