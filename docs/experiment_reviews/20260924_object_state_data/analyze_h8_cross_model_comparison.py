"""Audit the paired bulb-turn experiments with the archived geometry criterion."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np

from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps
from run_h8_cross_model_comparison import C, EPISODES, METHODS, O, PHYSICS, destination

R = C.parent


def archived_10b(physics, episode, method):
    if physics == 'm044_mu11':
        if method == 'guidance4':
            return C / f'episode_{episode:02d}' / 'guide4exec2_adaptive010_ddim4_scale50_m044_mu11'
        return R / f'four_reference_edit_noise_ddim_sweep_20260928/episode_{episode:02d}/edit_015_ddim4_seed44'
    return R / f'{physics}_guidance4_vs_edit015_20260928/episode_{episode:02d}/{method}'


def raw_baseline(physics, episode):
    if physics == 'm044_mu11':
        return C / f'episode_{episode:02d}/direct_m044_mu11'
    return R / f'{physics}_guidance4_vs_edit015_20260928/episode_{episode:02d}/raw'


def audit(seed, model, physics, episode, method):
    folder = archived_10b(physics, episode, method) if seed == 44 and model == '10b' else destination(seed, model, physics, episode, method)
    summary = json.loads((folder / 'summary.json').read_text())
    trace = json.loads((folder / 'trace.json').read_text())
    geom_file = folder / 'grasp_metrics.json'
    geom = json.loads(geom_file.read_text()) if geom_file.exists() else metrics(folder, all_frames=True)
    geo = geom['frames']
    ref = np.load(C / f'episode_{episode:02d}/reference_full.npz')['hand_target_rad']
    expanded, progress = interpolate_large_jumps(ref, .1)
    expanded = expanded[0]
    assert len(trace) == len(geo) == len(expanded) + 120
    assert summary['steps'] == summary['intended_steps'] == dict(settle=60, action=len(expanded), hold=60)
    assert summary['reference_interpolation_threshold'] == .1
    assert summary['execution_steps'] == 2
    if 'prior_noise_seed' in summary:
        assert summary['prior_noise_seed'] == seed
    assert summary['prior']['ddim'] == 4
    assert summary['mass_kg'] == PHYSICS[physics][0] and summary['friction'] == PHYSICS[physics][1]
    assert summary['settle_target_source'] == 'qpos' and not summary['stop_on_native_failure']
    assert summary['object_size_multiplier'] == 1
    assert summary['prior']['guidance_steps'] == (4 if method == 'guidance4' else (5 if model == 'h8' else 9))
    assert summary['prior']['guidance_scale'] == (50 if method == 'guidance4' else 0)
    if method == 'edit015':
        editor = summary['prior']['editor']
        assert editor['requested_noise_ratio'] == .15
        assert editor['future_reference_steps'] == (5 if model == 'h8' else 9)
        assert editor['timesteps'] == [8, 5, 3, 0]
    raw_folder = raw_baseline(physics, episode)
    with np.load(raw_folder / 'initial_state.npz') as a, np.load(folder / 'initial_state.npz') as b:
        assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    raw_trace = json.loads((raw_folder / 'trace.json').read_text())
    for field in ('object_pose', 'q', 'executed_target', 'relative_position',
                  'displacement_from_import_m', 'rotation_from_import_deg', 'vertical_error_deg'):
        assert all(a[field] == b[field] for a, b in zip(raw_trace[:60], trace[:60]))
    force_delta = float(np.max(np.abs(np.asarray([a['object_contact_force'] for a in raw_trace[:60]]) -
                                      np.asarray([b['object_contact_force'] for b in trace[:60]]))))
    assert force_delta < 1e-5
    pred = json.loads((folder / 'predictions.json').read_text())
    assert [p['reference_index'] for p in pred] == list(range(0, len(expanded), 2))
    assert all(p['seeds'] == [seed] for p in pred) if method == 'edit015' else True
    flags = []
    for row, g in zip(trace, geo):
        assert row['phase'] == g['phase'] and row['index'] == g['index']
        force = float(np.linalg.norm(row['object_contact_force']))
        flags.append((g['mesh_vertex_gap_m'] > .005 and force < .05) or g['mesh_vertex_gap_m'] > .02)
    loss = next((i for i in range(len(flags)-2) if all(flags[i:i+3])), None)
    assert loss is None or trace[loss]['phase'] in ('action', 'hold')
    cutoff = len(trace) if loss is None else loss
    longest = current = 0
    ordinary = False
    for i, (row, g) in enumerate(zip(trace, geo)):
        good = (i < cutoff and row['phase'] == 'action' and row['vertical_error_deg'] <= 30 and
                g['mesh_table_clearance_m'] > .08 and g['near_contact_link_count'] >= 2 and
                g['mesh_vertex_gap_m'] < .008 and np.linalg.norm(row['object_contact_force']) > .1)
        ordinary |= bool(good)
        current = current + 1 if good else 0
        longest = max(longest, current)
    actions = [row for row in trace if row['phase'] == 'action']
    commands = np.asarray([row['command'] for row in actions])
    limit = len(actions) if loss is None or trace[loss]['phase'] == 'hold' else trace[loss]['index']
    assert limit > 0
    return dict(seed=seed, model=model, physics=physics, episode=episode, method=method,
        folder=str(folder), first_separation_phase=None if loss is None else trace[loss]['phase'],
        first_separation_action_step=None if loss is None else (trace[loss]['index'] + 1 if trace[loss]['phase'] == 'action' else len(expanded)),
        first_separation_reference_progress=float(progress[min(limit, len(progress)-1)]),
        ordinary_turn=ordinary, stable_turn=longest >= 30, longest_vertical_contact_steps=longest,
        command_reference_rmse_before_separation_rad=float(np.sqrt(np.mean((commands[:limit] - expanded[:limit]) ** 2))),
        first_native_failure=summary['first_native_failure'],
        validation=dict(initial_all_fields_exact=True, settle_motion_exact=True,
                        settle_contact_force_max_delta_N=force_delta))


def audit_job(job):
    return audit(*job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[44, 45, 46])
    parser.add_argument('--models', nargs='+', choices=('1b', '10b', 'h8'), default=['1b', '10b', 'h8'])
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    jobs = [(seed, model, physics, episode, method) for seed in args.seeds
            for model in args.models for physics in PHYSICS for episode in EPISODES for method in METHODS]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(audit_job, jobs):
            results.append(row)
            print(row['seed'], row['model'], row['physics'], row['episode'], row['method'],
                row['ordinary_turn'], row['stable_turn'], row['first_separation_action_step'], flush=True)
    path = O / 'audited_results.json'
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print('AUDITED', len(results), 'rollouts', flush=True)


if __name__ == '__main__':
    main()
