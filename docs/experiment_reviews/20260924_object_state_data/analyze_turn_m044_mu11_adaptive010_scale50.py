"""Analyze raw, adaptive010 direct, and DDIM4 scale50 at 44 g / mu1.1."""

import concurrent.futures
import json
from pathlib import Path

import numpy as np

from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps


ROOT = Path(__file__).resolve().parent / 'reference_turn_baseline_20260926'
CASES = ROOT / 'qualified_comparison'
THRESHOLD = .10
OUT = ROOT / 'm044_mu11_adaptive010_ddim4_scale50_20260928'
EPISODES = (76, 34, 54, 2)
CONFIGS = (('guided', 4, 50),)
JOBS = [(ep, mode, ddim, scale) for ep in EPISODES
        for mode, ddim, scale in CONFIGS]


def folder_name(mode, ddim, scale):
    if mode == 'raw':
        return 'direct_m044_mu11'
    if mode == 'direct':
        return 'direct_adaptive010_m044_mu11'
    return f'guide1_adaptive010_ddim{ddim}_scale{scale}_m044_mu11'


def analyze(job):
    ep, mode, ddim, scale = job
    case = CASES / f'episode_{ep:02d}'
    folder = case / folder_name(mode, ddim, scale)
    saved = folder / 'm044_mu11_result.json'
    if saved.exists():
        return json.loads(saved.read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    raw = case / 'direct_m044_mu11'
    original = json.loads((raw / 'summary.json').read_text())
    source = np.load(case / 'reference_full.npz')['hand_target_rad']
    if mode == 'raw':
        expanded = source.copy()
        observed_progress = np.arange(1, source.shape[1] + 1, dtype=np.float64)
    else:
        expanded, expected_progress = interpolate_large_jumps(source, THRESHOLD)
        observed_progress = np.load(folder / 'reference_progress.npy')
        assert np.array_equal(observed_progress, expected_progress)
    assert summary['source_action_frames'] == original['source_action_frames']
    assert summary['steps']['action'] == len(observed_progress)
    if mode != 'raw':
        assert summary['adaptive_inserted_steps'] == len(observed_progress) - source.shape[1]
    assert summary['steps'] == summary['intended_steps']
    assert summary['reference_interpolation'] == 0
    assert summary['reference_interpolation_threshold'] == (None if mode == 'raw' else THRESHOLD)
    assert summary['reference_interpolation_equal_jump'] is None
    if mode != 'raw':
        assert summary['object_size_multiplier'] == 1.
        assert summary['settle_target_source'] == 'qpos'
    assert summary['native_protocol'] == original['native_protocol'] == PROTOCOL
    assert summary['mass_kg'] == original['mass_kg'] == .044
    assert summary['friction'] == original['friction'] == 1.1
    assert summary['vertical_scale_switch_angle_deg'] is None
    assert summary['vertical_scale_after'] is None
    assert summary['vertical_scale_trigger'] is None
    assert not summary['stop_on_native_failure'] and summary['action_limit'] is None

    if mode in ('raw', 'direct'):
        assert summary['mode'] == 'direct'
    else:
        assert summary['mode'] == 'guided'
        assert summary['execution_steps'] == 1
        assert summary['prior']['ddim'] == ddim
        assert summary['prior']['guidance_scale'] == scale
        assert summary['guidance_scale'] == scale
        assert summary['prior']['guidance_steps'] == 2
        assert summary['prior']['reference_interpolation_threshold'] == THRESHOLD
        predictions = json.loads((folder / 'predictions.json').read_text())
        assert [row['reference_index'] for row in predictions] == list(
            range(len(observed_progress)))
        assert all(row['guidance_scale'] == scale for row in predictions)

    with np.load(raw / 'initial_state.npz') as a, np.load(folder / 'initial_state.npz') as b:
        assert set(a.files) == set(b.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
    trace = json.loads((folder / 'trace.json').read_text())
    baseline = json.loads((raw / 'trace.json').read_text())
    assert len(trace) == sum(summary['steps'].values())
    static_keys = [key for key in baseline[0] if key != 'object_contact_force']
    assert all(all(x[key] == y[key] for key in static_keys)
               for x, y in zip(baseline[:60], trace[:60]))
    force_delta = float(np.max(np.abs(
        np.asarray([x['object_contact_force'] for x in baseline[:60]]) -
        np.asarray([x['object_contact_force'] for x in trace[:60]]))))
    assert force_delta < 1e-5
    if mode in ('raw', 'direct'):
        commands = np.asarray([x['command'] for x in trace if x['phase'] == 'action'])
        assert np.allclose(commands, expanded[0], atol=1e-7)

    data = metrics(folder, all_frames=True)
    evidence = data['frames']
    assert len(evidence) == len(trace)
    for row, geo in zip(trace, evidence):
        force = float(np.linalg.norm(row['object_contact_force']))
        geo['object_contact_force_norm_N'] = force
        geo['separated'] = bool(
            (geo['mesh_vertex_gap_m'] > .005 and force < .05) or
            geo['mesh_vertex_gap_m'] > .02)
    flags = [row['separated'] for row in evidence]
    index = next((i for i in range(len(flags) - 2) if all(flags[i:i + 3])), None)

    def source_progress(i):
        row = trace[i]
        if row['phase'] == 'action':
            return float(observed_progress[row['index']])
        return 0.0 if row['phase'] == 'settle' else float(source.shape[1])

    loss = None
    if index is not None:
        row = trace[index]
        progress = source_progress(index)
        loss = dict(zero_based_control_index=row['index'],
                    control_step=row['index'] + 1,
                    reference_action_number=progress,
                    source_action_frame=summary['source_start_frame'] + progress - 1,
                    trace_index=index, confirmation_control_steps=3,
                    **evidence[index])
    retention = dict(episode=ep, method=folder_name(mode, ddim, scale), first_separation=loss,
                     first_native_failure=summary['first_native_failure'],
                     action_count=summary['steps']['action'],
                     original_action_count=source.shape[1],
                     source_progress_map=(None if mode == 'raw' else 'reference_progress.npy'),
                     frames=evidence)
    (folder / 'retention.json').write_text(json.dumps(retention, indent=2) + '\n')

    cutoff = index if index is not None else len(trace)
    best = []
    current = []
    for i, (row, geo) in enumerate(zip(trace, evidence)):
        good = (i < cutoff and row['phase'] == 'action' and
                row['vertical_error_deg'] <= 30 and
                geo['mesh_table_clearance_m'] > .08 and
                geo['near_contact_link_count'] >= 2 and
                geo['mesh_vertex_gap_m'] < .008 and
                geo['object_contact_force_norm_N'] > .1)
        if good:
            current.append(row['index'])
            if len(current) > len(best):
                best = current.copy()
        else:
            current = []

    sensitivity = {}
    for gap in (.003, .005, .008, .01):
        trial = [((geo['mesh_vertex_gap_m'] > gap and
                   geo['object_contact_force_norm_N'] < .05) or
                  geo['mesh_vertex_gap_m'] > .02) for geo in evidence]
        first = next((i for i in range(len(trial) - 2)
                      if all(trial[i:i + 3])), None)
        sensitivity[str(gap)] = source_progress(first) if first is not None else None

    result = dict(
        episode=ep, mode=mode, ddim_steps=ddim, guidance_scale=scale,
        inserted_steps=summary['adaptive_inserted_steps'],
        executed_action_steps=summary['steps']['action'],
        first_separation=loss, completed_turn=len(best) >= 30,
        longest_vertical_contact_control_steps=len(best),
        vertical_contact_reference_interval=(
            [float(observed_progress[best[0]]), float(observed_progress[best[-1]])]
            if best else None),
        first_native_failure=summary['first_native_failure'],
        sensitivity_reference_action_number=sensitivity,
        validation=dict(initial_all_fields_exact=True,
                        settle_all_fields_except_object_force_exact=True,
                        settle_max_object_force_delta_N=force_delta,
                        progress_map_exact=True, native_protocol_exact=True,
                        full_tail_and_hold=True))
    saved.write_text(json.dumps(result, indent=2) + '\n')
    print('ANALYZED', ep, mode, ddim, scale,
          'loss', loss['reference_action_number'] if loss else None,
          'turn', result['completed_turn'], flush=True)
    return result


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(analyze, JOBS))
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
