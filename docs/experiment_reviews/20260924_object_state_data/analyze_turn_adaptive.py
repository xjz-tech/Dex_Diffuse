"""Analyze selective interpolation using the exact variable source-progress map."""

import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np

from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from reference_resampling import interpolate_large_jumps


ROOT = Path(__file__).resolve().parent / 'reference_turn_baseline_20260926'
CASES = ROOT / 'qualified_comparison'
THRESHOLD = float(sys.argv[1]) if len(sys.argv) > 1 else .12
assert THRESHOLD in (.09, .12, .14)
TAG = f'adaptive{round(THRESHOLD * 100):03d}'
OUT = ROOT / f'{TAG}_scale50_20260926'
EPISODES = (76, 34, 54, 2)
MODES = ('direct', 'guide1') if THRESHOLD != .12 else ('direct', 'guide1', 'guide2')
JOBS = [(ep, mode) for ep in EPISODES for mode in MODES]


def analyze(job):
    ep, mode = job
    case = CASES / f'episode_{ep:02d}'
    folder = case / f'{mode}_{TAG}'
    saved = folder / 'adaptive_result.json'
    if saved.exists():
        return json.loads(saved.read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    raw = case / 'direct'
    original = json.loads((raw / 'summary.json').read_text())
    source = np.load(case / 'reference_full.npz')['hand_target_rad']
    expanded, expected_progress = interpolate_large_jumps(source, THRESHOLD)
    observed_progress = np.load(folder / 'reference_progress.npy')
    assert np.array_equal(observed_progress, expected_progress)
    assert summary['source_action_frames'] == original['source_action_frames']
    assert summary['steps']['action'] == len(observed_progress)
    assert summary['adaptive_inserted_steps'] == len(observed_progress) - source.shape[1]
    assert summary['steps'] == summary['intended_steps']
    assert summary['reference_interpolation'] == 0
    assert summary['reference_interpolation_threshold'] == THRESHOLD
    assert summary['native_protocol'] == original['native_protocol'] == PROTOCOL
    assert summary['mass_kg'] == original['mass_kg'] == .17
    assert summary['friction'] == original['friction'] == 2.2
    assert summary['guidance_scale'] == 50
    assert not summary['stop_on_native_failure'] and summary['action_limit'] is None

    if mode == 'direct':
        assert summary['mode'] == 'direct'
    else:
        exe = int(mode[-1])
        assert summary['mode'] == 'guided'
        assert summary['execution_steps'] == exe
        assert summary['prior']['ddim'] == 4
        assert summary['prior']['guidance_steps'] == 2
        assert summary['prior']['reference_interpolation_threshold'] == THRESHOLD
        predictions = json.loads((folder / 'predictions.json').read_text())
        assert [row['reference_index'] for row in predictions] == list(
            range(0, len(observed_progress), exe))

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
    if mode == 'direct':
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
    retention = dict(episode=ep, method=mode, first_separation=loss,
                     first_native_failure=summary['first_native_failure'],
                     action_count=summary['steps']['action'],
                     original_action_count=source.shape[1],
                     source_progress_map='reference_progress.npy', frames=evidence)
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
        episode=ep, mode=mode, inserted_steps=summary['adaptive_inserted_steps'],
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
    print('ANALYZED', ep, mode,
          'loss', loss['reference_action_number'] if loss else None,
          'turn', result['completed_turn'], flush=True)
    return result


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(analyze, JOBS))
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
