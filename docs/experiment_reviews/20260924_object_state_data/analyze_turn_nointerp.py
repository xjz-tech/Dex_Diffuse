"""Audit un-interpolated scale-50 guided bulb turns against raw actions."""

import concurrent.futures
import json
from pathlib import Path

import numpy as np

import random4_analyze
from compare_corrected_rollouts import PROTOCOL


ROOT = Path(__file__).resolve().parent / 'reference_turn_baseline_20260926'
CASES = ROOT / 'qualified_comparison'
OUT = ROOT / 'nointerp_scale50_20260926'
random4_analyze.OUT = CASES
EPISODES = (76, 34, 54, 2)
JOBS = [(ep, exe) for ep in EPISODES for exe in (1, 2)]


def analyze(job):
    ep, exe = job
    case = CASES / f'episode_{ep:02d}'
    mode = f'guide{exe}_nointerp'
    folder = case / mode
    saved = folder / 'nointerp_result.json'
    if saved.exists():
        return json.loads(saved.read_text())
    raw = case / 'direct'
    summary = json.loads((folder / 'summary.json').read_text())
    raw_summary = json.loads((raw / 'summary.json').read_text())
    assert summary['prior']['ddim'] == 4
    assert summary['prior']['guidance_steps'] == 2
    assert summary['guidance_scale'] == 50
    assert summary['execution_steps'] == exe
    assert summary['reference_interpolation'] == 0
    assert summary['source_action_frames'] == raw_summary['source_action_frames']
    assert summary['steps'] == summary['intended_steps']
    assert summary['steps']['action'] == raw_summary['steps']['action']
    assert summary['native_protocol'] == raw_summary['native_protocol'] == PROTOCOL
    assert not summary['stop_on_native_failure'] and summary['action_limit'] is None
    assert summary['mass_kg'] == raw_summary['mass_kg'] == .17
    assert summary['friction'] == raw_summary['friction'] == 2.2

    predictions = json.loads((folder / 'predictions.json').read_text())
    assert [row['reference_index'] for row in predictions] == list(
        range(0, summary['steps']['action'], exe))

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

    retention = random4_analyze.analyze(ep, mode)
    cutoff = (retention['first_separation']['trace_index']
              if retention['first_separation'] else len(trace))
    best = []
    current = []
    for i, (row, geo) in enumerate(zip(trace, retention['frames'])):
        good = (i < cutoff and row['phase'] == 'action' and
                row['vertical_error_deg'] <= 30 and
                geo['mesh_table_clearance_m'] > .08 and
                geo['near_contact_link_count'] >= 2 and
                geo['mesh_vertex_gap_m'] < .008 and
                geo['object_contact_force_norm_N'] > .1)
        if good:
            current.append(row['index'] + 1)
            if len(current) > len(best):
                best = current.copy()
        else:
            current = []

    sensitivity = {}
    for gap in (.003, .005, .008, .01):
        flags = [((g['mesh_vertex_gap_m'] > gap and
                   g['object_contact_force_norm_N'] < .05) or
                  g['mesh_vertex_gap_m'] > .02) for g in retention['frames']]
        idx = next((i for i in range(len(flags) - 2)
                    if all(flags[i:i + 3])), None)
        sensitivity[str(gap)] = (1 + trace[idx]['index']
                                 if idx is not None and trace[idx]['phase'] == 'action'
                                 else None)

    actions = np.asarray([row['command'] for row in trace if row['phase'] == 'action'])
    reference = np.load(case / 'reference_full.npz')['hand_target_rad'][0]
    common_prefix = min(60, len(actions))
    early_rmse = float(np.sqrt(np.mean((actions[:common_prefix] -
                                        reference[:common_prefix]) ** 2)))
    result = dict(
        episode=ep, execution_steps=exe,
        first_separation=retention['first_separation'],
        completed_turn=len(best) >= 30,
        longest_vertical_contact_control_steps=len(best),
        vertical_contact_reference_interval=[best[0], best[-1]] if best else None,
        first_native_failure=summary['first_native_failure'],
        early_60_command_vs_reference_rmse_rad=early_rmse,
        sensitivity_reference_action_number=sensitivity,
        validation=dict(initial_all_fields_exact=True,
                        settle_all_fields_except_object_force_exact=True,
                        settle_max_object_force_delta_N=force_delta,
                        native_protocol_exact=True,
                        inference_reference_indices_exact=True,
                        full_tail_and_hold=True))
    (folder / 'nointerp_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ANALYZED', ep, exe,
          'loss', result['first_separation']['reference_action_number']
          if result['first_separation'] else None,
          'turn', result['completed_turn'], flush=True)
    return result


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(analyze, JOBS))
    (OUT / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
