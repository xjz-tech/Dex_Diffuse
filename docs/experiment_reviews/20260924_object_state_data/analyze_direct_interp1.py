"""Analyze the matched interpolation-only control for the qualified bulb turns."""

import concurrent.futures
import json
from pathlib import Path

import numpy as np

import random4_analyze
from compare_corrected_rollouts import PROTOCOL


ROOT = Path(__file__).resolve().parent / 'reference_turn_baseline_20260926'
CASES = ROOT / 'qualified_comparison'
random4_analyze.OUT = CASES
EPISODES = (76, 34, 54, 2)


def analyze_episode(ep):
    case = CASES / f'episode_{ep:02d}'
    control = case / 'direct_interp1'
    baseline = case / 'direct'
    s = json.loads((control / 'summary.json').read_text())
    d = json.loads((baseline / 'summary.json').read_text())
    assert s['mode'] == 'direct' and s['reference_interpolation'] == 1
    assert d['mode'] == 'direct' and d['reference_interpolation'] == 0
    assert s['source_action_frames'] == d['source_action_frames']
    assert s['native_protocol'] == d['native_protocol'] == PROTOCOL
    assert s['steps'] == s['intended_steps']
    assert s['steps']['action'] == 2 * d['steps']['action'] - 1
    assert s['action_limit'] is None and not s['stop_on_native_failure']
    assert s['mass_kg'] == d['mass_kg'] == .17
    assert s['friction'] == d['friction'] == 2.2

    with np.load(control / 'initial_state.npz') as a, np.load(baseline / 'initial_state.npz') as b:
        assert set(a.files) == set(b.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files)

    tr = json.loads((control / 'trace.json').read_text())
    old = json.loads((baseline / 'trace.json').read_text())
    assert len(tr) == sum(s['steps'].values())
    static_keys = [key for key in old[0] if key != 'object_contact_force']
    assert all(all(x[key] == y[key] for key in static_keys)
               for x, y in zip(old[:60], tr[:60]))
    force_delta = float(np.max(np.abs(
        np.asarray([x['object_contact_force'] for x in old[:60]]) -
        np.asarray([x['object_contact_force'] for x in tr[:60]]))))
    assert force_delta < 1e-5

    reference = np.load(case / 'reference_full.npz')['hand_target_rad'][0]
    expected = np.empty((2 * len(reference) - 1, reference.shape[1]))
    expected[::2] = reference
    expected[1::2] = (reference[:-1] + reference[1:]) / 2
    commands = np.asarray([x['command'] for x in tr if x['phase'] == 'action'])
    assert np.allclose(commands, expected, atol=1e-7)

    retention = random4_analyze.analyze(ep, 'direct_interp1')
    cutoff = (retention['first_separation']['trace_index']
              if retention['first_separation'] else len(tr))
    best = []
    current = []
    for i, (row, geo) in enumerate(zip(tr, retention['frames'])):
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

    return dict(
        episode=ep,
        first_separation=retention['first_separation'],
        completed_turn=len(best) >= 30,
        longest_vertical_contact_control_steps=len(best),
        vertical_contact_reference_interval=(
            [1 + (best[0] - 1) / 2, 1 + (best[-1] - 1) / 2] if best else None),
        first_native_failure=s['first_native_failure'],
        full_action_steps=s['steps']['action'],
        validation=dict(initial_all_fields_exact=True,
                        settle_all_fields_except_object_force_exact=True,
                        settle_max_object_force_delta_N=force_delta,
                        expanded_commands_exact=True,
                        native_protocol_exact=True,
                        full_tail_and_hold=True),
    )


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(analyze_episode, EPISODES))
    output = ROOT / 'direct_interp1_comparison_20260926'
    output.mkdir(exist_ok=True)
    (output / 'RESULTS.json').write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(row['episode'], 'separation', row['first_separation']['reference_action_number']
              if row['first_separation'] else None, 'turn', row['completed_turn'], flush=True)


if __name__ == '__main__':
    main()
