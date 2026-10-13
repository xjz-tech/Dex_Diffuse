"""Verify that the selected recorded videos reproduce the complete batch traces."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/all_full_episodes'
METHODS = ('direct', 'guide2_exec2', 'guide2_exec1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episodes', type=int, nargs='+')
    args = parser.parse_args()
    results = []
    for episode in args.episodes:
        for method in METHODS:
            base = ROOT / f'episode_{episode:02d}'
            original = base / 'full' / method
            recorded = base / 'video_full' / method
            a = json.loads((original / 'summary.json').read_text())
            b = json.loads((recorded / 'summary.json').read_text())
            assert a['steps'] == b['steps'] == a['intended_steps'] == b['intended_steps']
            assert a['source_episode'] == b['source_episode'] == episode
            assert a['source_start_frame'] == b['source_start_frame']
            assert a['reference'] == b['reference']
            assert a['initial_object_pose_wrist'] == b['initial_object_pose_wrist']
            assert not a['video_recorded'] and b['video_recorded']
            assert not a['stop_on_native_failure'] and not b['stop_on_native_failure']
            initial_a = np.load(original / 'initial_state.npz')
            initial_b = np.load(recorded / 'initial_state.npz')
            assert set(initial_a.files) == set(initial_b.files)
            assert all(np.array_equal(initial_a[key], initial_b[key])
                       for key in initial_a.files)
            trace_a = json.loads((original / 'trace.json').read_text())
            trace_b = json.loads((recorded / 'trace.json').read_text())
            assert trace_a == trace_b, (episode, method, 'trace mismatch')
            results.append(dict(episode=episode, method=method,
                                steps=len(trace_a), exact_trace=True,
                                exact_initial_state=True))
    target = ROOT / 'selected_full_video_trace_audit.json'
    target.write_text(json.dumps(results, indent=2) + '\n')
    print(target, len(results), 'exact matches')


if __name__ == '__main__':
    main()
