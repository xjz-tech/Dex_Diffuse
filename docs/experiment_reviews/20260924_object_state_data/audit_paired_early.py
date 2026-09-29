"""Check per-episode initial and settling equivalence across the three controllers."""

import json
from pathlib import Path

import numpy as np


P = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/all_full_episodes'
METHODS = ['direct', 'guide2_exec2', 'guide2_exec1']
STATIC_FIELDS = ['q', 'command', 'object_pose', 'relative_position',
                 'vertical_deg', 'native_failure']


def audit_folder(folder):
    summaries = [json.loads((folder / method / 'summary.json').read_text())
                 for method in METHODS]
    ids = summaries[0]['source_episode_ids']
    assert all(s['source_episode_ids'] == ids for s in summaries)
    snapshots = [np.load(folder / method / 'initial_state.npz') for method in METHODS]
    traces = [np.load(folder / method / 'trajectory.npz') for method in METHODS]
    fields = [key for key in snapshots[0].files if all(
        key in snap.files and snap[key].shape == snapshots[0][key].shape and
        snap[key].dtype.kind in 'ifub' for snap in snapshots)]
    assert len(fields) == 27, (folder, len(fields))
    output = []
    for pos, episode in enumerate(ids):
        initial = all(np.array_equal(
                          snapshots[0][key][pos] if snapshots[0][key].shape[0] == len(ids) else snapshots[0][key],
                          snap[key][pos] if snap[key].shape[0] == len(ids) else snap[key])
                      for snap in snapshots[1:] for key in fields)
        static_lengths = [s['steps']['settle'] for s in summaries]
        static = (len(set(static_lengths)) == 1 and
                  all(np.array_equal(traces[0][key][:static_lengths[0], pos],
                                     trace[key][:static_lengths[0], pos])
                      for trace in traces[1:] for key in STATIC_FIELDS))
        output.append(dict(episode=episode, folder=str(folder), initial_exact=initial,
                           static_exact=static, static_steps=static_lengths))
    return output


def main():
    records = []
    for first in range(0, 80, 2):
        folder = P / f'pair_{first:02d}_{first+1:02d}'
        if all((folder / method / 'summary.json').exists() for method in METHODS):
            records.extend(audit_folder(folder))
    affected = [r['episode'] for r in records if not r['initial_exact'] or not r['static_exact']]
    report = dict(completed=len(records), exact=sum(r['initial_exact'] and r['static_exact'] for r in records),
                  affected=affected, rows=records)
    (P / 'pairing_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(completed=report['completed'], exact=report['exact'],
                          affected=affected), indent=2))


if __name__ == '__main__':
    main()
