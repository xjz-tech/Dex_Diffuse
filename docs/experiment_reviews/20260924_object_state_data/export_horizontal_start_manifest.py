"""Export the distinct imported hand and bulb states for all 80 starts."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/all_full_episodes'


def main():
    selected = json.loads((ROOT / 'selection.json').read_text())['episodes']
    assert len(selected) == 80
    rows = []
    for episode, item in enumerate(selected):
        with np.load(ROOT / f'episode_{episode:02d}/reference_full.npz') as reference:
            hand = reference['hand_qpos_rad'][0, 0]
            bulb = reference['object_pose_wrist'][0, 0]
        assert hand.shape == (22,) and bulb.shape == (4, 4)
        row = dict(episode=episode,
                   source_start_frame=item['source_start_frame'],
                   source_end_state_frame=item['source_end_state_frame'],
                   lifted_start=item['lifted_start'])
        row.update({f'hand_q_{joint:02d}_rad': float(value)
                    for joint, value in enumerate(hand)})
        row.update({f'bulb_in_wrist_{i}{j}': float(bulb[i, j])
                    for i in range(4) for j in range(4)})
        rows.append(row)
    assert len({tuple(row[f'hand_q_{i:02d}_rad'] for i in range(22))
                for row in rows}) == 80
    assert len({tuple(row[f'bulb_in_wrist_{i}3'] for i in range(3))
                for row in rows}) == 80
    target = ROOT / 'horizontal_start_states.csv'
    with target.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(target)


if __name__ == '__main__':
    main()
