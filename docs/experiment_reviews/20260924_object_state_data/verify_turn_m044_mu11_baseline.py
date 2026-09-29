"""Validate raw turn starts at 44 g / friction 1.1 before guidance tests."""

import concurrent.futures
import json
import shutil
from pathlib import Path

import cv2

from compare_corrected_rollouts import PROTOCOL
from random4_geometry import metrics
from turn_baseline_screen import check


ROOT = Path(__file__).resolve().parent / 'reference_turn_baseline_20260926/qualified_comparison'
EPISODES = (76, 34, 54, 2)


def verify(ep):
    case = ROOT / f'episode_{ep:02d}'
    run = case / 'direct_m044_mu11'
    out = run / 'baseline_verified.json'
    if out.exists():
        result = json.loads(out.read_text())
        assert result['baseline_verified']
        return result
    screen = check(run)
    summary = json.loads((run / 'summary.json').read_text())
    trace = json.loads((run / 'trace.json').read_text())
    assert summary['mass_kg'] == .044 and summary['friction'] == 1.1
    assert summary['native_protocol'] == PROTOCOL
    assert summary['steps'] == summary['intended_steps']
    assert summary['reference_interpolation'] == 0
    assert summary['reference_interpolation_threshold'] is None
    assert summary['video_recorded'] and summary['grasp_evidence']
    assert screen['preliminary_pass']
    first = screen['block'][0] - 1
    evidence = run / 'turn_evidence'
    evidence.mkdir(exist_ok=True)
    for name in ('summary.json', 'initial_state.npz'):
        shutil.copy2(run / name, evidence / name)
    (evidence / 'trace.json').write_text(json.dumps(trace[:60] + trace[60 + first:60 + first + 30]))
    geometry = metrics(evidence, all_frames=True)['frames'][60:]
    assert len(geometry) == 30
    assert all(row['near_contact_link_count'] >= 2 and
               row['mesh_vertex_gap_m'] < .008 and
               row['mesh_table_clearance_m'] > .08 for row in geometry)
    cap = cv2.VideoCapture(str(run / 'direct_front.mp4'))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1 + len(trace)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 61 + first + 29)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (544, 1280, 3)
    still = run / 'verified_turn_frame.jpg'
    assert cv2.imwrite(str(still), frame)
    screen.update(baseline_verified=True,
                  mesh_confirmed_30_steps=[first + 1, first + 30],
                  min_near_contact_links=min(x['near_contact_link_count'] for x in geometry),
                  min_table_clearance_m=min(x['mesh_table_clearance_m'] for x in geometry),
                  max_mesh_gap_m=max(x['mesh_vertex_gap_m'] for x in geometry),
                  video_frame=str(still))
    out.write_text(json.dumps(screen, indent=2) + '\n')
    print('BASELINE VERIFIED', ep, screen['block'], flush=True)
    return screen


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(verify, EPISODES))
