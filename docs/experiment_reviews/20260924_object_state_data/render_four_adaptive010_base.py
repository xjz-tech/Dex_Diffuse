"""Four-episode raw reference versus DDIM4/scale25 at 44 g / friction 1.1."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from reference_resampling import interpolate_large_jumps
from render_full_episode53_axes import camera_projector, draw_axes, object_pose


P = Path(__file__).resolve().parent
ROOT = P / 'reference_turn_baseline_20260926'
CASES = ROOT / 'qualified_comparison'
OUT = ROOT / 'm044_mu11_adaptive010_ddim4_scale25_four_video_20260927'
OUT.mkdir(exist_ok=True)
VIDEO = OUT / 'four_episodes_raw_vs_ddim4_scale25_m044_mu11.mp4'
EPISODES = (76, 34, 54, 2)
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
LABELS = ('原始 reference · 仿真正面', 'DDIM4 / scale25 · 仿真正面',
          '原始 reference · 含桌面宽景', 'DDIM4 / scale25 · 含桌面宽景')
PANELS = ((0, 64), (640, 64), (0, 608), (640, 608))


def read_to(run, index):
    while run['last_video_index'] < index:
        ok, frame = run['cap'].read()
        if not ok:
            raise RuntimeError(f"video truncated: {run['folder']} at frame {index}")
        assert frame.shape == (544, 1280, 3), frame.shape
        run['frame'] = frame
        run['last_video_index'] += 1
    return run['frame']


def load_run(folder, mode):
    summary = json.loads((folder / 'summary.json').read_text())
    trace = json.loads((folder / 'trace.json').read_text())
    cap = cv2.VideoCapture(str(folder / ('direct_front.mp4' if mode == 'direct' else 'guided_front.mp4')))
    assert cap.isOpened(), folder
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1 + len(trace)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 30) < 1e-6
    assert summary['steps'] == summary['intended_steps']
    assert summary['steps']['settle'] == summary['steps']['hold'] == 60
    assert summary['video_recorded']
    assert summary['mass_kg'] == .044 and summary['friction'] == 1.1
    projectors = []
    for meta in summary['camera_metadata']:
        projectors.append(camera_projector(np.asarray(meta['eye'], dtype=float),
                                            np.asarray(meta['target'], dtype=float),
                                            meta['horizontal_fov']))
    assert len(projectors) == 2
    return dict(folder=str(folder), summary=summary, trace=trace, cap=cap,
                projectors=projectors, last_video_index=-1, frame=None)


def episode_data(ep):
    case = CASES / f'episode_{ep:02d}'
    direct = load_run(case / 'direct_m044_mu11', 'direct')
    guide = load_run(case / 'guide1_adaptive010_ddim4_scale25_m044_mu11_video', 'guided')
    baseline = case / 'guide1_adaptive010_ddim4_scale25_m044_mu11'
    with np.load(baseline / 'initial_state.npz') as a, np.load(Path(guide['folder']) / 'initial_state.npz') as b, np.load(case / 'direct_m044_mu11/initial_state.npz') as c:
        assert all(np.array_equal(a[k], b[k]) and np.array_equal(a[k], c[k]) for k in a.files)
    assert guide['summary']['object_size_multiplier'] == 1.
    assert guide['summary']['settle_target_source'] == 'qpos'
    assert guide['summary']['execution_steps'] == 1 and guide['summary']['prior']['guidance_steps'] == 2
    prior_trace = json.loads((baseline / 'trace.json').read_text())
    assert len(prior_trace) == len(guide['trace'])
    for key in ('command', 'q', 'object_pose', 'vertical_error_deg', 'native_failure', 'hand_body_pose'):
        assert all(x[key] == y[key] for x, y in zip(prior_trace, guide['trace'])), (ep, key)
    with np.load(case / 'reference_full.npz') as ref:
        source_actions = ref['hand_target_rad'].shape[1]
        expanded, progress = interpolate_large_jumps(ref['hand_target_rad'], .10)
    assert np.array_equal(progress, np.load(Path(guide['folder']) / 'reference_progress.npy'))
    assert expanded.shape[1] == guide['summary']['steps']['action']
    assert source_actions == direct['summary']['steps']['action']
    assert direct['summary']['source_start_frame'] == guide['summary']['source_start_frame']
    assert direct['summary']['native_protocol'] == guide['summary']['native_protocol']
    assert guide['summary']['prior']['ddim'] == 4
    assert guide['summary']['guidance_scale'] == 25
    assert guide['summary']['reference_interpolation_threshold'] == .10
    assert guide['summary']['vertical_scale_switch_angle_deg'] is None
    losses = [json.loads((case / 'direct_m044_mu11' / 'm044_mu11_result.json').read_text())['first_separation'],
              json.loads((baseline / 'm044_mu11_result.json').read_text())['first_separation']]
    return dict(episode=ep, case=case, direct=direct, guide=guide,
                progress=progress, source_actions=source_actions,
                losses=losses)

