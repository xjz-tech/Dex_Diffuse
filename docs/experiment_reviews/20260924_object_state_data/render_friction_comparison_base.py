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
    assert summary['mass_kg'] == .044 and summary['friction'] in (1.1, 1.4)
    projectors = []
    for meta in summary['camera_metadata']:
        projectors.append(camera_projector(np.asarray(meta['eye'], dtype=float),
                                            np.asarray(meta['target'], dtype=float),
                                            meta['horizontal_fov']))
    assert len(projectors) == 2
    return dict(folder=str(folder), summary=summary, trace=trace, cap=cap,
                projectors=projectors, last_video_index=-1, frame=None)

