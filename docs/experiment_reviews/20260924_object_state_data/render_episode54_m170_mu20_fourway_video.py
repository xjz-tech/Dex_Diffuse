"""Compose Episode 54 at 170 g / mu=2.0: three simulations plus real front."""
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_full_episode53_axes import camera_projector, draw_axes, object_pose


P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
CASE = R / 'qualified_comparison/episode_54'
BASE = R / 'm170_mu20_guidance4_vs_edit015_20260928' / 'episode_54'
OUT = R / 'm170_mu20_episode54_fourway_video_20260928'
DATA = Path('/home/carus/Data/Object_state_data/episode_54/front')
VIDEO = OUT / 'episode54_m170_mu20_direct_guidance_edit_real_front.mp4'
DOWNLOAD = Path('/home/carus/Downloads/episode54_170g_mu2.0_direct_guidance_edit_real.mp4')
METHODS = ('direct_interp', 'guidance4', 'edit015')
LABELS = {
    'direct_interp': '同规则插值 direct · 仿真正面',
    'guidance4': '传统 guidance · DDIM4 / scale50 / guide4 exec2',
    'edit015': '动作编辑 · noise0.15 / DDIM4 / exec2',
}
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
POSITIONS = ((0, 64), (640, 64), (0, 608))
REAL_POSITION = (640, 608)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_to(run, index):
    while run['last_video_index'] < index:
        ok, frame = run['cap'].read()
        if not ok:
            raise RuntimeError(f"video truncated: {run['folder']} at frame {index}")
        assert frame.shape == (544, 1280, 3), frame.shape
        run['frame'] = frame
        run['last_video_index'] += 1
    return run['frame']


def load_run(method):
    folder = OUT / method
    baseline = BASE / method
    summary = json.loads((folder / 'summary.json').read_text())
    trace = json.loads((folder / 'trace.json').read_text())
    baseline_trace = json.loads((baseline / 'trace.json').read_text())
    analysis = json.loads((baseline / 'analysis.json').read_text())
    stem = 'direct_front.mp4' if method == 'direct_interp' else 'guided_front.mp4'
    cap = cv2.VideoCapture(str(folder / stem))
    assert cap.isOpened(), folder
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 1 + len(trace)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 30) < 1e-6
    assert len(trace) == len(baseline_trace)
    for key in ('command', 'q', 'object_pose', 'vertical_error_deg', 'native_failure'):
        assert all(a[key] == b[key] for a, b in zip(trace, baseline_trace)), (method, key)
    with np.load(folder / 'initial_state.npz') as a, np.load(baseline / 'initial_state.npz') as b:
        assert set(a.files) == set(b.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
    progress = np.load(folder / 'reference_progress.npy')
    assert np.array_equal(progress, np.load(baseline / 'reference_progress.npy'))
    assert len(progress) == summary['steps']['action'] == 612
    assert summary['steps']['settle'] == summary['steps']['hold'] == 60
    assert summary['mass_kg'] == .17 and summary['friction'] == 2.0
    assert summary['reference_interpolation_threshold'] == .1
    assert summary['video_recorded'] and summary['front_only'] and summary['audit_recording']
    if method == 'direct_interp':
        assert summary['mode'] == 'direct' and summary['execution_steps'] == 1
    elif method == 'guidance4':
        assert summary['mode'] == 'guided' and summary['execution_steps'] == 2
        assert summary['guidance_scale'] == 50
        assert summary['prior']['ddim'] == 4 and summary['prior']['guidance_steps'] == 4
    else:
        assert summary['mode'] == 'guided' and summary['execution_steps'] == 2
        assert summary['guidance_scale'] == 0
        preds = json.loads((folder / 'predictions.json').read_text())
        assert preds and all(abs(x['noise_ratio'] - .15) < 1e-12 for x in preds)
        assert summary['prior']['editor']['timesteps'] == [8, 5, 3, 0]
    meta = summary['camera_metadata'][0]
    projector = camera_projector(np.asarray(meta['eye'], dtype=float),
                                 np.asarray(meta['target'], dtype=float),
                                 meta['horizontal_fov'])
    return dict(method=method, folder=str(folder), summary=summary, trace=trace,
                analysis=analysis, progress=progress, cap=cap, projector=projector,
                last_video_index=-1, frame=None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [load_run(method) for method in METHODS]
    assert all(np.array_equal(runs[0]['progress'], r['progress']) for r in runs[1:])
    progress = runs[0]['progress']
    action_steps = len(progress)
    with np.load(CASE / 'reference_full.npz') as ref:
        source_ids = ref['source_state_frame_indices'][0].astype(int)
        original_actions = ref['hand_target_rad'].shape[1]
    assert original_actions == 462 and len(source_ids) == original_actions + 1
    assert source_ids[0] == 110 and all((DATA/f'{i:06d}.png').is_file() for i in source_ids)
    timeline = ([('import', 0)] * 30 + [('settle', i) for i in range(60)] +
                [('ready', 59)] * 30 + [('action', i) for i in range(action_steps)] +
                [('hold', i) for i in range(60)])
    verification = dict(video=str(VIDEO), download=str(DOWNLOAD), episode=54,
        mass_kg=.17, friction=2.0, fps=30,
        panels=[LABELS[m] for m in METHODS] + ['原数据集 front · 真机'],
        simulation_view='front camera only',
        object_axes='simulation object-local XYZ: red, green, blue',
        alignment=('simulations use the same expanded control step and original-reference '
                   'progress; real frame uses source_state_frame_indices at floor(progress)'),
        traces_exactly_match_no_video=True, initial_states_exactly_match_no_video=True,
        results={m: json.loads((BASE/m/'analysis.json').read_text()) for m in METHODS})
    writer = imageio.get_writer(str(VIDEO), fps=30, codec='libx264', quality=7,
        macro_block_size=16,
        ffmpeg_params=['-preset', 'fast', '-threads', '2', '-movflags', '+faststart'])
    frames = 0
    source_cache = {}
    try:
        for phase, j in timeline:
            if phase == 'import':
                video_index = 0
                trace_index = None
                source_index = 0
                phase_text = '真实初态导入（定格 1 秒）'
                progress_value = 0.
            elif phase in ('settle', 'ready'):
                video_index = 1 + j
                trace_index = j
                source_index = 0
                phase_text = '共同动作起点（定格 1 秒）' if phase == 'ready' else f'静置检查 {j + 1}/60'
                progress_value = 0.
            elif phase == 'action':
                video_index = 61 + j
                trace_index = 60 + j
                progress_value = float(progress[j])
                source_index = min(int(np.floor(progress_value)), original_actions)
                phase_text = f'动作步 {j + 1}/{action_steps}'
            else:
                video_index = 61 + action_steps + j
                trace_index = 60 + action_steps + j
                source_index = original_actions
                progress_value = float(original_actions)
                phase_text = f'完整动作结束后保持 {j + 1}/60'

            source_id = int(source_ids[source_index])
            if source_id not in source_cache:
                image = Image.open(DATA/f'{source_id:06d}.png').convert('RGB')
                assert image.size == (640, 480)
                source_cache[source_id] = image
            canvas = Image.new('RGB', (1280, 1152), (17, 24, 33))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 2), f'Episode 54 · 170g · 摩擦系数 2.0 · {phase_text}',
                      font=FONT, fill='white')
            draw.text((12, 34),
                      f'原始 reference 进度 {progress_value:g}/462 · 仿真灯泡局部轴 X/Y/Z：红/绿/蓝',
                      font=SMALL, fill=(190, 225, 250))
            for (x, y), run in zip(POSITIONS, runs):
                frame = read_to(run, video_index)
                rgb = cv2.cvtColor(frame[:, :640], cv2.COLOR_BGR2RGB)
                canvas.paste(Image.fromarray(rgb), (x, y))
                row = None if trace_index is None else run['trace'][trace_index]
                if row is not None:
                    draw_axes(canvas, object_pose(row), run['projector'], (x, y))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((x, y, x + 639, y + 63), fill=(25, 36, 49))
                draw.text((x + 8, y), LABELS[run['method']], font=FONT, fill='white')
                a = run['analysis']
                detail = (f"分离进度 {a['first_separation_reference_progress']:g} · "
                          f"最长有效竖直保持 {a['longest_vertical_contact_steps']} 步")
                draw.text((x + 8, y + 35), detail, font=SMALL, fill=(188, 231, 208))

            rx, ry = REAL_POSITION
            canvas.paste(source_cache[source_id], (rx, ry + 64))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((rx, ry, rx + 639, ry + 63), fill=(25, 36, 49))
            draw.text((rx + 8, ry), '原数据集 front · 真机', font=FONT, fill='white')
            draw.text((rx + 8, ry + 35),
                      f'源帧 {source_id} · 原始动作进度 {progress_value:g}',
                      font=SMALL, fill=(188, 231, 208))
            writer.append_data(np.asarray(canvas))
            frames += 1
    finally:
        writer.close()
        for run in runs:
            run['cap'].release()

    cap = cv2.VideoCapture(str(VIDEO))
    assert cap.isOpened()
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (1152, 1280, 3)
        decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert decoded == frames == len(timeline) and abs(fps - 30) < 1e-6
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(VIDEO, DOWNLOAD)
    assert sha256(VIDEO) == sha256(DOWNLOAD)
    verification.update(frames=decoded, duration_s=decoded/30, codec='H.264',
                        fully_decoded=True, sha256=sha256(VIDEO),
                        download_copy_exact=True)
    (OUT/'video_verification.json').write_text(json.dumps(verification, ensure_ascii=False,
                                                          indent=2)+'\n')
    print('VIDEO', VIDEO, decoded, decoded/30, flush=True)
    print('DOWNLOAD', DOWNLOAD, DOWNLOAD.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
