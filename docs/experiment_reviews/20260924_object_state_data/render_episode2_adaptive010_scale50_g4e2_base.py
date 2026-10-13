"""Four-episode raw reference versus DDIM4/scale50 at 44 g / friction 1.1."""

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
OUT = ROOT / 'm044_mu11_adaptive010_ddim4_scale50_g4e2_episode2_video_20260928'
OUT.mkdir(exist_ok=True)
VIDEO = OUT / 'four_episodes_raw_vs_ddim4_scale50_m044_mu11.mp4'
EPISODES = (2,)
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
LABELS = ('原始 reference · 仿真正面', 'DDIM4 / scale50 · 仿真正面',
          '原始 reference · 含桌面宽景', 'DDIM4 / scale50 · 含桌面宽景')
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
    guide = load_run(case / 'guide4exec2_adaptive010_ddim4_scale50_m044_mu11_video', 'guided')
    baseline = case / 'guide4exec2_adaptive010_ddim4_scale50_m044_mu11'
    prior_trace = json.loads((baseline / 'trace.json').read_text())
    assert len(prior_trace) == len(guide['trace'])
    for key in ('command', 'q', 'object_pose', 'vertical_error_deg', 'native_failure'):
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
    assert guide['summary']['prior']['guidance_steps'] == 4
    assert guide['summary']['execution_steps'] == 2
    with np.load(baseline / 'initial_state.npz') as a, np.load(Path(guide['folder']) / 'initial_state.npz') as b:
        assert set(a.files) == set(b.files)
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
    assert guide['summary']['guidance_scale'] == 50
    assert guide['summary']['reference_interpolation_threshold'] == .10
    assert guide['summary']['vertical_scale_switch_angle_deg'] is None
    losses = [json.loads((case / 'direct_m044_mu11' / 'm044_mu11_result.json').read_text())['first_separation'],
              json.loads((baseline / 'm044_mu11_result.json').read_text())['first_separation']]
    return dict(episode=ep, case=case, direct=direct, guide=guide,
                progress=progress, source_actions=source_actions,
                losses=losses)


def main():
    verification = dict(video=str(VIDEO), episodes=[], fps=30,
                        alignment='Original source-action progress; original direct repeats while guided executes inserted midpoints. Simulated physics times differ.',
                        panels=LABELS, object_axes='simulation object-local XYZ: red, green, blue')
    frame_count = 0
    writer = imageio.get_writer(str(VIDEO), fps=30, codec='libx264', quality=7,
                                macro_block_size=16,
                                ffmpeg_params=['-preset', 'fast', '-threads', '2', '-movflags', '+faststart'])
    try:
        for ep in EPISODES:
            data = episode_data(ep)
            raw, guide = data['direct'], data['guide']
            n = data['source_actions']
            ng = len(data['progress'])
            timeline = ([('title', 0)] * 30 + [('import', 0)] * 30 +
                        [('settle', i) for i in range(60)] + [('ready', 59)] * 30 +
                        [('action', i) for i in range(ng)] +
                        [('hold', i) for i in range(60)])
            episode_first = frame_count
            for local_index, (phase, j) in enumerate(timeline):
                canvas = Image.new('RGB', (1280, 1152), (17, 24, 33))
                draw = ImageDraw.Draw(canvas)
                if phase == 'action':
                    source_progress = float(data['progress'][j])
                    header = f'episode {ep:02d} · 原始 reference 进度 {source_progress:g} / {n}'
                elif phase == 'hold':
                    header = f'episode {ep:02d} · 全部动作结束，末尾保持 {j + 1}/60'
                else:
                    header = f'episode {ep:02d} · 自身源帧 {raw["summary"]["source_start_frame"]} 的横抓起点'
                draw.text((12, 1), header, font=FONT, fill='white')
                text = {'title':'下一条 episode · 原始与 DDIM4/scale50 对照',
                        'import':'导入瞬间（定格 1 秒）', 'settle':'仿真静置检查',
                        'ready':'共同动作起点（定格 1 秒）',
                        'action':'按原始动作进度对齐；插值组执行的物理步更多',
                        'hold':'完整动作尾段后保持 2 秒'}[phase]
                draw.text((12, 32), text + ' · 灯泡局部 XYZ：红 / 绿 / 蓝',
                          font=SMALL, fill=(190, 225, 250))
                if phase == 'title':
                    writer.append_data(np.asarray(canvas))
                    frame_count += 1
                    continue
                if phase == 'import':
                    raw_video_index = guide_video_index = 0
                    raw_row = guide_row = None
                elif phase in ('settle', 'ready'):
                    raw_video_index = guide_video_index = 1 + j
                    raw_row = raw['trace'][j]
                    guide_row = guide['trace'][j]
                elif phase == 'action':
                    raw_action = int(np.floor(data['progress'][j])) - 1
                    raw_video_index = 61 + raw_action
                    guide_video_index = 61 + j
                    raw_row = raw['trace'][60 + raw_action]
                    guide_row = guide['trace'][60 + j]
                else:
                    raw_video_index = 61 + n + j
                    guide_video_index = 61 + ng + j
                    raw_row = raw['trace'][60 + n + j]
                    guide_row = guide['trace'][60 + ng + j]
                raw_frame = read_to(raw, raw_video_index)
                guide_frame = read_to(guide, guide_video_index)
                for panel, ((x, y), label) in enumerate(zip(PANELS, LABELS)):
                    run = raw if panel in (0, 2) else guide
                    frame = raw_frame if panel in (0, 2) else guide_frame
                    view_index = 0 if panel < 2 else 1
                    rgb = cv2.cvtColor(frame[:, view_index * 640:(view_index + 1) * 640],
                                       cv2.COLOR_BGR2RGB)
                    canvas.paste(Image.fromarray(rgb), (x, y))
                    row = raw_row if panel in (0, 2) else guide_row
                    if row is not None:
                        draw_axes(canvas, object_pose(row), run['projectors'][view_index], (x, y))
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((x, y, x + 639, y + 63), fill=(25, 36, 49))
                    draw.text((x + 8, y), label, font=FONT, fill='white')
                    if row is None:
                        detail = '真实导入位姿；下一段为 60 步静置'
                    else:
                        loss = data['losses'][0 if panel in (0, 2) else 1]
                        separated = bool(loss and
                            (phase == 'hold' or
                             (phase == 'action' and row['index'] >= loss['zero_based_control_index'])))
                        detail = (f'{phase} {row["index"] + 1} | 竖直角 {row["vertical_error_deg"]:.1f}° | '
                                  f'几何分离标记 {"已到" if separated else "未到"}')
                    draw.text((x + 8, y + 35), detail, font=SMALL,
                              fill=(255, 150, 130) if '已到' in detail else (188, 231, 208))
                if phase == 'action' and j in (0, ng // 2, ng - 1):
                    canvas.save(OUT / f'episode_{ep:02d}_action_{j:04d}.jpg', quality=90)
                writer.append_data(np.asarray(canvas))
                frame_count += 1
            raw['cap'].release()
            guide['cap'].release()
            verification['episodes'].append(dict(episode=ep, source_action_count=n,
                expanded_action_count=ng, video_frames=len(timeline),
                start_frame=episode_first, end_frame=frame_count - 1,
                guide_trace_exactly_matches_no_video=True))
            print('COMPOSED', ep, len(timeline), 'frames', flush=True)
    finally:
        writer.close()
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
    assert decoded == frame_count and abs(fps - 30) < 1e-6
    verification.update(frames=decoded, duration_s=decoded / 30,
                        codec='H.264', fully_decoded=True)
    (OUT / 'video_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print('VIDEO', VIDEO, decoded, decoded / 30, flush=True)


if __name__ == '__main__':
    main()
