"""Replace the wide simulated view with the source dataset's front camera."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import render_four_adaptive010_base as base
from render_full_episode53_axes import draw_axes, object_pose


DATA = Path('/home/carus/Data/Object_state_data')
VIDEO = base.OUT / 'four_episodes_adaptive010_raw_guide_real.mp4'
LABELS = ('原速 reference · 仿真正面', '>0.1 插值 · DDIM4 / scale25',
          '原数据集 front · 真机')


def main():
    verification = dict(
        video=str(VIDEO), episodes=[], fps=30, panels=LABELS,
        alignment=('Dataset source_state_frame_indices follow the original reference '
                   'action progress; integer action p displays recorded state p, '
                   'inserted midpoint uses preceding recorded frame. Simulation '
                   'panels show post-control states and therefore need not be '
                   'pixel-time-synchronous with dataset camera exposure.'),
        object_axes='simulation object-local XYZ: red, green, blue',
    )
    writer = imageio.get_writer(str(VIDEO), fps=30, codec='libx264', quality=7,
                                macro_block_size=16,
                                ffmpeg_params=['-preset', 'fast', '-threads', '2',
                                               '-movflags', '+faststart'])
    total_frames = 0
    try:
        for ep in base.EPISODES:
            data = base.episode_data(ep)
            raw, guide = data['direct'], data['guide']
            with np.load(data['case'] / 'reference_full.npz') as ref:
                source_ids = ref['source_state_frame_indices'][0].astype(int)
            assert len(source_ids) == data['source_actions'] + 1
            assert source_ids[0] == raw['summary']['source_start_frame']
            source_dir = DATA / f'episode_{ep}' / 'front'
            assert source_dir.is_dir()
            assert all((source_dir / f'{int(i):06d}.png').is_file()
                       for i in source_ids)
            last_source_id = None
            source_image = None
            n, ng = data['source_actions'], len(data['progress'])
            timeline = ([('title', 0)] * 30 + [('import', 0)] * 30 +
                        [('settle', i) for i in range(60)] + [('ready', 59)] * 30 +
                        [('action', i) for i in range(ng)] +
                        [('hold', i) for i in range(60)])
            first_frame = total_frames
            for phase, j in timeline:
                canvas = Image.new('RGB', (1920, 608), (18, 25, 35))
                draw = ImageDraw.Draw(canvas)
                if phase == 'action':
                    progress = float(data['progress'][j])
                    source_state_index = min(int(np.floor(progress)), n)
                    raw_action = max(0, source_state_index - 1)
                    raw_video_index = 61 + raw_action
                    guide_video_index = 61 + j
                    raw_row = raw['trace'][60 + raw_action]
                    guide_row = guide['trace'][60 + j]
                    header = f'episode {ep:02d} · 原始 reference 进度 {progress:g} / {n}'
                elif phase == 'hold':
                    source_state_index = n
                    raw_video_index = 61 + n + j
                    guide_video_index = 61 + ng + j
                    raw_row = raw['trace'][60 + n + j]
                    guide_row = guide['trace'][60 + ng + j]
                    header = f'episode {ep:02d} · 动作结束，末尾保持 {j + 1}/60'
                else:
                    source_state_index = 0
                    if phase == 'import' or phase == 'title':
                        raw_video_index = guide_video_index = 0
                        raw_row = guide_row = None
                    else:
                        raw_video_index = guide_video_index = 1 + j
                        raw_row = raw['trace'][j]
                        guide_row = guide['trace'][j]
                    header = f'episode {ep:02d} · 自身源帧 {source_ids[0]} 的横抓起点'
                source_id = int(source_ids[source_state_index])
                subtitle = {
                    'title': '下一条 episode · 原始 reference / DDIM4 scale25 / 原数据集 front',
                    'import': '物理运行前 · 按导入状态FK重建（定格1秒）',
                    'settle': '仿真静置检查；真机停留在起点帧',
                    'ready': '共同动作起点（定格 1 秒）',
                    'action': '按原始动作进度对齐；插值组实际控制步数更多',
                    'hold': '完整动作尾段后保持 2 秒；真机停留在尾帧',
                }[phase]
                draw.text((12, 0), header, font=base.FONT, fill='white')
                draw.text((12, 31), subtitle, font=base.SMALL,
                          fill=(190, 225, 250))
                if phase == 'title':
                    writer.append_data(np.asarray(canvas))
                    total_frames += 1
                    continue

                raw_frame = base.read_to(raw, raw_video_index)
                guide_frame = base.read_to(guide, guide_video_index)
                if source_id != last_source_id:
                    source_image = Image.open(source_dir / f'{source_id:06d}.png').convert('RGB')
                    assert source_image.size == (640, 480)
                    last_source_id = source_id
                assert source_image is not None
                for panel, run, frame, row in (
                    (0, raw, raw_frame, raw_row),
                    (1, guide, guide_frame, guide_row),
                ):
                    x = panel * 640
                    rgb = cv2.cvtColor(frame[:, :640], cv2.COLOR_BGR2RGB)
                    if phase == 'import':
                        zero = Image.open(base.OUT / f'episode{ep}_zero_physics.png').convert('RGB')
                        rgb = np.vstack([np.zeros((64,640,3),dtype=np.uint8),np.asarray(zero)])
                    canvas.paste(Image.fromarray(rgb), (x, 64))
                    if row is not None:
                        draw_axes(canvas, object_pose(row), run['projectors'][0], (x, 64))
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((x, 64, x + 639, 127), fill=(25, 36, 49))
                    draw.text((x + 8, 64), LABELS[panel], font=base.FONT, fill='white')
                    if row is None:
                        detail = '导入位姿；下一段为 60 步静置'
                    else:
                        loss = data['losses'][panel]
                        separated = bool(loss and (
                            phase == 'hold' or
                            (phase == 'action' and row['index'] >=
                             loss['zero_based_control_index'])))
                        detail = (f'{phase} {row["index"] + 1} | 竖直角 '
                                  f'{row["vertical_error_deg"]:.1f}° | '
                                  f'几何分离标记 {"已到" if separated else "未到"}')
                    draw.text((x + 8, 99), detail, font=base.SMALL,
                              fill=(255, 150, 130) if '已到' in detail
                              else (188, 231, 208))

                canvas.paste(source_image, (1280, 128))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((1280, 64, 1919, 127), fill=(25, 36, 49))
                draw.text((1288, 64), LABELS[2], font=base.FONT, fill='white')
                draw.text((1288, 99),
                          f'源帧 {source_id} | 原始动作进度 '
                          f'{0 if phase not in ("action", "hold") else (progress if phase == "action" else n):g}',
                          font=base.SMALL, fill=(188, 231, 208))
                if phase == 'action' and j in (0, ng // 2, ng - 1):
                    canvas.save(base.OUT / f'episode_{ep:02d}_real_action_{j:04d}.jpg',
                                quality=90)
                writer.append_data(np.asarray(canvas))
                total_frames += 1
            raw['cap'].release()
            guide['cap'].release()
            verification['episodes'].append(dict(
                episode=ep, source_start_frame=int(source_ids[0]),
                source_end_frame=int(source_ids[-1]),
                source_action_count=n, expanded_action_count=ng,
                video_frames=len(timeline), start_frame=first_frame,
                end_frame=total_frames - 1,
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
        assert frame.shape == (608, 1920, 3)
        decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert decoded == total_frames and abs(fps - 30) < 1e-6
    verification.update(frames=decoded, duration_s=decoded / 30,
                        codec='H.264', fully_decoded=True)
    (base.OUT / 'real_front_video_verification.json').write_text(
        json.dumps(verification, indent=2) + '\n')
    print('VIDEO', VIDEO, decoded, decoded / 30, flush=True)


if __name__ == '__main__':
    main()
