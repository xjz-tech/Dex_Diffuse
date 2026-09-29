"""Synchronize the recorded episode53 cameras with case467's two real rollouts."""

from pathlib import Path
import json

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53')
OUTPUT = ROOT / 'direct_case467'
CASE = 467
FIRST_SOURCE_FRAME = 90
LAST_SOURCE_FRAME = 165
FIRST_TRAJECTORY_FRAME = 60
FRAMES = 105
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)

records = {}
captures = {}
for name, folder in [('direct', OUTPUT), ('guided', ROOT / 'repeat')]:
    archive = np.load(folder / 'trajectory.npz')
    records[name] = {key: archive[key] for key in ['phase', 'index', 'failure', 'vertical_error_deg']}
    cap = cv2.VideoCapture(str(folder / 'case467_clear.mp4'))
    assert cap.isOpened()
    captures[name] = cap

source_frames = {}
for view in ['front', 'wrist']:
    source_frames[view] = [
        Image.open(SOURCE / view / f'{frame:06d}.png').convert('RGB')
        for frame in range(FIRST_SOURCE_FRAME, LAST_SOURCE_FRAME + 1)
    ]
    assert all(image.size == (640, 480) for image in source_frames[view])

fast_path = OUTPUT / 'original_vs_direct_vs_prior.mp4'
slow_path = OUTPUT / 'original_vs_direct_vs_prior_slow.mp4'
with imageio.get_writer(str(fast_path), fps=30, codec='libx264', quality=8) as fast, imageio.get_writer(
    str(slow_path), fps=15, codec='libx264', quality=8
) as slow:
    for video_frame in range(FRAMES):
        trajectory_frame = FIRST_TRAJECTORY_FRAME + video_frame
        phase = str(records['direct']['phase'][trajectory_frame])
        assert phase == str(records['guided']['phase'][trajectory_frame])
        source_frame = min(FIRST_SOURCE_FRAME + video_frame + 1, LAST_SOURCE_FRAME)
        held_source = phase == 'hold'
        assert (source_frame == LAST_SOURCE_FRAME) == (video_frame >= 74)

        canvas = Image.new('RGB', (1280, 1152), (14, 19, 25))
        draw = ImageDraw.Draw(canvas)
        phase_text = '动作执行' if phase == 'reference' else '仿真末态保持'
        draw.text((12, 0), 'episode53 真机原视频 + case467 仿真对比  |  按名义30 Hz对齐', font=FONT, fill='white')
        draw.text(
            (12, 33),
            f'{phase_text} {(video_frame + 1) / 30:.2f} s  |  原生wrist / 170 g / 摩擦2.2  |  绿箭头：灯泡顶部方向',
            font=SMALL,
            fill=(190, 225, 245),
        )

        panels = [
            ('真机原视频：前视角', 'front', 0, 64),
            ('真机原视频：腕部视角', 'wrist', 640, 64),
            ('仿真：原始动作直接回放', 'direct', 0, 608),
            ('仿真：10B prior + 轨迹引导', 'guided', 640, 608),
        ]
        for label, name, x, y in panels:
            draw.rectangle((x, y, x + 639, y + 63), fill=(21, 30, 40))
            draw.text((x + 12, y + 1), label, font=FONT, fill='white')
            if name in source_frames:
                image = source_frames[name][source_frame - FIRST_SOURCE_FRAME]
                detail = f'原始帧 {source_frame}' + ('  |  片段结束，末帧定格' if held_source else '')
            else:
                ok, video_image = captures[name].read()
                assert ok
                assert video_image.shape == (544, 1280, 3)
                image = Image.fromarray(cv2.cvtColor(video_image[64:, 640:], cv2.COLOR_BGR2RGB))
                failed = bool(records[name]['failure'][: trajectory_frame + 1, CASE].any())
                angle = float(records[name]['vertical_error_deg'][trajectory_frame, CASE])
                detail = f'距竖直 {angle:.1f}°  |  原生failure: {failed}'
            draw.text((x + 12, y + 34), detail, font=SMALL, fill=(255, 170, 150) if 'True' in detail else (190, 225, 245))
            canvas.paste(image, (x, y + 64))

        draw.line((640, 64, 640, 1151), fill=(110, 130, 145), width=2)
        draw.line((0, 608, 1279, 608), fill=(110, 130, 145), width=2)
        rgb = np.asarray(canvas)
        fast.append_data(rgb)
        slow.append_data(rgb)
        if trajectory_frame in [60, 97, 124, 134, 164]:
            canvas.save(OUTPUT / f'original_vs_sim_{trajectory_frame:03d}.jpg')

for cap in captures.values():
    ok, _ = cap.read()
    assert not ok
    cap.release()

verification = {
    'source_episode': 53,
    'source_frame_mapping': 'simulation action j -> source state frame 90+j+1 for j=0..74; source frame165 stays visible during simulation hold',
    'source_views': ['original front PNG', 'original wrist PNG'],
    'simulation_views': ['direct case467 side camera', 'guided case467 side camera'],
    'source_frame_first': 91,
    'source_frame_last': 165,
    'source_final_frame_frozen_during_hold': True,
    'video_frames': FRAMES,
    'videos': {},
}
for path, expected_fps in [(fast_path, 30), (slow_path, 15)]:
    cap = cv2.VideoCapture(str(path))
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (1152, 1280, 3)
        count += 1
    cap.release()
    assert count == FRAMES and fps == expected_fps
    verification['videos'][path.name] = {'frames': count, 'fps': fps, 'duration_s': count / fps}

(OUTPUT / 'original_vs_sim_verification.json').write_text(json.dumps(verification, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(verification, indent=2, ensure_ascii=False))
