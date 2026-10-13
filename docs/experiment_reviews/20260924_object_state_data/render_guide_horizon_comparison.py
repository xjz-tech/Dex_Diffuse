"""Four-way real/direct/guide2/guide9 video for the corrected episode-53 case."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
OUT = P / 'corrected_direct_vs_reference'
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53/front')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 19)


def main(comparison='horizon'):
    if comparison == 'horizon':
        names = ('direct', 'guided', 'guided9')
        labels = {'direct': '记录动作直接执行', 'guided': '10B prior + guide2 / exec2',
                  'guided9': '10B prior + guide9 / exec2'}
        title = 'episode53 真机 / 直接执行 / guide2-exec2 / guide9-exec2'
        stem = 'guide_horizon'
        path = OUT / 'episode53_direct_guide2_guide9_4p5s.mp4'
    elif comparison == 'execution':
        names = ('direct', 'guided9', 'guided9_exec4')
        labels = {'direct': '记录动作直接执行', 'guided9': '10B prior + guide9 / exec2',
                  'guided9_exec4': '10B prior + guide9 / exec4'}
        title = 'episode53 真机 / 直接执行 / guide9-exec2 / guide9-exec4'
        stem = 'execution_steps'
        path = OUT / 'episode53_direct_guide9_exec2_exec4_4p5s.mp4'
    else:
        raise ValueError(comparison)
    caps = {}
    traces = {}
    for name in names:
        traces[name] = json.loads((OUT / name / 'trace.json').read_text())
        video_stem = 'direct' if name == 'direct' else 'guided'
        caps[name] = cv2.VideoCapture(str(OUT / name / f'{video_stem}_two_views.mp4'))
        assert caps[name].isOpened()
    shots = {0: 'start', 74: 'action_end', 134: 'plus_two_seconds'}
    with imageio.get_writer(str(path), fps=30, codec='libx264', quality=8,
                            macro_block_size=16, ffmpeg_log_level='error') as video:
        for frame in range(135):
            real_frame = frame + 91
            canvas = Image.new('RGB', (1280, 1152), (14, 19, 25))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 0), title,
                      font=FONT, fill='white')
            segment = '动作' if frame < 75 else '增加的2秒：真机继续，仿真保持末关节目标'
            draw.text((12, 33), f'{frame / 30:.2f}–{(frame + 1) / 30:.2f}s  |  {segment}',
                      font=SMALL, fill=(190, 225, 245))
            panels = [('real', 0, 64), ('direct', 640, 64),
                      (names[1], 0, 608), (names[2], 640, 608)]
            for name, x, y in panels:
                draw.rectangle((x, y, x + 639, y + 63), fill=(21, 30, 40))
                if name == 'real':
                    image = Image.open(SOURCE / f'{real_frame:06d}.png').convert('RGB')
                    detail = f'真实前视角 · 源帧 {real_frame}'
                    label = '真机原视频'
                else:
                    ok, raw = caps[name].read()
                    assert ok and raw.shape == (544, 1280, 3)
                    image = Image.fromarray(cv2.cvtColor(raw[64:, 640:], cv2.COLOR_BGR2RGB))
                    rec = traces[name][60 + frame]
                    assert rec['phase'] == ('action' if frame < 75 else 'hold')
                    detail = (f'距世界竖直 {rec["vertical_error_deg"]:.1f}° · '
                              f'相对初态偏移 {rec["displacement_from_import_m"] * 100:.1f}cm · '
                              f'failure {rec["native_failure"]}')
                    label = labels[name]
                draw.text((x + 12, y + 1), label, font=FONT, fill='white')
                draw.text((x + 12, y + 34), detail, font=SMALL, fill=(190, 225, 245))
                canvas.paste(image, (x, y + 64))
            draw.line((640, 64, 640, 1151), fill=(110, 130, 145), width=2)
            draw.line((0, 608, 1279, 608), fill=(110, 130, 145), width=2)
            video.append_data(np.asarray(canvas))
            if frame in shots:
                canvas.save(OUT / f'{stem}_{shots[frame]}.png')
    for cap in caps.values():
        ok, _ = cap.read()
        assert not ok
        cap.release()
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (1152, 1280, 3)
        count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert count == 135 and fps == 30
    result = dict(video=str(path), frames=count, fps=fps, duration_s=count / fps,
                  panels=['real_front', *names],
                  source_frames=[91, 225], action_steps=75, hold_steps=60)
    (OUT / f'{stem}_video_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
