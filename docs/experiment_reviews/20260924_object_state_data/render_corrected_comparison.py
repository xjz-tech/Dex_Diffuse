"""Render real episode 53, corrected direct replay, and corrected guided replay."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


P = Path(__file__).resolve().parent
OUT = P / 'corrected_direct_vs_reference'
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 19)
FRAMES = 135  # 75 recorded-action steps + 60 held-command steps at 30 Hz


def main():
    traces = {}
    caps = {}
    for mode in ('direct', 'guided'):
        traces[mode] = json.loads((OUT / mode / 'trace.json').read_text())
        caps[mode] = cv2.VideoCapture(str(OUT / mode / f'{mode}_two_views.mp4'))
        assert caps[mode].isOpened()
    video = OUT / 'episode53_real_direct_reference_4p5s.mp4'
    stills = {0: 'start', 74: 'action_end', 134: 'plus_two_seconds'}
    with imageio.get_writer(str(video), fps=30, codec='libx264', quality=8,
                            macro_block_size=16, ffmpeg_log_level='error') as writer:
        for k in range(FRAMES):
            real_frame = 91 + k
            phase = '动作执行' if k < 75 else '动作后 2 秒'
            canvas = Image.new('RGB', (1280, 1152), (14, 19, 25))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 0), 'episode53 真机原视频 / 原始动作直接执行 / 10B prior + reference',
                      font=FONT, fill='white')
            draw.text((12, 33),
                f'{phase}  {k / 30:.2f}–{(k + 1) / 30:.2f}s  |  前2.5秒动作对齐；之后真机继续，仿真保持末关节目标',
                font=SMALL, fill=(190, 225, 245))
            for view, x in (('front', 0), ('wrist', 640)):
                image = Image.open(SOURCE / view / f'{real_frame:06d}.png').convert('RGB')
                assert image.size == (640, 480)
                draw.rectangle((x, 64, x + 639, 127), fill=(21, 30, 40))
                title = '真机原视频：前视角' if view == 'front' else '真机原视频：腕部视角'
                draw.text((x + 12, 65), title, font=FONT, fill='white')
                draw.text((x + 12, 98), f'源帧 {real_frame}  |  真实连续画面',
                          font=SMALL, fill=(190, 225, 245))
                canvas.paste(image, (x, 128))
            for mode, x in (('direct', 0), ('guided', 640)):
                ok, raw = caps[mode].read()
                assert ok and raw.shape == (544, 1280, 3)
                # Both runs recorded the same fixed side-camera location.
                image = Image.fromarray(cv2.cvtColor(raw[64:, 640:], cv2.COLOR_BGR2RGB))
                rec = traces[mode][60 + k]
                assert rec['phase'] == ('action' if k < 75 else 'hold')
                draw.rectangle((x, 608, x + 639, 671), fill=(21, 30, 40))
                title = '仿真：真机原始关节目标直接执行' if mode == 'direct' else '仿真：10B prior + reference 引导'
                draw.text((x + 12, 609), title, font=FONT, fill='white')
                detail = (f'距世界竖直 {rec["vertical_error_deg"]:.1f}°  |  '
                          f'相对初态位移 {rec["displacement_from_import_m"] * 100:.1f}cm  |  '
                          f'failure {rec["native_failure"]}')
                draw.text((x + 12, 642), detail, font=SMALL,
                          fill=(255, 170, 150) if rec['native_failure'] else (190, 225, 245))
                canvas.paste(image, (x, 672))
            draw.line((640, 64, 640, 1151), fill=(110, 130, 145), width=2)
            draw.line((0, 608, 1279, 608), fill=(110, 130, 145), width=2)
            writer.append_data(np.asarray(canvas))
            if k in stills:
                canvas.save(OUT / f'episode53_{stills[k]}.png')
    for cap in caps.values():
        ok, _ = cap.read()
        assert not ok
        cap.release()
    verify = cv2.VideoCapture(str(video))
    count = 0
    while True:
        ok, frame = verify.read()
        if not ok:
            break
        assert frame.shape == (1152, 1280, 3)
        count += 1
    fps = verify.get(cv2.CAP_PROP_FPS)
    verify.release()
    assert count == FRAMES and fps == 30
    result = dict(video=str(video), frames=count, fps=fps, duration_s=count / fps,
        source_frame_first=91, source_frame_last=225,
        simulation='75 recorded-action control steps then 60 steps holding final command',
        real='continuous original frames, including 2 seconds beyond recorded action segment')
    (OUT / 'video_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
