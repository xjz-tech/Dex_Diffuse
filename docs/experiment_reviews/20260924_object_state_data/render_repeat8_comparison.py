"""Compare an eight-times-held recorded reference with direct playback."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = Path(__file__).resolve().parent
OUT = P / 'corrected_direct_vs_reference/repeat8_guide8'
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 19)
HOLD_STEPS = 60


def main(out=OUT, source_repeats=8):
    action_steps = 75 * source_repeats
    variant = 'repeat8' if source_repeats == 8 else 'tile8'
    captures = {}
    traces = {}
    for mode in ('direct', 'guided'):
        traces[mode] = json.loads((out / mode / 'trace.json').read_text())
        captures[mode] = cv2.VideoCapture(str(out / mode / f'{mode}_two_views.mp4'))
        assert captures[mode].isOpened()
    # Cache source images: each real image is shown for eight 30 Hz video frames.
    source = {
        view: [Image.open(SOURCE / view / f'{frame:06d}.png').convert('RGB')
               for frame in range(91, 166)]
        for view in ('front', 'wrist')
    }
    video = out / f'episode53_{variant}_direct_vs_guide8_{(action_steps+HOLD_STEPS)//30}s.mp4'
    stills = ({0: 'start', 479: 'sixteen_seconds', 539: 'eighteen_seconds',
               570: 'nineteen_seconds', 599: 'action_end', 659: 'hold_end'}
              if source_repeats == 8 else
              {0: 'start', 119: 'four_seconds', 149: 'action_end', 209: 'hold_end'})
    with imageio.get_writer(str(video), fps=30, codec='libx264', quality=7,
                            macro_block_size=16, ffmpeg_log_level='error') as writer:
        for k in range(action_steps + HOLD_STEPS):
            is_hold = k >= action_steps
            source_index = min(k // source_repeats, 74)
            source_frame = 91 + source_index
            canvas = Image.new('RGB', (1280, 1152), (14, 19, 25))
            draw = ImageDraw.Draw(canvas)
            title = ('每个记录关节目标连续执行8步' if source_repeats == 8
                     else '每次将当前记录目标复制为8个guide点，执行2步')
            draw.text((12, 0), f'episode53：{title}',
                      font=FONT, fill='white')
            phase = ('仿真保持末关节目标；真机末帧定格' if is_hold
                     else f'真机每帧显示{source_repeats}个控制步')
            draw.text((12, 33), f'{k / 30:.2f}–{(k+1) / 30:.2f}s  |  {phase}',
                      font=SMALL, fill=(190, 225, 245))
            panels = [('front', 0, 64), ('wrist', 640, 64),
                      ('direct', 0, 608), ('guided', 640, 608)]
            for name, x, y in panels:
                draw.rectangle((x, y, x + 639, y + 63), fill=(21, 30, 40))
                if name in source:
                    image = source[name][source_index]
                    label = '真机原视频：前视角' if name == 'front' else '真机原视频：腕部视角'
                    detail = f'源帧 {source_frame} · ' + ('定格' if is_hold else f'每帧重复{source_repeats}次')
                else:
                    ok, raw = captures[name].read()
                    assert ok and raw.shape == (544, 1280, 3)
                    image = Image.fromarray(cv2.cvtColor(raw[64:, 640:], cv2.COLOR_BGR2RGB))
                    rec = traces[name][60 + k]
                    assert rec['phase'] == ('hold' if is_hold else 'action')
                    label = (f'仿真：记录关节目标直接执行，每目标{source_repeats}步'
                             if name == 'direct' else '仿真：10B prior + 当前目标复制guide8')
                    detail = (f'距世界竖直 {rec["vertical_error_deg"]:.1f}° · '
                              f'相对导入偏移 {rec["displacement_from_import_m"]*100:.1f}cm · '
                              f'failure {rec["native_failure"]}')
                draw.text((x+12, y+1), label, font=FONT, fill='white')
                draw.text((x+12, y+34), detail, font=SMALL, fill=(190, 225, 245))
                canvas.paste(image, (x, y+64))
            draw.line((640, 64, 640, 1151), fill=(110, 130, 145), width=2)
            draw.line((0, 608, 1279, 608), fill=(110, 130, 145), width=2)
            writer.append_data(np.asarray(canvas))
            if k in stills:
                canvas.save(out / f'{variant}_{stills[k]}.png')
    for cap in captures.values():
        ok, _ = cap.read()
        assert not ok
        cap.release()
    cap = cv2.VideoCapture(str(video))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (1152, 1280, 3)
        count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert count == action_steps + HOLD_STEPS and fps == 30
    info = dict(video=str(video), frames=count, fps=fps, duration_s=count/fps,
                source_frames=[91, 165], source_repeats=source_repeats,
                action_steps=action_steps, hold_steps=HOLD_STEPS,
                real_image_frozen_during_hold=True)
    (out / 'video_verification.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
