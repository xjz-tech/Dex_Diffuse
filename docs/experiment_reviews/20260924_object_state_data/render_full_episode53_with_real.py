"""Add the source episode-53 front camera to the full three-simulation comparison."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


P = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/full_episode53'
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53/front')
INPUT = P / 'episode53_full_direct_vs_guide2_exec2_exec1_front.mp4'
OUTPUT = P / 'episode53_full_real_direct_guide2_exec2_exec1_front.mp4'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)


def main():
    cap = cv2.VideoCapture(str(INPUT))
    assert cap.isOpened()
    writer = imageio.get_writer(str(OUTPUT), fps=30, codec='libx264', quality=7,
                                macro_block_size=16)
    last_source_frame = None
    real = None
    try:
        for shown in range(773):
            ok, comparison = cap.read()
            assert ok and comparison.shape == (544, 1920, 3), shown
            source_frame = 90 + shown // 2 if shown < 713 else 447
            if source_frame != last_source_frame:
                real = Image.open(SOURCE / f'{source_frame:06d}.png').convert('RGB')
                assert real.size == (640, 480)
                last_source_frame = source_frame
            canvas = Image.new('RGB', (1280, 1088), (20, 28, 39))
            draw = ImageDraw.Draw(canvas)
            canvas.paste(real, (0, 64))
            draw.rectangle((0, 0, 639, 63), fill=(25, 36, 49))
            draw.text((8, 0), '真机原视频 · episode 53 正面', font=FONT, fill='white')
            phase = '动作' if shown < 713 else '末态定格'
            draw.text((8, 35), f'{phase} | 源帧 {source_frame} | 源进度 {(source_frame-90)/30:.2f}s',
                      font=SMALL, fill=(194, 229, 212))
            for k, xy in enumerate([(640, 0), (0, 544), (640, 544)]):
                panel = comparison[:, k * 640:(k + 1) * 640]
                canvas.paste(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), xy)
            writer.append_data(np.asarray(canvas))
            if shown in (148, 432, 600, 603, 712, 772):
                canvas.save(P / f'episode53_full_real_4panel_{shown:03d}.png')
        assert not cap.read()[0]
    finally:
        writer.close()
        cap.release()
    verify = cv2.VideoCapture(str(OUTPUT))
    count = 0
    while True:
        ok, frame = verify.read()
        if not ok:
            break
        assert frame.shape == (1088, 1280, 3)
        count += 1
    verify.release()
    assert count == 773, count
    (P / 'real_video_verification.json').write_text(json.dumps(dict(
        video=str(OUTPUT), frames=count, fps=30, duration_s=count / 30,
        panels=['real front source', 'direct simulated original actions',
                '10B guide2/exec2 scale50 insert1', '10B guide2/exec1 scale50 insert1'],
        source_actions=[90, 446], final_source_state=447,
        alignment='Real and direct panels repeat each action-progress frame twice; guided panels show every simulation step. Real terminal source frame 447 is frozen during simulation hold.',
    ), indent=2) + '\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
