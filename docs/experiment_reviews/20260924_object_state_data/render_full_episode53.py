"""Three front-view simulation rollouts from episode-53 frame 90 to the end."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


P = Path(__file__).resolve().parent / 'corrected_direct_vs_reference/full_episode53'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)
SPECS = [
    ('原始动作直接仿真', P / 'direct_original', 'direct'),
    ('10B prior · guide2 / exec2 · scale50', P / 'insert1_scale50_guide2_exec2', 'guided'),
    ('10B prior · guide2 / exec1 · scale50', P / 'insert1_scale50_guide2_exec1', 'guided'),
]


def main():
    runs = []
    for title, folder, stem in SPECS:
        cap = cv2.VideoCapture(str(folder / f'{stem}_front.mp4'))
        assert cap.isOpened(), folder
        trace = json.loads((folder / 'trace.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        runs.append(dict(title=title, cap=cap, trace=trace, summary=summary,
                         last=-1, frame=None))
    assert [r['summary']['steps']['action'] for r in runs] == [357, 713, 713]
    assert [r['summary']['steps']['hold'] for r in runs] == [60, 60, 60]
    video = P / 'episode53_full_direct_vs_guide2_exec2_exec1_front.mp4'
    writer = imageio.get_writer(str(video), fps=30, codec='libx264', quality=7,
                                macro_block_size=16)
    try:
        for shown in range(713 + 60):
            canvas = Image.new('RGB', (1920, 544), (20, 28, 39))
            draw = ImageDraw.Draw(canvas)
            for k, run in enumerate(runs):
                if shown < 713:
                    index = shown // 2 if k == 0 else shown
                    source_frame = 90 + shown // 2
                    phase = '动作'
                else:
                    index = run['summary']['steps']['action'] + shown - 713
                    source_frame = 447
                    phase = '末态保持'
                while run['last'] < index:
                    ok, frame = run['cap'].read()
                    assert ok, (k, index, run['last'])
                    run['last'] += 1
                    run['frame'] = frame
                front = run['frame'][64:]
                assert front.shape == (480, 640, 3)
                x = k * 640
                canvas.paste(Image.fromarray(cv2.cvtColor(front, cv2.COLOR_BGR2RGB)), (x, 64))
                row = run['trace'][60 + index]
                draw.rectangle((x, 0, x + 639, 63), fill=(25, 36, 49))
                draw.text((x + 8, 0), run['title'], font=FONT, fill='white')
                sim_time = index / 30
                message = (f'{phase} | 源帧 {source_frame} | 仿真 {sim_time:.2f}s | '
                           f'距竖直 {row["vertical_error_deg"]:.1f}° | failure {row["native_failure"]}')
                draw.text((x + 8, 35), message, font=SMALL,
                          fill=(255, 155, 135) if row['native_failure'] else (194, 229, 212))
            writer.append_data(np.asarray(canvas))
            if shown in (148, 318, 518, 712, 772):
                canvas.save(P / f'episode53_full_source{source_frame:03d}.png')
    finally:
        writer.close()
        for run in runs:
            run['cap'].release()
    cap = cv2.VideoCapture(str(video))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (544, 1920, 3)
        count += 1
    cap.release()
    assert count == 773, count
    (P / 'video_verification.json').write_text(json.dumps(dict(
        video=str(video), frames=count, fps=30, duration_s=count / 30,
        panels=[item[0] for item in SPECS],
        alignment='source action progress; direct sim frames repeated twice; guided frames shown once',
        source_actions=[90, 446], source_final_state=447,
        guide_interpolation=1, terminal_hold_sim_steps=60,
    ), indent=2) + '\n')
    print(video)


if __name__ == '__main__':
    main()
