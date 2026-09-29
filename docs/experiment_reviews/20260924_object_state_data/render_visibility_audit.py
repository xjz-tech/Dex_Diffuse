"""Show the omitted initialization/settle segment and where the bulb goes."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'visibility_audit_20260926'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)


def main():
    output = ROOT / 'four_episodes_import_settle_wide.mp4'
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 544))
    assert writer.isOpened()
    total = 0
    for episode in (0, 2, 28, 51):
        folder = ROOT / f'episode_{episode:02d}'
        summary = json.loads((folder / 'summary.json').read_text())
        trace = json.loads((folder / 'trace.json').read_text())
        original = BASE / f'corrected_direct_vs_reference/all_full_episodes/episode_{episode:02d}/full/direct'
        assert trace == json.loads((original / 'trace.json').read_text())
        source_start = summary['source_start_frame']
        actions = summary['steps']['action']
        cap = cv2.VideoCapture(str(folder / 'direct_two_views.mp4'))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(trace) + 1
        for index in range(len(trace) + 1):
            ok, frame = cap.read()
            assert ok
            canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, 1279, 63), fill=(20, 28, 39))
            if index == 0:
                phase = '导入后，尚未执行物理步（此画面停留 1 秒）'
                detail = f'真机源帧 {source_start} | 原始动作直接仿真'
                repeats = 30
            else:
                row = trace[index - 1]
                t = index / 30
                phase_name = {'settle': '静置：reference 尚未开始',
                              'action': '执行 reference', 'hold': '末端保持'}[row['phase']]
                phase = f'{phase_name} | 导入后 {t:.2f} s'
                source_frame = source_start + min(row['index'], actions - 1) if row['phase'] == 'action' else source_start
                detail = f'物体离导入位置 {row["displacement_from_import_m"]*100:.1f} cm | failure {row["native_failure"]}'
                if row['phase'] == 'action':
                    detail += f' | 源动作帧 {source_frame}'
                repeats = 1
            draw.text((9, 0), f'episode {episode:02d} · 原近景', font=FONT, fill='white')
            draw.text((649, 0), '全景：显示手、灯泡和仿真桌面', font=FONT, fill='white')
            draw.text((9, 34), phase, font=SMALL, fill=(255, 219, 161))
            draw.text((649, 34), detail, font=SMALL, fill=(220, 236, 247))
            encoded = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            for _ in range(repeats):
                writer.write(encoded)
                total += 1
        cap.release()
    writer.release()
    # One image makes the omitted segment and the camera crop explicit.
    ep0 = ROOT / 'episode_00'
    imported = Image.open(ep0 / 'direct_import000.jpg').convert('RGB')
    settled = Image.open(ep0 / 'direct_settle059.jpg').convert('RGB')
    real = Image.open('/home/carus/Data/Object_state_data/episode_0/front/000076.png').convert('RGB')
    panels = [(real, '真机源帧 76：灯泡靠近真实桌面'),
              (imported.crop((640, 64, 1280, 544)), '仿真导入瞬间：灯泡已放入，桌面在下方'),
              (settled.crop((0, 64, 640, 544)), '静置 2 秒后：旧录像近景只剩手'),
              (settled.crop((640, 64, 1280, 544)), '同一时刻全景：灯泡已经落在仿真桌上')]
    canvas = Image.new('RGB', (1280, 1088), (20, 28, 39))
    draw = ImageDraw.Draw(canvas)
    for i, (picture, title) in enumerate(panels):
        x, y = (i % 2)*640, (i // 2)*544
        canvas.paste(picture, (x, y+64))
        draw.text((x+9, y+16), title, font=FONT, fill='white')
    canvas.save(ROOT / 'episode00_initialization_proof.png')
    (ROOT / 'render_verification.json').write_text(json.dumps(dict(
        episodes=[0, 2, 28, 51], methods=['direct'], frames=total, fps=30,
        duration_s=total/30, initial_frame_display_repetitions=30,
        entire_physics_traces_equal_to_original=True,
        left='Original close camera', right='New fixed wide camera'), indent=2))
    print(output, total, total/30)


if __name__ == '__main__':
    main()
