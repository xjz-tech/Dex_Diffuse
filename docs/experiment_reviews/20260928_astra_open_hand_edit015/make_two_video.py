"""Reference hand, direct execution, and edit. No noise ratio and no opening percentage."""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/Astra-controller')
BASE = ROOT / 'outputs/astra_open_hand'
PLAN = json.loads((ROOT / 'docs/astra_controller/20260922_open_hand/plan_spec.json').read_text())
REF_FRAMES = BASE / 'comparison' / 'reference_hand_frames'
RUNS = {
    'direct': BASE / 'seed42_direct_open',
    'edit': BASE / 'seed42_edit_through_open',
}
LABELS = ['direct', 'edit']
TITLES = ['参考', '直接执行', 'edit']


def opening(q, q0, mask):
    q = np.asarray(q, dtype=np.float64)
    return float(np.dot((q0 - q)[mask], q0[mask]) / np.dot(q0[mask], q0[mask]) * 100)


def curve(label):
    spec_q0 = np.asarray(PLAN['initial_target'], dtype=np.float64)
    names = PLAN['joint_names']
    mask = np.array([n.endswith(('_FE', '_PIP', '_DIP', '_IP')) or n == 'right_pinky_CMC' for n in names])
    run = RUNS[label]
    manifest = json.loads((run / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (run / 'trajectory.jsonl').read_text().splitlines()]
    series = [dict(step=0, actual_open_pct=0., failure=False)]
    for row in rows:
        series.append(dict(step=row['step'],
                           actual_open_pct=opening(row['state']['qpos'], spec_q0, mask),
                           failure=row['failure']))
    assert series[-1]['failure']
    return series


def main():
    curves = {key: curve(key) for key in LABELS}
    ends = [curves[key][-1]['step'] for key in LABELS]
    video = BASE / 'comparison' / 'open_hand_reference_direct_edit.mp4'
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    large = ImageFont.truetype(font, 30, index=2)
    small = ImageFont.truetype(font, 22, index=2)
    fps, hold = 30, 15
    size = (1920, 680)
    ref_end = 128
    encoder = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    try:
        for step in range(ref_end + 1 + hold):
            canvas = Image.new('RGB', size, '#101722')
            draw = ImageDraw.Draw(canvas)
            panels = []
            ref_step = min(step, ref_end)
            with Image.open(REF_FRAMES / f'{ref_step:06d}.png') as frame:
                panels.append(frame.convert('RGB').resize((600, 450), Image.Resampling.LANCZOS))
            statuses = ['指令中的手势']
            colors = ['#b7c8df']
            for key, end in zip(LABELS, ends):
                shown = min(step, end)
                row = curves[key][shown]
                with Image.open(RUNS[key] / 'frames' / f'{shown:06d}.png') as frame:
                    panels.append(frame.convert('RGB').resize((600, 450), Image.Resampling.LANCZOS))
                if row['failure']:
                    statuses.append('已脱手')
                    colors.append('#ff998b')
                else:
                    statuses.append('仍握着')
                    colors.append('#b6e6cb')
            for i, frame in enumerate(panels):
                x = i * 640
                draw.text((x + 18, 14), TITLES[i], font=large, fill='white')
                canvas.paste(frame, (x + 16, 58))
                if i > 0:
                    draw.rectangle((x + 16, 58, x + 615, 100), fill='#202936')
                draw.text((x + 18, 520), statuses[i], font=small, fill=colors[i])
            raw = canvas.tobytes()
            encoder.stdin.write(raw)
            encoder.stdin.write(raw)
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('encoding failed')
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert frames == 2 * (ref_end + 1 + hold)
    print(video)
    print(frames, round(seconds, 2))


if __name__ == '__main__':
    main()
