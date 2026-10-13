"""Three-arm open-hand video: direct, scale50 guide4, edit 0.15 DDIM4."""
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
RUNS = {
    'direct': BASE / 'seed42_direct',
    'guided50_g4': BASE / 'seed42_guided50_guide4',
    'edit015': BASE / 'seed42_edit015_ddim4',
}
LABELS = ['direct', 'guided50_g4', 'edit015']
TITLES = ['直接执行', 'guidance scale 50 · guide 4', 'edit 0.15 · DDIM4']
SUBS = ['关节目标原样执行', 'exec2 / DDIM4', '噪声比 0.15 / exec2 / 无梯度 guidance']


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
    series = [dict(step=0, seconds=0., actual_open_pct=0., failure=False)]
    for row in rows:
        series.append(dict(step=row['step'], seconds=row['step'] * manifest['control_dt'],
                           actual_open_pct=opening(row['state']['qpos'], spec_q0, mask),
                           failure=row['failure']))
    assert series[-1]['failure']
    return series


def main():
    curves = {key: curve(key) for key in LABELS}
    ends = [curves[key][-1]['step'] for key in LABELS]
    out = BASE / 'comparison'
    out.mkdir(exist_ok=True)
    video = out / 'open_hand_direct_scale50_guide4_edit015.mp4'
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    large = ImageFont.truetype(font, 28, index=2)
    small = ImageFont.truetype(font, 22, index=2)
    fps, hold = 30, 60
    size = (1920, 700)
    slots = [0, 640, 1280]
    encoder = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    try:
        for step in range(max(ends) + 1 + hold):
            canvas = Image.new('RGB', size, '#101722')
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 10), '同初态、同一份渐开参考 | seed42 | 1倍仿真时间 | 失败后面板冻结', font=large, fill='white')
            for i, key in enumerate(LABELS):
                x = slots[i]
                actual = min(step, ends[i])
                row = curves[key][actual]
                draw.text((x + 12, y_title := 48), TITLES[i], font=large, fill='white')
                draw.text((x + 12, 82), SUBS[i], font=small, fill='#b7c8df')
                with Image.open(RUNS[key] / 'frames' / f'{actual:06d}.png') as frame:
                    assert frame.size == (960, 720)
                    canvas.paste(frame.convert('RGB').resize((620, 465), Image.Resampling.LANCZOS), (x + 10, 114))
                draw.rectangle((x + 10, 114, x + 629, 158), fill='#202936')
                draw.text((x + 18, 120), f"实际张开 {row['actual_open_pct']:.0f}%", font=small, fill='white')
                status = '运行中'
                if step >= ends[i]:
                    status = '原生失败'
                    if step > ends[i]:
                        status += '；终帧冻结'
                draw.text((x + 12, 590), f"step {actual} | {status}", font=small,
                          fill='#ff998b' if row['failure'] else '#b6e6cb')
            encoder.stdin.write(canvas.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('encoding failed')
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert frames == max(ends) + 1 + hold
    meta = dict(video=str(video), frames=frames, seconds=round(seconds, 2), fps=fps,
                final_actual_open_pct={key: round(curves[key][-1]['actual_open_pct']) for key in LABELS},
                final_steps=dict(zip(LABELS, ends)), terminal_hold_seconds=2,
                playback='1x simulation time; ended arms visibly freeze',
                layout='direct | scale50 guide4 | edit 0.15 DDIM4')
    (HERE / 'three_video.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
