"""Five-way open-hand comparison: original four arms plus edit 0.15 / DDIM4."""
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
OLD = json.loads((ROOT / 'docs/astra_controller/20260922_open_hand/curves.json').read_text())
RUNS = {
    'direct': BASE / 'seed42_direct',
    'guided25': BASE / 'seed42_guided25',
    'guided50': BASE / 'seed42_guided50',
    'prior_only': BASE / 'seed42_prior_only',
    'edit015': BASE / 'seed42_edit015_ddim4',
    'guided50_g4': BASE / 'seed42_guided50_guide4',
}
LABELS = ['direct', 'guided25', 'guided50', 'prior_only', 'edit015', 'guided50_g4']
TITLES = [
    'Astra 原始张手动作',
    'guidance scale25 · guide9',
    'guidance scale50 · guide9',
    '10B Prior，无 guidance',
    'edit 0.15 DDIM4',
    'guidance scale50 · guide4',
]
SUBS = [
    '直接执行关节目标',
    'exec2 / DDIM4',
    'exec2 / DDIM4',
    'scale0：不使用张手参考',
    '噪声比0.15 / exec2 / 无梯度 guidance',
    'exec2 / DDIM4 / fresh noise',
]


def opening(q, q0, mask):
    q = np.asarray(q)
    return float(np.dot((q0 - q)[mask], q0[mask]) / np.dot(q0[mask], q0[mask]) * 100)


def edit_curve():
    run = RUNS['edit015']
    spec_q0 = np.asarray(PLAN['initial_target'])
    names = PLAN['joint_names']
    mask = np.array([n.endswith(('_FE', '_PIP', '_DIP', '_IP')) or n == 'right_pinky_CMC' for n in names])
    manifest = json.loads((run / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (run / 'trajectory.jsonl').read_text().splitlines()]
    curve = [dict(step=0, seconds=0., reference_open_pct=0., actual_open_pct=0., failure=False)]
    for step, row in enumerate(rows, 1):
        start = ((step - 1) // 8) * 8
        expected = np.asarray(PLAN['plans'][str(start)], dtype=np.float32)[step - 1 - start]
        np.testing.assert_array_equal(row['astra_reference'], expected)
        curve.append(dict(step=step, seconds=step * manifest['control_dt'],
                          reference_open_pct=opening(row['astra_reference'], spec_q0, mask),
                          actual_open_pct=opening(row['state']['qpos'], spec_q0, mask),
                          failure=row['failure']))
    return curve, manifest


def timetable_curve(label):
    run = RUNS[label]
    spec_q0 = np.asarray(PLAN['initial_target'], dtype=np.float64)
    names = PLAN['joint_names']
    mask = np.array([n.endswith(('_FE', '_PIP', '_DIP', '_IP')) or n == 'right_pinky_CMC' for n in names])
    manifest = json.loads((run / 'manifest.json').read_text())
    rows = [json.loads(line) for line in (run / 'trajectory.jsonl').read_text().splitlines()]
    guided9 = [json.loads(line) for line in (RUNS['guided50'] / 'trajectory.jsonl').read_text().splitlines()]
    curve = [dict(step=0, seconds=0., reference_open_pct=0., actual_open_pct=0., failure=False)]
    for row in rows:
        step = row['step']
        alpha = np.clip((step - 8) / 120.0, 0, 1)
        expected = spec_q0 * (1 - alpha)
        np.testing.assert_allclose(row['astra_reference'], expected, atol=1e-5, rtol=0)
        if step <= len(guided9):
            np.testing.assert_allclose(row['astra_reference'], guided9[step - 1]['astra_reference'], atol=1e-5, rtol=0)
        curve.append(dict(step=step, seconds=step * manifest['control_dt'],
                          reference_open_pct=opening(row['astra_reference'], spec_q0, mask),
                          actual_open_pct=opening(row['state']['qpos'], spec_q0, mask),
                          failure=row['failure']))
    return curve, manifest


def main():
    curves = {key: OLD[key] for key in ('direct', 'guided25', 'guided50', 'prior_only')}
    curves['edit015'], manifest = edit_curve()
    curves['guided50_g4'], _ = timetable_curve('guided50_g4')
    model = json.loads((RUNS['edit015'] / 'model.json').read_text())
    editor = model['reference_editor']
    SUBS[4] = (f"实际噪声比 {editor['actual_noise_ratio']:.3f} / 时间步 {editor['timesteps']} / exec2")
    ends = [curves[key][-1]['step'] for key in LABELS]
    out = BASE / 'comparison'
    out.mkdir(exist_ok=True)
    video = out / 'open_hand_with_edit015_and_scale50_guide4.mp4'
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    large = ImageFont.truetype(font, 26, index=2)
    small = ImageFont.truetype(font, 20, index=2)
    fps, hold = 30, 60
    size = (1920, 1240)
    slots = [(0, 56), (640, 56), (1280, 56), (0, 640), (640, 640), (1280, 640)]
    encoder = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    try:
        for step in range(max(ends) + 1 + hold):
            canvas = Image.new('RGB', size, '#101722')
            draw = ImageDraw.Draw(canvas)
            draw.text((18, 12), '同初态、同4秒渐开指令 | seed42 | 1倍仿真时间 | 失败后面板冻结', font=large, fill='white')
            for i, key in enumerate(LABELS):
                x, y = slots[i]
                actual = min(step, ends[i])
                row = curves[key][actual]
                draw.text((x + 12, y + 2), TITLES[i], font=large, fill='white')
                draw.text((x + 12, y + 34), SUBS[i], font=small, fill='#b7c8df')
                with Image.open(RUNS[key] / 'frames' / f'{actual:06d}.png') as frame:
                    assert frame.size == (960, 720)
                    canvas.paste(frame.convert('RGB').resize((620, 465), Image.Resampling.LANCZOS), (x + 10, y + 64))
                draw.rectangle((x + 10, y + 64, x + 629, y + 108), fill='#202936')
                draw.text((x + 18, y + 70),
                          f"关节伸展：实际 {row['actual_open_pct']:.1f}% / 参考 {row['reference_open_pct']:.1f}%",
                          font=small, fill='white')
                status = '运行中'
                if step >= ends[i]:
                    status = '原生失败' if row['failure'] else '观察截止，未失败'
                    if step > ends[i]:
                        status += '；终帧冻结'
                draw.text((x + 12, y + 536), f"step {actual} / {row['seconds']:.2f}s | {status}", font=small,
                          fill='#ff998b' if row['failure'] else '#b6e6cb')
            draw.text((18, 1204), '伸展%是屈伸关节投影，不是任务成功率。edit 使用固定噪声 seed44；旧 guidance 臂仍是当时的 fresh noise。', font=small, fill='#cbd5e3')
            encoder.stdin.write(canvas.tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('encoding failed')
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert frames == max(ends) + 1 + hold
    meta = dict(video=str(video), frames=frames, seconds=seconds, fps=fps,
                final_steps=dict(zip(LABELS, ends)), terminal_hold_seconds=2,
                editor=editor,                 edit_failure=curves['edit015'][-1]['failure'],
                edit_final_step=curves['edit015'][-1]['step'],
                guide4_final_step=curves['guided50_g4'][-1]['step'],
                guide4_failure=curves['guided50_g4'][-1]['failure'],
                control_dt=manifest['control_dt'],
                playback='1x simulation time; ended arms visibly freeze',
                layout='top: direct, scale25 guide9, scale50 guide9; bottom: prior-only, edit 0.15, scale50 guide4')
    (HERE / 'video.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
