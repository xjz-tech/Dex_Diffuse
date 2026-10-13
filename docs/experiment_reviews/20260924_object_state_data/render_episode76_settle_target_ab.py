"""Front-only comparison of fixed measured-q and recorded-command targets."""

import json

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from render_full_episode53_axes import camera_projector, draw_axes, object_pose
from run_episode76_settle_target_ab import OUT


VIDEO = OUT / 'episode76_fixed_q_vs_fixed_recorded_cmd_front.mp4'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)


def load_run(mode):
    folder = OUT / (mode + '_video')
    summary = json.loads((folder / 'summary.json').read_text())
    trace = json.loads((folder / 'trace.json').read_text())
    baseline = json.loads((OUT / mode / 'trace.json').read_text())
    assert len(trace) == len(baseline) == 60
    assert summary['static_only'] and summary['video_recorded']
    assert summary['mass_kg'] == .044 and summary['friction'] == 1.1
    assert summary['settle_target_source'] == mode
    keys = ('q', 'qd', 'command', 'executed_target', 'object_pose', 'native_failure')
    assert all(a[k] == b[k] for a, b in zip(trace, baseline) for k in keys)
    with np.load(folder / 'initial_state.npz') as a, np.load(OUT / mode / 'initial_state.npz') as b:
        assert all(np.array_equal(a[k], b[k]) for k in a.files)
        q0 = a['q'][0].copy()
    cap = cv2.VideoCapture(str(folder / 'direct_front.mp4'))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 61
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (544, 1280, 3)
        frames.append(cv2.cvtColor(frame[:, :640], cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(frames) == 61
    # The pre-physics import preview may still contain stale body transforms.
    # Show only actual post-physics frames, explicitly labeled from step 1.
    meta = summary['camera_metadata'][0]
    projector = camera_projector(np.asarray(meta['eye']), np.asarray(meta['target']),
                                 meta['horizontal_fov'])
    return dict(trace=trace, frames=frames[1:], q0=q0, projector=projector)


def main():
    runs = [load_run(mode) for mode in ('qpos', 'reference_action')]
    source = Image.open('/home/carus/Data/Object_state_data/episode_76/front/000114.png').convert('RGB')
    assert source.size == (640, 480)
    labels = ['A · 固定目标 = 初始实测关节角', 'B · 固定目标 = 第114帧原始cmd',
              '真机 front · 第114帧初态定格']
    timeline = ([('first', 0)] * 60 +
                [('settle', j) for j in range(60) for _ in range(3)] +
                [('last', 59)] * 60)
    writer = imageio.get_writer(str(VIDEO), fps=30, codec='libx264', quality=8,
                                macro_block_size=16,
                                ffmpeg_params=['-preset', 'fast', '-threads', '2',
                                               '-movflags', '+faststart'])
    try:
        for video_index, (phase, j) in enumerate(timeline):
            canvas = Image.new('RGB', (1920, 608), (18, 25, 35))
            draw = ImageDraw.Draw(canvas)
            draw.text((12, 0), 'episode 76 · 固定目标静置 A/B · 44 g / 摩擦 1.1 · 无 prior',
                      font=FONT, fill='white')
            caption = {'first': '首个实际物理步定格；两组均从相同实测关节和灯泡位姿导入',
                       'settle': f'静置第 {j + 1}/60 控制步 · 仿真时间 {(j + 1)/30:.2f} s · 放慢3倍',
                       'last': '第60步末态定格；真机画面仅用于源初态参照'}[phase]
            draw.text((12, 33), caption, font=SMALL, fill=(195, 225, 245))
            for k, run in enumerate(runs):
                x = k * 640
                row = run['trace'][j]
                canvas.paste(Image.fromarray(run['frames'][j]), (x, 64))
                draw_axes(canvas, object_pose(row), run['projector'], (x, 64))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((x, 64, x + 639, 127), fill=(25, 36, 49))
                draw.text((x + 8, 64), labels[k], font=FONT, fill='white')
                detail = (f'位移 {row["displacement_from_import_m"]*1000:.2f} mm | '
                          f'旋转 {row["rotation_from_import_deg"]:.2f}° | '
                          f'q偏差RMSE {np.sqrt(np.mean((np.array(row["q"])-run["q0"])**2)):.3f} rad')
                draw.text((x + 8, 101), detail, font=SMALL, fill=(195, 230, 210))
            canvas.paste(source, (1280, 128))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((1280, 64, 1919, 127), fill=(25, 36, 49))
            draw.text((1288, 64), labels[2], font=FONT, fill='white')
            draw.text((1288, 101), '源图保持不动；未将动态真机动作当作静置对照',
                      font=SMALL, fill=(195, 230, 210))
            writer.append_data(np.asarray(canvas))
            if video_index in (0, 150, len(timeline)-1):
                canvas.save(OUT / f'video_frame_{video_index:03d}.jpg', quality=92)
    finally:
        writer.close()
    cap = cv2.VideoCapture(str(VIDEO))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (608, 1920, 3)
        decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert decoded == len(timeline) and fps == 30
    record = dict(video=str(VIDEO), frames=decoded, duration_s=decoded/fps, fps=fps,
                  fully_decoded=True, video_rollouts_exactly_match_no_video=True,
                  panel_labels=labels, source_frame=114, source_frame_frozen=True,
                  static_physics_steps=60, playback_slowdown=3,
                  import_preview_omitted='Potential stale body rendering before the first physics step; video starts with the verified first physics step.')
    (OUT/'video_verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(VIDEO, decoded, 'frames', flush=True)


if __name__ == '__main__':
    main()
