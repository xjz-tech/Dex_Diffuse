"""Focused post-flip real/reference-direct/reference-edit comparison for ep54, 170g/mu2.0."""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from render_full_episode53_axes import camera_projector, draw_axes, object_pose


P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
CASE = R / 'qualified_comparison/episode_54'
RUNS = R / 'm170_mu20_episode54_fourway_video_20260928'
BASE = R / 'm170_mu20_guidance4_vs_edit015_20260928/episode_54'
OUT = R / 'm170_mu20_episode54_postflip_right_turn_20260928'
DATA = Path('/home/carus/Data/Object_state_data/episode_54/front')
VIDEO = OUT / 'episode54_m170_postflip_reference_vs_edit.mp4'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
START = 101  # zero-based action index: displayed action step 102
END = 189    # zero-based action index: displayed action step 190


def read_to(run, index):
    while run['last'] < index:
        ok, frame = run['cap'].read()
        if not ok:
            raise RuntimeError(f"video truncated: {run['folder']} at {index}")
        assert frame.shape == (544, 1280, 3)
        run['frame'] = frame
        run['last'] += 1
    return run['frame']


def load(method):
    folder = RUNS / method
    stem = 'direct_front.mp4' if method == 'direct_interp' else 'guided_front.mp4'
    trace = json.loads((folder/'trace.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    baseline = json.loads((BASE/method/'trace.json').read_text())
    assert len(trace) == len(baseline)
    assert all(a['object_pose'] == b['object_pose'] and a['command'] == b['command']
               for a, b in zip(trace, baseline))
    cap = cv2.VideoCapture(str(folder/stem))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(trace)+1
    meta = summary['camera_metadata'][0]
    projector = camera_projector(np.asarray(meta['eye']), np.asarray(meta['target']),
                                 meta['horizontal_fov'])
    return dict(method=method, folder=str(folder), trace=trace, summary=summary,
                cap=cap, projector=projector, last=-1, frame=None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    direct, edit = load('direct_interp'), load('edit015')
    direct_result = json.loads((BASE/'direct_interp/analysis.json').read_text())
    edit_result = json.loads((BASE/'edit015/analysis.json').read_text())
    geometry = json.loads((BASE/'edit015/grasp_metrics.json').read_text())['frames']
    progress = np.load(BASE/'edit015/reference_progress.npy')
    assert np.array_equal(progress, np.load(BASE/'direct_interp/reference_progress.npy'))
    with np.load(CASE/'reference_full.npz') as z:
        source_ids = z['source_state_frame_indices'][0].astype(int)
        reference_pose = z['object_pose_wrist'][0]
    assert float(progress[START]) == 80 and float(progress[END]) == 145

    start_ref = int(np.floor(progress[START]))
    end_ref = int(np.floor(progress[186]))
    delta = (Rotation.from_matrix(reference_pose[end_ref, :3, :3]) *
             Rotation.from_matrix(reference_pose[start_ref, :3, :3]).inv()).as_rotvec()
    axis_world = Rotation.from_quat(edit['summary']['initial_wrist'][3:]).apply(
        delta/np.linalg.norm(delta))
    orientations = Rotation.from_quat(np.asarray([
        edit['trace'][60+j]['object_pose'][3:] for j in range(START, END+1)]))
    increments = np.degrees((orientations[1:]*orientations[:-1].inv()).as_rotvec() @ axis_world)
    cumulative = np.r_[0., np.cumsum(increments)]

    timeline = [('ready', START)]*30 + [('action', j) for j in range(START, END+1)] + [('end', END)]*30
    writer = imageio.get_writer(str(VIDEO), fps=30, codec='libx264', quality=7,
        macro_block_size=16,
        ffmpeg_params=['-preset', 'fast', '-threads', '2', '-movflags', '+faststart'])
    frames = 0
    try:
        for phase, j in timeline:
            prog = float(progress[j])
            source_index = min(int(np.floor(prog)), len(source_ids)-1)
            source_id = int(source_ids[source_index])
            real = Image.open(DATA/f'{source_id:06d}.png').convert('RGB')
            assert real.size == (640, 480)
            canvas = Image.new('RGB', (1920, 640), (17, 24, 33))
            draw = ImageDraw.Draw(canvas)
            phase_text = {'ready':'稳定翻转起点（定格1秒）', 'action':'同期播放',
                          'end':'首次几何分离（定格1秒）'}[phase]
            draw.text((12, 6),
                      f'Episode 54 · 170g / 摩擦2.0 · {phase_text} · 动作 {j+1} · reference进度 {prog:g}/462',
                      font=FONT, fill='white')
            draw.text((12, 39),
                      '相同插值reference进度对齐 · reference edit输入：最近3步动作 + 未来9步reference',
                      font=SMALL, fill=(190, 225, 250))

            # Real reference panel.
            canvas.paste(real, (0, 160))
            draw.rectangle((0, 96, 639, 159), fill=(25, 36, 49))
            draw.text((8, 96), '真机 reference · 数据集 front', font=FONT, fill='white')
            draw.text((8, 131), f'源帧 {source_id} · 原始进度 {prog:g}',
                      font=SMALL, fill=(188, 231, 208))

            # Simulation panels.
            for x, run, label, result in (
                (640, direct, '同规则插值 direct · 仿真正面', direct_result),
                (1280, edit, 'reference edit · history3 + reference9', edit_result)):
                row = run['trace'][60+j]
                frame = read_to(run, 61+j)
                rgb = cv2.cvtColor(frame[:, :640], cv2.COLOR_BGR2RGB)
                canvas.paste(Image.fromarray(rgb), (x, 96))
                draw_axes(canvas, object_pose(row), run['projector'], (x, 96))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((x, 96, x+639, 159), fill=(25, 36, 49))
                draw.text((x+8, 96), label, font=FONT, fill='white')
                separated = prog >= result['first_separation_reference_progress']
                if run['method'] == 'edit015':
                    geo = geometry[60+j]
                    force = float(np.linalg.norm(row['object_contact_force']))
                    contact = (geo['near_contact_link_count'] >= 2 and
                               geo['mesh_vertex_gap_m'] < .008 and force > .1)
                    angle = cumulative[j-START]
                    detail = f'累计右转 {angle:+.1f}° · 有效接触 {"是" if contact else "否"}'
                else:
                    detail = f'分离进度 {result["first_separation_reference_progress"]:g} · 当前已分离 {"是" if separated else "否"}'
                draw.text((x+8, 131), detail, font=SMALL,
                          fill=(255, 150, 130) if ('否' in detail or separated) else (188, 231, 208))
            writer.append_data(np.asarray(canvas))
            frames += 1
    finally:
        writer.close()
        direct['cap'].release(); edit['cap'].release()

    cap = cv2.VideoCapture(str(VIDEO)); assert cap.isOpened(); decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        assert frame.shape == (640, 1920, 3); decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    assert decoded == frames == len(timeline) and abs(fps-30) < 1e-6
    verification = dict(video=str(VIDEO), frames=decoded, fps=fps,
        duration_s=decoded/fps, fully_decoded=True, episode=54, mass_kg=.17,
        friction=2.0, start_action_step=START+1, end_action_step=END+1,
        start_reference_progress=float(progress[START]),
        end_reference_progress=float(progress[END]),
        panels=['real_reference_front','interpolated_direct','reference_edit'],
        simulation_recordings_match_numeric_rollouts=True,
        alignment='same expanded action index and original-reference progress',
        downloads_written=False)
    (OUT/'video_verification.json').write_text(json.dumps(verification, ensure_ascii=False,
                                                          indent=2)+'\n')
    print('VIDEO', VIDEO, decoded, decoded/fps, flush=True)


if __name__ == '__main__':
    main()
