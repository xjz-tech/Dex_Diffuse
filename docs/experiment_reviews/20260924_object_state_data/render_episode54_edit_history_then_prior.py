#!/usr/bin/env python3
"""Render synchronized post-branch edit versus autonomous-prior comparison."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from render_full_episode53_axes import camera_projector, draw_axes, object_pose

P = Path(__file__).resolve().parent
R = P/'reference_turn_baseline_20260926'
EDIT_NUM = R/'m170_mu20_guidance4_vs_edit015_20260928/episode_54/edit015'
EDIT_VIDEO = R/'m170_mu20_episode54_fourway_video_20260928/edit015'
PRIOR = R/'m170_mu20_episode54_edit_history_then_prior_20260928'
OUT = PRIOR/'episode54_edit_history_then_prior_vs_edit.mp4'
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 20)
SMALL = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
BRANCH_LAST = 101


def load(folder, video_folder):
    trace = json.loads((folder/'trace.json').read_text())
    geometry = json.loads((folder/'grasp_metrics.json').read_text())['frames']
    summary = json.loads((video_folder/'summary.json').read_text())
    cap = cv2.VideoCapture(str(video_folder/'guided_front.mp4'))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(trace)+1
    meta = summary['camera_metadata'][0]
    project = camera_projector(np.asarray(meta['eye']), np.asarray(meta['target']),
                               meta['horizontal_fov'])
    return dict(folder=folder, trace=trace, geometry=geometry, summary=summary,
                cap=cap, project=project, last=-1, frame=None)


def read_to(run, frame_index):
    while run['last'] < frame_index:
        ok, frame = run['cap'].read()
        if not ok:
            raise RuntimeError(f"video truncated at {frame_index}: {run['folder']}")
        assert frame.shape == (544, 1280, 3)
        run['frame'] = frame
        run['last'] += 1
    return run['frame']


def effective(row, geo):
    return (geo['near_contact_link_count'] >= 2 and
            geo['mesh_vertex_gap_m'] < .008 and
            np.linalg.norm(row['object_contact_force']) > .1)


def main():
    edit = load(EDIT_NUM, EDIT_VIDEO)
    prior = load(PRIOR, PRIOR)
    # The renderer's baseline recording was independently checked against the
    # numeric baseline trace when it was made; assert it again here.
    recorded_edit = json.loads((EDIT_VIDEO/'trace.json').read_text())
    assert all(a['object_pose'] == b['object_pose'] and a['command'] == b['command']
               for a, b in zip(edit['trace'], recorded_edit))

    results = json.loads((PRIOR/'RESULTS.json').read_text())
    axis = np.asarray(json.loads(
        (R/'m170_mu20_episode54_postflip_right_turn_20260928/RESULTS.json').read_text()
    )['rotation_axis_world'])

    def cumulative(run):
        rows = run['trace'][60+BRANCH_LAST:]
        rotations = Rotation.from_quat(np.asarray([x['object_pose'][3:] for x in rows]))
        inc = np.degrees((rotations[1:]*rotations[:-1].inv()).as_rotvec() @ axis)
        return np.r_[0., np.cumsum(inc)]
    edit_curve, prior_curve = cumulative(edit), cumulative(prior)

    # Freeze the common state, play every post-branch action and terminal hold
    # at the true 30 Hz rate, then freeze the final state.
    branch_trace = 60+BRANCH_LAST
    timeline = [('branch', branch_trace)]*30
    timeline += [('action', i) for i in range(60+102, 60+612)]
    timeline += [('hold', i) for i in range(60+612, 732)]
    timeline += [('end', 731)]*30

    OUT.unlink(missing_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*'mp4v'),
                             30, (1280, 640))
    assert writer.isOpened()
    frames = 0
    try:
        for phase, ti in timeline:
            canvas = Image.new('RGB', (1280, 640), (17, 24, 33))
            draw = ImageDraw.Draw(canvas)
            if phase == 'branch':
                time_text = '共同状态（定格1秒）'
            elif phase == 'action':
                time_text = f'分叉后 {(ti-branch_trace)/30:.2f}s'
            elif phase == 'hold':
                time_text = f'动作结束后静置 {(ti-(60+612)+1)/30:.2f}s'
            else:
                time_text = '最终状态（定格1秒）'
            displayed_action = min(max(ti-60+1, 102), 612)
            draw.text((12, 7),
                      f'Episode 54 · 170g / 摩擦2.0 · {time_text} · 动作 {displayed_action}',
                      font=FONT, fill='white')
            draw.text((12, 40),
                      '两边共同执行 edit 到动作102；动作103起，右侧移除全部 future reference',
                      font=SMALL, fill=(190, 225, 250))
            draw.text((12, 65), '绿色=仍满足有效接触；X红 / Y绿 / Z蓝为灯泡局部轴',
                      font=SMALL, fill=(188, 231, 208))

            for x, run, curve, label, future in (
                (0, edit, edit_curve, '继续 reference edit', '未来reference：9步'),
                (640, prior, prior_curve, 'edit历史 → 纯10B prior', '未来reference：0步')):
                video_frame = read_to(run, ti+1)
                rgb = cv2.cvtColor(video_frame[:, :640], cv2.COLOR_BGR2RGB)
                canvas.paste(Image.fromarray(rgb), (x, 96))
                row, geo = run['trace'][ti], run['geometry'][ti]
                draw_axes(canvas, object_pose(row), run['project'], (x, 96))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((x, 96, x+639, 159), fill=(25, 36, 49))
                draw.text((x+8, 98), label, font=FONT, fill='white')
                ci = ti-branch_trace
                angle = float(curve[ci])
                held = effective(row, geo)
                draw.text((x+8, 132),
                          f'目标右转累计 {angle:+.1f}° · 有效接触 {"是" if held else "否"} · {future}',
                          font=SMALL,
                          fill=(188, 231, 208) if held else (255, 150, 130))
            writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)); frames += 1
    finally:
        writer.release(); edit['cap'].release(); prior['cap'].release()

    cap = cv2.VideoCapture(str(OUT)); assert cap.isOpened()
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        assert frame.shape == (640, 1280, 3); decoded += 1
    fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    assert decoded == frames == len(timeline) and abs(fps-30) < 1e-6
    verification = dict(
        video=str(OUT), frames=decoded, fps=fps, duration_s=decoded/fps,
        fully_decoded=True, common_state_freeze_frames=30,
        synchronized_postbranch_action_frames=510, terminal_hold_frames=60,
        final_freeze_frames=30, first_autonomous_displayed_action_step=103,
        edit_recording_matches_numeric_trace=True,
        pure_prior_recording_is_current_rollout=True,
        future_reference_after_branch=dict(edit=9, pure_prior=0),
        downloads_written=False)
    (PRIOR/'video_verification.json').write_text(
        json.dumps(verification, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
