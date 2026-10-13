"""Overlay object-local XYZ axes on the real/simulation four-panel video."""

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import yaml


ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
P = ROOT / 'docs/experiment_reviews/20260924_object_state_data/corrected_direct_vs_reference/full_episode53'
INPUT = P / 'episode53_full_real_direct_guide2_exec2_exec1_front.mp4'
OUTPUT = P / 'episode53_full_real_direct_guide2_exec2_exec1_xyz_axes.mp4'
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53')
CALIBRATION = Path('/home/carus/Program/dex_sim_bench/configs/bulb_task_default.yaml')
FONT = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)
AXIS_LENGTH_M = .035
COLORS = [(255, 75, 70), (55, 240, 110), (70, 145, 255)]


def camera_projector(eye, target, horizontal_fov_deg):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., 0., 1.]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = 320 / np.tan(np.deg2rad(horizontal_fov_deg) / 2)

    def project(points):
        d = np.asarray(points) - eye
        depth = d @ forward
        if np.any(depth <= 0):
            return None
        return np.column_stack((320 + focal * (d @ right) / depth,
                                240 - focal * (d @ up) / depth))

    return project


def real_projector():
    config = yaml.safe_load(CALIBRATION.read_text())['cameras']['front']
    camera_from_base = np.linalg.inv(np.asarray(config['world_from_camera'], dtype=float))
    focal = 320 / np.tan(np.deg2rad(config['horizontal_fov']) / 2)

    def project(points):
        homogeneous = np.column_stack((np.asarray(points), np.ones(len(points))))
        camera = (camera_from_base @ homogeneous.T).T[:, :3]
        if np.any(camera[:, 2] <= 0):
            return None
        return np.column_stack((320 + focal * camera[:, 0] / camera[:, 2],
                                240 + focal * camera[:, 1] / camera[:, 2]))

    return project


def object_pose(record):
    pose = np.eye(4)
    pose[:3, 3] = record['object_pose'][:3]
    pose[:3, :3] = Rotation.from_quat(record['object_pose'][3:7]).as_matrix()
    return pose


def draw_axes(canvas, pose, project, corner):
    world_points = pose[:3, 3][None] + np.vstack((np.zeros(3),
        AXIS_LENGTH_M * pose[:3, :3].T))
    pixels = project(world_points)
    if pixels is None:
        return
    pixels = np.rint(pixels).astype(int)
    x, y = corner
    panel = canvas.crop((x, y + 64, x + 640, y + 544))
    draw = ImageDraw.Draw(panel)
    origin = tuple(pixels[0])
    for axis in range(3):
        endpoint = tuple(pixels[axis + 1])
        color = COLORS[axis]
        draw.line((origin, endpoint), fill=(0, 0, 0), width=8)
        draw.line((origin, endpoint), fill=color, width=5)
        direction = pixels[axis + 1] - pixels[0]
        norm = np.linalg.norm(direction)
        if norm > 1:
            along = direction / norm
            cross = np.array([-along[1], along[0]])
            tip = pixels[axis + 1]
            head = [tuple(np.rint(tip).astype(int)),
                    tuple(np.rint(tip - 12 * along + 5 * cross).astype(int)),
                    tuple(np.rint(tip - 12 * along - 5 * cross).astype(int))]
            draw.polygon(head, fill=color, outline='black')
            label = 'XYZ'[axis]
            draw.text(tuple(tip + 5 * along + 2 * cross), label, font=FONT, fill=color,
                      stroke_width=2, stroke_fill='black')
    draw.ellipse((origin[0]-4, origin[1]-4, origin[0]+4, origin[1]+4),
                 fill='white', outline='black', width=2)
    canvas.paste(panel, (x, y + 64))


def main():
    real_pose = np.load(SOURCE / 'obj_state.npy').reshape(-1, 4, 4)
    folders = [P / 'direct_original', P / 'insert1_scale50_guide2_exec2',
               P / 'insert1_scale50_guide2_exec1']
    traces = [json.loads((f / 'trace.json').read_text()) for f in folders]
    summary = [json.loads((f / 'summary.json').read_text()) for f in folders]
    sim_project = []
    for run in summary:
        wrist = np.asarray(run['initial_wrist'])
        local_object = np.asarray(run['initial_object_pose_wrist'])
        target = wrist[:3] + Rotation.from_quat(wrist[3:]).apply(local_object[:3, 3])
        eye = target + np.array([-.32, .24, -.08])
        eye[2] = max(eye[2], -.32)
        sim_project.append(camera_projector(eye, target, 42))
    project_real = real_projector()
    cap = cv2.VideoCapture(str(INPUT))
    assert cap.isOpened()
    writer = imageio.get_writer(str(OUTPUT), fps=30, codec='libx264', quality=8,
                                macro_block_size=16)
    try:
        for shown in range(773):
            ok, frame = cap.read()
            assert ok and frame.shape == (1088, 1280, 3), shown
            canvas = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(canvas)
            source_frame = 90 + shown // 2 if shown < 713 else 447
            draw_axes(canvas, real_pose[source_frame], project_real, (0, 0))
            for k, corner in enumerate(((640, 0), (0, 544), (640, 544))):
                if shown < 713:
                    index = shown // 2 if k == 0 else shown
                else:
                    index = summary[k]['steps']['action'] + shown - 713
                pose = object_pose(traces[k][60 + index])
                draw_axes(canvas, pose, sim_project[k], corner)
            draw.rectangle((5, 514, 335, 541), fill=(17, 25, 35))
            draw.text((10, 517), 'X 红  Y 绿  Z 蓝 · 真机为近似标定投影',
                      font=FONT, fill='white')
            writer.append_data(np.asarray(canvas))
            if shown in (0, 148, 600, 603, 712, 772):
                canvas.save(P / f'episode53_full_xyz_axes_{shown:03d}.png')
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
    (P / 'axes_video_verification.json').write_text(json.dumps(dict(
        video=str(OUTPUT), frames=count, fps=30, duration_s=count / 30,
        object_axes='local +X red, +Y green, +Z blue; 0.035 m from recorded/simulated object origin',
        real_pose='Object_state_data/episode_53/obj_state.npy',
        real_camera='approximate front calibration from dex_sim_bench/configs/bulb_task_default.yaml; not surveyed for episode 53',
        simulation_pose='trace.json object_pose for each rollout',
        simulation_camera='per-run fixed camera at initial object position +[-0.32,0.24,-0.08], clamped z>=-0.32, looking at initial object, 42 deg HFOV',
    ), indent=2) + '\n')
    print(OUTPUT)


if __name__ == '__main__':
    main()
