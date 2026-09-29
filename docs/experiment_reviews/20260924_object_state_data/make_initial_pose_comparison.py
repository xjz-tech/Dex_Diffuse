"""Compare source frame90 with exact saved import/settled geometry for case467.

The sim panels use saved root/joint states and the same URDF visual meshes,
rendered offline. They are geometric reconstructions, not physics frames.
"""

from pathlib import Path
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pyrender
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import trimesh


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'initial_pose_diagnostic'
CONTROLLER = Path('/home/carus/Program/dex-controller')
SOURCE = Path('/home/carus/Data/Object_state_data/episode_53')
HAND_URDF = CONTROLLER / 'maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf'
OBJECT_MESH = CONTROLLER / 'data/NOKOV-v3/object/mesh/bulb2.obj'
CASE = 467
sys.path.insert(0, '/home/carus/Program/dex_sim_bench/vendor')
from sharpa_fk_standalone.sharpa_fk import SharpAFK


def matrix(xyz, quaternion):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    t[:3, 3] = xyz
    return t


def origin(element):
    if element is None:
        return np.eye(4)
    t = np.eye(4)
    t[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def camera_pose(eye, target):
    # OpenGL cameras look toward -Z with +Y up.
    backward = eye - target
    backward /= np.linalg.norm(backward)
    up = np.array([0., 0., 1.])
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([right, up, backward])
    t[:3, 3] = eye
    return t


def load_visuals():
    root = ET.parse(HAND_URDF).getroot()
    visuals = []
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            mesh_element = visual.find('geometry/mesh')
            if mesh_element is None:
                continue
            path = (HAND_URDF.parent / mesh_element.get('filename')).resolve()
            mesh = trimesh.load(path, force='mesh', process=False)
            scale = np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' ')
            mesh.vertices *= scale
            color_element = visual.find('material/color')
            color = ([.79, .82, .93, 1.] if color_element is None
                     else np.fromstring(color_element.get('rgba'), sep=' ').tolist())
            mesh.visual.vertex_colors = (np.array(color) * 255).astype(np.uint8)
            visuals.append((link.get('name'), origin(visual.find('origin')), pyrender.Mesh.from_trimesh(mesh, smooth=False)))
    bulb = trimesh.load(OBJECT_MESH, force='mesh', process=False)
    bulb.visual.vertex_colors = np.array([221, 112, 23, 255], dtype=np.uint8)
    return visuals, bulb


def render_state(renderer, visuals, bulb, poses, q, wrist, obj, scale, eye, target):
    scene = pyrender.Scene(bg_color=[18, 24, 31], ambient_light=[.48, .48, .48])
    hand_root = matrix(wrist[:3], wrist[3:])
    for name, visual_origin, mesh in visuals:
        scene.add(mesh, pose=hand_root @ poses[name] @ visual_origin)
    bulb_mesh = bulb.copy()
    bulb_mesh.vertices *= scale
    scene.add(pyrender.Mesh.from_trimesh(bulb_mesh, smooth=False), pose=matrix(obj[:3], obj[3:]))
    scene.add(pyrender.PerspectiveCamera(yfov=2 * math.atan(math.tan(math.radians(42) / 2) * 480 / 640)),
              pose=camera_pose(eye, target))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=camera_pose(eye, target))
    image, _ = renderer.render(scene)
    return Image.fromarray(image)


def main():
    OUT.mkdir(exist_ok=True)
    initial = np.load(ROOT / 'repeat/initial_state.npz')
    trace = np.load(ROOT / 'repeat/trajectory.npz')
    direct_initial = np.load(ROOT / 'direct_case467/initial_state.npz')
    direct_trace = np.load(ROOT / 'direct_case467/trajectory.npz')
    reference = np.load(ROOT / 'reference/reference.npz')
    fk = SharpAFK(HAND_URDF)
    visuals, bulb = load_visuals()
    scale = float(initial['object_scale'][CASE])
    source_wrist = initial['wrist'][CASE, :7]
    recorded_state = reference['recorded_state31'][1, 0]
    first = recorded_state[3:6] / np.linalg.norm(recorded_state[3:6])
    second = recorded_state[6:9] - np.dot(first, recorded_state[6:9]) * first
    second /= np.linalg.norm(second)
    source_tcp_columns = np.eye(4)
    source_tcp_columns[:3, :3] = np.column_stack([first, second, np.cross(first, second)])
    source_tcp_columns[:3, 3] = recorded_state[:3]
    column_source_relative = np.linalg.inv(source_tcp_columns) @ reference['object_pose_base'][1, 0]
    row_source_relative = reference['object_pose_wrist'][1, 0]
    source_object = matrix(source_wrist[:3], source_wrist[3:]) @ column_source_relative
    source_object_pose = np.r_[source_object[:3, 3], Rotation.from_matrix(source_object[:3, :3]).as_quat()]
    row_source_object = matrix(source_wrist[:3], source_wrist[3:]) @ row_source_relative
    row_source_object_pose = np.r_[row_source_object[:3, 3], Rotation.from_matrix(row_source_object[:3, :3]).as_quat()]
    states = dict(
        source_columns=(reference['hand_qpos_rad'][1, 0], source_wrist, source_object_pose),
        source_rows=(reference['hand_qpos_rad'][1, 0], source_wrist, row_source_object_pose),
        imported=(initial['q'][CASE], initial['wrist'][CASE, :7], initial['object'][CASE, :7]),
        settled=(trace['q'][59, CASE], trace['wrist_pose'][59, CASE], trace['object_pose'][59, CASE]),
    )
    target = states['settled'][2][:3]
    offsets = [[.12, .42, .14], [-.32, .24, -.08]]
    renderer = pyrender.OffscreenRenderer(640, 480)
    rendered = {}
    for label, (q, wrist, obj) in states.items():
        poses = dict(zip(fk.link_names, fk.link_poses(q, fk.link_names, joint_limit_mode='ignore')))
        for view, offset in enumerate(offsets):
            eye = target + np.array(offset)
            eye[2] = max(eye[2], -.32)
            image = render_state(renderer, visuals, bulb, poses, q, wrist, obj, scale, eye, target)
            image.save(OUT / f'case467_{label}_geometry_view{view}.png')
            rendered[(label, view)] = image
    renderer.delete()

    source = {view: Image.open(SOURCE / view / '000090.png').convert('RGB') for view in ['front', 'wrist']}
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 23)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
    canvas = Image.new('RGB', (1920, 1110), (16, 22, 29))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), 'episode53 / case467  手—灯泡初始位姿对照', font=font, fill='white')
    draw.text((16, 42), '左：真机第90帧  |  中：导入瞬间  |  右：静置2秒后（原对比视频起点）', font=small, fill=(204, 225, 246))
    titles = [('真机前视角', '真机腕部视角'), ('仿真掌侧 / 几何重建', '仿真侧面 / 几何重建'),
              ('仿真掌侧 / 几何重建', '仿真侧面 / 几何重建')]
    panels = [source['front'], source['wrist'], rendered[('imported', 0)], rendered[('imported', 1)],
              rendered[('settled', 0)], rendered[('settled', 1)]]
    for col in range(3):
        for row in range(2):
            x, y = 640 * col, 90 + 510 * row
            draw.text((x + 10, y), titles[col][row], font=small, fill=(233, 239, 246))
            canvas.paste(panels[col * 2 + row], (x, y + 30))
    draw.text((16, 1084), '仿真列按保存的关节/根位姿与原URDF网格重建；两个仿真时刻使用完全相同的虚拟相机。真机相机外参不同。',
              font=small, fill=(207, 221, 235))
    canvas.save(OUT / 'case467_source_import_settled.png')

    matched = Image.new('RGB', (1920, 1110), (16, 22, 29))
    matched_draw = ImageDraw.Draw(matched)
    matched_draw.text((16, 8), 'case467  相同 wrist、相同相机：源相对位姿 → 扰动导入 → 静置2秒', font=font, fill='white')
    matched_draw.text((16, 42), '左：6D旋转按列的候选解释  |  中：上一轮实际导入  |  右：静置60步后',
                      font=small, fill=(204, 225, 246))
    for col, label in enumerate(['source_columns', 'imported', 'settled']):
        for row in range(2):
            x, y = 640 * col, 90 + 510 * row
            matched_draw.text((x + 10, y), '掌侧' if row == 0 else '侧面', font=small, fill=(233, 239, 246))
            matched.paste(rendered[(label, row)], (x, y + 30))
    matched_draw.text((16, 1084), '三列均使用原生 wrist、相同相机、相同手和灯泡网格；左列是未运行物理的反事实几何重建。',
                      font=small, fill=(207, 221, 235))
    matched.save(OUT / 'case467_matched_camera_three_stages.png')
    parser_canvas = Image.new('RGB', (2560, 1110), (16, 22, 29))
    parser_draw = ImageDraw.Draw(parser_canvas)
    parser_draw.text((16, 8), 'case467  6D旋转约定冲突：按列 vs 按行', font=font, fill='white')
    parser_draw.text((16, 42), '四列使用同一个原生 wrist、同一虚拟相机、同一手和灯泡网格', font=small, fill=(204, 225, 246))
    for col, (label, title) in enumerate([('source_columns', '按列候选'), ('source_rows', '本次按行解释'),
                                         ('imported', '按行 + 位姿扰动'), ('settled', '静置2秒后')]):
        for row in range(2):
            x, y = 640 * col, 90 + 510 * row
            parser_draw.text((x + 10, y), title + (' / 掌侧' if row == 0 else ' / 侧面'),
                             font=small, fill=(233, 239, 246))
            parser_canvas.paste(rendered[(label, row)], (x, y + 30))
    parser_draw.text((16, 1084), '两种6D解释的相对位置相差约20cm、方向相差90.6°；仅凭现有数据不能确认采集时使用哪种。',
                     font=small, fill=(207, 221, 235))
    parser_canvas.save(OUT / 'case467_rotation_parser_comparison.png')
    source_rel = column_source_relative
    wrist_init, object_init = states['imported'][1:]
    wrist_rot = Rotation.from_quat(wrist_init[3:])
    imported_rel_pos = wrist_rot.inv().apply(object_init[:3] - wrist_init[:3])
    imported_rel_rot = wrist_rot.inv() * Rotation.from_quat(object_init[3:])
    settled_rel_pos = trace['relative_position'][59, CASE]
    settled_rel_rot = Rotation.from_quat(trace['relative_quaternion'][59, CASE])
    source_rel_rot = Rotation.from_matrix(source_rel[:3, :3])
    angle = lambda a, b: float(np.degrees((a.inv() * b).magnitude()))
    metrics = dict(
        case=CASE, source_episode=53, source_frame=90,
        column_interpretation_relative_position_m=source_rel[:3, 3].tolist(),
        row_interpretation_relative_position_m=row_source_relative[:3, 3].tolist(),
        imported_relative_position_m=imported_rel_pos.tolist(),
        settled_relative_position_m=settled_rel_pos.tolist(),
        rows_vs_columns_position_m=float(np.linalg.norm(row_source_relative[:3, 3] - source_rel[:3, 3])),
        rows_vs_columns_rotation_deg=angle(source_rel_rot, Rotation.from_matrix(row_source_relative[:3, :3])),
        import_vs_rows_position_m=float(np.linalg.norm(imported_rel_pos - row_source_relative[:3, 3])),
        import_vs_rows_rotation_deg=angle(Rotation.from_matrix(row_source_relative[:3, :3]), imported_rel_rot),
        import_vs_columns_position_m=float(np.linalg.norm(imported_rel_pos - source_rel[:3, 3])),
        import_vs_columns_rotation_deg=angle(source_rel_rot, imported_rel_rot),
        settled_vs_import_position_m=float(np.linalg.norm(settled_rel_pos - imported_rel_pos)),
        settled_vs_import_rotation_deg=angle(imported_rel_rot, settled_rel_rot),
        settled_vs_columns_position_m=float(np.linalg.norm(settled_rel_pos - source_rel[:3, 3])),
        settled_vs_columns_rotation_deg=angle(source_rel_rot, settled_rel_rot),
        first_settle_step_position_change_m=float(np.linalg.norm(trace['relative_position'][0, CASE] - imported_rel_pos)),
        imported_q_matches_source=bool(np.array_equal(initial['q'][CASE], reference['hand_qpos_rad'][1, 0])),
        settled_q_max_change_rad=float(np.max(np.abs(trace['q'][59, CASE] - initial['q'][CASE]))),
        settled_q_rms_change_rad=float(np.sqrt(np.mean((trace['q'][59, CASE] - initial['q'][CASE]) ** 2))),
        wrist_unchanged_during_settle=bool(np.allclose(trace['wrist_pose'][:60, CASE], wrist_init, atol=1e-6)),
        direct_guided_initial_equal=bool(all(np.array_equal(initial[key], direct_initial[key]) for key in initial.files)),
        direct_guided_settle_equal=bool(all(np.array_equal(trace[key][:60, CASE], direct_trace[key][:60, CASE])
                                            for key in ['q', 'object_pose', 'wrist_pose', 'failure'])),
    )
    assert metrics['imported_q_matches_source']
    assert metrics['wrist_unchanged_during_settle']
    assert metrics['direct_guided_initial_equal'] and metrics['direct_guided_settle_equal']
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'comparison_manifest.json').write_text(json.dumps(dict(
        source_frame=90, source_images=[str(SOURCE / view / '000090.png') for view in ['front', 'wrist']],
        simulation_state_files=[str(ROOT / 'repeat/initial_state.npz'), str(ROOT / 'repeat/trajectory.npz')],
        object_scale=scale, simulation_meshes='original native hand URDF visual meshes and bulb2.obj',
        renderer='offline pyrender geometry reconstruction; no physics advancement',
        simulation_camera_target=target.tolist(), simulation_camera_offsets=offsets,
        outputs=[str(OUT / 'case467_source_import_settled.png'), str(OUT / 'case467_matched_camera_three_stages.png'),
                 str(OUT / 'case467_rotation_parser_comparison.png')],
    ), indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
