"""Offline exact-FK zero-step geometry, checked against recorded Isaac Gym bodies."""

import os
import argparse
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import pyrender
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, '/home/carus/Program/dex_sim_bench/vendor/sharpa_fk_standalone')
from sharpa_fk import SharpAFK

P = Path(__file__).resolve().parent
ROOT = P / 'reference_turn_baseline_20260926'
CASE = ROOT / 'qualified_comparison/episode_76'
OUT = ROOT / 'episode76_zero_step_audit_20260927'
HAND = Path('/home/carus/Program/dex-controller/maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf')
BULB = Path('/home/carus/Program/dex-controller/data/NOKOV-v3/object/bulb2.urdf')


def matrix(pose):
    t = np.eye(4)
    t[:3,:3] = Rotation.from_quat(np.asarray(pose)[3:7]).as_matrix()
    t[:3,3] = np.asarray(pose)[:3]
    return t


def meshes(urdf, kind):
    result = {}
    for link in ET.parse(urdf).getroot().findall('link'):
        parts = []
        for visual in link.findall(kind):
            element = visual.find('geometry/mesh')
            if element is None:
                continue
            mesh = trimesh.load(urdf.parent / element.get('filename'), force='mesh', process=False)
            mesh.vertices *= np.fromstring(element.get('scale', '1 1 1'), sep=' ')
            origin = visual.find('origin')
            if origin is not None:
                t = np.eye(4)
                t[:3,:3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
                t[:3,3] = np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                mesh.apply_transform(t)
            parts.append(mesh)
        if parts:
            result[link.get('name')] = trimesh.util.concatenate(parts)
    return result


def camera_pose(eye, target):
    eye, target = np.asarray(eye), np.asarray(target)
    z = eye-target
    z /= np.linalg.norm(z)
    x = np.cross([0,0,1],z)
    x /= np.linalg.norm(x)
    y = np.cross(z,x)
    t = np.eye(4)
    t[:3,:3] = np.column_stack([x,y,z])
    t[:3,3] = eye
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=CASE / 'direct_m044_mu11')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--include-index', action='store_true')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run = args.run
    initial = np.load(run/'initial_state.npz')
    summary = json.loads((run/'summary.json').read_text())
    trace = json.loads((run/'trace.json').read_text())
    fk = SharpAFK(HAND)
    wrist = matrix(initial['wrist'][0])
    names = summary['hand_body_names']
    checks = []
    for i in (0,59):
        predicted = fk.link_poses(trace[i]['q'], names, output_frame='world',
                                  wrist_pose=wrist, joint_limit_mode='ignore')
        actual = np.array(trace[i]['hand_body_pose'])
        dp = float(np.linalg.norm(predicted[:,:3,3]-actual[:,:3],axis=1).max())
        dr = float(np.degrees((Rotation.from_matrix(predicted[:,:3,:3]).inv()*
                              Rotation.from_quat(actual[:,3:])).magnitude()).max())
        assert dp < 1e-6 and dr < .001
        checks.append(dict(control_step=i+1,max_body_position_error_m=dp,
                           max_body_rotation_error_deg=dr))
    visual = meshes(HAND,'visual')
    collision = meshes(HAND,'collision')
    bulb_visual = meshes(BULB,'visual')['base']
    bulb_collision = meshes(BULB,'collision')['base']
    scale = float(initial['object_scale'].ravel()[0])
    bulb_visual.vertices *= scale
    bulb_collision.vertices *= scale
    frames = [('zero_physics', initial['q'][0], initial['object'][0,:7]),
              ('first_control_step', trace[0]['q'], trace[0]['object_pose']),
              ('after_60_steps', trace[59]['q'], trace[59]['object_pose'])]
    camera_metadata = summary['camera_metadata']
    if not camera_metadata:
        camera_metadata = json.loads((CASE / 'direct_m044_mu11/summary.json').read_text())['camera_metadata']
    meta = camera_metadata[0]
    cp = camera_pose(meta['eye'],meta['target'])
    yfov = 2*np.arctan(np.tan(np.deg2rad(meta['horizontal_fov'])/2)*480/640)
    renderer = pyrender.OffscreenRenderer(640,480)
    rendered = []
    stats = {}
    try:
        for label,q,ob in frames:
            poses = fk.link_poses(q, fk.link_names, output_frame='world',
                                  wrist_pose=wrist, joint_limit_mode='ignore')
            poses = dict(zip(fk.link_names,poses))
            obj_pose = matrix(ob)
            scene = pyrender.Scene(bg_color=[.025,.035,.05,1],ambient_light=[.5,.5,.5])
            for name,mesh in visual.items():
                color = [.05,.85,.95,1] if name.startswith('right_middle') else [.65,.67,.70,1]
                if args.include_index and name.startswith('right_index'):
                    color = [.95,.2,.65,1]
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=0,roughnessFactor=.8,doubleSided=True)
                scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False),pose=poses[name])
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1,.39,.03,1],metallicFactor=0,roughnessFactor=.8,doubleSided=True)
            scene.add(pyrender.Mesh.from_trimesh(bulb_visual,material=material,smooth=False),pose=obj_pose)
            scene.add(pyrender.PerspectiveCamera(yfov=yfov,znear=.001,zfar=5),pose=cp)
            scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.5),pose=cp)
            rgb,_ = renderer.render(scene)
            Image.fromarray(rgb).save(out/f'{label}_front.png')
            rendered.append(Image.fromarray(rgb))
            world_bulb = bulb_collision.copy()
            world_bulb.apply_transform(obj_pose)
            bulb_hull = world_bulb.convex_hull
            per_link = {}
            for name,mesh in collision.items():
                if not (name.startswith('right_middle') or (args.include_index and name.startswith('right_index'))):
                    continue
                world_hand = mesh.copy()
                world_hand.apply_transform(poses[name])
                hand_hull = world_hand.convex_hull
                near,distance,_ = trimesh.proximity.closest_point(bulb_hull,hand_hull.vertices)
                reverse,reverse_distance,_ = trimesh.proximity.closest_point(hand_hull,bulb_hull.vertices)
                min_distance = float(min(distance.min(),reverse_distance.min()))
                # A positive projected interval gap is a rigorous separation
                # certificate for both complete meshes, including triangle interiors.
                pair_index = int(distance.argmin())
                axis = hand_hull.vertices[pair_index]-near[pair_index]
                axis /= np.linalg.norm(axis)
                hand_projection = world_hand.vertices@axis
                bulb_projection = world_bulb.vertices@axis
                gap = max(float(hand_projection.min()-bulb_projection.max()),
                          float(bulb_projection.min()-hand_projection.max()))
                per_link[name] = dict(convex_hull_vertex_to_surface_distance_mm=min_distance*1000,
                                      separating_axis_gap_mm=gap*1000)
            stats[label] = per_link
            print(label,json.dumps(per_link),flush=True)
    finally:
        renderer.delete()
    source = Image.open('/home/carus/Data/Object_state_data/episode_76/front/000114.png').convert('RGB')
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    canvas=Image.new('RGB',(1920,568),(18,25,35))
    draw=ImageDraw.Draw(canvas)
    titles=['零物理步 · 原始q与物体位姿的FK重建','首个控制步后 · 同模型同视角','真机 front · episode76 第114帧']
    for k,im in enumerate([rendered[0],rendered[1],source]):
        canvas.paste(im,(640*k,88))
        draw.text((640*k+8,8),titles[k],font=font,fill='white')
        text='青色为中指；未运行物理' if k==0 else ('青色为中指；按记录状态重建' if k==1 else '原始数据集图像；相机视角未标定')
        draw.text((640*k+8,45),text,font=small,fill=(180,230,245))
    canvas.save(out/'zero_step_vs_first_step_vs_real.png')
    output=dict(episode=76,source_frame=114,hand_urdf=str(HAND),bulb_urdf=str(BULB),
                object_scale=scale,fk_validation=checks,middle_geometry=stats,
                camera_metadata=meta, included_index=args.include_index,
                geometry_note='Convex hulls of URDF collision meshes. Positive separating-axis interval gap across ALL original mesh vertices proves complete triangle surfaces disjoint, before PhysX contact offsets. Vertex-to-hull-surface distance is a sampled estimate, not an exact minimum.',
                physics_steps_for_initial_image=0)
    (out/'geometry_audit.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    main()
