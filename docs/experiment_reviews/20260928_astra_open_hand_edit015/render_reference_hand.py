"""Kinematic hand for the shared open-hand reference. No physics, no object."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyrender
import trimesh
from PIL import Image
from scipy.spatial.transform import Rotation
import json

ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/Astra-controller')
EDIT = ROOT / 'outputs/astra_open_hand/seed42_edit015_ddim4'
OUT = ROOT / 'outputs/astra_open_hand/comparison/reference_hand_frames'
OUT.mkdir(parents=True, exist_ok=True)
manifest = json.loads((EDIT / 'manifest.json').read_text())
rows = [json.loads(line) for line in (EDIT / 'trajectory.jsonl').read_text().splitlines()]
urdf = Path(rows[0]['state']['hand_urdf'])
names = json.loads((ROOT / 'docs/astra_controller/20260922_open_hand/plan_spec.json').read_text())['joint_names']
tree = ET.parse(urdf).getroot()


def origin(node):
    t = np.eye(4)
    if node is not None:
        t[:3, 3] = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(node.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


joints = []
for joint in tree.findall('joint'):
    joints.append(dict(name=joint.get('name'), parent=joint.find('parent').get('link'),
                       child=joint.find('child').get('link'), origin=origin(joint.find('origin')),
                       axis=None if joint.get('type') == 'fixed' else np.fromstring(joint.find('axis').get('xyz'), sep=' ')))
wrist = np.asarray(manifest['initial']['wrist_state'], dtype=np.float64)
base = np.eye(4)
base[:3, :3] = Rotation.from_quat(wrist[3:7]).as_matrix()
base[:3, 3] = wrist[:3]


def fk(q):
    angles = dict(zip(names, q))
    poses = {'world': base}
    remaining = list(joints)
    while remaining:
        before = len(remaining)
        for joint in remaining[:]:
            if joint['parent'] not in poses:
                continue
            motion = np.eye(4)
            if joint['axis'] is not None:
                motion[:3, :3] = Rotation.from_rotvec(joint['axis'] * angles[joint['name']]).as_matrix()
            poses[joint['child']] = poses[joint['parent']] @ joint['origin'] @ motion
            remaining.remove(joint)
        if len(remaining) >= before:
            raise RuntimeError('unresolved URDF tree')
    return poses


poses = fk(rows[0]['state']['qpos'])
errors = [float(np.linalg.norm(poses[link][:3, 3] - np.asarray(actual[:3])))
          for link, actual in rows[0]['state']['body_pose_world'].items()]
if max(errors) > 1e-4:
    raise RuntimeError(f'FK mismatch {max(errors)}')

width, height = 960, 720
scene = pyrender.Scene(bg_color=[0.16, 0.18, 0.22, 1.0], ambient_light=[0.55, 0.55, 0.55])
visuals = []
for link in tree.findall('link'):
    for visual in link.findall('visual'):
        mesh = visual.find('geometry/mesh')
        tm = trimesh.load(str(urdf.parent / mesh.get('filename')), force='mesh', process=False)
        tm.apply_scale(np.fromstring(mesh.get('scale', '1 1 1'), sep=' '))
        color = visual.find('material/color')
        rgba = np.fromstring(color.get('rgba'), sep=' ') if color is not None else np.array([0.8, 0.83, 0.9, 1.0])
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=rgba, metallicFactor=0.15, roughnessFactor=0.65)
        node = scene.add(pyrender.Mesh.from_trimesh(tm, material=material, smooth=False))
        visuals.append((node, link.get('name'), origin(visual.find('origin'))))

eye = np.array(manifest['camera_position'])
center = np.array(manifest['camera_target'])
back = (eye - center) / np.linalg.norm(eye - center)
right = np.cross(np.array([0.0, 0.0, 1.0]), back)
right /= np.linalg.norm(right)
up = np.cross(back, right)
camera_pose = np.eye(4)
camera_pose[:3, :3] = np.column_stack((right, up, back))
camera_pose[:3, 3] = eye
yfov = 2 * math.atan(math.tan(math.radians(manifest['camera_fov']) / 2) * height / width)
scene.add(pyrender.PerspectiveCamera(yfov=yfov, znear=0.01, zfar=5.0), pose=camera_pose)
scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.8), pose=camera_pose)
renderer = pyrender.OffscreenRenderer(width, height)

targets = [np.asarray(manifest['initial']['qpos'], dtype=np.float64)]
targets.extend(np.asarray(row['astra_reference'], dtype=np.float64) for row in rows)
q0 = np.asarray(json.loads((ROOT / 'docs/astra_controller/20260922_open_hand/plan_spec.json').read_text())['initial_target'], dtype=np.float64)
# Fully open at trajectory step 128. Frames 0..58 already match the executed reference.
for step in range(len(targets), 129):
    alpha = np.clip((step - 8) / 120.0, 0, 1)
    targets.append(q0 * (1 - alpha))
for step, q in enumerate(targets):
    poses = fk(q)
    for node, link, local in visuals:
        scene.set_pose(node, poses[link] @ local)
    color, _ = renderer.render(scene)
    Image.fromarray(color).save(OUT / f'{step:06d}.png')
renderer.delete()
print(f'wrote {len(targets)} frames to {OUT}')
