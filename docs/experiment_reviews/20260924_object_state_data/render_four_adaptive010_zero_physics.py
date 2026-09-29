"""Accurate pre-physics opening image, avoiding cached simulator body poses."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import json
import argparse
from pathlib import Path
import numpy as np
import pyrender
from PIL import Image
from audit_episode76_zero_step import meshes, matrix, camera_pose, SharpAFK, HAND, BULB

P=Path(__file__).resolve().parent
R=P/'reference_turn_baseline_20260926'
parser=argparse.ArgumentParser();parser.add_argument('--episode',type=int,required=True);args=parser.parse_args()
C=R/f'qualified_comparison/episode_{args.episode:02d}/direct_m044_mu11'
O=R/'m044_mu11_adaptive010_ddim4_scale25_four_video_20260927'
O.mkdir(exist_ok=True)
ini=np.load(C/'initial_state.npz');summary=json.loads((C/'summary.json').read_text());trace=json.loads((C/'trace.json').read_text())
fk=SharpAFK(HAND);wrist=matrix(ini['wrist'][0])
# Check the FK used for the zero frame against the first actual physics frame.
names=summary['hand_body_names'];pred=fk.link_poses(trace[0]['q'],names,output_frame='world',wrist_pose=wrist,joint_limit_mode='ignore')
assert np.linalg.norm(pred[:,:3,3]-np.array(trace[0]['hand_body_pose'])[:,:3],axis=1).max()<1e-6
poses=dict(zip(fk.link_names,fk.link_poses(ini['q'][0],fk.link_names,output_frame='world',wrist_pose=wrist,joint_limit_mode='ignore')))
scene=pyrender.Scene(bg_color=[.025,.035,.05,1],ambient_light=[.5,.5,.5])
for name,mesh in meshes(HAND,'visual').items():
 mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.65,.67,.70,1],metallicFactor=0,roughnessFactor=.8,doubleSided=True)
 scene.add(pyrender.Mesh.from_trimesh(mesh,material=mat,smooth=False),pose=poses[name])
bulb=meshes(BULB,'visual')['base'];bulb.vertices*=float(ini['object_scale'][0])
mat=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1,.39,.03,1],metallicFactor=0,roughnessFactor=.8,doubleSided=True)
scene.add(pyrender.Mesh.from_trimesh(bulb,material=mat,smooth=False),pose=matrix(ini['object'][0,:7]))
meta=summary['camera_metadata'][0];cp=camera_pose(meta['eye'],meta['target'])
yfov=2*np.arctan(np.tan(np.deg2rad(meta['horizontal_fov'])/2)*480/640)
scene.add(pyrender.PerspectiveCamera(yfov=yfov,znear=.001,zfar=5),pose=cp)
scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.5),pose=cp)
renderer=pyrender.OffscreenRenderer(640,480)
try:
 rgb,_=renderer.render(scene);Image.fromarray(rgb).save(O/f'episode{args.episode}_zero_physics.png')
finally:renderer.delete()
print('Rendered exact zero-physics pose')
