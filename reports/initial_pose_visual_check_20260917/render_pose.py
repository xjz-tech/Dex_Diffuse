"""Offline URDF hand-pose reconstruction; no object pose is available in source H5."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
meta=json.loads((ROOT/'reports/initial_pose_sample_20260917.json').read_text())
urdf=Path('/home/frankagvl/workspace/dex_setup/dex-controller_sapgv2/maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf')
# Convert only the ASCII palm mesh to OBJ for TinyRenderer compatibility.
tree=ET.parse(urdf)
for mesh in tree.findall('.//mesh'):
    source=(urdf.parent/mesh.attrib['filename']).resolve()
    mesh.set('filename',str(OUT/'palm_visual.obj') if source.name=='right_hand_C_MC_visual.STL' else str(source))
render_urdf=OUT/'render_hand.urdf'
tree.write(render_urdf)
p.connect(p.DIRECT)
body=p.loadURDF(str(render_urdf),useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE)
for link in range(-1,p.getNumJoints(body)):
 p.changeVisualShape(body,link,rgbaColor=[0.68,0.73,0.80,1])
joints={p.getJointInfo(body,j)[1].decode():j for j in range(p.getNumJoints(body))}
assert all(n in joints for n in meta['real_joint_names'])
width,height=640,600
canvas=Image.new('RGB',(3*width,2*(height+45)+65),'white')
draw=ImageDraw.Draw(canvas)
draw.text((16,12),'URDF HAND RECONSTRUCTION ONLY - original image / bulb pose / wrist world pose NOT recorded',fill='black')
draw.text((16,32),'Top: selected dataset step=0. Bottom: current active ROTATE. Arbitrary display cameras; not grasp verification.',fill='black')
manifest={'urdf':str(urdf),'bulb_rendered':False,'base_pose':'identity, arbitrary display frame; not recovered world pose','poses':{}}
for row,(label,values) in enumerate([('SELECTED DATASET POSE (inactive backup)',meta['new_real_rad']),('CURRENT ACTIVE ROTATE',meta['old_real_rad'])]):
 for name,value in zip(meta['real_joint_names'],values):p.resetJointState(body,joints[name],value)
 assert np.allclose([p.getJointState(body,joints[n])[0] for n in meta['real_joint_names']],values,atol=1e-12,rtol=0)
 boxes=[p.getAABB(body,j) for j in range(p.getNumJoints(body))]
 lo=np.min([v[0] for v in boxes],axis=0);hi=np.max([v[1] for v in boxes],axis=0)
 center=(lo+hi)/2
 print(label,'bounds',lo,hi,'center',center)
 for col,yaw in enumerate((40,160,280)):
  view=p.computeViewMatrixFromYawPitchRoll(center.tolist(),.40,yaw,-18,0,2)
  proj=p.computeProjectionMatrixFOV(48,width/height,.01,2)
  image=p.getCameraImage(width,height,view,proj,renderer=p.ER_TINY_RENDERER,lightDirection=[-1,-1,2],shadow=1)
  pixels=np.asarray(image[2],dtype=np.uint8).reshape(height,width,4)[:,:,:3]
  tile=Image.fromarray(pixels)
  y=65+row*(height+45)
  canvas.paste(tile,(col*width,y+35))
  draw.text((col*width+12,y+10),label+' | view '+str(col+1),fill='black')
 canvas.save(OUT/'hand_pose_comparison.png')
 manifest['poses'][label]=dict(joint_names=meta['real_joint_names'],qpos_real_rad=values)
p.disconnect()
(OUT/'render_metadata.json').write_text(json.dumps(manifest,indent=2)+'\n')
