from isaacgym import gymapi
import numpy as np,json
from pathlib import Path
from scipy.spatial.transform import Rotation
import cv2
OUT=Path(__file__).resolve().parent;rows=json.loads((OUT/'selected.json').read_text())
g=gymapi.acquire_gym();p=gymapi.SimParams();p.up_axis=gymapi.UP_AXIS_Z;p.gravity=gymapi.Vec3(0,0,0);p.use_gpu_pipeline=False;p.physx.use_gpu=True
s=g.create_sim(0,0,gymapi.SIM_PHYSX,p);assert s is not None
opts=gymapi.AssetOptions();opts.fix_base_link=True;opts.disable_gravity=True;opts.collapse_fixed_joints=False;opts.use_mesh_materials=True;opts.default_dof_drive_mode=gymapi.DOF_MODE_POS;opts.vhacd_enabled=True
hand=g.load_asset(s,'/home/carus/Program/dex-controller/maniptrans_envs/assets/sharpa_hand','v3right_sharpa_wave-forhammer5.urdf',opts)
o=gymapi.AssetOptions();o.fix_base_link=True;o.disable_gravity=True;o.use_mesh_materials=True
bulb=g.load_asset(s,'/home/carus/Program/dex-controller/data/NOKOV-v3/object','bulb2.urdf',o)
items=[]
for i,row in enumerate(rows):
 en=g.create_env(s,gymapi.Vec3(-1,-1,-1),gymapi.Vec3(1,1,1),3)
 h=g.create_actor(en,hand,gymapi.Transform(),'dexhand',i,0)
 w=np.asarray(row['wrist']);obj=np.asarray(row['object']);rot=Rotation.from_quat(w[3:7]);rel=rot.inv().apply(obj[:3]-w[:3]);quat=(rot.inv()*Rotation.from_quat(obj[3:7])).as_quat()
 t=gymapi.Transform();t.p=gymapi.Vec3(*rel);t.r=gymapi.Quat(*quat)
 b=g.create_actor(en,bulb,t,'bulb',i,0);g.set_actor_scale(en,b,row['scale'])
 for j in range(g.get_actor_rigid_body_count(en,b)):g.set_rigid_body_color(en,b,j,gymapi.MESH_VISUAL,gymapi.Vec3(.95,.43,.07))
 state=g.get_actor_dof_states(en,h,gymapi.STATE_ALL);state['pos']=row['q'];state['vel']=0;g.set_actor_dof_states(en,h,state,gymapi.STATE_ALL);g.set_actor_dof_position_targets(en,h,np.asarray(row['q'],np.float32))
 cams=[]
 for offset in [(0.35,-.32,.16),(-.35,-.16,.12)]:
  props=gymapi.CameraProperties();props.width=600;props.height=480;props.horizontal_fov=44
  c=g.create_camera_sensor(en,props);target=(rel+np.array([0,0,0]))/2
  g.set_camera_location(c,en,gymapi.Vec3(*(target+offset)),gymapi.Vec3(*target));cams.append(c)
 items.append((en,h,cams))
g.prepare_sim(s)
# No simulation step: render saved joint/root state only.
g.step_graphics(s);g.render_all_camera_sensors(s)
for i,(en,h,cams) in enumerate(items):
 actual=g.get_actor_dof_states(en,h,gymapi.STATE_POS)['pos'];assert np.max(np.abs(actual-np.asarray(rows[i]['q'])))<1e-6
 for v,c in enumerate(cams):
  rgba=np.asarray(g.get_camera_image(s,en,c,gymapi.IMAGE_COLOR)).reshape(480,600,4)
  cv2.imwrite(str(OUT/f'pose_{i}_view{v}.png'),cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR))
print('Rendered 6 actual saved initial states, 2 views; wrist orientation aligned; zero physics steps.',flush=True)
g.destroy_sim(s)
