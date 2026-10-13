from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse');P=Path(__file__).resolve().parent;D=Path('/home/carus/Data/Object_state_data');C=Path('/home/carus/Program/dex-controller');B=Path('/home/carus/Program/dex_sim_bench');refdir=P/'reference';refdir.mkdir(exist_ok=True)
specs=[dict(episode=51,start=105,end=180),dict(episode=53,start=90,end=165)]
arrays={k:[] for k in ['hand_qpos_rad','hand_target_rad','recorded_state31','object_pose_base','wrist_pose_base','object_pose_wrist','source_state_frame_indices']};audit=[]
for c in specs:
 ep=D/f'episode_{c["episode"]}';s=np.load(ep/'state.npy');a=np.load(ep/'action.npy');obj=np.load(ep/'obj_state.npy').reshape(-1,4,4);st,en=c['start'],c['end'];state=s[st:en+1];o=obj[st:en+1].astype(float);wr=np.repeat(np.eye(4)[None],len(state),axis=0)
 r1=state[:,3:6].astype(float);r1/=np.linalg.norm(r1,axis=1,keepdims=True);r2=state[:,6:9].astype(float);r2-=np.sum(r1*r2,axis=1,keepdims=True)*r1;r2/=np.linalg.norm(r2,axis=1,keepdims=True);wr[:,:3,:3]=np.stack([r1,r2,np.cross(r1,r2)],axis=1);wr[:,:3,3]=state[:,:3]
 rel=np.linalg.inv(wr)@o
 for key,value in dict(hand_qpos_rad=state[:,9:],hand_target_rad=a[st:en,9:],recorded_state31=state,object_pose_base=o,wrist_pose_base=wr,object_pose_wrist=rel,source_state_frame_indices=np.arange(st,en+1)).items():arrays[key].append(value)
 axes=R.from_matrix(o[:,:3,:3]).apply([0,1,0]);ang=np.degrees(np.arccos(np.clip(axes[:,2],-1,1)))
 audit.append(dict(**c,source=str(ep),frames=len(s),object_rotation_orthogonality_max=float(abs(o[:,:3,:3]@o[:,:3,:3].transpose(0,2,1)-np.eye(3)).max()),object_rotation_det_range=[float(np.linalg.det(o[:,:3,:3]).min()),float(np.linalg.det(o[:,:3,:3]).max())],object_bottom_row_valid=bool(np.allclose(o[:,3],[0,0,0,1])),initial_position_wrist=rel[0,:3,3].tolist(),start_world_angle_deg=float(ang[0]),end_world_angle_deg=float(ang[-1]),source_hashes={f:hashlib.sha256((ep/f).read_bytes()).hexdigest() for f in ['state.npy','action.npy','obj_state.npy']},selection='front and wrist images visually confirmed lifted hand regrasp before socket contact; endpoint source images retained'))
np.savez_compressed(refdir/'reference.npz',**{k:np.stack(v) for k,v in arrays.items()})
names=json.loads((ROOT/'docs/experiment_reviews/20260924_horizontal_to_vertical/reference/initial_state.json').read_text())['hand_joint_names']
(refdir/'initial_state.json').write_text(json.dumps(dict(hand_joint_names=names,specs=specs),indent=2))
# Both current native evaluator hand and existing real replay hand use identical fixed palm mounting.
def mount(path):return ET.parse(path).getroot().find("joint[@name='base_joint']/origin").attrib
native=C/'maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf';real=B/'assets/robots/fr3_sharpa_real/hand.urdf';assert mount(native)==mount(real)
meshchecks={f:hashlib.sha256((C/'data/NOKOV-v3/object/mesh'/f).read_bytes()).hexdigest()==hashlib.sha256((B/'assets/source_objects/mesh'/f).read_bytes()).hexdigest() for f in ['bulb1.obj','bulb2.obj','bulb1_col.obj']}
(refdir/'verification.json').write_text(json.dumps(dict(episodes=audit,hand_mounts_equal=True,mount=mount(native),mesh_equal_to_existing_real_replay=meshchecks,coordinate_assumption='Existing dex_sim_bench reader treats matching31+16 format as robot-base xyz+row6D TCP and row-major4x4 object pose; use inv(T_base_TCP)@T_base_object. TCP=hand URDF root remains calibration assumption.',timing='No timestamps; nominal30Hz matches prior experiment',selection='episode51 f105-180 and episode53 f90-165; earlier candidates37/63 rejected because object-axis tracking inconsistent with visual/socket phase'),indent=2))
print(json.dumps(audit,indent=2))
