from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926'
mesh=Path('/home/carus/Program/dex-controller/data/NOKOV-v3/object/mesh/bulb2.obj')
v=np.array([[float(x) for x in l.split()[1:4]] for l in mesh.read_text().splitlines() if l.startswith('v ')])
def check(folder):
 s=json.load(open(folder/'summary.json'));t=json.load(open(folder/'trace.json'));scale=float(np.load(folder/'initial_state.npz')['object_scale'].ravel()[0]);settle=t[59]
 static_ok=65<=settle['vertical_error_deg']<=115 and settle['displacement_from_import_m']<.035 and settle['rotation_from_import_deg']<30
 blocks=[];cur=[]
 for row in t:
  if row['phase']!='action':continue
  ob=np.array(row['object_pose']);clear=float(Rotation.from_quat(ob[3:]).apply(v*scale)[:,2].min()+ob[2]+.385)
  force=np.linalg.norm(row['object_contact_force']);bodies=np.array(row['hand_body_pose']);cf=np.linalg.norm(row['hand_contact_force'],axis=-1)
  near=np.count_nonzero((np.linalg.norm(bodies[:,:3]-ob[:3],axis=-1)<.12)&(cf>.05))
  good=row['vertical_error_deg']<=30 and clear>.08 and force>.1 and near>=2 and np.linalg.norm(np.array(row['relative_position'])-settle['relative_position'])<.08
  if good:cur.append(row['index'])
  else:
   if cur:blocks.append(cur)
   cur=[]
 if cur:blocks.append(cur)
 best=max(blocks,key=len,default=[])
 out=dict(episode=s['source_episode'],start=s['source_start_frame'],settle_angle=settle['vertical_error_deg'],settle_drift_cm=settle['displacement_from_import_m']*100,settle_rotation_deg=settle['rotation_from_import_deg'],static_ok=bool(static_ok),longest_vertical_contact_steps=len(best),block=[best[0]+1,best[-1]+1] if best else None,preliminary_pass=bool(static_ok and len(best)>=30),needs_mesh_and_video_verification=True)
 (folder/'turn_screen.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 out=[]
 for f in sorted(R.glob('ep*/direct_probe/summary.json')):
  row=check(f.parent);out.append(row);print(row)
 (R/'screen_results.json').write_text(json.dumps(out,indent=2))
