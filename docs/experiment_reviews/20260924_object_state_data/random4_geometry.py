from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree, ConvexHull
import trimesh
P=Path(__file__).resolve().parent
OUT=P/'random4_ep53_standard_20260926'
HAND=Path('/home/carus/Program/dex-controller/maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf')
BULB=Path('/home/carus/Program/dex-controller/data/NOKOV-v3/object/mesh/bulb1_col.obj')
VIS=BULB.with_name('bulb2.obj')
def geometry():
 links={}
 for l in ET.parse(HAND).getroot().findall('link'):
  vv=[]
  for c in l.findall('collision'):
   m=c.find('geometry/mesh')
   if m is None:continue
   v=trimesh.load(HAND.parent/m.get('filename'),force='mesh',process=False).vertices.copy()
   v*=np.fromstring(m.get('scale','1 1 1'),sep=' ')
   o=c.find('origin')
   if o is not None:v=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).apply(v)+np.fromstring(o.get('xyz','0 0 0'),sep=' ')
   vv.append(v)
  if vv:links[l.get('name')]=np.concatenate(vv)
 return links,trimesh.load(BULB,force='mesh',process=False).vertices.copy(),trimesh.load(VIS,force='mesh',process=False).vertices.copy()
def metrics(folder,all_frames=False):
 s=json.load(open(folder/'summary.json'));t=json.load(open(folder/'trace.json'))
 ini=np.load(folder/'initial_state.npz');scale=float(ini['object_scale'].ravel()[0]);links,bv,vv=geometry();bv*=scale;vv*=scale;tree=cKDTree(bv)
 # Conservative sphere bounds skip pairs whose capped vertex distance is
 # certainly 3cm. This changes analysis cost, not its geometric definition.
 bcenter=bv.mean(0);bradius=float(np.linalg.norm(bv-bcenter,axis=1).max())
 bounds={name:(verts.mean(0),float(np.linalg.norm(verts-verts.mean(0),axis=1).max())) for name,verts in links.items()}
 bmin,bmax=bv.min(0)-.03,bv.max(0)+.03
 selected=t if all_frames else t[-30:]
 rows=[]
 for row in selected:
  ob=np.array(row['object_pose']);rot=Rotation.from_quat(ob[3:]);clear=float(rot.apply(vv)[:,2].min()+ob[2]+.385)
  object_center=rot.apply(bcenter)+ob[:3]
  near=[];dist=[];forces=np.linalg.norm(row['hand_contact_force'],axis=-1)
  for name,pose,force in zip(s['hand_body_names'],row['hand_body_pose'],forces):
   if name not in links:continue
   pose=np.array(pose);linkrot=Rotation.from_quat(pose[3:]);center,radius=bounds[name]
   if np.linalg.norm(linkrot.apply(center)+pose[:3]-object_center)-radius-bradius>.03:
    dist.append(.03)
    continue
   world=linkrot.apply(links[name])+pose[:3]
   local=rot.inv().apply(world-ob[:3]);mask=np.all((local>=bmin)&(local<=bmax),axis=1)
   d=min(.03,float(tree.query(local[mask],workers=1)[0].min())) if mask.any() else .03;dist.append(d)
   if d<.008 and force>.05:near.append(name)
  rows.append(dict(phase=row['phase'],index=row['index'],mesh_vertex_gap_m=min(dist),near_contact_links=near,near_contact_link_count=len(near),mesh_table_clearance_m=clear))
 poses=np.array([r['object_pose'] for r in t[-30:]]);trans=float(np.linalg.norm(poses[:,:3]-poses[0,:3],axis=1).max());rots=Rotation.from_quat(poses[:,3:]);angle=float(np.degrees((rots[0].inv()*rots).magnitude()).max())
 stat=dict(last30_translation_range_m=trans,last30_rotation_range_deg=angle,final_drift_m=t[59]['displacement_from_import_m'],final_rotation_deg=t[59]['rotation_from_import_deg'])
 if not all_frames:
  stat.update(min_airborne_clearance_m=min(r['mesh_table_clearance_m'] for r in rows),minimum_near_contact_links=min(r['near_contact_link_count'] for r in rows),max_hand_mesh_gap_m=max(r['mesh_vertex_gap_m'] for r in rows))
  stat['numeric_pass']=bool(trans<.005 and angle<5 and stat['final_drift_m']<.035 and stat['final_rotation_deg']<30 and stat['min_airborne_clearance_m']>.05 and stat['minimum_near_contact_links']>=2 and stat['max_hand_mesh_gap_m']<.008)
 result=dict(static=stat,frames=rows,method='minimum distance between all URDF collision vertices and bulb collision vertices, capped at 3cm using an exact AABB lower-bound exclusion; supporting link requires <8mm vertex gap and net contact force >0.05N. Net contact force is not pair-specific. Table clearance uses visual mesh and exact table top z=-.385. Independent video verification required.')
 (folder/'grasp_metrics.json').write_text(json.dumps(result,indent=2))
 return result
if __name__=='__main__':
 import sys
 for ep in map(int,sys.argv[1:]):
  f=OUT/f'episode_{ep:02d}'/'static'
  if (f/'summary.json').exists():print(ep,metrics(f)['static'],flush=True)
