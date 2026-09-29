import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,pickle,ast
import numpy as np,torch,pytorch_kinematics as pk
from scipy.spatial import cKDTree
OUT=Path(__file__).resolve().parent;ROOT=Path('/home/carus/Program/dex-controller');torch.set_num_threads(1);torch.set_num_interop_threads(1)
mod=ast.parse((OUT.parents[2]/'eval/real/hardware.py').read_text())
for node in mod.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY_SHARPA_DOF_NAMES' for t in node.targets):names=ast.literal_eval(node.value)
urdf=ROOT/'maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf';chain=pk.build_chain_from_urdf(urdf.read_bytes()).to(dtype=torch.float32,device='cpu');order=[names.index(n) for n in chain.get_joint_parameter_names()]
links=['right_thumb_fingertip','right_index_fingertip','right_middle_fingertip','right_ring_fingertip','right_pinky_fingertip']
def fk(q):
 out=[]
 with torch.no_grad():
  for start in range(0,len(q),2048):
   poses=chain.forward_kinematics(torch.tensor(q[start:start+2048,order],dtype=torch.float32));out.append(torch.stack([poses[n].get_matrix()[:,:3,3] for n in links],dim=1).numpy())
 return np.concatenate(out)
ref=[];rid=[];frame=[]
for i in range(150):
 with (ROOT/f'data/retargeting/NOKOV-v3/mano2sharpa_rh/bulb2/{i:03d}.pkl').open('rb') as f:d=pickle.load(f)
 q=np.asarray(d['opt_dof_pos'])[::2];ref.append(q);rid.append(np.full(len(q),i));frame.append(np.arange(len(q)))
ref=np.concatenate(ref);rid=np.concatenate(rid);frame=np.concatenate(frame);real=np.load(OUT/'real_states.npz');rt=fk(real['qpos']);st=fk(ref);print('FK done',rt.shape,st.shape,flush=True)
tree=cKDTree(st.reshape(-1,15));ds,idx=tree.query(rt.reshape(-1,15),workers=1);rms=ds/np.sqrt(5)*100;maxerr=np.linalg.norm(rt-st[idx],axis=2).max(1)*100
fold=np.random.default_rng(20260912).permutation(150)%5;cal=[]
for k in range(5):
 train=fold[rid]!=k;query=(fold[rid]==k)&(frame%10==0);dist,_=cKDTree(st[train].reshape(-1,15)).query(st[query].reshape(-1,15),workers=1);cal.extend(dist/np.sqrt(5)*100)
cutoffs=np.percentile(cal,[50,90,95,99]);contact=real['contact_count'].sum(1)>0;groups=[]
for name,mask in [('all',np.ones(len(rms),bool)),('tactile_any',contact),('tactile_two_or_more',(real['contact_count']>0).sum(1)>=2)]:
 groups.append(dict(group=name,frames=int(mask.sum()),rms_cm_percentiles=np.percentile(rms[mask],[10,25,50,75,90,95]).tolist(),threshold_coverage={str(t):float((rms[mask]<=t).mean()) for t in [.5,1,1.5,2,3]},calibrated95_coverage=float((rms[mask]<=cutoffs[2]).mean()),all_five_within_2cm=float((maxerr[mask]<=2).mean()),all_five_within_3_6cm=float((maxerr[mask]<=3.6).mean())))
np.savez_compressed(OUT/'fingertips.npz',real_tips=rt,reference_tips=st,rms_cm=rms,max_cm=maxerr,nearest_ref_index=rid[idx],nearest_ref_frame=frame[idx],nearest_ref_qpos=ref[idx],calibration_cm=np.array(cal))
(OUT/'fingertip_coverage.json').write_text(json.dumps(dict(metric='Root mean squared 3D distance across five fingertip link origins, same URDF in wrist frame; nearest by joint fingertip position vector; not contact/object coverage',calibration_p50_p90_p95_p99_cm=cutoffs.tolist(),groups=groups,urdf=str(urdf),policy_joint_names=names),indent=2));print(json.dumps(dict(calibration=cutoffs.tolist(),groups=groups),indent=2),flush=True)
