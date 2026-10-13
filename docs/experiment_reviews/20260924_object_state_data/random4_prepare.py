"""Select on initial geometry only; freeze randomized order before controller evaluation."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from prepare_all_full_episodes import frame_data, MESH, TABLE_Z_M
P=Path(__file__).resolve().parent
OUT=P/'random4_ep53_standard_20260926'
OUT.mkdir(exist_ok=True)
state,_,_,_,rel,_=frame_data(53)
q0=state[90,9:]; p0=rel[90,:3,3]; r0=Rotation.from_matrix(rel[90,:3,:3])
v=np.array([[float(x) for x in l.split()[1:4]] for l in MESH.read_text().splitlines() if l.startswith('v ')])
v=v[ConvexHull(v).vertices]
records=[]
for ep in range(80):
 if ep==53:continue
 s,a,o,w,r,ang=frame_data(ep)
 clear=np.min(v@o[:,2,:3].T,axis=0)+o[:,2,3]-TABLE_Z_M
 qe=np.sqrt(((s[:,9:]-q0)**2).mean(-1)); pe=np.linalg.norm(r[:,:3,3]-p0,axis=-1)
 full_re=np.degrees((r0.inv()*Rotation.from_matrix(r[:,:3,:3])).magnitude())
 re=np.degrees(np.arccos(np.clip(r[:,:3,1]@rel[90,:3,1],-1,1)))
 valid=(ang>=70)&(ang<=110)&(clear>=.01)&(np.arange(len(s))<=len(s)-81)&(qe<=.45)&(pe<=.035)&(re<=25)
 if not valid.any():continue
 score=qe/.3+pe/.05+re/30;score[~valid]=np.inf
 start=int(score.argmin());end=len(s)-1
 folder=OUT/f'episode_{ep:02d}';folder.mkdir(exist_ok=True)
 np.savez_compressed(folder/'reference_full.npz',hand_qpos_rad=s[None,start:,9:],hand_target_rad=a[None,start:end,9:],recorded_state31=s[None,start:],object_pose_base=o[None,start:],wrist_pose_base=w[None,start:],object_pose_wrist=r[None,start:],source_state_frame_indices=np.arange(start,end+1)[None])
 records.append(dict(episode=ep,start=start,end=end,actions=end-start,q_rmse_rad=float(qe[start]),relative_position_error_m=float(pe[start]),long_axis_error_deg=float(re[start]),full_rotation_error_deg=float(full_re[start]),real_estimated_clearance_m=float(clear[start]),vertical_deg=float(ang[start])))
rng=np.random.default_rng(20260926)
order=rng.permutation(len(records));records=[records[i] for i in order]
manifest=dict(selection_design_note='An initial full-SO3 + qRMSE .25 filter admitted only one episode before any simulator screening; use directed long-axis similarity for nearly axisymmetric bulb and qRMSE .45; retain original full rotation, never rotate candidates to fit.',seed=20260926,template=dict(episode=53,frame=90),criterion='horizontal 70..110deg, estimated real mesh clearance >=1cm, >=80 actions remaining, qRMSE<=.45rad, relative position difference<=3.5cm, bulb directed longitudinal axis difference<=25deg (axial roll retained from source) vs ep53f90; best geometric score per episode; no fallback; random order before controller outcomes',static_gate='last 30 settle steps: airborne mesh clearance >5cm, translation range <5mm and rotation range <5deg; final drift no worse than ep53 + small numeric margin (3.5cm/30deg); hand geometry/contact and video inspection; accept first four passing in frozen random order',candidates=records)
(OUT/'selection_precommitted.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
