from pathlib import Path
import json
import numpy as np
from scipy.spatial import ConvexHull
from prepare_all_full_episodes import frame_data,MESH,TABLE_Z_M
P=Path(__file__).resolve().parent;OUT=P/'reference_turn_baseline_20260926';OUT.mkdir(exist_ok=True)
v=np.asarray([[float(x) for x in l.split()[1:4]] for l in MESH.read_text().splitlines() if l.startswith('v ')]);v=v[ConvexHull(v).vertices]
records=[]
for ep,old in [(50,111),(76,94),(70,73),(40,93)]:
 s,a,o,w,r,angle=frame_data(ep);clear=(v@o[:,2,:3].T).min(0)+o[:,2,3]-TABLE_Z_M
 valid=np.where((angle>=70)&(angle<=110)&(clear>.01)&(np.arange(len(s))>=old)&(np.arange(len(s))<=old+65))[0]
 chosen=sorted(set([int(valid[np.argmin(abs(valid-target))]) for target in [old,old+5,old+10,old+20,old+30,old+45,old+60]]))
 for start in chosen:
  f=OUT/f'ep{ep:02d}_f{start:04d}';f.mkdir(exist_ok=True);end=len(s)-1
  np.savez_compressed(f/'reference_full.npz',hand_qpos_rad=s[None,start:,9:],hand_target_rad=a[None,start:end,9:],recorded_state31=s[None,start:],object_pose_base=o[None,start:],wrist_pose_base=w[None,start:],object_pose_wrist=r[None,start:],source_state_frame_indices=np.arange(start,end+1)[None])
  records.append(dict(episode=ep,start=start,folder=str(f),real_angle=float(angle[start]),real_estimated_clearance_m=float(clear[start])))
(OUT/'candidates.json').write_text(json.dumps(dict(purpose='user requires raw reference completes turn before guidance comparison; same four episodes, source states unchanged; stage1 only vary recorded horizontal start frame',direct_acceptance='settled long axis 65..115 degrees from world +Z, object airborne and in hand; action achieves <=30 degrees for >=30 consecutive control steps while airborne with object contact and no hand separation; verify geometry and video; fixed raw actions at30Hz, no interpolation',candidates=records),indent=2))
print([(r['episode'],r['start']) for r in records])
