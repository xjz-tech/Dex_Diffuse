"""Geometry of reference windows, with no policy or simulator execution."""
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
a = np.load(P/'paired_commands.npz')['reference9'][4:]
r = np.load(ROOT/'docs/experiment_reviews/20260924_object_state_data/reference/reference.npz')['hand_target_rad'][1].astype(float)
e = np.empty((149,22)); e[::2] = r; e[1::2] = (r[:-1]+r[1:])/2
out = {}
for label,w in [('astra_left_31_windows',a),
                ('real_raw_34_windows',np.array([r[i:i+9] for i in range(0,len(r)-8,2)])),
                ('real_insert1_71_windows',np.array([e[i:i+9] for i in range(0,len(e)-8,2)]))]:
    d = np.diff(w,axis=1)
    length = np.linalg.norm(d,axis=2).sum(axis=1)
    valid = length>1e-8
    out[label] = dict(windows=len(w),
        mean_path_straightness=float(np.mean(np.linalg.norm(w[valid,-1]-w[valid,0],axis=1)/length[valid])),
        second_difference_rms_rad=float(np.sqrt(np.mean(np.diff(w,n=2,axis=1)**2))),
        first_to_ninth_joint_rms_rad=float(np.sqrt(np.mean((w[:,-1]-w[:,0])**2))))
(P/'reference_geometry.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
