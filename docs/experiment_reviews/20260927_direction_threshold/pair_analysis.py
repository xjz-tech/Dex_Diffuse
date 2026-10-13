"""Measure predefined illustrative opposite-direction pairs from saved native runs."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
rows=json.loads((P/'analysis.json').read_text())
idx={(r['noise_seed'],r['lambda_direction']):r for r in rows}
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
out=[]
for noise,ls in [(49,[.125,-.125]),(48,[-.5,-.625]),(50,[0,-.5])]:
    runs=[Path(idx[noise,l]['video']).parent for l in ls]
    ts=[[json.loads(s) for s in (r/'trajectory.jsonl').read_text().splitlines()] for r in runs]
    ds=[np.load(r/'prediction_000008.npz') for r in runs]
    np.testing.assert_array_equal(ds[0]['observation'],ds[1]['observation'])
    da=ds[0]['prediction'][0,0].astype(float)-ds[1]['prediction'][0,0]
    states=[t[8]['state'] for t in ts]
    kp=np.asarray(ts[0][7]['state']['dof_stiffness_nm_per_rad'])
    req=json.loads((runs[0]/'request_0002.json').read_text())
    tips=[b.replace('_elastomer','_fingertip') for b in req['state']['contact_body_names']]
    all_da=np.asarray([r['applied_target'] for r in ts[0]])[8:]-np.asarray([r['applied_target'] for r in ts[1]])[8:]
    out.append(dict(noise=noise,lambdas=ls,net_deg=[idx[noise,l]['twist_after_initial8_deg'] for l in ls],
        reference_prefix2_difference_rms_rad=rms(ds[0]['reference'][0,:2].astype(float)-ds[1]['reference'][0,:2]),
        command_first_difference_rms_rad=rms(da),command_first_difference_rms_deg=float(np.rad2deg(rms(da))),
        P_term_difference_rms_Nm=rms(kp*da),
        measured_joint_total_torque_difference_rms_Nm=rms(np.asarray(states[0]['joint_total_torque_nm'])-states[1]['joint_total_torque_nm']),
        object_net_contact_force_difference_N=float(np.linalg.norm(np.asarray(states[0]['object_net_contact_force_world'])-states[1]['object_net_contact_force_world'])),
        measured_tip_difference_mean_mm=float(np.mean([1000*np.linalg.norm(np.asarray(states[0]['body_pose_world'][b][:3])-states[1]['body_pose_world'][b][:3]) for b in tips])),
        full_280steps_command_difference_rms_rad=rms(all_da),
        full_280steps_max_perstep_command_difference_rms_rad=float(np.sqrt(np.mean(all_da**2,axis=-1)).max())))
(P/'direction_flip_pairs.json').write_text(json.dumps(out,indent=2)+'\n')
print('pairs',len(out))
