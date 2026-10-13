"""Audit paired trajectories, reference/command/torque perturbations and rotation."""
import json,sys,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sys.path.insert(0,str(P/'source/eval'))
from astra_kinematic_step import point_jacobian
def rms(x):return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def read(run):return [json.loads(s) for s in (run/'trajectory.jsonl').read_text().splitlines()]
runs={}
for f in sorted((P/'runs').glob('native_*/trial_summary.json')):
    s=json.loads(f.read_text());runs[(s['noise_seed'],s['lambda_direction'])]=(f.parent,s,read(f.parent))
records=[]
expected={'failure_obj_pos_thres_m':.05,'failure_tip_pos_thres_m':.1,'failure_obj_rot_thres_deg':180.,'invalid_obj_pos_thres_m':.15,'failure_tolerance_scale':10000.,'fixed_tolerance_steps':20000,'reset_on_reach_goal':False,'cross_trajectory_goal_prob':.3,'traj_steps_limit':12000}
source=ROOT/'.worktrees/Astra-controller/outputs/astra_gait_pause9/no_gait_e3577_n48'
original=read(source)
for (noise,lam),(run,s,rows) in sorted(runs.items()):
    m=json.loads((run/'manifest.json').read_text());model=json.loads((run/'model.json').read_text())
    assert m['protocol']==expected and m['data_indices']=='000-149'
    assert (model['prior_ddim_steps'],model['guidance_steps'],model['guidance_scale'])==(4,9,25)
    axis=np.asarray(m['axis_world']);q=np.asarray([r['state']['object_xyzw'] for r in rows])
    tilt=np.rad2deg(np.arccos(np.clip(Rotation.from_quat(q).apply([0,1,0])@axis,-1,1)))
    pos=np.array([r['state']['object_position'] for r in rows])
    initial_pos=np.asarray(rows[7]['state']['object_position'])
    s=dict(s,max_axis_tilt_deg=float(tilt.max()),max_position_from_step8_m=float(np.linalg.norm(pos-initial_pos,axis=1).max()),
        net_direction='failure' if s['native_failure'] else ('left' if s['twist_after_initial8_deg']>5 else 'right' if s['twist_after_initial8_deg']< -5 else 'small'),
        last1s_direction='left' if s['final_1s_net_deg']>1 else 'right' if s['final_1s_net_deg']< -1 else 'small')
    contacts=np.array([np.linalg.norm(r['state']['object_net_contact_force_world']) for r in rows])
    s['object_net_contact_force_norm_min_after8_N']=float(contacts[8:].min())
    s['object_net_contact_force_norm_final_N']=float(contacts[-1])
    # Angular velocities are checked, not silently used to infer torque.
    w=np.asarray([r['state']['object_velocity'][3:] for r in rows])
    fd=(Rotation.from_quat(q[1:])*Rotation.from_quat(q[:-1]).inv()).as_rotvec()/m['control_dt']
    s['root_omega_vs_pose_difference_correlation']=float(np.corrcoef(((w[1:]+w[:-1])/2).ravel(),fd.ravel())[0,1])
    if (noise,0.) in runs:
        base,_,br=runs[(noise,0.)]
        prefix_error=max(float(np.max(np.abs(np.asarray(r['state'][key])-np.asarray(b['state'][key])))) for r,b in zip(rows[:8],br[:8]) for key in ['qpos','qvel','object_position','object_xyzw','target_before','object_velocity'])
        assert prefix_error==0
        s['paired_first8_state_max_error']=prefix_error
        d=np.load(run/'prediction_000008.npz');d0=np.load(base/'prediction_000008.npz')
        np.testing.assert_array_equal(d['observation'],d0['observation'])
        da=(d['prediction']-d0['prediction'])[0]
        dr=(d['reference']-d0['reference'])[0]
        req=json.loads((run/'request_0002.json').read_text());kp=np.asarray(rows[7]['state']['dof_stiffness_nm_per_rad'])
        kd=np.asarray(rows[7]['state']['dof_damping_nm_s_per_rad']);eff=np.asarray(rows[7]['state']['dof_effort_limit_nm'])
        qbefore=np.asarray(req['state']['qpos']);vbefore=np.asarray(req['state']['qvel'])
        nominal=kp*(d['prediction'][0,0]-qbefore)-kd*vbefore
        nominal0=kp*(d0['prediction'][0,0]-qbefore)-kd*vbefore
        tip_deltas=[]
        actual_tip_deltas=[]
        for body in req['state']['contact_body_names']:
            tip=body.replace('_elastomer','_fingertip')
            _,jac=point_jacobian(req['state'],req['joint_names'],tip)
            tip_deltas.append(np.linalg.norm(jac@da[0])*1000)
            actual_tip_deltas.append(1000*np.linalg.norm(np.asarray(rows[8]['state']['body_pose_world'][tip][:3])-np.asarray(br[8]['state']['body_pose_world'][tip][:3])))
        s.update(first_reference_prefix2_delta_rms_rad=rms(dr[:2]),first_reference9_delta_rms_rad=rms(dr),
            first_command_prefix2_delta_rms_rad=rms(da),first_command_action1_delta_rms_rad=rms(da[0]),
            first_command_action1_max_abs_delta_rad=float(np.abs(da[0]).max()),
            first_command_tip_linearized_mean_delta_mm=float(np.mean(tip_deltas)),
            first_command_tip_linearized_max_delta_mm=float(np.max(tip_deltas)),
            step9_measured_tip_position_mean_delta_mm=float(np.mean(actual_tip_deltas)),
            step9_measured_tip_position_max_delta_mm=float(np.max(actual_tip_deltas)),
            first_command_delta_P_term_rms_Nm=rms(kp*da[0]),first_command_delta_P_term_max_abs_Nm=float(np.max(np.abs(kp*da[0]))),
            nominal_PD_above_effort_limit_count=int(np.sum((np.abs(nominal)>eff)|(np.abs(nominal0)>eff))),
            step9_measured_joint_total_torque_delta_rms_Nm=rms(np.asarray(rows[8]['state']['joint_total_torque_nm'])-br[8]['state']['joint_total_torque_nm']),
            step9_object_net_contact_force_delta_norm_N=float(np.linalg.norm(np.asarray(rows[8]['state']['object_net_contact_force_world'])-br[8]['state']['object_net_contact_force_world'])),
            step9_twist_difference_deg=float(rows[8]['twist_degrees']-br[8]['twist_degrees']))
    if noise==48 and lam==1:
        s['historical_reproduction_max_error']=max(float(np.max(np.abs(np.asarray(r['state'][key])-np.asarray(b['state'][key])))) for r,b in zip(rows,original) for key in ['qpos','qvel','object_position','object_xyzw','target_before','object_velocity'])
        assert s['historical_reproduction_max_error']==0
    records.append(s)
(P/'analysis.json').write_text(json.dumps(records,indent=2)+'\n')
with (P/'summary.csv').open('w') as f:
    fields=sorted(set().union(*(r.keys() for r in records)));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(records)
print('n lambda steps fail net_deg tail_deg perturb_rad delta_P_Nm')
for r in records:print(r['noise_seed'],r['lambda_direction'],r['steps'],r['native_failure'],round(r['twist_after_initial8_deg'],3),round(r['final_1s_net_deg'],3),r.get('first_command_action1_delta_rms_rad'),r.get('first_command_delta_P_term_rms_Nm'))
