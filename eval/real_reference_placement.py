"""Controlled object-property overrides and static-only placement search."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

def restore_domains(env, archive, placements, mass, friction):
    """Repeat the three archived native domains; override bulb mass/friction only."""
    import torch
    from isaacgym import gymapi
    source=np.load(archive,allow_pickle=False)
    for i,(ep,case) in enumerate(zip(env.envs,placements)):
        d=int(case['domain_index']);g=env.gym
        hand=g.find_actor_handle(ep,'dexhand');obj=g.find_actor_handle(ep,'manip_obj')
        props=g.get_actor_rigid_body_properties(ep,hand)
        for j,p in enumerate(props):p.mass=float(source['hand_mass'][d,j])
        # Match native mass randomization: recompute inertia for restored masses.
        g.set_actor_rigid_body_properties(ep,hand,props,True)
        shapes=g.get_actor_rigid_shape_properties(ep,hand)
        for j,p in enumerate(shapes):
            for key in ['friction','rolling_friction','torsion_friction','restitution']:
                setattr(p,key,float(source['hand_'+key][d,j]))
        g.set_actor_rigid_shape_properties(ep,hand,shapes)
        dofs=g.get_actor_dof_properties(ep,hand)
        dofs['stiffness']=source['hand_dof_stiffness'][d]
        dofs['damping']=source['hand_dof_damping'][d]
        g.set_actor_dof_properties(ep,hand,dofs)
        g.set_actor_scale(ep,obj,float(source['object_scale'][d]))
        shapes=g.get_actor_rigid_shape_properties(ep,obj)
        for j,p in enumerate(shapes):
            for key in ['friction','rolling_friction','torsion_friction','restitution']:
                setattr(p,key,float(source['object_'+key][d,j]))
            if friction is not None:p.friction=float(friction)
        g.set_actor_rigid_shape_properties(ep,obj,shapes)
        props=g.get_actor_rigid_body_properties(ep,obj)
        assert len(props)==1
        props[0].mass=float(mass if mass is not None else source['object_mass'][d])
        props[0].com=gymapi.Vec3(*source['object_com'][d].tolist())
        g.set_actor_rigid_body_properties(ep,obj,props,True)
        actual=g.get_actor_rigid_body_properties(ep,obj)[0].mass
        assert abs(actual-props[0].mass)<1e-6
        env.manip_obj_mass[i]=actual
        env.random_force_prob[i]=float(source['random_force_probability'][d])
        case['archived_wrist_pose']=source['wrist'][d,:7].tolist()
    # Synchronize CPU-side PhysX property updates before final tensor placement.
    env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True)
    env._refresh()
    env._base_state[:,:7]=torch.tensor([p['archived_wrist_pose'] for p in placements],device=env.device,dtype=torch.float32)
    env.apply_forces.zero_();env.apply_torque.zero_()

def generate(root, base_file):
    from scipy.spatial.transform import Rotation as R
    base=json.loads(Path(base_file).read_text())['selected_placement']
    cases=[];specs=[]
    shifts=list(itertools.product([-.012,-.006,0,.006,.012],repeat=3))
    rotations=[np.zeros(3)]+[np.eye(3)[axis]*angle for axis in range(3) for angle in [-12,12]]
    for shift in shifts:
        for rot in rotations:
            pose_id=len(specs)
            case=dict(base)
            case.update(pose_id=pose_id,search_translation_wrist_m=list(shift),search_rotation_wrist_deg=rot.tolist())
            case['object_position_wrist']=(np.array(base['object_position_wrist'])+shift).tolist()
            case['object_quaternion_wrist_xyzw']=(R.from_rotvec(np.deg2rad(rot))*R.from_quat(base['object_quaternion_wrist_xyzw'])).as_quat().tolist()
            specs.append(case)
            for domain in range(3):cases.append(dict(case,domain_index=domain))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    (root/'search_cases.json').write_text(json.dumps({'placements':cases,'unique_pose_count':len(specs),'domains_per_pose':3,'search':'5^3 position grid, 7 orientations, static hold only'},indent=2)+'\n')
    (root/'search_protocol.json').write_text(json.dumps(dict(
        mass_kg=.150,object_friction=2.2,hand_friction='preserve archived native values',
        pose_count=len(specs),case_count=len(cases),hold_steps=90,control_hz=30,
        ranking='all 3 domains survive; persistent >=3 contacting fingertips; low late position drift and joint error; no reference actions used',
        reference_control='10B EMA, DDIM4, exec2, guidance first2, scale25; same 75 recorded actions',
        final_selection='up to 3 diverse statically ranked poses plus old-placement control, each in the same 3 archived domains',
        base_placement=base,position_offsets_m=[-.012,-.006,0,.006,.012],rotation_offsets_deg=[-12,0,12],
    ),indent=2)+'\n')
    print('generated',len(specs),'poses,',len(cases),'static cases')

def rank(root):
    from scipy.spatial.transform import Rotation as R
    root=Path(root);fit=root/'search';t=np.load(fit/'trajectory.npz',allow_pickle=False)
    placements=json.loads((root/'search_cases.json').read_text())['placements']
    init=np.load(fit/'initial_state.npz',allow_pickle=False)
    assert np.allclose(init['object_mass'],.15,atol=1e-7)
    assert np.allclose(init['object_friction'],2.2,atol=1e-6)
    assert np.allclose(init['cached_object_mass'],.15,atol=1e-7)
    rel=t['relative_position'];q=t['q'];command=t['command']
    failed=t['failure'].any(0)
    contacts=(t['contact_force_norm']>.05).sum(-1)
    # The last second measures settled stability, separate from initial accommodation.
    late=rel[-30:];late_motion=np.linalg.norm(np.diff(late,axis=0),axis=-1).sum(0)
    spread=np.linalg.norm(late-late.mean(0),axis=-1).max(0)
    qrmse=np.sqrt(((q[-30:]-command[-30:])**2).mean((0,2)))
    contact3=(contacts[-30:]>=3).mean(0)
    median_contact=np.median(contacts[-30:],axis=0)
    pose_rows=[]
    for start in range(0,len(placements),3):
        sl=slice(start,start+3)
        score=float(100*failed[sl].sum()+.05*(3-contact3[sl].sum())+late_motion[sl].max()+2*spread[sl].max()+.1*qrmse[sl].max())
        pose_rows.append(dict(pose_id=placements[start]['pose_id'],case_indices=list(range(start,start+3)),
            placement=placements[start],native_failures=int(failed[sl].sum()),
            minimum_three_tip_contact_fraction=float(contact3[sl].min()),
            median_tip_counts=median_contact[sl].tolist(),late_motion_m=late_motion[sl].tolist(),
            late_position_spread_m=spread[sl].tolist(),q_rmse_rad=qrmse[sl].tolist(),static_score=score))
    pose_rows.sort(key=lambda r:r['static_score'])
    selected=[]
    for row in pose_rows:
        if row['native_failures']:continue
        p=row['placement'];rp=R.from_quat(p['object_quaternion_wrist_xyzw'])
        diverse=True
        for old in selected:
            op=old['placement'];distance=np.linalg.norm(np.array(p['object_position_wrist'])-op['object_position_wrist'])
            angle=np.degrees((rp.inv()*R.from_quat(op['object_quaternion_wrist_xyzw'])).magnitude())
            if distance<.005 and angle<8:diverse=False
        if diverse:selected.append(row)
        if len(selected)==3:break
    baseline=next(r for r in pose_rows if np.allclose(r['placement']['search_translation_wrist_m'],0) and np.allclose(r['placement']['search_rotation_wrist_deg'],0))
    out=[]
    for label,row in [('old_pose',baseline)]+[('new_pose_%d'%(i+1),row) for i,row in enumerate(selected)]:
        for d in range(3):out.append(dict(row['placement'],domain_index=d,group=label))
    result=dict(selected=selected,old_pose=baseline,total_poses=len(pose_rows),
        robust_no_failure_poses=sum(r['native_failures']==0 for r in pose_rows),ranked=pose_rows)
    (root/'static_ranking.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'evaluation_cases.json').write_text(json.dumps({'placements':out},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='ranked'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','rank']);p.add_argument('root',type=Path)
    p.add_argument('--base',type=Path,default=Path('/home/carus/Data/bulb_right_turn_reference_260909/sim_test_20260924/fit/placement.json'))
    a=p.parse_args()
    if a.action=='generate':generate(a.root,a.base)
    else:rank(a.root)
