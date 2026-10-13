from pathlib import Path
import shutil
P=Path(__file__).resolve().parent;ROOT=P.parents[2];OLD=ROOT/'docs/experiment_reviews/20260924_horizontal_to_vertical/native_wrist_170g_mu22';s=(OLD/'sim.py').read_text()
s=s.replace("P.parent/'reference/", "P/'reference/")
a=s.index(" fit=json.loads");b=s.index(" os.chdir",a)
s=s[:a]+''' ref=np.load(P/'reference/reference.npz');specs=json.loads((P/'reference/initial_state.json').read_text())['specs']
 offsets=[[0,0,0],[.01,0,0],[-.01,0,0],[0,.01,0],[0,-.01,0],[0,0,.01],[0,0,-.01]];poses=[];cases=[]
 for ri,spec in enumerate(specs):
  T=ref['object_pose_wrist'][ri,0]
  for offset in offsets:
   for deg in [-10,0,10]:
    pose=dict(reference_id=ri,episode=spec['episode'],start=spec['start'],end=spec['end'],local_offset_m=offset,local_x_rotation_deg=deg,pose_id=len(poses),object_position_wrist=(T[:3,3]+offset).tolist(),object_quaternion_wrist_xyzw=(R.from_euler('x',deg,degrees=True)*R.from_matrix(T[:3,:3])).as_quat().tolist());poses.append(pose)
    for seed in [42,43,44]:cases.append(dict(pose,case_id=len(cases),noise_seed=seed))
 n=len(cases);assert n==126
''' +s[b:]
s=s.replace("lib=native._import_local_maniptrans(C)","lib=native._import_local_maniptrans(C)\n from astra_demo_cache import install_demo_cache\n install_demo_cache(P/'demo_cache')")
s=s.replace("q0=np.clip(ref['hand_qpos_rad'][0],cpu(lower),cpu(upper))", "q0=np.stack([np.clip(ref['hand_qpos_rad'][c['reference_id'],0],cpu(lower),cpu(upper)) for c in cases])")
pos=s.index("  # Keep every")
s=s[:pos]+'''  # Associate each recorded grasp with the closest native demonstration, using
  # only initialization q and wrist-relative position, never rollout outcome.
  demos=cpu(env.demo_data['opt_dof_pos']);lengths=cpu(env.demo_data['seq_len']).astype(int)
  dwr=R.from_rotvec(cpu(env.demo_data['opt_wrist_rot']).reshape(-1,3));dwp=cpu(env.demo_data['opt_wrist_pos']).reshape(-1,3);dob=cpu(env.demo_data['obj_trajectory']).reshape(-1,4,4)
  drel=dwr.inv().apply(dob[:,:3,3]-dwp).reshape(demos.shape[0],demos.shape[1],3);matches=[]
  for ri in range(len(specs)):
   qerr=np.sqrt(((demos-ref['hand_qpos_rad'][ri,0])**2).mean(-1));perr=np.linalg.norm(drel-ref['object_pose_wrist'][ri,0,:3,3],axis=-1);score=qerr/.3+perr/.05
   for di,le in enumerate(lengths):score[di,le:]=np.inf
   di,fr=np.unravel_index(score.argmin(),score.shape);matches.append(dict(demo_index=int(di),demo_frame=int(fr),initial_q_rmse_rad=float(qerr[di,fr]),initial_object_position_error_m=float(perr[di,fr])))
  for c in cases:c.update(matches[c['reference_id']])
  dump(out/'native_demo_matches.json',matches);dump(out/'cases.json',cases)
''' +s[pos:]
s=s.replace("step(np.repeat(q0[None],n,axis=0),'settle',j)","step(q0,'settle',j)")
s=s.replace("range(90)","range(75)").replace("'seeds':[c['noise_seed'] for c in cases]","'seeds':[c['noise_seed'] for c in cases],'reference_ids':[c['reference_id'] for c in cases]")
s=s.replace("native_goal_index=cpu(env.global_cur_idx)","native_goal_index=cpu(env.global_cur_idx),applied_forces=cpu(env.apply_forces),qd=cpu(env._qd),object_velocity=cpu(env._manip_obj_root_state[:,7:]),executed_target=cpu(env.curr_targets)")
s=s.replace("initialization='real qpos frame160;63 near bulb poses x3 native nuisance/noise instances',reference_frames=[160,250]", "initialization='Object_state_data 2 recorded grasps; each21near poses x3native nuisance/noise instances',reference_specs=specs,coordinate_assumption='robot-base object pose and TCP from xyz+row6D; TCP equals native hand root',control_steps=dict(settle=60,reference=75,hold=30)")
s=s.replace("j in [0,29,59,89]","j in [0,29,59,74]")
(P/'sim.py').write_text(s)
s=(OLD/'server.py').read_text().replace("assert 0<=j<len(reference)","assert 0<=j<reference.shape[1]").replace("len(reference)-1","reference.shape[1]-1").replace("target=np.repeat(reference[indices][None],batch,axis=0)","target=np.stack([reference[int(ri),indices] for ri in msg['reference_ids']])")
(P/'server.py').write_text(s)
shutil.copy(OLD/'case110_direct_vs_guided/astra_demo_cache.py',P/'astra_demo_cache.py')
s=(OLD/'run.py').read_text().replace("ROOT=P.parents[3]","ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')").replace("P.parent/'reference/reference.npz'","P/'reference/reference.npz'").replace("'/tmp/dex_native170_'","'/tmp/dex_objectstate_'")
s=s.replace("if len(sys.argv)>2:cmd+=['--record-ids',sys.argv[2]]","cmd+=['--record-ids',sys.argv[2] if len(sys.argv)>2 else ','.join(map(str,list(range(9))+list(range(63,72))))]")
(P/'run.py').write_text(s)
for f in ['sim.py','server.py','run.py']:compile((P/f).read_text(),str(P/f),'exec')
print('built scripts',P)
