"""Native-wrist 170g/mu2.2 reference-guided pose sweep; no task code changes."""
from pathlib import Path
import os,sys,json,argparse,random
import numpy as np
P=Path(__file__).resolve().parent
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
C=Path('/home/carus/Program/dex-controller')
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--socket',required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--record-ids',default='3,4,5,66,67,68,129,130,131');args=a.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True)
 from isaacgym import gymapi,gymtorch
 import torch,cv2
 from scipy.spatial.transform import Rotation as R
 from omegaconf import OmegaConf
 sys.path.insert(0,str(ROOT/'eval'));import sim_eval as native
 from ipc import connect_unix,send_message,recv_message
 torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.manual_seed_all(42);np.random.seed(42);random.seed(42)
 ref=np.load(P/'reference/reference.npz');specs=json.loads((P/'reference/initial_state.json').read_text())['specs']
 offsets=[[0,0,0],[.01,0,0],[-.01,0,0],[0,.01,0],[0,-.01,0],[0,0,.01],[0,0,-.01]];poses=[];cases=[]
 for ri,spec in enumerate(specs):
  T=ref['object_pose_wrist'][ri,0]
  for offset in offsets:
   for deg in [-10,0,10]:
    pose=dict(reference_id=ri,episode=spec['episode'],start=spec['start'],end=spec['end'],local_offset_m=offset,local_x_rotation_deg=deg,pose_id=len(poses),object_position_wrist=(T[:3,3]+offset).tolist(),object_quaternion_wrist_xyzw=(R.from_euler('x',deg,degrees=True)*R.from_matrix(T[:3,:3])).as_quat().tolist());poses.append(pose)
    for seed in [42,43,44]:cases.append(dict(pose,case_id=len(cases),noise_seed=seed))
 n=len(cases);assert n==126
 os.chdir(str(C));lib=native._import_local_maniptrans(C)
 from astra_demo_cache import install_demo_cache
 install_demo_cache(P/'demo_cache')
 resets=dict(failureObjPosThres=.05,failureThumbTipPosThres=.1,failureIndexTipPosThres=.1,failureMiddleTipPosThres=.1,failurePinkyTipPosThres=.1,failureRingTipPosThres=.1,failureObjRotThres=180.,invalidObjPosThres=.15,FailureToleranceScale=10000.,fixedToleranceSteps=20000,trajStepsLimit=12000,resetOnReachGoal=False,enableCrossTrajectoryReset=True,crossTrajectoryGoalProb=.3)
 cfg=native._make_task_config(Path('/home/carus/Data/exp_data/hydra_config.yaml'),n,['v3:bulb2@%03d'%i for i in range(150)],C/'data/NOKOV-v3',C/'data/retargeting/NOKOV-v3',resets);cfg.env.enableCameraSensors=True
 asset=native._install_sharpa_asset_override(native._resolve_sharpa_urdf(C/'maniptrans_envs/assets/sharpa_hand'));env=None;sock=None;writers={};cams={}
 try:
  env=lib.make(sim_device='cuda:0',rl_device='cuda:0',graphics_device_id=0,multi_gpu=False,cfg=cfg,display=False,record=False,has_headless_arg=True,headless=True);native._restore_sharpa_asset_override(asset);asset=None
  env.compute_observations();env.reset();env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True);env.compute_observations();g=env.gym;s=env.sim
  def cpu(x):return x.detach().cpu().numpy().copy()
  lower,upper=native._validate_environment(env,native._load_manifest(Path('/home/carus/Data/exp_data/hydra_config.yaml')))
  ref=np.load(P/'reference/reference.npz');q0=np.stack([np.clip(ref['hand_qpos_rad'][c['reference_id'],0],cpu(lower),cpu(upper)) for c in cases]);names=json.loads((P/'reference/initial_state.json').read_text())['hand_joint_names'];assert names==list(env.dexhand.dof_names)
  # Associate each recorded grasp with the closest native demonstration, using
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
  # Keep every native wrist and nuisance realization exactly as reset.
  native_wrist=cpu(env._base_state);np.save(out/'native_wrist_before_pairing.npy',native_wrist)
  for i,c in enumerate(cases):
   en=env.envs[i];h=g.find_actor_handle(en,'dexhand');o=g.find_actor_handle(en,'manip_obj')
   hs=g.get_actor_rigid_shape_properties(en,h)
   for z in hs:z.friction=2.2
   g.set_actor_rigid_shape_properties(en,h,hs)
   sp=g.get_actor_rigid_shape_properties(en,o)
   for z in sp:z.friction=2.2
   g.set_actor_rigid_shape_properties(en,o,sp)
   bp=g.get_actor_rigid_body_properties(en,o);bp[0].mass=.17;g.set_actor_rigid_body_properties(en,o,bp,True)
   assert abs(g.get_actor_rigid_body_properties(en,o)[0].mass-.17)<1e-6
   assert all(abs(z.friction-2.2)<1e-6 for z in g.get_actor_rigid_shape_properties(en,o)+g.get_actor_rigid_shape_properties(en,h))
  env.manip_obj_mass[:]=.17
  base=cpu(env._base_state[:,:7]);br=R.from_quat(base[:,3:]);op=br.apply([c['object_position_wrist'] for c in cases])+base[:,:3];oq=(br*R.from_quat([c['object_quaternion_wrist_xyzw'] for c in cases])).as_quat()
  env._q[:]=torch.tensor(q0,device=env.device);env._qd.zero_();env.curr_targets[:]=env._q;env.prev_targets[:]=env._q;env._pos_control[:]=env._q
  env._manip_obj_root_state[:,:3]=torch.tensor(op,device=env.device);env._manip_obj_root_state[:,3:7]=torch.tensor(oq,device=env.device);env._manip_obj_root_state[:,7:]=0
  env.envidx_to_demoidx[:]=torch.tensor([c['demo_index'] for c in cases],device=env.device)
  for k in ['global_cur_idx','progress_buf']:getattr(env,k)[:]=torch.tensor([c['demo_frame'] for c in cases],device=env.device)
  for k in ['failure_progress_buf','reset_buf','failure_buf','success_buf','running_progress_buf','stable_frames_buf','traj_steps_counter','is_target_cross','error_buf']:getattr(env,k).zero_()
  hi=env._global_dexhand_indices.flatten();oi=env._global_manip_obj_indices.flatten();ids=torch.cat([hi,oi]);g.set_dof_state_tensor_indexed(s,gymtorch.unwrap_tensor(env._dof_state),gymtorch.unwrap_tensor(hi),len(hi));g.set_actor_root_state_tensor_indexed(s,gymtorch.unwrap_tensor(env._root_state),gymtorch.unwrap_tensor(ids),len(ids));g.set_dof_position_target_tensor(s,gymtorch.unwrap_tensor(env._pos_control));env.compute_observations()
  native._dump_initial_state(env,native._current_policy_observation(env,'qpos-target-residual'),out/'initial_state.npz');(out/'config.yaml').write_text(OmegaConf.to_yaml(cfg))
  record_ids=[int(x) for x in args.record_ids.split(',')];W,H=640,480
  for i in record_ids:
   pair=[]
   for local in [[.12,.42,.14],[-.32,.24,-.08]]:
    cp=gymapi.CameraProperties();cp.width=W;cp.height=H;cp.horizontal_fov=42;cam=g.create_camera_sensor(env.envs[i],cp);eye=op[i]+np.asarray(local);eye[2]=max(eye[2],-.32);g.set_camera_location(cam,env.envs[i],gymapi.Vec3(*eye),gymapi.Vec3(*op[i]));pair.append(cam)
   cams[i]=pair;v=cv2.VideoWriter(str(out/f'case{i:03d}_two_views.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(2*W,H+64));assert v.isOpened();writers[i]=v
  traces=[];failed=np.zeros(n,bool);sock=connect_unix(args.socket);sock.settimeout(600);send_message(sock,{'type':'hello'});info,_=recv_message(sock);assert info['ok']
  def step(command,phase,j):
   env.step(native._absolute_targets_to_env_action(command,lower,upper,env.device));f=cpu(env.failure_buf).astype(bool);failed[:]|=f
   wr=cpu(env._base_state[:,:7]);ob=cpu(env._manip_obj_root_state[:,:7]);rot=R.from_quat(ob[:,3:]);axes=rot.apply([0,1,0]);vertical=np.degrees(np.arccos(np.clip(axes[:,2],-1,1)));bw=R.from_quat(wr[:,3:]);relp=bw.inv().apply(ob[:,:3]-wr[:,:3]);relq=(bw.inv()*rot).as_quat();force=np.stack([cpu(env.net_cf[:,env.dexhand_handles[k]]) for k in env.dexhand.contact_body_names],axis=1)
   traces.append(dict(phase=phase,index=j,q=cpu(env._q),command=command.copy(),object_pose=ob,wrist_pose=wr,relative_position=relp,relative_quaternion=relq,vertical_error_deg=vertical,failure=f,contact_force_norm=np.linalg.norm(force,axis=-1),native_goal_index=cpu(env.global_cur_idx),applied_forces=cpu(env.apply_forces),qd=cpu(env._qd),object_velocity=cpu(env._manip_obj_root_state[:,7:]),executed_target=cpu(env.curr_targets)))
   position_delta=np.max(np.abs(wr[:,:3]-base[:,:3]));rotation_delta=np.max(np.minimum(np.linalg.norm(wr[:,3:]-base[:,3:],axis=1),np.linalg.norm(wr[:,3:]+base[:,3:],axis=1)))
   if position_delta>1e-5 or rotation_delta>1e-5:
    np.savez(out/'wrist_mismatch.npz',expected=base,actual=wr)
    raise AssertionError('native wrist changed: position %.8f quaternion %.8f'%(position_delta,rotation_delta))
   g.fetch_results(s,True);g.step_graphics(s);g.render_all_camera_sensors(s)
   for i,pair in cams.items():
    panel=np.zeros((H+64,2*W,3),np.uint8)
    for k,cam in enumerate(pair):
     rgba=g.get_camera_image(s,env.envs[i],cam,gymapi.IMAGE_COLOR).reshape(H,W,4);panel[64:,k*W:(k+1)*W]=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_RGB2BGR)
    cv2.putText(panel,f'case {i} | pose {cases[i]["pose_id"]} | noise {cases[i]["noise_seed"]} | 170g mu2.2 | native wrist',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.64,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(panel,f'{phase} {j/30:.2f}s | vertical error {vertical[i]:.1f}deg | failure {failed[i]} | palm / side views',(10,50),cv2.FONT_HERSHEY_SIMPLEX,.58,(190,230,255),1,cv2.LINE_AA);writers[i].write(panel)
    if j in [0,29,59,74]:cv2.imwrite(str(out/f'case{i:03d}_{phase}{j:03d}.jpg'),panel)
   if j%30==0:print(json.dumps(dict(phase=phase,step=j,failed=int(failed.sum()),total=n)),flush=True)
  for j in range(60):step(q0,'settle',j)
  settled_failure=failed.copy();hist=np.repeat(native._current_policy_observation(env,'qpos-target-residual')[:,None],4,axis=1)
  for j in range(75):
   if j%2==0:
    send_message(sock,{'type':'predict','reference_index':j,'seeds':[c['noise_seed'] for c in cases],'reference_ids':[c['reference_id'] for c in cases]},hist.astype(np.float32));msg,plan=recv_message(sock);assert msg['ok']
   step(plan[:,j%2],'reference',j);hist=np.concatenate([hist[:,1:],native._current_policy_observation(env,'qpos-target-residual')[:,None]],axis=1)
  last=cpu(env.curr_targets)
  for j in range(30):step(last,'hold',j)
  np.savez_compressed(out/'trajectory.npz',**{k:np.stack([t[k] for t in traces]) for k in traces[0]})
  dump(out/'metadata.json',dict(prior=info,recorded_cases=record_ids,settle_failure=settled_failure.tolist(),any_failure=failed.tolist(),native_protocol=resets,control_dt=float(env.dt*env.control_freq_inv),wrist='every environment retains its own native reset wrist; no wrist override',physical_changes='bulb170g; hand/object sliding friction2.2; all remaining native nuisance properties retained independently per environment',initialization='Object_state_data 2 recorded grasps; each21near poses x3native nuisance/noise instances',reference_specs=specs,coordinate_assumption='robot-base object pose and TCP from xyz+row6D; TCP equals native hand root',control_steps=dict(settle=60,reference=75,hold=30)))
  print('COMPLETE',out,flush=True)
 finally:
  for w in writers.values():w.release()
  if sock is not None:
   try:send_message(sock,{'type':'shutdown'});sock.close()
   except Exception:pass
  if asset is not None:native._restore_sharpa_asset_override(asset)
  native._destroy_environment(env)
if __name__=='__main__':main()
