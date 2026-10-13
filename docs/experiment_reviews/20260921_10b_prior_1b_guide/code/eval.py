import os,sys,json,time,socket,itertools
from pathlib import Path
import numpy as np
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse');sys.path.insert(0,str(ROOT/'eval'))
import sim_eval as setup
from isaacgym import gymapi,gymtorch
import torch,cv2
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from ipc import send_message,recv_message
from argparse import Namespace
out=Path(sys.argv[1]).resolve();arm=int(sys.argv[2]);C=Path('/home/carus/Program/dex-controller');os.chdir(C);lib=setup._import_local_maniptrans(C)
seed=20260920;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
initials=[('demo079',25,1129),('demo082',19,2182),('demo094',42,244),('real_pose50',50,32)]
cases=[]
for init,physical,noise in itertools.product(initials,[(.044,1.312),(.044,2.572),(.240,1.312),(.240,2.572)],[8,19,25]):
 cases.append({'case':len(cases),'initialization':init[0],'source_seed':init[1],'source_env':init[2],'mass_kg':physical[0],'object_friction':physical[1],'policy_seed':noise})
N=len(cases)
a=Namespace(failure_obj_pos_thres_m=.05,failure_tip_pos_thres_m=.1,failure_obj_rot_thres_deg=180.,invalid_obj_pos_thres_m=.15,failure_tolerance_scale=10000.,fixed_tolerance_steps=20000,traj_steps_limit=12000,reset_on_reach_goal=0,cross_trajectory_goal_prob=.3)
overrides=setup._reset_overrides_from_args(a)
cfg=setup._make_task_config(Path('/home/carus/Data/exp_data/hydra_config.yaml'),N,['v3:bulb2@%03d'%i for i in range(150)],C/'data/NOKOV-v3',C/'data/retargeting/NOKOV-v3',overrides);cfg.env.randomObjectScales=[1.,1.];cfg.env.enableCameraSensors=True
asset=setup._install_sharpa_asset_override(setup._resolve_sharpa_urdf(str(C/'maniptrans_envs/assets/sharpa_hand')))
try:e=lib.make(sim_device='cuda:0',rl_device='cuda:0',graphics_device_id=0,multi_gpu=False,cfg=cfg,display=False,record=False,has_headless_arg=True,headless=True)
finally:setup._restore_sharpa_asset_override(asset)
e.compute_observations();e.reset();g=e.gym;s=e.sim
archive={seed:np.load(ROOT/f'docs/experiment_reviews/20260918_historical_3k_restore/seed{seed}_scale25/initial_parameters.npz') for seed in [25,19,42]};real=np.load(ROOT/'docs/experiment_reviews/20260917_1b_initialization_audit/initial_state.npz')
def cpu(x):return x.detach().cpu().numpy().copy()
q=[];qd=[];wr=[];ob=[];demos=[];frames=[];physical=[]
for i,c in enumerate(cases):
 en=e.envs[i];h=g.find_actor_handle(en,'dexhand');o=g.find_actor_handle(en,'manip_obj');idx=c['source_env'];z=archive.get(c['source_seed'])
 if z is not None:
  q.append(z['q'][idx]);qd.append(z['qd'][idx]);wr.append(z['wrist'][idx]);ob.append(z['object'][idx]);demos.append(int(z['demo'][idx]));frames.append(int(z['frame'][idx]))
  hp=g.get_actor_rigid_body_properties(en,h)
  for j,p in enumerate(hp):p.mass=float(z['hand_mass'][idx,j])
  g.set_actor_rigid_body_properties(en,h,hp,False)
  hs=g.get_actor_rigid_shape_properties(en,h)
  for j,p in enumerate(hs):p.friction=float(z['hand_friction'][idx,j])
  g.set_actor_rigid_shape_properties(en,h,hs);dp=g.get_actor_dof_properties(en,h)
  for key in ['stiffness','damping']:dp[key]=z[key][idx]
  g.set_actor_dof_properties(en,h,dp);e.random_force_prob[i]=float(z['force_prob'][idx]);rest=float(z['restitution'][idx])
 else:
  q.append(real['q'][idx]);qd.append(np.zeros(22,np.float32));wr.append(real['wrist'][idx]);ob.append(real['object'][idx]);demos.append(131);frames.append(469)
  # Fixed native nuisance realization for this real-pose block, shared across all 12 cases.
  src=e.envs[36];sh=g.find_actor_handle(src,'dexhand');so=g.find_actor_handle(src,'manip_obj')
  g.set_actor_rigid_body_properties(en,h,g.get_actor_rigid_body_properties(src,sh),False);g.set_actor_rigid_shape_properties(en,h,g.get_actor_rigid_shape_properties(src,sh));g.set_actor_dof_properties(en,h,g.get_actor_dof_properties(src,sh));e.random_force_prob[i]=e.random_force_prob[36];rest=float(g.get_actor_rigid_shape_properties(src,so)[0].restitution)
 props=g.get_actor_rigid_body_properties(en,o);props[0].mass=c['mass_kg'];g.set_actor_rigid_body_properties(en,o,props,True)
 sp=g.get_actor_rigid_shape_properties(en,o)
 for p in sp:p.friction=c['object_friction'];p.restitution=rest;p.rolling_friction=0.;p.torsion_friction=0.
 g.set_actor_rigid_shape_properties(en,o,sp);p=g.get_actor_rigid_body_properties(en,o)[0];sp=g.get_actor_rigid_shape_properties(en,o);hp=g.get_actor_rigid_body_properties(en,h);hs=g.get_actor_rigid_shape_properties(en,h);dp=g.get_actor_dof_properties(en,h)
 assert abs(p.mass-c['mass_kg'])<1e-6 and all(abs(x.friction-c['object_friction'])<1e-6 for x in sp)
 physical.append({**c,'actual_mass_kg':float(p.mass),'actual_object_friction':[float(x.friction) for x in sp],'scale':float(g.get_actor_scale(en,o)),'restitution':float(sp[0].restitution),'hand_mass':[float(x.mass) for x in hp],'hand_friction':[float(x.friction) for x in hs],'stiffness':dp['stiffness'].tolist(),'damping':dp['damping'].tolist(),'force_probability':float(e.random_force_prob[i]),'inertia':[[getattr(getattr(p.inertia,a),b) for b in 'xyz'] for a in 'xyz']})
e.manip_obj_mass[:]=torch.as_tensor([c['mass_kg'] for c in cases],device=e.device,dtype=torch.float32)
g.simulate(s);g.fetch_results(s,True)
for field in ['dof_state','actor_root_state','rigid_body_state','net_contact_force']:getattr(g,'refresh_'+field+'_tensor')(s)
for target,data in [(e._q,q),(e._qd,qd),(e._base_state,wr),(e._manip_obj_root_state,ob),(e.envidx_to_demoidx,demos),(e.global_cur_idx,frames),(e.progress_buf,frames)]:target[:]=torch.as_tensor(np.asarray(data),device=e.device,dtype=target.dtype)
e.curr_targets[:]=e._q;e.prev_targets[:]=e._q;e._pos_control[:]=e._q
hi=e._global_dexhand_indices.flatten();oi=e._global_manip_obj_indices.flatten();ids=torch.cat([hi,oi]);g.set_dof_state_tensor_indexed(s,gymtorch.unwrap_tensor(e._dof_state),gymtorch.unwrap_tensor(hi),len(hi));g.set_actor_root_state_tensor_indexed(s,gymtorch.unwrap_tensor(e._root_state),gymtorch.unwrap_tensor(ids),len(ids));g.set_dof_position_target_tensor(s,gymtorch.unwrap_tensor(e._pos_control));e.compute_observations()
initial_wrist=e._base_state.clone();np.savez_compressed(out/'initial_state.npz',q=cpu(e._q),qd=cpu(e._qd),wrist=cpu(e._base_state),object=cpu(e._manip_obj_root_state),target=cpu(e.curr_targets),demo=cpu(e.envidx_to_demoidx),frame=cpu(e.global_cur_idx),torch_rng=cpu(torch.cuda.get_rng_state()))
(out/'cases.json').write_text(json.dumps(cases,indent=2));(out/'physical_parameters.json').write_text(json.dumps(physical,indent=2));(out/'simulation_config.yaml').write_text(OmegaConf.to_yaml(cfg))
# Dedicated CUDA random stream for the native pre_physics_step. Same stream/shape in all seven arms.
# Force distribution, event probability, decay and LOCAL_SPACE application remain native.
pre=e.pre_physics_step;force_state=torch.Generator(device='cuda:0').manual_seed(620260920).get_state()
def paired_pre(actions):
 global force_state
 outer=torch.cuda.get_rng_state();torch.cuda.set_rng_state(force_state)
 try:pre(actions);force_state=torch.cuda.get_rng_state()
 finally:torch.cuda.set_rng_state(outer)
e.pre_physics_step=paired_pre
config={'method':out.name,'arm':arm,'prior_ddim':4,'guide_ddim':4,'execution_steps':2,'guidance_steps':2,'guide_scale':25 if arm==2 else None,'cap_steps':12000,'native_overrides':overrides,'control_dt':float(cfg.sim.dt)*int(cfg.env.controlFrequencyInv),'environment_seed':seed,'dedicated_native_prephysics_rng_seed':620260920,'recording':'three actual paired cases; native camera after env.step','real_pose_nuisance':'new seed20260920 native draw, held fixed across methods and physics; same archived real q/wrist/object pose as earlier seed50','note':'same prior noise and native local disturbance stream across methods; goal switching remains native and state-dependent; physical cache synchronized to actual mass'}
(out/'config.json').write_text(json.dumps(config,indent=2));dt=config['control_dt']
record_ids=[next(i for i,c in enumerate(cases) if c['initialization']==init and c['mass_kg']==.044 and c['object_friction']==2.572 and c['policy_seed']==ns) for init,ns in [('demo079',8),('demo082',19),('demo094',25)]]
cams={};videos={}
for i in record_ids:
 cp=gymapi.CameraProperties();cp.width=640;cp.height=480;cp.horizontal_fov=55;cam=g.create_camera_sensor(e.envs[i],cp);origin=np.asarray(wr[i])[:3];g.set_camera_location(cam,e.envs[i],gymapi.Vec3(*(origin+[.12,.55,.15])),gymapi.Vec3(*(origin+[-.08,0,-.12])));cams[i]=cam;v=cv2.VideoWriter(str(out/f'case{i:02d}_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,540));assert v.isOpened();videos[i]=v
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.connect(os.environ['AUDIT_SOCKET']);obs=setup._current_policy_observation(e,'qpos-target-residual');hist=np.repeat(obs[:,None],4,axis=1);generators=[np.random.default_rng(c['policy_seed']) for c in cases];gg=[np.random.default_rng(c['policy_seed']+100000) for c in cases]
alive=np.ones(N,bool);results=[None]*N;pred=np.zeros((N,2,22),np.float32);record={k:[] for k in ['q','raw','object','error_m','failure_progress','active']};start=time.monotonic()
try:
 for step in range(12000):
  before_alive=alive.copy()
  if step%2==0:
   active=np.flatnonzero(alive);noise=np.stack([generators[i].standard_normal((12,22)).astype(np.float32) for i in active]);gnoise=np.stack([gg[i].standard_normal((12,22)).astype(np.float32) for i in active]);send_message(sock,{'type':'predict','arm':arm,'noise':noise.tolist(),'guide_noise':gnoise.tolist()},hist[active]);msg,p=recv_message(sock);assert msg['ok'] and p.shape==(len(active),2,22);pred[active]=p
  raw=pred[:,step%2].copy();raw[~alive]=cpu(e._q)[~alive];d=cpu(e.envidx_to_demoidx);f=cpu(e.global_cur_idx);data=e.demo_data;target_local=Rotation.from_rotvec(cpu(data['opt_wrist_rot'][d,f])).inv().apply(cpu(data['obj_trajectory'][d,f,:3,3]-data['opt_wrist_pos'][d,f]))
  actions=setup._absolute_targets_to_env_action(raw,e.dexhand_dof_lower_limits,e.dexhand_dof_upper_limits,e.device);_,reward,done,info=e.step(actions);err=np.linalg.norm(cpu(e.states['manip_obj_pos_rel_wrist'])-target_local,axis=1)
  if step==0:
   jump=torch.minimum(torch.norm(e._base_state[:,3:7]-initial_wrist[:,3:7],dim=-1),torch.norm(e._base_state[:,3:7]+initial_wrist[:,3:7],dim=-1));assert float(jump.max())<1e-4;np.savez_compressed(out/'first_actions.npz',raw=raw,wrist_delta=cpu(jump))
  ended=cpu(done).astype(bool);fail=cpu(e.failure_buf).astype(bool)
  for i in np.flatnonzero(alive & (ended | (step==11999))):
   results[i]={**cases[i],'method':out.name,'steps':step+1,'seconds':(step+1)*dt,'reason':'failure' if fail[i] else ('done' if ended[i] else 'timeout'),'final_target_error_m':float(err[i]),'error_flag':bool(e.error_buf[i])};alive[i]=False
  for key,value in [('q',cpu(e._q)),('raw',raw),('object',cpu(e._manip_obj_root_state)),('error_m',err),('failure_progress',cpu(e.failure_progress_buf)),('active',before_alive)]:record[key].append(value)
  obs=setup._current_policy_observation(e,'qpos-target-residual');hist=np.concatenate([hist[:,1:],obs[:,None]],axis=1)
  recording=[i for i in record_ids if before_alive[i]]
  if recording:
   g.fetch_results(s,True);g.step_graphics(s);g.render_all_camera_sensors(s)
   for i in recording:
    rgba=np.asarray(g.get_camera_image(s,e.envs[i],cams[i],gymapi.IMAGE_COLOR)).reshape(480,640,4);panel=np.zeros((540,640,3),np.uint8);panel[60:]=cv2.cvtColor(rgba[...,:3],cv2.COLOR_RGB2BGR)
    cv2.putText(panel,f'{out.name} | {cases[i]["initialization"]} noise{cases[i]["policy_seed"]}',(8,23),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1);cv2.putText(panel,f'44g mu2.572 | {dt*(step+1):.2f}s | native failure={int(fail[i])}',(8,47),cv2.FONT_HERSHEY_SIMPLEX,.47,(90,110,255) if fail[i] else (190,245,190),1);videos[i].write(panel)
    if step==0 or not alive[i] or step==11999:cv2.imwrite(str(out/f'case{i:02d}_frame{step+1:05d}.png'),panel)
   # A terminated case has no further video frames; release its camera only.
   # Physics, native stepping, actions, RNG, and all recorded frames are unchanged.
   for i in recording:
    if not alive[i]:g.destroy_camera_sensor(s,e.envs[i],cams.pop(i))
  if (step+1)%300==0 or not alive.any():
   (out/'progress.json').write_text(json.dumps({'steps':step+1,'sim_seconds':dt*(step+1),'alive':int(alive.sum()),'total':N,'wall_seconds':time.monotonic()-start},indent=2));(out/'partial_results.json').write_text(json.dumps([x for x in results if x is not None],indent=2));print(step+1,int(alive.sum()),time.monotonic()-start,flush=True)
  if not alive.any():break
 np.savez_compressed(out/'rollout.npz',**{k:np.asarray(v) for k,v in record.items()});(out/'results.json').write_text(json.dumps({'complete':True,'results':results,'wall_seconds':time.monotonic()-start,'recorded_cases':record_ids},indent=2));print('DONE',out.name,flush=True)
finally:
 for v in videos.values():v.release()
 send_message(sock,{'type':'stop'});sock.close();setup._destroy_environment(e)
