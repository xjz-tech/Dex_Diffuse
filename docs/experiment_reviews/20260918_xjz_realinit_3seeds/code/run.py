"""Native xjz task evaluation; only initialization and direct video capture added."""
import os,sys,json,socket,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
sys.path.insert(0,str(ROOT/'eval'))
import sim_eval as setup
from isaacgym import gymapi,gymtorch
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from ipc import send_message,recv_message
import cv2
out=Path(sys.argv[1]).resolve();seed=int(sys.argv[2]);arm=int(sys.argv[3]);init_mode=sys.argv[4] if len(sys.argv)>4 else 'real'
assert arm in (0,2)
torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
controller=Path('/home/carus/Program/dex-controller');os.chdir(str(controller))
lib=setup._import_local_maniptrans(controller)
from argparse import Namespace
args=Namespace(failure_obj_pos_thres_m=.05,failure_tip_pos_thres_m=.1,failure_obj_rot_thres_deg=180.,invalid_obj_pos_thres_m=.15,failure_tolerance_scale=10000.,fixed_tolerance_steps=20000,traj_steps_limit=12000,reset_on_reach_goal=0,cross_trajectory_goal_prob=.3)
overrides=setup._reset_overrides_from_args(args)
cfg=setup._make_task_config(Path('/home/carus/Data/exp_data/hydra_config.yaml'),1,['v3:bulb2@%03d'%i for i in range(150)],controller/'data/NOKOV-v3',controller/'data/retargeting/NOKOV-v3',overrides)
cfg.env.enableCameraSensors=True
asset=setup._install_sharpa_asset_override(setup._resolve_sharpa_urdf(str(controller/'maniptrans_envs/assets/sharpa_hand')))
try:e=lib.make(sim_device='cuda:0',rl_device='cuda:0',graphics_device_id=0,multi_gpu=False,cfg=cfg,display=False,record=False,has_headless_arg=True,headless=True)
finally:setup._restore_sharpa_asset_override(asset)
e.compute_observations();e.reset()
g=e.gym;s=e.sim
init_info={'mode':init_mode,'seed':seed}
if init_mode=='real':
 old=ROOT/'docs/experiment_reviews/20260917_1b_initialization_audit'
 ini=np.load(old/'initial_state.npz');idx=32+seed-50
 placements=json.loads((old/'placements.json').read_text())['per_run']
 name=['bulb_nohold_1','bulb_nohold_2','bulb_hold_1','bulb_hold_2'][(seed-50)%4];placement=placements[name]
 e.envidx_to_demoidx[:]=placement['demo'];e.global_cur_idx[:]=placement['frame'];e.progress_buf[:]=placement['frame']
 e._q[:]=torch.as_tensor(ini['q'][idx],device=e.device);e._qd.zero_()
 e._base_state[:]=torch.as_tensor(ini['wrist'][idx],device=e.device)
 e._manip_obj_root_state[:]=torch.as_tensor(ini['object'][idx],device=e.device)
 for name2 in ['curr_targets','prev_targets','_pos_control']:getattr(e,name2)[:]=e._q
 ids=torch.cat([e._global_dexhand_indices.flatten(),e._global_manip_obj_indices.flatten()])
 g.set_dof_state_tensor(s,gymtorch.unwrap_tensor(e._dof_state))
 g.set_actor_root_state_tensor_indexed(s,gymtorch.unwrap_tensor(e._root_state),gymtorch.unwrap_tensor(ids),len(ids))
 g.set_dof_position_target_tensor(s,gymtorch.unwrap_tensor(e._pos_control))
 for field in ['dof_state','actor_root_state','rigid_body_state','net_contact_force']:getattr(g,'refresh_'+field+'_tensor')(s)
 e.compute_observations()
 init_info.update(source=str(old/'initial_state.npz'),source_env=idx,source_pose=name,demo=placement['demo'],frame=placement['frame'])
for k,v in overrides.items():assert cfg.env[k]==v,(k,cfg.env[k],v)
assert e.enable_latency_tracking and e.failure_tolerance_scale==10000 and e.fixed_tolerance_steps==20000
assert e.invalid_obj_pos_thres==.15 and e.failure_obj_pos_thres==.05
dt=float(cfg.sim.dt)*int(cfg.env.controlFrequencyInv)
config={'seed':seed,'arm':'ordinary_1b' if arm==0 else 'guide10k','initialization':init_info,'protocol':'eval/xjz_test.sh native env.step + compute_imitation_reward_latency_tracking','overrides':overrides,'prior_ddim_steps':4,'guide_ddim_steps':4,'execute_steps':2,'guidance_steps':2,'guide_scale':25 if arm==2 else None,'control_dt':dt,'max_steps':12000,'wait':'synchronous inference; no extra physics while waiting','recording':'live camera image captured after every env.step; 30fps','fresh_noise':True,'guide_seed':seed+100000,'randomization':'original xjz task configuration retained; explicit real initialization replaces reset pose only'}
(out/'config.json').write_text(json.dumps(config,indent=2));(out/'simulation_config.yaml').write_text(OmegaConf.to_yaml(cfg))
def cpu(x):return x.detach().cpu().numpy().copy()
np.savez_compressed(out/'initial_state.npz',q=cpu(e._q),qd=cpu(e._qd),wrist=cpu(e._base_state),object=cpu(e._manip_obj_root_state),target=cpu(e.curr_targets),demo=cpu(e.envidx_to_demoidx),frame=cpu(e.global_cur_idx),torch_rng=cpu(torch.cuda.get_rng_state()),mass=cpu(e.manip_obj_mass))
props=gymapi.CameraProperties();props.width=800;props.height=600;props.horizontal_fov=55
cam=g.create_camera_sensor(e.envs[0],props);assert cam>=0
origin=cpu(e._base_state)[0,:3]
g.set_camera_location(cam,e.envs[0],gymapi.Vec3(*(origin+[.12,.55,.15])),gymapi.Vec3(*(origin+[-.08,0,-.12])))
video=cv2.VideoWriter(str(out/'live_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(800,680));assert video.isOpened()
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.connect(os.environ['AUDIT_SOCKET'])
obs=setup._current_policy_observation(e,'qpos-target-residual');hist=np.repeat(obs[:,None],4,axis=1)
gen=np.random.default_rng(seed);gg=np.random.default_rng(seed+100000)
rec={k:[] for k in ['q','action','raw','object','target_error_m','failure_progress','goal_demo','goal_frame']};reason='timeout';start=time.monotonic()
try:
 for step in range(12000):
  if step%2==0:
   send_message(sock,{'type':'predict','arm':arm,'noise':gen.standard_normal((1,12,22)).astype(np.float32).tolist(),'guide_noise':gg.standard_normal((1,12,22)).astype(np.float32).tolist()},hist)
   msg,pred=recv_message(sock);assert msg['ok'] and pred.shape==(1,2,22)
  raw=pred[:,step%2];demo=int(e.envidx_to_demoidx[0]);frame=int(e.global_cur_idx[0])
  d=e.demo_data
  target_local=Rotation.from_rotvec(cpu(d['opt_wrist_rot'][demo,frame])).inv().apply(cpu(d['obj_trajectory'][demo,frame,:3,3]-d['opt_wrist_pos'][demo,frame]))
  act=setup._absolute_targets_to_env_action(raw,e.dexhand_dof_lower_limits,e.dexhand_dof_upper_limits,e.device)
  _,reward,done,info=e.step(act)
  actual=cpu(e._q);obj=cpu(e._manip_obj_root_state);target=cpu(e.curr_targets)
  error=float(np.linalg.norm(cpu(e.states['manip_obj_pos_rel_wrist'])[0]-target_local))
  assert np.isfinite(actual).all() and np.isfinite(obj).all()
  for key,val in [('q',actual[0]),('action',target[0]),('raw',raw[0].copy()),('object',obj[0]),('target_error_m',error),('failure_progress',float(e.failure_progress_buf[0])),('goal_demo',demo),('goal_frame',frame)]:rec[key].append(val)
  obs=setup._current_policy_observation(e,'qpos-target-residual');hist=np.concatenate([hist[:,1:],obs[:,None]],axis=1)
  ended=bool(done[0]);failed=bool(e.failure_buf[0])
  g.fetch_results(s,True);g.step_graphics(s);g.render_all_camera_sensors(s)
  rgba=g.get_camera_image(s,e.envs[0],cam,gymapi.IMAGE_COLOR)
  img=cv2.cvtColor(np.asarray(rgba).reshape(600,800,4)[...,:3],cv2.COLOR_RGB2BGR)
  panel=np.zeros((680,800,3),np.uint8);panel[80:]=img
  label='1B prior' if arm==0 else '1B + 10k guide (scale 25)'
  cv2.putText(panel,f'LIVE SIM | {label} | seed {seed}',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),2)
  cv2.putText(panel,f't={(step+1)*dt:.2f}s | target error={error*100:.2f}cm | xjz native failure={int(failed)}',(12,59),cv2.FONT_HERSHEY_SIMPLEX,.59,(80,110,255) if failed else (190,245,190),2)
  video.write(panel)
  if step in (0,5,29,299) or ended:cv2.imwrite(str(out/f'frame_{step+1:05d}.png'),panel)
  if (step+1)%300==0 or ended:
   status={'seed':seed,'arm':config['arm'],'steps':step+1,'seconds':(step+1)*dt,'error_m':error,'failure':failed,'wall_seconds':time.monotonic()-start}
   (out/'progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
  if ended:
   reason='failure' if failed else ('success' if bool(e.success_buf[0]) else 'done');break
 np.savez_compressed(out/'rollout.npz',**{k:np.asarray(v) for k,v in rec.items()})
 result={'complete':True,'seed':seed,'arm':config['arm'],'steps':len(rec['q']),'seconds':len(rec['q'])*dt,'reason':reason,'final_target_error_m':rec['target_error_m'][-1],'final_failure_progress':rec['failure_progress'][-1],'error_flag':bool(e.error_buf[0]),'wall_seconds':time.monotonic()-start}
 (out/'result.json').write_text(json.dumps(result,indent=2));print('DONE',json.dumps(result),flush=True)
finally:
 video.release();send_message(sock,{'type':'stop'});sock.close();setup._destroy_environment(e)
