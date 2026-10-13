import sys, os
from pathlib import Path
from argparse import Namespace
import numpy as np
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
sys.path.insert(0,str(ROOT/'eval'))
import sim_eval as setup
from isaacgym import gymapi
import torch
C=Path('/home/carus/Program/dex-controller');os.chdir(C)
lib=setup._import_local_maniptrans(C)
a=Namespace(failure_obj_pos_thres_m=.05,failure_tip_pos_thres_m=.1,failure_obj_rot_thres_deg=180.,invalid_obj_pos_thres_m=.15,failure_tolerance_scale=10000.,fixed_tolerance_steps=20000,traj_steps_limit=12000,reset_on_reach_goal=0,cross_trajectory_goal_prob=.3)
cfg=setup._make_task_config(Path('/home/carus/Data/exp_data/hydra_config.yaml'),16,['v3:bulb2@%03d'%i for i in range(150)],C/'data/NOKOV-v3',C/'data/retargeting/NOKOV-v3',setup._reset_overrides_from_args(a))
asset=setup._install_sharpa_asset_override(setup._resolve_sharpa_urdf(str(C/'maniptrans_envs/assets/sharpa_hand')))
try:e=lib.make(sim_device='cuda:0',rl_device='cuda:0',graphics_device_id=0,multi_gpu=False,cfg=cfg,display=False,record=False,has_headless_arg=True,headless=True)
finally:setup._restore_sharpa_asset_override(asset)
e.compute_observations();e.reset()
qs=[];tips=[];names=[]
try:
 for i in range(4):
  actions=setup._absolute_targets_to_env_action(e._q.cpu().numpy(), e.dexhand_dof_lower_limits,e.dexhand_dof_upper_limits,e.device)
  e.step(actions)
  qs.append(e._q.cpu().numpy().copy())
  ids=[e.dexhand.body_names.index(n) for n in e.dexhand.fingertip_body_names]
  tips.append(e.states['joints_state_rel_wrist'][:,ids,:3].cpu().numpy().copy())
 np.savez_compressed(ROOT/'docs/experiment_reviews/20260923_fk_guidance_scale/fk_sim_states.npz',q=qs,tips=tips,dof_names=e.dexhand.dof_names,tip_names=e.dexhand.fingertip_body_names)
finally:setup._destroy_environment(e)
