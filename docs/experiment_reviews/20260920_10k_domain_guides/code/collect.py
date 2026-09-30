import sys,os,json,hashlib,time,random
from pathlib import Path
import numpy as np
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse'); OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eval'))
import sim_eval as setup
from isaacgym import gymapi,gymtorch
import torch
from omegaconf import OmegaConf
from rl_games.algos_torch import model_builder
C=Path('/home/carus/Program/dex-controller');os.chdir(str(C));lib=setup._import_local_maniptrans(C)
# Use the same native SAPG expert checkpoint as original collection, rank0/block5.
checkpoint=next((C/'runs/v3sharpa_use/bulb2').glob('*/nn/1_v3sharpa/bulb2/*.pth'))
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='44d9b4830d52756be63209e0d2c32dabb18211451feb5925aa6b3e7331a4c374'
full=OmegaConf.load('/home/carus/Data/exp_data/hydra_config.yaml')
seed=20260920;np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(4)
N=600;B=150;L=250;WARM=4
cfg=setup._make_task_config(Path('/home/carus/Data/exp_data/hydra_config.yaml'),N,['v3:bulb2@%03d'%i for i in range(150)],C/'data/NOKOV-v3',C/'data/retargeting/NOKOV-v3')
cfg.env.actStyle='flat_hora';cfg.env.randomObjectScales=[1.,1.]
# Collection only: physical DR drawn at construction then frozen per slot. No RL defaults edited.
for actor,props in cfg.task.randomization_params.actor_params.items():
 for prop,attrs in props.items():
  if prop=='color':continue
  for attr,spec in attrs.items():spec.setup_only=True
asset=setup._install_sharpa_asset_override(setup._resolve_sharpa_urdf(str(C/'maniptrans_envs/assets/sharpa_hand')))
try:e=lib.make(sim_device='cuda:0',rl_device='cuda:0',graphics_device_id=0,multi_gpu=False,cfg=cfg,display=False,record=False,has_headless_arg=True,headless=True)
finally:setup._restore_sharpa_asset_override(asset)
e.compute_observations();e.reset();g=e.gym;s=e.sim
def cpu(x):return x.detach().cpu().numpy().copy()
rng=np.random.default_rng(seed+1);um=rng.random(B);uf=rng.random(B)
domains=['light_low','light_high','heavy_low','heavy_high'];mass=np.concatenate([.04+.04*um,.04+.04*um,.18+.12*um,.18+.12*um]);mu=np.concatenate([.5+uf,2.5+1.5*uf,.5+uf,2.5+1.5*uf])
physical=[]
for i,en in enumerate(e.envs):
 base=i%B;src=e.envs[base];hand=g.find_actor_handle(en,'dexhand');src_hand=g.find_actor_handle(src,'dexhand');obj=g.find_actor_handle(en,'manip_obj')
 if i>=B:
  g.set_actor_rigid_body_properties(en,hand,g.get_actor_rigid_body_properties(src,src_hand),False)
  g.set_actor_rigid_shape_properties(en,hand,g.get_actor_rigid_shape_properties(src,src_hand))
  g.set_actor_dof_properties(en,hand,g.get_actor_dof_properties(src,src_hand))
 op=g.get_actor_rigid_body_properties(en,obj);op[0].mass=float(mass[i]);g.set_actor_rigid_body_properties(en,obj,op,True)
 sp=g.get_actor_rigid_shape_properties(en,obj)
 src_obj=g.find_actor_handle(src,'manip_obj');surface=g.get_actor_rigid_shape_properties(src,src_obj)[0]
 for p in sp:p.friction=float(mu[i]);p.restitution=float(surface.restitution);p.rolling_friction=0.;p.torsion_friction=0.
 g.set_actor_rigid_shape_properties(en,obj,sp)
 props=g.get_actor_rigid_body_properties(en,obj)[0];sp=g.get_actor_rigid_shape_properties(en,obj);hp=g.get_actor_rigid_body_properties(en,hand);hs=g.get_actor_rigid_shape_properties(en,hand);dp=g.get_actor_dof_properties(en,hand)
 assert abs(props.mass-mass[i])<1e-6 and all(abs(p.friction-mu[i])<1e-6 for p in sp)
 physical.append({'env':i,'base_env':base,'domain':domains[i//B],'mass_kg':float(props.mass),'object_friction':[float(p.friction) for p in sp],'scale':float(g.get_actor_scale(en,obj)),'inertia':[[getattr(getattr(props.inertia,a),b) for b in 'xyz'] for a in 'xyz'],'com':[props.com.x,props.com.y,props.com.z],'restitution':float(sp[0].restitution),'hand_mass':[float(p.mass) for p in hp],'hand_friction':[float(p.friction) for p in hs],'stiffness':dp['stiffness'].tolist(),'damping':dp['damping'].tolist()})
e.manip_obj_mass[:]=torch.as_tensor(mass,device=e.device,dtype=torch.float32)
e.random_force_prob[:]=e.random_force_prob[:B].repeat(4)
# Flush PhysX CPU property updates before scored/recorded trajectories.
g.simulate(s);g.fetch_results(s,True)
for field in ['dof_state','actor_root_state','rigid_body_state','net_contact_force']:getattr(g,'refresh_'+field+'_tensor')(s)
params=OmegaConf.to_container(full.rl_train.params,resolve=True);params['model']['name']='continuous_a2c_logstd';params['network']['name']='actor_critic';params['network'].pop('dict_feature_encoder',None);params['network']['space']['continuous']['fixed_sigma']='coef_cond'
keys=list(e.observation_space.spaces.keys());width=sum(int(np.prod(e.observation_space.spaces[k].shape)) for k in keys);assert width==142,(keys,width)
model=model_builder.ModelBuilder().load(params).build({'actions_num':22,'input_shape':(143,),'num_seqs':N,'value_size':1,'normalize_value':True,'normalize_input':True,'type':'extra_param','coef_ids':torch.linspace(50.,0.,6,device='cuda:0'),'coef_id_idx':142}).to('cuda:0').eval()
ck=torch.load(str(checkpoint),map_location='cpu');model.load_state_dict(ck[0]['model'],strict=True);del ck
(OUT/'collection_config.yaml').write_text(OmegaConf.to_yaml(cfg));(OUT/'physical_slots.json').write_text(json.dumps(physical,indent=2))
(OUT/'collection_manifest.json').write_text(json.dumps({'expert':str(checkpoint),'expert_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'block':5,'coef_id':0,'seed':seed,'domains':domains,'mass_ranges_kg':[[.04,.08],[.18,.30]],'object_friction_ranges':[[.5,1.5],[2.5,4.]],'selection':'40 shared base-env initializations with 250 contiguous post-warmup transitions, no native failure and >=2 target reaches in each domain; no replacement within a dataset','warmup_steps':WARM,'L':L,'fixed_geometry_scale':1,'paired_hand_properties':True,'collection_protocol':'original SAPG collection task thresholds, NOT DP evaluation; native dynamics; fixed per-slot physics; source mass cache updated explicitly','observation_keys':keys},indent=2))
selected={};attempts=[];pool={};start=time.monotonic()
try:
 for round_id in range(10):
  ids=torch.arange(N,device=e.device,dtype=torch.long);e.reset_idx(ids)
  # Clone the initial q/qd, wrist-relative object pose, target demo/frame from base block.
  q=e._q[:B].clone();qd=e._qd[:B].clone();w=e._base_state[:B].clone();obj=e._manip_obj_root_state[:B].clone();demo=e.envidx_to_demoidx[:B].clone();frame=e.global_cur_idx[:B].clone();prog=e.progress_buf[:B].clone()
  for j in range(4):
   sl=slice(j*B,(j+1)*B);translation=e._base_state[sl,:3].clone()-w[:,:3]
   e._q[sl]=q;e._qd[sl]=qd;e._base_state[sl]=w;e._base_state[sl,:3]+=translation;e._manip_obj_root_state[sl]=obj;e._manip_obj_root_state[sl,:3]+=translation
   e.envidx_to_demoidx[sl]=demo;e.global_cur_idx[sl]=frame;e.progress_buf[sl]=prog
  e.curr_targets[:]=e._q;e.prev_targets[:]=e._q;e._pos_control[:]=e._q
  hi=e._global_dexhand_indices.flatten();oi=e._global_manip_obj_indices.flatten();allids=torch.cat([hi,oi])
  g.set_dof_state_tensor_indexed(s,gymtorch.unwrap_tensor(e._dof_state),gymtorch.unwrap_tensor(hi),len(hi));g.set_actor_root_state_tensor_indexed(s,gymtorch.unwrap_tensor(e._root_state),gymtorch.unwrap_tensor(allids),len(allids));g.set_dof_position_target_tensor(s,gymtorch.unwrap_tensor(e._pos_control));e.compute_observations()
  states=tuple(torch.zeros_like(t,device=e.device) for t in model.get_default_rnn_state());failed=np.zeros(N,bool);reaches=np.zeros(N,int);firstfail=np.full(N,-1,int);buf={k:[] for k in ['obs','action','demo','frame','object_position_wrist','object_velocity']};initial={'q':cpu(e._q),'wrist':cpu(e._base_state),'object':cpu(e._manip_obj_root_state),'demo':cpu(e.envidx_to_demoidx),'frame':cpu(e.global_cur_idx)}
  for step in range(WARM+L):
   flat=torch.cat([e.obs_dict[k].reshape(N,-1) for k in keys],dim=-1);inp=torch.cat([flat,torch.zeros((N,1),device=e.device)],-1)
   with torch.no_grad():res=model({'is_train':False,'prev_actions':None,'obs':inp,'rnn_states':states});states=res['rnn_states'];act=res['mus'].clamp(-1,1)
   before=setup._current_policy_observation(e,'qpos-target-residual');d=cpu(e.envidx_to_demoidx);f=cpu(e.global_cur_idx)
   _,reward,done,info=e.step(act)
   now=cpu(e.failure_buf).astype(bool);firstfail[(firstfail<0)&now]=step;failed|=now;reaches+=cpu(e.reach_final_goal).astype(int)
   if step>=WARM:
    for k,v in [('obs',before),('action',cpu(e.curr_targets)),('demo',d),('frame',f),('object_position_wrist',cpu(e.states['manip_obj_pos_rel_wrist'])),('object_velocity',cpu(e._manip_obj_root_state[:,7:13]))]:buf[k].append(v)
  arrays={k:np.stack(v) for k,v in buf.items()};ok=(~failed)&(reaches>=2);common=np.all(ok.reshape(4,B),axis=0)
  attempts.append({'round':round_id,'eligible_by_domain':[int(x.sum()) for x in ok.reshape(4,B)],'common':int(common.sum()),'first_failure_step':firstfail.tolist(),'reaches':reaches.tolist()})
  for base in np.flatnonzero(common):
   if int(base) in selected:continue
   selected[int(base)]={'round':round_id,'base_env':int(base),'initial_demo':int(initial['demo'][base]),'initial_frame':int(initial['frame'][base])}
   for j,name in enumerate(domains):
    slot=j*B+base
    for k,v in arrays.items():pool[f'b{base}_{name}_{k}']=v[:,slot].copy()
    for k,v in initial.items():pool[f'b{base}_{name}_initial_{k}']=v[slot].copy()
  (OUT/'collection_progress.json').write_text(json.dumps({'round':round_id,'common_initializations':len(selected),'wall_seconds':time.monotonic()-start}));print('ROUND',round_id,'common',common.sum(),'total',len(selected),flush=True)
  if len(selected)>=40:break
 assert len(selected)>=40,('insufficient shared successful segments',len(selected))
 chosen=sorted(np.random.default_rng(seed+2).choice(sorted(selected),40,replace=False).tolist());keep={k:v for k,v in pool.items() if int(k.split('_')[0][1:]) in chosen}
 np.savez_compressed(OUT/'collection_pool.npz',**keep);(OUT/'collection_selected.json').write_text(json.dumps([selected[i] for i in chosen],indent=2));(OUT/'collection_attempts.json').write_text(json.dumps(attempts,indent=2))
 (OUT/'collection_done.json').write_text(json.dumps({'complete':True,'base_envs':chosen,'domain_transitions':10000,'domains':domains,'wall_seconds':time.monotonic()-start},indent=2));print('COLLECTION DONE',flush=True)
finally:setup._destroy_environment(e)
