import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'eval'))
import sim_eval as sim
import numpy as np
import torch
from omegaconf import OmegaConf
OUT=Path(os.environ['RESTORE_OUT'])
original=sim._validate_environment
def capture(e,manifest):
 result=original(e,manifest)
 before=torch.cuda.get_rng_state().clone()
 def cpu(x):return x.detach().cpu().numpy().copy()
 data={k:[] for k in ['mass','friction','rolling_friction','torsion_friction','restitution','hand_mass','hand_friction','stiffness','damping','scale']}
 for en in e.envs:
  obj=e.gym.find_actor_handle(en,'manip_obj');hand=e.gym.find_actor_handle(en,'dexhand')
  data['mass'].append(e.gym.get_actor_rigid_body_properties(en,obj)[0].mass)
  props=e.gym.get_actor_rigid_shape_properties(en,obj)[0]
  for key in ['friction','rolling_friction','torsion_friction','restitution']:data[key].append(getattr(props,key))
  data['hand_mass'].append([x.mass for x in e.gym.get_actor_rigid_body_properties(en,hand)])
  data['hand_friction'].append([x.friction for x in e.gym.get_actor_rigid_shape_properties(en,hand)])
  dof=e.gym.get_actor_dof_properties(en,hand)
  for key in ['stiffness','damping']:data[key].append(dof[key].copy())
  data['scale'].append(e.gym.get_actor_scale(en,obj))
 data={k:np.asarray(v) for k,v in data.items()}
 data.update(q=cpu(e._q),qd=cpu(e._qd),wrist=cpu(e._base_state),object=cpu(e._manip_obj_root_state),demo=cpu(e.envidx_to_demoidx),frame=cpu(e.global_cur_idx),cached_mass=cpu(e.manip_obj_mass),force_prob=cpu(e.random_force_prob),torch_rng=cpu(before))
 assert torch.equal(before,torch.cuda.get_rng_state())
 np.savez_compressed(OUT/'initial_parameters.npz',**data)
 (OUT/'simulation_config.yaml').write_text(OmegaConf.to_yaml(e.cfg))
 (OUT/'capture.json').write_text(json.dumps({'num_envs':len(e.envs),'mass_source':'gym.get_actor_rigid_body_properties, after initial env.reset','rng_unchanged_by_capture':True},indent=2))
 print('INITIAL PARAMETERS CAPTURED',OUT,flush=True)
 return result
sim._validate_environment=capture
sim.main()
