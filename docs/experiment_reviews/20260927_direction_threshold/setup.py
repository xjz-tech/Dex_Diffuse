"""Private source snapshot, with read-only force instrumentation."""
from pathlib import Path
import shutil,hashlib,json,difflib
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
SRC=ROOT/'.worktrees/Astra-controller'
DST=P/'source'
DST.mkdir(exist_ok=True)
(DST/'eval').mkdir(exist_ok=True)
for f in (SRC/'eval').iterdir():
    if f.is_file() and f.suffix in ('.py','.sh'):shutil.copy2(f,DST/'eval'/f.name)
shutil.copytree(SRC/'maniptrans_envs',DST/'maniptrans_envs',symlinks=True,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
for name in ['diffusion_policy']:
    if not (DST/name).exists():(DST/name).symlink_to(SRC/name,target_is_directory=True)
task=DST/'maniptrans_envs/lib/envs/tasks/sindexhandmanip_sh.py'
diff=[] # No changes to task physics. Extra object force sensor pilot was rejected.
f=DST/'eval/astra_sim.py';s=f.read_text();old=s
s=s.replace('from isaacgym import gymapi','from isaacgym import gymapi, gymtorch')
needle='        self.initial = self.state()'
patch='''        hand_handle = env.gym.find_actor_handle(env.envs[0], 'dexhand')
        self.actual_dof_properties = env.gym.get_actor_dof_properties(env.envs[0], hand_handle)
'''
assert s.count(needle)==1;s=s.replace(needle,patch+needle)
needle='        return state'
patch='''        state.update(
            joint_total_torque_nm=array(env.dof_force)[0].tolist(),
            object_net_contact_force_world=array(env.net_cf)[0, env._manip_obj_rigid_body_handle].tolist(),
            applied_random_force_local=array(env.apply_forces)[0, env._manip_obj_rigid_body_handle].tolist(),
            dof_stiffness_nm_per_rad=np.asarray(self.actual_dof_properties['stiffness']).tolist(),
            dof_damping_nm_s_per_rad=np.asarray(self.actual_dof_properties['damping']).tolist(),
            dof_effort_limit_nm=np.asarray(self.actual_dof_properties['effort']).tolist(),
        )
'''
assert s.count(needle)==1;s=s.replace(needle,patch+needle);f.write_text(s)
diff+=list(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='original/astra_sim.py',tofile='instrumented/astra_sim.py'))
(P/'instrumentation.diff').write_text(''.join(diff))
hashes={str(p.relative_to(DST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [task,f,DST/'eval/astra_model_server.py',DST/'eval/xjz_test.sh',DST/'eval/sim_eval.py']}
(P/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
