"""Check imported transforms against the actual RL relative-pose function."""

import sys
sys.path.insert(0,'/home/carus/opt/isaacgym/python')
import isaacgym
import torch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_apply
import ast
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

P=Path(__file__).resolve().parent
ROOT=P/'reference_turn_baseline_20260926'
CASE=ROOT/'qualified_comparison/episode_76'
OUT=ROOT/'episode76_zero_step_audit_20260927'
PROJECT=P.parents[2]
TASK=PROJECT/'maniptrans_envs/lib/envs/tasks/sindexhandmanip_sh.py'
REAL=Path('/home/carus/Program/TacMP/third_party/diffusion_policy/direct_robot_env.py')


def functions(path,names,namespace):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    module=ast.Module(body=nodes,type_ignores=[])
    exec(compile(module,str(path),'exec'),namespace)
    return namespace


def pose7(t):
    return np.r_[t[:3,3],Rotation.from_matrix(t[:3,:3]).as_quat()]


def main():
    real=functions(REAL,['_normalize_vector','rotation_6d_to_matrix','arm9_to_pose_matrix'],
                   dict(np=np,ARM_DIM=9))
    native=functions(TASK,['calculate_relative_pose'],dict(torch=torch,
                     quat_conjugate=quat_conjugate,quat_mul=quat_mul,quat_apply=quat_apply))
    state=np.load('/home/carus/Data/Object_state_data/episode_76/state.npy')[114]
    obj=np.load('/home/carus/Data/Object_state_data/episode_76/obj_state.npy')[114].reshape(4,4)
    wrist=real['arm9_to_pose_matrix'](state[:9])
    reference=np.load(CASE/'reference_full.npz')
    assert np.allclose(wrist,reference['wrist_pose_base'][0,0],atol=1e-7)
    rel=reference['object_pose_wrist'][0,0]
    calculated=native['calculate_relative_pose'](
        torch.tensor(pose7(obj)[None],dtype=torch.float64),
        torch.tensor(pose7(wrist)[None],dtype=torch.float64)).numpy()[0]
    initial=np.load(CASE/'direct_m044_mu11/initial_state.npz')
    imported=native['calculate_relative_pose'](
        torch.tensor(initial['object'][:,:7],dtype=torch.float64),
        torch.tensor(initial['wrist'][:,:7],dtype=torch.float64)).numpy()[0]
    position_error=float(np.linalg.norm(calculated[:3]-imported[:3]))
    rotation_error=float(np.degrees((Rotation.from_quat(calculated[3:]).inv()*
                                     Rotation.from_quat(imported[3:])).magnitude()))
    assert position_error<1e-6 and rotation_error<.001
    output=dict(source_wrist_matches_dataset_decoder=True,
        relative_position_error_m=position_error,relative_rotation_error_deg=rotation_error,
        reference_matrix_position_error_m=float(np.linalg.norm(rel[:3,3]-calculated[:3])),
        native_task_source=str(TASK),function='calculate_relative_pose',
        object_scale=float(initial['object_scale'].ravel()[0]),
        native_reward='target_states[manip_obj_pos_rel_wrist] - states[manip_obj_pos_rel_wrist]',
        native_target_source='Retargeted NOKOV-v3 opt_wrist_pos/rot and opt_dof_pos paired with obj_trajectory; not Object_state_data live TCP/hand rows.',
        hand_urdf='/home/carus/Program/dex-controller/maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf',
        hand_base_fixed_xyz=[-.01,-.03,-.02],hand_base_fixed_yaw_rad=-1.57079632679,
        fixed_joint_included_in_FK=True)
    OUT.mkdir(exist_ok=True)
    (OUT/'rl_coordinate_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
