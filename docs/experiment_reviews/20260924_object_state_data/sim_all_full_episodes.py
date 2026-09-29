"""Vectorized native-protocol full-tail replay for selected Object_state_data grasps."""

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np


P = Path(__file__).resolve().parent
ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
CONTROLLER = Path('/home/carus/Program/dex-controller')
DATA = P / 'corrected_direct_vs_reference/all_full_episodes'
PROTOCOL = dict(
    failureObjPosThres=.05, failureThumbTipPosThres=.1,
    failureIndexTipPosThres=.1, failureMiddleTipPosThres=.1,
    failurePinkyTipPosThres=.1, failureRingTipPosThres=.1,
    failureObjRotThres=180., invalidObjPosThres=.15,
    FailureToleranceScale=10000., fixedToleranceSteps=20000,
    trajStepsLimit=12000, resetOnReachGoal=False,
    enableCrossTrajectoryReset=True, crossTrajectoryGoalProb=.3,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['direct', 'guided'], required=True)
    parser.add_argument('--execution-steps', type=int, choices=[1, 2], default=2)
    parser.add_argument('--episodes', default='all',
                        help='Comma-separated dataset IDs, or all for 0–79')
    parser.add_argument('--socket')
    parser.add_argument('--stop-when-all-failed', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'guided' and not args.socket:
        parser.error('guided requires --socket')
    ids = list(range(80)) if args.episodes == 'all' else [int(x) for x in args.episodes.split(',')]
    assert ids and len(set(ids)) == len(ids) and all(0 <= i < 80 for i in ids)
    n = len(ids)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from isaacgym import gymtorch
    import torch
    from scipy.spatial.transform import Rotation
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT / 'eval'))
    import sim_eval as native
    from ipc import connect_unix, recv_message, send_message
    from reference_resampling import interpolate_actions

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    selection = json.loads((DATA / 'selection.json').read_text())['episodes']
    packed = np.load(DATA / 'packed_reference.npz')
    original_actions = packed['hand_target_rad'][ids]
    original_lengths = packed['action_lengths'][ids]
    assert all(int(original_lengths[j]) == selection[e]['actions'] for j, e in enumerate(ids))
    interpolation = 1 if args.mode == 'guided' else 0
    targets = interpolate_actions(original_actions, interpolation)
    lengths = (original_lengths - 1) * (interpolation + 1) + 1
    max_action_steps = int(lengths.max())
    references = [np.load(DATA / f'episode_{e:02d}/reference_full.npz') for e in ids]
    q0 = np.stack([r['hand_qpos_rad'][0, 0] for r in references])
    local_object = np.stack([r['object_pose_wrist'][0, 0] for r in references])

    os.chdir(CONTROLLER)
    lib = native._import_local_maniptrans(CONTROLLER)
    from astra_demo_cache import install_demo_cache
    install_demo_cache(P / 'demo_cache')
    cfg = native._make_task_config(
        Path('/home/carus/Data/exp_data/hydra_config.yaml'), n,
        [f'v3:bulb2@{i:03d}' for i in range(150)],
        CONTROLLER / 'data/NOKOV-v3',
        CONTROLLER / 'data/retargeting/NOKOV-v3', PROTOCOL)
    cfg.env.enableCameraSensors = False
    asset = native._install_sharpa_asset_override(
        native._resolve_sharpa_urdf(CONTROLLER / 'maniptrans_envs/assets/sharpa_hand'))
    env = None
    sock = None
    try:
        env = lib.make(sim_device='cuda:0', rl_device='cuda:0', graphics_device_id=0,
                       multi_gpu=False, cfg=cfg, display=False, record=False,
                       has_headless_arg=True, headless=True)
        native._restore_sharpa_asset_override(asset)
        asset = None
        env.compute_observations()
        env.reset()
        gym, sim = env.gym, env.sim
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        env.compute_observations()

        def cpu(value):
            return value.detach().cpu().numpy().copy()

        lower, upper = native._validate_environment(
            env, native._load_manifest(Path('/home/carus/Data/exp_data/hydra_config.yaml')))
        assert json.loads((P / 'reference/initial_state.json').read_text())['hand_joint_names'] == list(env.dexhand.dof_names)
        held = np.clip(q0, cpu(lower), cpu(upper))
        demos = cpu(env.demo_data['opt_dof_pos'])
        demo_lengths = cpu(env.demo_data['seq_len']).astype(int)
        demo_wrist_rot = Rotation.from_rotvec(cpu(env.demo_data['opt_wrist_rot']).reshape(-1, 3))
        demo_wrist_pos = cpu(env.demo_data['opt_wrist_pos']).reshape(-1, 3)
        demo_objects = cpu(env.demo_data['obj_trajectory']).reshape(-1, 4, 4)
        demo_relative = demo_wrist_rot.inv().apply(
            demo_objects[:, :3, 3] - demo_wrist_pos).reshape(demos.shape[0], demos.shape[1], 3)
        match_ids = []
        match_frames = []
        for j in range(n):
            qerror = np.sqrt(np.mean((demos - q0[j])**2, axis=-1))
            perror = np.linalg.norm(demo_relative - local_object[j, :3, 3], axis=-1)
            score = qerror / .3 + perror / .05
            for di, valid_length in enumerate(demo_lengths):
                score[di, valid_length:] = np.inf
            di, frame = np.unravel_index(score.argmin(), score.shape)
            match_ids.append(int(di))
            match_frames.append(int(frame))

        for j in range(n):
            arena = env.envs[j]
            hand = gym.find_actor_handle(arena, 'dexhand')
            obj = gym.find_actor_handle(arena, 'manip_obj')
            for actor in (hand, obj):
                shapes = gym.get_actor_rigid_shape_properties(arena, actor)
                for shape in shapes:
                    shape.friction = 2.2
                gym.set_actor_rigid_shape_properties(arena, actor, shapes)
            body = gym.get_actor_rigid_body_properties(arena, obj)
            body[0].mass = .17
            gym.set_actor_rigid_body_properties(arena, obj, body, True)
            assert abs(gym.get_actor_rigid_body_properties(arena, obj)[0].mass - .17) < 1e-6
        env.manip_obj_mass[:] = .17

        # PhysX must flush property changes before importing object/hand roots.
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        env.compute_observations()

        wrist = cpu(env._base_state[:, :7])
        wrist_rotation = Rotation.from_quat(wrist[:, 3:])
        world_position = wrist_rotation.apply(local_object[:, :3, 3]) + wrist[:, :3]
        world_quat = (wrist_rotation * Rotation.from_matrix(local_object[:, :3, :3])).as_quat()
        env._q[:] = torch.tensor(q0, device=env.device)
        env._qd.zero_()
        env.curr_targets[:] = torch.tensor(held, device=env.device)
        env.prev_targets[:] = env.curr_targets
        env._pos_control[:] = env.curr_targets
        env._manip_obj_root_state[:, :3] = torch.tensor(world_position, device=env.device)
        env._manip_obj_root_state[:, 3:7] = torch.tensor(world_quat, device=env.device)
        env._manip_obj_root_state[:, 7:] = 0
        env.envidx_to_demoidx[:] = torch.tensor(match_ids, device=env.device)
        env.global_cur_idx[:] = torch.tensor(match_frames, device=env.device)
        env.progress_buf[:] = torch.tensor(match_frames, device=env.device)
        for name in ('failure_progress_buf', 'reset_buf', 'failure_buf', 'success_buf',
                     'running_progress_buf', 'stable_frames_buf', 'traj_steps_counter',
                     'is_target_cross', 'error_buf'):
            getattr(env, name).zero_()
        hand_ids = env._global_dexhand_indices.flatten()
        object_ids = env._global_manip_obj_indices.flatten()
        root_ids = torch.cat([hand_ids, object_ids])
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(env._dof_state),
                                         gymtorch.unwrap_tensor(hand_ids), len(hand_ids))
        assert gym.set_actor_root_state_tensor_indexed(
            sim, gymtorch.unwrap_tensor(env._root_state), gymtorch.unwrap_tensor(root_ids), len(root_ids))
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env._pos_control))
        env.compute_observations()
        native._dump_initial_state(env,
            native._current_policy_observation(env, 'qpos-target-residual'),
            out / 'initial_state.npz')
        (out / 'config.yaml').write_text(OmegaConf.to_yaml(cfg))

        info = None
        if args.mode == 'guided':
            sock = connect_unix(args.socket)
            sock.settimeout(600)
            send_message(sock, {'type': 'hello'})
            info, _ = recv_message(sock)
            assert info['ok'] and info['guidance_steps'] == 2
            assert info['guidance_scale'] == 50 and info['execution_steps'] == args.execution_steps
            assert info['reference_interpolation'] == 1
        all_rows = dict(q=[], command=[], object_pose=[], relative_position=[],
                        vertical_deg=[], native_failure=[])
        ever_failed = np.zeros(n, dtype=bool)
        last_command = held.copy()

        def step(command):
            env.step(native._absolute_targets_to_env_action(command, lower, upper, env.device))
            current_wrist = cpu(env._base_state[:, :7])
            assert np.allclose(current_wrist, wrist, atol=1e-5)
            obj = cpu(env._manip_obj_root_state[:, :7])
            obj_rotation = Rotation.from_quat(obj[:, 3:])
            relative_position = wrist_rotation.inv().apply(obj[:, :3] - wrist[:, :3])
            vertical = np.degrees(np.arccos(np.clip(obj_rotation.apply([0, 1, 0])[:, 2], -1, 1)))
            all_rows['q'].append(cpu(env._q))
            all_rows['command'].append(command.astype(np.float32).copy())
            all_rows['object_pose'].append(obj.astype(np.float32))
            all_rows['relative_position'].append(relative_position.astype(np.float32))
            all_rows['vertical_deg'].append(vertical.astype(np.float32))
            failure = cpu(env.failure_buf).astype(bool)
            all_rows['native_failure'].append(failure)
            ever_failed[:] |= failure
            return bool(ever_failed.all())

        settle_steps = 0
        for j in range(60):
            all_failed = step(held)
            settle_steps += 1
            if all_failed and args.stop_when_all_failed:
                break
        history = np.repeat(native._current_policy_observation(env, 'qpos-target-residual')[:, None], 4, axis=1)
        planned = np.zeros((n, args.execution_steps, 22), dtype=np.float32)
        action_steps = 0
        for j in range(max_action_steps):
            if args.stop_when_all_failed and ever_failed.all():
                break
            active = np.flatnonzero((j < lengths) &
                                    (~ever_failed if args.stop_when_all_failed else np.ones(n, bool)))
            if args.mode == 'direct':
                command = targets[np.arange(n), np.minimum(j, lengths - 1)].copy()
                if args.stop_when_all_failed:
                    command[ever_failed] = last_command[ever_failed]
            else:
                command = last_command.copy()
                if j % args.execution_steps == 0 and len(active):
                    send_message(sock, dict(type='predict', reference_index=j,
                        seeds=[44] * len(active), reference_ids=[ids[i] for i in active]),
                        history[active].astype(np.float32))
                    message, plan = recv_message(sock)
                    assert message['ok'] and plan.shape == (len(active), args.execution_steps, 22)
                    planned[active] = plan
                command[active] = planned[active, j % args.execution_steps]
            step(command)
            action_steps += 1
            last_command = command.copy()
            if args.mode == 'guided':
                history = np.concatenate((history[:, 1:],
                    native._current_policy_observation(env, 'qpos-target-residual')[:, None]), axis=1)
            if j % 100 == 0:
                print(json.dumps(dict(mode=args.mode, execution_steps=args.execution_steps,
                    action_step=j, action_total=max_action_steps,
                    active=len(active), failed=int(np.any(np.stack(all_rows['native_failure']), axis=0).sum()))),
                    flush=True)
        hold_steps = 0
        for j in range(60):
            if args.stop_when_all_failed and ever_failed.all():
                break
            step(last_command)
            hold_steps += 1
        trajectory = {key: np.stack(value) for key, value in all_rows.items()}
        np.savez_compressed(out / 'trajectory.npz', **trajectory)
        summary = dict(mode=args.mode, execution_steps=args.execution_steps,
            source_episode_ids=ids, episode_count=n,
            original_action_lengths=original_lengths.tolist(), action_lengths=lengths.tolist(),
            intended_steps=dict(settle=60, global_action=max_action_steps, hold=60),
            steps=dict(settle=settle_steps, global_action=action_steps, hold=hold_steps),
            stop_when_all_failed=args.stop_when_all_failed,
            native_protocol=PROTOCOL, prior=info, mass_kg=.17, friction=2.2,
            guidance_scale=50 if args.mode == 'guided' else None,
            guidance_steps=2 if args.mode == 'guided' else None,
            reference_interpolation=interpolation,
            wrist='native fixed wrist for each environment',
            init_order='properties; simulate/fetch/observe; root import',
            native_demo_match=[dict(index=di, frame=fr) for di, fr in zip(match_ids, match_frames)],
            initial_object_pose_wrist=local_object.tolist(), initial_wrist=wrist.tolist(),
            trajectory=str(out / 'trajectory.npz'))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('COMPLETE', args.mode, args.execution_steps, out, flush=True)
    finally:
        if sock is not None:
            try:
                send_message(sock, {'type': 'shutdown'})
                sock.close()
            except Exception:
                pass
        if asset is not None:
            native._restore_sharpa_asset_override(asset)
        native._destroy_environment(env)


if __name__ == '__main__':
    main()
