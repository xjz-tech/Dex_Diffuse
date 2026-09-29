"""Paired episode-53 direct-action and reference-guided rollouts.

Run each arm in a fresh, identically seeded native environment.  In particular,
advance/fetch once after changing rigid-body mass and before importing roots.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
CONTROLLER = Path('/home/carus/Program/dex-controller')
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
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=P / 'reference/reference.npz')
    parser.add_argument('--reference-id', type=int, default=1)
    parser.add_argument('--source-episode', type=int, default=53)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--stop-on-native-failure', action='store_true')
    parser.add_argument('--socket')
    parser.add_argument('--guidance-steps', type=int, default=2)
    parser.add_argument('--guidance-scale', type=float, default=25.)
    parser.add_argument('--object-mass-kg', type=float, default=.17)
    parser.add_argument('--object-size-multiplier', type=float, default=1.,
                        help='Multiply the native randomized bulb actor scale before state import')
    parser.add_argument('--friction', type=float, default=2.2)
    parser.add_argument('--vertical-scale-switch-angle-deg', type=float, default=None,
                        help='Once the simulated bulb +Y axis reaches this angle from vertical, lower guidance scale for subsequent predictions')
    parser.add_argument('--vertical-scale-after', type=float, default=None,
                        help='Guidance scale after the vertical-angle trigger; requires --vertical-scale-switch-angle-deg')
    parser.add_argument('--execution-steps', type=int, default=2)
    parser.add_argument('--reference-repeat', type=int, default=1)
    parser.add_argument('--reference-interpolation', type=int, default=0)
    parser.add_argument('--reference-interpolation-threshold', type=float, default=None,
                        help='Insert one midpoint only when any adjacent joint jump exceeds this threshold (rad)')
    parser.add_argument('--reference-interpolation-equal-jump', type=float, default=None,
                        help='Insert one midpoint only when the maximum adjacent joint jump equals this value (rad, absolute tolerance 1e-6)')
    parser.add_argument('--reference-mode', choices=['expanded', 'tile_per_call'], default='expanded')
    parser.add_argument('--front-only', action='store_true')
    parser.add_argument('--audit-recording', action='store_true',
                        help='Record import and settle, plus a wide view; physics is unchanged')
    parser.add_argument('--grasp-evidence', action='store_true',
                        help='Save contact forces and hand body poses without changing dynamics')
    parser.add_argument('--static-only', action='store_true',
                        help='Only import and settle; used to screen starts before controller outcomes')
    parser.add_argument('--settle-target-source', choices=['qpos', 'reference_action'], default='qpos',
                        help='Fixed settle target: initial measured joint angles (default), or first original recorded action; diagnostic variable')
    parser.add_argument('--action-limit', type=int, default=None,
                        help='Explicit diagnostic prefix length; source reference is not modified')
    parser.add_argument('--prior-noise-seed', type=int, default=44)
    parser.add_argument('--object-wrist-offset', type=float, nargs=3, default=[0.,0.,0.])
    args = parser.parse_args()
    if args.mode == 'guided' and not args.socket:
        parser.error('--socket is required in guided mode')
    if not np.isfinite(args.object_mass_kg) or args.object_mass_kg <= 0:
        parser.error('--object-mass-kg must be positive and finite')
    if not np.isfinite(args.object_size_multiplier) or args.object_size_multiplier <= 0:
        parser.error('--object-size-multiplier must be positive and finite')
    if not np.isfinite(args.friction) or args.friction < 0:
        parser.error('--friction must be nonnegative and finite')
    if args.reference_repeat < 1:
        parser.error('--reference-repeat must be positive')
    if args.reference_interpolation < 0 or (args.reference_interpolation and
       (args.reference_repeat != 1 or args.reference_mode != 'expanded')):
        parser.error('interpolation requires expanded mode and repeat=1')
    if args.reference_interpolation_threshold is not None and (
            args.reference_interpolation != 0 or args.reference_repeat != 1 or
            args.reference_mode != 'expanded'):
        parser.error('threshold interpolation requires expanded mode, repeat=1, no uniform insertion')
    if args.reference_interpolation_equal_jump is not None and (
            args.reference_interpolation_threshold is not None or
            args.reference_interpolation != 0 or args.reference_repeat != 1 or
            args.reference_mode != 'expanded'):
        parser.error('equal-jump interpolation requires expanded mode, repeat=1, no other insertion')
    if args.execution_steps < 1 or args.execution_steps > args.guidance_steps:
        parser.error('--execution-steps must be between 1 and --guidance-steps')
    if (args.vertical_scale_switch_angle_deg is None) != (args.vertical_scale_after is None):
        parser.error('vertical scale switch angle and after-scale must be specified together')
    if args.vertical_scale_switch_angle_deg is not None and (
            args.mode != 'guided' or not 0 < args.vertical_scale_switch_angle_deg < 180 or
            not np.isfinite(args.vertical_scale_after) or args.vertical_scale_after < 0 or
            args.vertical_scale_after > args.guidance_scale):
        parser.error('vertical scale switch requires guided mode, an angle in (0,180), and after-scale between 0 and initial scale')
    if args.reference_mode == 'tile_per_call' and args.reference_repeat != 1:
        parser.error('tile_per_call uses the original reference once per two-step inference call')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    from isaacgym import gymapi, gymtorch
    import cv2
    import torch
    from scipy.spatial.transform import Rotation
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT / 'eval'))
    import sim_eval as native
    from ipc import connect_unix, recv_message, send_message
    from reference_resampling import (interpolate_actions, interpolate_large_jumps,
                                      interpolate_equal_jumps)

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    ref = np.load(args.reference)
    target_repeat = args.execution_steps if args.reference_mode == 'tile_per_call' else args.reference_repeat
    if args.reference_interpolation_equal_jump is not None:
        expanded_targets, reference_progress = interpolate_equal_jumps(
            ref['hand_target_rad'], args.reference_interpolation_equal_jump)
        np.save(out / 'reference_progress.npy', reference_progress)
    elif args.reference_interpolation_threshold is None:
        expanded_targets = interpolate_actions(np.repeat(ref['hand_target_rad'], target_repeat, axis=1),
                                               args.reference_interpolation)
        reference_progress = None
    else:
        expanded_targets, reference_progress = interpolate_large_jumps(
            ref['hand_target_rad'], args.reference_interpolation_threshold)
        np.save(out / 'reference_progress.npy', reference_progress)
    action_steps = expanded_targets.shape[1]
    if args.action_limit is not None:
        assert args.action_limit > 0
        action_steps = min(action_steps, args.action_limit)
    ri = args.reference_id
    source_frames = ref['source_state_frame_indices'][ri]
    source_start, source_end = int(source_frames[0]), int(source_frames[-1])
    assert source_end > source_start
    assert ref['hand_target_rad'].shape[1] == source_end - source_start

    os.chdir(CONTROLLER)
    lib = native._import_local_maniptrans(CONTROLLER)
    from astra_demo_cache import install_demo_cache
    install_demo_cache(P / 'demo_cache')
    cfg = native._make_task_config(
        Path('/home/carus/Data/exp_data/hydra_config.yaml'), 1,
        ['v3:bulb2@%03d' % i for i in range(150)],
        CONTROLLER / 'data/NOKOV-v3',
        CONTROLLER / 'data/retargeting/NOKOV-v3', PROTOCOL)
    cfg.env.enableCameraSensors = True
    if args.object_size_multiplier != 1.:
        cfg.env.randomObjectScales = [float(s) * args.object_size_multiplier
                                      for s in cfg.env.randomObjectScales]
    asset = native._install_sharpa_asset_override(
        native._resolve_sharpa_urdf(CONTROLLER / 'maniptrans_envs/assets/sharpa_hand'))
    env = None
    sock = None
    writer = None
    try:
        env = lib.make(sim_device='cuda:0', rl_device='cuda:0',
                       graphics_device_id=0, multi_gpu=False, cfg=cfg,
                       display=False, record=False, has_headless_arg=True,
                       headless=True)
        native._restore_sharpa_asset_override(asset)
        asset = None
        env.compute_observations()
        env.reset()
        g, sim = env.gym, env.sim
        g.simulate(sim)
        g.fetch_results(sim, True)
        env.compute_observations()

        def cpu(x):
            return x.detach().cpu().numpy().copy()

        lower, upper = native._validate_environment(
            env, native._load_manifest(Path('/home/carus/Data/exp_data/hydra_config.yaml')))
        assert json.loads((P / 'reference/initial_state.json').read_text())['hand_joint_names'] == list(env.dexhand.dof_names)
        q0 = ref['hand_qpos_rad'][ri, 0][None].copy()
        settle_target = (q0 if args.settle_target_source == 'qpos' else
                         ref['hand_target_rad'][ri, 0][None].copy())
        held = np.clip(settle_target, cpu(lower), cpu(upper))
        demos = cpu(env.demo_data['opt_dof_pos'])
        lengths = cpu(env.demo_data['seq_len']).astype(int)
        dwr = Rotation.from_rotvec(cpu(env.demo_data['opt_wrist_rot']).reshape(-1, 3))
        dwp = cpu(env.demo_data['opt_wrist_pos']).reshape(-1, 3)
        dob = cpu(env.demo_data['obj_trajectory']).reshape(-1, 4, 4)
        drel = dwr.inv().apply(dob[:, :3, 3] - dwp).reshape(demos.shape[0], demos.shape[1], 3)
        qerr = np.sqrt(((demos - q0[0]) ** 2).mean(-1))
        perr = np.linalg.norm(drel - ref['object_pose_wrist'][ri, 0, :3, 3], axis=-1)
        score = qerr / .3 + perr / .05
        for di, length in enumerate(lengths):
            score[di, length:] = np.inf
        demo_index, demo_frame = np.unravel_index(score.argmin(), score.shape)

        e = env.envs[0]
        hand = g.find_actor_handle(e, 'dexhand')
        obj = g.find_actor_handle(e, 'manip_obj')
        original_object_scale = float(g.get_actor_scale(e, obj)) / args.object_size_multiplier
        for actor in (hand, obj):
            shapes = g.get_actor_rigid_shape_properties(e, actor)
            for shape in shapes:
                shape.friction = args.friction
            g.set_actor_rigid_shape_properties(e, actor, shapes)
        body = g.get_actor_rigid_body_properties(e, obj)
        body[0].mass = args.object_mass_kg
        g.set_actor_rigid_body_properties(e, obj, body, True)
        env.manip_obj_mass[:] = args.object_mass_kg
        assert abs(g.get_actor_rigid_body_properties(e, obj)[0].mass - args.object_mass_kg) < 1e-6

        # Isaac Gym GPU PhysX otherwise ignores the root import on its next step.
        g.simulate(sim)
        g.fetch_results(sim, True)
        env.compute_observations()

        wrist = cpu(env._base_state[:, :7])
        wr = Rotation.from_quat(wrist[:, 3:])
        local_obj = ref['object_pose_wrist'][ri, 0].copy()
        local_obj[:3,3] += np.asarray(args.object_wrist_offset)
        world_pos = wr.apply(local_obj[:3, 3][None]) + wrist[:, :3]
        world_quat = (wr * Rotation.from_matrix(local_obj[:3, :3])).as_quat()
        env._q[:] = torch.tensor(q0, device=env.device)
        env._qd.zero_()
        env.curr_targets[:] = torch.tensor(held, device=env.device)
        env.prev_targets[:] = env.curr_targets
        env._pos_control[:] = env.curr_targets
        env._manip_obj_root_state[:, :3] = torch.tensor(world_pos, device=env.device)
        env._manip_obj_root_state[:, 3:7] = torch.tensor(world_quat, device=env.device)
        env._manip_obj_root_state[:, 7:] = 0
        env.envidx_to_demoidx[:] = int(demo_index)
        env.global_cur_idx[:] = int(demo_frame)
        env.progress_buf[:] = int(demo_frame)
        for name in ('failure_progress_buf', 'reset_buf', 'failure_buf',
                     'success_buf', 'running_progress_buf', 'stable_frames_buf',
                     'traj_steps_counter', 'is_target_cross', 'error_buf'):
            getattr(env, name).zero_()
        hi = env._global_dexhand_indices.flatten()
        oi = env._global_manip_obj_indices.flatten()
        ids = torch.cat([hi, oi])
        g.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(env._dof_state),
                                       gymtorch.unwrap_tensor(hi), len(hi))
        assert g.set_actor_root_state_tensor_indexed(
            sim, gymtorch.unwrap_tensor(env._root_state), gymtorch.unwrap_tensor(ids), len(ids))
        g.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env._pos_control))
        env.compute_observations()
        native._dump_initial_state(env,
            native._current_policy_observation(env, 'qpos-target-residual'), out / 'initial_state.npz')
        (out / 'config.yaml').write_text(OmegaConf.to_yaml(cfg))

        cameras = []
        camera_metadata = []
        if not args.no_video:
            camera_offsets = ([-.32, .24, -.08],) if args.front_only else ([.12, .42, .14], [-.32, .24, -.08])
            if args.audit_recording:
                camera_offsets = ([-.32, .24, -.08], [-.85, .65, .40])
            for camera_index, offset in enumerate(camera_offsets):
                props = gymapi.CameraProperties()
                props.width, props.height, props.horizontal_fov = 640, 480, 42
                target = world_pos[0].copy() - wr.apply(np.asarray(args.object_wrist_offset)[None])[0]
                if args.audit_recording and camera_index == 1:
                    target[2] = -.20
                    props.horizontal_fov = 65
                cam = g.create_camera_sensor(e, props)
                eye = target + np.asarray(offset)
                eye[2] = max(eye[2], -.32)
                g.set_camera_location(cam, e, gymapi.Vec3(*eye), gymapi.Vec3(*target))
                cameras.append(cam)
                camera_metadata.append(dict(eye=eye.tolist(), target=target.tolist(),
                                            horizontal_fov=props.horizontal_fov))
            video_name = f'{args.mode}_front.mp4' if args.front_only else f'{args.mode}_two_views.mp4'
            width = 640 * len(cameras)
            writer = cv2.VideoWriter(str(out / video_name),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, 544))
            assert writer.isOpened()

        if args.mode == 'guided':
            sock = connect_unix(args.socket)
            sock.settimeout(600)
            send_message(sock, {'type': 'hello'})
            info, _ = recv_message(sock)
            assert info['ok']
            assert info['guidance_steps'] == args.guidance_steps
            assert info['guidance_scale'] == args.guidance_scale
            assert info['execution_steps'] == args.execution_steps
            assert info['reference_repeat'] == args.reference_repeat
            assert info['reference_interpolation'] == args.reference_interpolation
            assert info['reference_interpolation_threshold'] == args.reference_interpolation_threshold
            assert info['reference_interpolation_equal_jump'] == args.reference_interpolation_equal_jump
            assert info['reference_mode'] == args.reference_mode
        else:
            info = None

        trace = []
        def render_frame(phase, index, vertical, failure):
            g.fetch_results(sim, True)
            g.step_graphics(sim)
            g.render_all_camera_sensors(sim)
            panel = np.zeros((544, width, 3), dtype=np.uint8)
            for k, cam in enumerate(cameras):
                rgba = g.get_camera_image(sim, e, cam, gymapi.IMAGE_COLOR).reshape(480, 640, 4)
                panel[64:, k * 640:(k + 1) * 640] = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            label = 'raw recorded action' if args.mode == 'direct' else '10B prior + reference'
            cv2.putText(panel, f'episode {args.source_episode} | {label} | {args.object_mass_kg * 1000:g}g, mu {args.friction:g}',
                        (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .66, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(panel, f'{phase} {index / 30:.2f}s | vertical {vertical:.1f}deg | failure {failure}',
                        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .62, (190, 230, 255), 1, cv2.LINE_AA)
            if args.audit_recording:
                cv2.putText(panel, 'same front camera | wide view includes table', (670, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(panel)
            if ((phase == 'action' and index in (0, action_steps // 2 - 1, action_steps - 1)) or
                (phase == 'hold' and index in (0, 29, 59)) or
                (args.audit_recording and phase == 'settle' and index in (0, 5, 15, 59)) or
                phase == 'import'):
                cv2.imwrite(str(out / f'{args.mode}_{phase}{index:03d}.jpg'), panel)

        def step(command, phase, index, record):
            env.step(native._absolute_targets_to_env_action(command, lower, upper, env.device))
            actual_wrist = cpu(env._base_state[:, :7])
            assert np.allclose(actual_wrist, wrist, atol=1e-5)
            ob = cpu(env._manip_obj_root_state[:, :7])
            rot = Rotation.from_quat(ob[:, 3:])
            relp = wr.inv().apply(ob[:, :3] - wrist[:, :3])[0]
            relrot = wr.inv() * rot
            dpos = float(np.linalg.norm(relp - local_obj[:3, 3]))
            drot = float(np.degrees((Rotation.from_matrix(local_obj[:3, :3]).inv() * relrot).magnitude()[0]))
            vertical = float(np.degrees(np.arccos(np.clip(rot.apply([0, 1, 0])[0, 2], -1, 1))))
            failure = bool(cpu(env.failure_buf)[0])
            trace.append(dict(phase=phase, index=index, object_pose=ob[0].tolist(),
                              q=cpu(env._q)[0].tolist(), command=command[0].tolist(),
                              executed_target=cpu(env.curr_targets)[0].tolist(),
                              relative_position=relp.tolist(),
                              displacement_from_import_m=dpos,
                              rotation_from_import_deg=drot,
                              vertical_error_deg=vertical, native_failure=failure))
            if args.grasp_evidence:
                handles = [env.dexhand_handles[k] for k in env.dexhand.body_names]
                trace[-1].update(qd=cpu(env._qd)[0].tolist(),
                                 hand_body_pose=cpu(env._rigid_body_state[0, handles, :7]).tolist(),
                                 hand_contact_force=cpu(env.net_cf[0, handles]).tolist(),
                                 object_contact_force=cpu(env._manip_obj_cf)[0].tolist(),
                                 object_velocity=cpu(env._manip_obj_root_state)[0, 7:].tolist())
            if record and not args.no_video:
                render_frame(phase, index, vertical, failure)
            if index % 30 == 0:
                print(json.dumps(dict(mode=args.mode, phase=phase, index=index,
                                      displacement_m=dpos, rotation_deg=drot,
                                      failure=failure)), flush=True)
            return failure

        first_failure = None
        if args.audit_recording and not args.no_video:
            imported_vertical = float(np.degrees(np.arccos(np.clip(
                Rotation.from_quat(world_quat).apply([0, 1, 0])[0, 2], -1, 1))))
            render_frame('import', 0, imported_vertical, False)
        for j in range(60):
            failure = step(held, 'settle', j, args.audit_recording)
            if failure and first_failure is None:
                first_failure = dict(phase='settle', index=j)
            if failure and args.stop_on_native_failure:
                break
        vertical_scale_switched = bool(
            args.vertical_scale_switch_angle_deg is not None and
            trace[-1]['vertical_error_deg'] <= args.vertical_scale_switch_angle_deg)
        vertical_scale_trigger = (dict(phase='settle', index=59,
                                       vertical_error_deg=trace[-1]['vertical_error_deg'])
                                  if vertical_scale_switched else None)
        if not args.static_only and not (first_failure and args.stop_on_native_failure):
            history = np.repeat(native._current_policy_observation(env, 'qpos-target-residual')[:, None], 4, axis=1)
            for j in range(action_steps):
                if args.mode == 'direct':
                    command = expanded_targets[ri, j][None]
                else:
                    if j % args.execution_steps == 0:
                        inference_scale = (args.vertical_scale_after if vertical_scale_switched
                                           else args.guidance_scale)
                        send_message(sock, dict(type='predict', reference_index=j,
                            seeds=[args.prior_noise_seed], reference_ids=[ri],
                            guidance_scale=inference_scale), history.astype(np.float32))
                        msg, plan = recv_message(sock)
                        assert msg['ok']
                        assert msg['guidance_scale'] == inference_scale
                    command = plan[:, j % args.execution_steps]
                failure = step(command, 'action', j, True)
                if (not vertical_scale_switched and
                        args.vertical_scale_switch_angle_deg is not None and
                        trace[-1]['vertical_error_deg'] <= args.vertical_scale_switch_angle_deg):
                    vertical_scale_switched = True
                    vertical_scale_trigger = dict(phase='action', index=j,
                        vertical_error_deg=trace[-1]['vertical_error_deg'])
                if failure and first_failure is None:
                    first_failure = dict(phase='action', index=j)
                if failure and args.stop_on_native_failure:
                    break
                history = np.concatenate([history[:, 1:],
                    native._current_policy_observation(env, 'qpos-target-residual')[:, None]], axis=1)
        if not args.static_only and not (first_failure and args.stop_on_native_failure):
            last = cpu(env.curr_targets)
            for j in range(60):
                failure = step(last, 'hold', j, True)
                if failure and first_failure is None:
                    first_failure = dict(phase='hold', index=j)
                if failure and args.stop_on_native_failure:
                    break

        (out / 'trace.json').write_text(json.dumps(trace, indent=2) + '\n')
        phase_rows = {phase: [row for row in trace if row['phase'] == phase]
                      for phase in ('settle', 'action', 'hold')}
        summary = dict(prior_noise_seed=args.prior_noise_seed, object_wrist_offset_m=args.object_wrist_offset, mode=args.mode, source_episode=args.source_episode, source_start_frame=source_start,
            source_action_frames=[source_start, source_end - 1], control_hz=30,
            reference=str(args.reference), reference_id=ri,
            reference_repeat=args.reference_repeat,
            reference_interpolation=args.reference_interpolation,
            reference_interpolation_threshold=args.reference_interpolation_threshold,
            reference_interpolation_equal_jump=args.reference_interpolation_equal_jump,
            adaptive_inserted_steps=(int(action_steps - ref['hand_target_rad'].shape[1])
                                     if reference_progress is not None else None),
            reference_mode=args.reference_mode,
            front_only=args.front_only, video_recorded=not args.no_video,
            audit_recording=args.audit_recording, camera_metadata=camera_metadata,
            static_only=args.static_only, grasp_evidence=args.grasp_evidence,
            settle_target_source=args.settle_target_source,
            settle_target_rad=held[0].tolist(),
            action_limit=args.action_limit,
            hand_body_names=list(env.dexhand.body_names) if args.grasp_evidence else None,
            execution_steps=args.execution_steps,
            guidance_scale=args.guidance_scale,
            vertical_scale_switch_angle_deg=args.vertical_scale_switch_angle_deg,
            vertical_scale_after=args.vertical_scale_after,
            vertical_scale_trigger=vertical_scale_trigger,
            intended_steps=dict(settle=60, action=0 if args.static_only else action_steps,
                                hold=0 if args.static_only else 60),
            steps={phase: len(rows) for phase, rows in phase_rows.items()},
            stop_on_native_failure=args.stop_on_native_failure,
            first_native_failure=first_failure,
            native_protocol=PROTOCOL, prior=info,
            initial_object_pose_wrist=local_obj.tolist(), initial_wrist=wrist[0].tolist(),
            native_demo_match=dict(index=int(demo_index), frame=int(demo_frame)),
            mass_kg=args.object_mass_kg, friction=args.friction,
            object_size_multiplier=args.object_size_multiplier,
            original_object_scale=original_object_scale,
            actual_object_scale=float(g.get_actor_scale(e, obj)),
            init_order='properties; simulate/fetch/observe; root import',
            settle_end=phase_rows['settle'][-1],
            action_end=phase_rows['action'][-1] if phase_rows['action'] else None,
            hold_end=phase_rows['hold'][-1] if phase_rows['hold'] else None,
            any_native_failure=any(t['native_failure'] for t in trace))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('COMPLETE', args.mode, out, flush=True)
    finally:
        if writer is not None:
            writer.release()
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
