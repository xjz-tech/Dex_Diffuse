"""Check recorded grasps before any policy action; optional offset is exploratory."""

from pathlib import Path
import json
import os
import random
import sys

import numpy as np

P = Path(__file__).resolve().parent
ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
CONTROLLER = Path('/home/carus/Program/dex-controller')
OFFSET_X_M = float(os.environ.get('STATIC_EXPLORATORY_OFFSET_X_M', '0'))
OFFSET_Z_M = float(os.environ.get('STATIC_EXPLORATORY_OFFSET_Z_M', '0'))
OFFSET_WORLD_Z_M = float(os.environ.get('STATIC_EXPLORATORY_OFFSET_WORLD_Z_M', '0'))
STEPS = int(os.environ.get('STATIC_STEPS', '60'))
USE_FULL_ROOT_SET = os.environ.get('STATIC_USE_FULL_ROOT_SET', '0') == '1'
OBJECT_ONLY_ROOT_SET = os.environ.get('STATIC_OBJECT_ONLY_ROOT_SET', '0') == '1'
CPU_SIM = os.environ.get('STATIC_CPU_SIM', '0') == '1'
SKIP_PROPERTY_SET = os.environ.get('STATIC_SKIP_PROPERTY_SET', '0') == '1'
PROPERTY_MODE = 'none' if SKIP_PROPERTY_SET else os.environ.get('STATIC_PROPERTY_MODE', 'all')
FLUSH_AFTER_PROPERTIES = os.environ.get('STATIC_FLUSH_AFTER_PROPERTIES', '0') == '1'
DIRECT_SIM = os.environ.get('STATIC_DIRECT_SIM', '0') == '1'
SKIP_REFRESH_AFTER_IMPORT = os.environ.get('STATIC_SKIP_REFRESH_AFTER_IMPORT', '0') == '1'
OUT = P / ('static_unperturbed' if OFFSET_X_M == 0 and OFFSET_Z_M == 0 and OFFSET_WORLD_Z_M == 0 else f'static_exploratory_x_{OFFSET_X_M:+.3f}m_z_{OFFSET_Z_M:+.3f}m_worldz_{OFFSET_WORLD_Z_M:+.3f}m')
if USE_FULL_ROOT_SET:
    OUT = Path(str(OUT) + '_fullroot')
if OBJECT_ONLY_ROOT_SET:
    OUT = Path(str(OUT) + '_objectonly')
if CPU_SIM:
    OUT = Path(str(OUT) + '_cpusim')
if SKIP_PROPERTY_SET:
    OUT = Path(str(OUT) + '_skipprops')
elif PROPERTY_MODE != 'all':
    OUT = Path(str(OUT) + '_property_' + PROPERTY_MODE)
if FLUSH_AFTER_PROPERTIES:
    OUT = Path(str(OUT) + '_flushprops')
if DIRECT_SIM:
    OUT = Path(str(OUT) + '_directsim')
if SKIP_REFRESH_AFTER_IMPORT:
    OUT = Path(str(OUT) + '_skiprefresh')


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    OUT.mkdir(exist_ok=True)
    from isaacgym import gymapi, gymtorch
    import cv2
    import torch
    from scipy.spatial.transform import Rotation
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT / 'eval'))
    import sim_eval as native

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    os.chdir(CONTROLLER)
    lib = native._import_local_maniptrans(CONTROLLER)
    from astra_demo_cache import install_demo_cache
    install_demo_cache(P / 'demo_cache')
    ref = np.load(P / 'reference/reference.npz')
    specs = json.loads((P / 'reference/initial_state.json').read_text())['specs']
    n = len(specs)
    resets = dict(
        failureObjPosThres=.05, failureThumbTipPosThres=.1,
        failureIndexTipPosThres=.1, failureMiddleTipPosThres=.1,
        failurePinkyTipPosThres=.1, failureRingTipPosThres=.1,
        failureObjRotThres=180., invalidObjPosThres=.15,
        FailureToleranceScale=10000., fixedToleranceSteps=20000,
        trajStepsLimit=12000, resetOnReachGoal=False,
        enableCrossTrajectoryReset=True, crossTrajectoryGoalProb=.3,
    )
    cfg = native._make_task_config(
        Path('/home/carus/Data/exp_data/hydra_config.yaml'), n,
        ['v3:bulb2@%03d' % i for i in range(150)],
        CONTROLLER / 'data/NOKOV-v3',
        CONTROLLER / 'data/retargeting/NOKOV-v3', resets,
    )
    cfg.env.enableCameraSensors = True
    if CPU_SIM:
        cfg.sim.use_gpu_pipeline = False
        cfg.sim.physx.use_gpu = False
    asset = native._install_sharpa_asset_override(
        native._resolve_sharpa_urdf(CONTROLLER / 'maniptrans_envs/assets/sharpa_hand'))
    env = None
    writers = []
    try:
        device_name = 'cpu' if CPU_SIM else 'cuda:0'
        env = lib.make(sim_device=device_name, rl_device=device_name, graphics_device_id=0,
                       multi_gpu=False, cfg=cfg, display=False, record=False,
                       has_headless_arg=True, headless=True)
        native._restore_sharpa_asset_override(asset)
        asset = None
        env.compute_observations()
        env.reset()
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env.compute_observations()
        g, sim = env.gym, env.sim

        def cpu(tensor):
            return tensor.detach().cpu().numpy().copy()

        lower, upper = native._validate_environment(
            env, native._load_manifest(Path('/home/carus/Data/exp_data/hydra_config.yaml')))
        assert json.loads((P / 'reference/initial_state.json').read_text())['hand_joint_names'] == list(env.dexhand.dof_names)
        q0 = np.stack([ref['hand_qpos_rad'][i, 0] for i in range(n)])
        # Preserve the recorded state exactly. The native action path applies
        # the simulator's joint limits to the held target (at most 0.0036 rad).
        held_target = np.clip(q0, cpu(lower), cpu(upper))
        demos = cpu(env.demo_data['opt_dof_pos'])
        lengths = cpu(env.demo_data['seq_len']).astype(int)
        demo_wr = Rotation.from_rotvec(cpu(env.demo_data['opt_wrist_rot']).reshape(-1, 3))
        demo_wp = cpu(env.demo_data['opt_wrist_pos']).reshape(-1, 3)
        demo_obj = cpu(env.demo_data['obj_trajectory']).reshape(-1, 4, 4)
        demo_rel = demo_wr.inv().apply(demo_obj[:, :3, 3] - demo_wp).reshape(demos.shape[0], demos.shape[1], 3)
        matches = []
        for i in range(n):
            qerr = np.sqrt(((demos - q0[i]) ** 2).mean(-1))
            perr = np.linalg.norm(demo_rel - ref['object_pose_wrist'][i, 0, :3, 3], axis=-1)
            score = qerr / .3 + perr / .05
            for di, length in enumerate(lengths):
                score[di, length:] = np.inf
            matches.append(np.unravel_index(score.argmin(), score.shape))
        if PROPERTY_MODE != 'none':
            for i in range(n):
                e = env.envs[i]
                hand = g.find_actor_handle(e, 'dexhand')
                obj = g.find_actor_handle(e, 'manip_obj')
                if PROPERTY_MODE in ('all', 'hand_friction'):
                    hand_shapes = g.get_actor_rigid_shape_properties(e, hand)
                    for shape in hand_shapes:
                        shape.friction = 2.2
                    g.set_actor_rigid_shape_properties(e, hand, hand_shapes)
                    assert all(abs(s.friction - 2.2) < 1e-6 for s in g.get_actor_rigid_shape_properties(e, hand))
                if PROPERTY_MODE in ('all', 'object_friction'):
                    obj_shapes = g.get_actor_rigid_shape_properties(e, obj)
                    for shape in obj_shapes:
                        shape.friction = 2.2
                    g.set_actor_rigid_shape_properties(e, obj, obj_shapes)
                    assert all(abs(s.friction - 2.2) < 1e-6 for s in g.get_actor_rigid_shape_properties(e, obj))
                if PROPERTY_MODE in ('all', 'object_mass'):
                    body = g.get_actor_rigid_body_properties(e, obj)
                    body[0].mass = .17
                    g.set_actor_rigid_body_properties(e, obj, body, True)
                    assert abs(g.get_actor_rigid_body_properties(e, obj)[0].mass - .17) < 1e-6
            if PROPERTY_MODE in ('all', 'object_mass'):
                env.manip_obj_mass[:] = .17
        if FLUSH_AFTER_PROPERTIES:
            g.simulate(sim)
            g.fetch_results(sim, True)
            env.compute_observations()
        native_before_import = cpu(env._manip_obj_root_state[:, :13])
        wrist = cpu(env._base_state[:, :7])
        wrist_rot = Rotation.from_quat(wrist[:, 3:])
        local_obj = ref['object_pose_wrist'][:, 0].copy()
        local_obj[:, 0, 3] += OFFSET_X_M
        local_obj[:, 2, 3] += OFFSET_Z_M
        local_obj[:, :3, 3] += wrist_rot.inv().apply(np.tile([0., 0., OFFSET_WORLD_Z_M], (n, 1)))
        world_pos = wrist_rot.apply(local_obj[:, :3, 3]) + wrist[:, :3]
        world_quat = (wrist_rot * Rotation.from_matrix(local_obj[:, :3, :3])).as_quat()
        env._q[:] = torch.tensor(q0, device=env.device)
        env._qd.zero_()
        env.curr_targets[:] = torch.tensor(held_target, device=env.device)
        env.prev_targets[:] = env.curr_targets
        env._pos_control[:] = env.curr_targets
        env._manip_obj_root_state[:, :3] = torch.tensor(world_pos, device=env.device)
        env._manip_obj_root_state[:, 3:7] = torch.tensor(world_quat, device=env.device)
        env._manip_obj_root_state[:, 7:] = 0
        env.envidx_to_demoidx[:] = torch.tensor([x[0] for x in matches], device=env.device)
        for name in ['global_cur_idx', 'progress_buf']:
            getattr(env, name)[:] = torch.tensor([x[1] for x in matches], device=env.device)
        for name in ['failure_progress_buf', 'reset_buf', 'failure_buf', 'success_buf',
                     'running_progress_buf', 'stable_frames_buf', 'traj_steps_counter',
                     'is_target_cross', 'error_buf']:
            getattr(env, name).zero_()
        hi = env._global_dexhand_indices.flatten()
        oi = env._global_manip_obj_indices.flatten()
        assert all(int(oi[i]) == g.get_actor_index(
            env.envs[i], g.find_actor_handle(env.envs[i], 'manip_obj'), gymapi.DOMAIN_SIM)
            for i in range(n))
        ids = torch.cat([hi, oi])
        g.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(env._dof_state), gymtorch.unwrap_tensor(hi), len(hi))
        if OBJECT_ONLY_ROOT_SET:
            root_set_result = g.set_actor_root_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(env._root_state), gymtorch.unwrap_tensor(oi), len(oi))
        elif USE_FULL_ROOT_SET:
            root_set_result = g.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(env._root_state))
        else:
            root_set_result = g.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(env._root_state), gymtorch.unwrap_tensor(ids), len(ids))
        g.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(env._pos_control))
        actor_readback = []
        for i in range(n):
            try:
                actor = g.find_actor_handle(env.envs[i], 'manip_obj')
                states = g.get_actor_rigid_body_states(env.envs[i], actor, gymapi.STATE_ALL)
                actor_readback.append(dict(pos=states['pose']['p'][0].tolist(),
                                           quat=states['pose']['r'][0].tolist()))
            except Exception as exc:
                actor_readback.append(dict(error=repr(exc)))
        if not SKIP_REFRESH_AFTER_IMPORT:
            env.compute_observations()
        if DIRECT_SIM:
            before = cpu(env._manip_obj_root_state[:, :13])
            frames = []
            for _ in range(3):
                g.simulate(sim)
                g.fetch_results(sim, True)
                g.refresh_actor_root_state_tensor(sim)
                frames.append(cpu(env._manip_obj_root_state[:, :13]))
            save_json(OUT / 'direct_sim.json', dict(before=before.tolist(), frames=[x.tolist() for x in frames],
                      root_set_result=str(root_set_result),
                      displacement_m=[np.linalg.norm(x[:, :3] - before[:, :3], axis=-1).tolist() for x in frames]))
            return
        native._dump_initial_state(env, native._current_policy_observation(env, 'qpos-target-residual'), OUT / 'initial_state.npz')
        (OUT / 'config.yaml').write_text(OmegaConf.to_yaml(cfg))
        save_json(OUT / 'setup.json', dict(specs=specs, native_protocol=resets,
              wrist='native reset, unchanged', source_pose='row-6D; wrist-local X translation applied only when exploratory offset is nonzero',
              exploratory_wrist_local_x_offset_m=OFFSET_X_M,
              exploratory_wrist_local_z_offset_m=OFFSET_Z_M,
              exploratory_world_z_offset_m=OFFSET_WORLD_Z_M,
              full_root_tensor_set=USE_FULL_ROOT_SET,
              native_object_before_import=native_before_import.tolist(),
              tensor_object_after_import=cpu(env._manip_obj_root_state[:, :13]).tolist(),
              actor_object_readback_after_import=actor_readback,
              object_sim_indices=cpu(oi).tolist(),
              hand_joint_target=f'recorded first frame, native-limit clipped and held for {STEPS} steps',
              maximum_native_joint_target_clip_rad=float(np.max(abs(q0 - held_target))),
              mass_kg=.17, hand_and_object_friction=2.2,
              matches=[dict(demo_index=int(x[0]), demo_frame=int(x[1])) for x in matches]))

        cameras = []
        for i in range(n):
            pair = []
            for offset in [[.12, .42, .14], [-.32, .24, -.08]]:
                props = gymapi.CameraProperties()
                props.width, props.height, props.horizontal_fov = 640, 480, 42
                cam = g.create_camera_sensor(env.envs[i], props)
                eye = world_pos[i] + np.asarray(offset)
                eye[2] = max(eye[2], -.32)
                g.set_camera_location(cam, env.envs[i], gymapi.Vec3(*eye), gymapi.Vec3(*world_pos[i]))
                pair.append(cam)
            cameras.append(pair)
            writer = cv2.VideoWriter(str(OUT / f'episode{specs[i]["episode"]}_static.mp4'),
                                     cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 544))
            assert writer.isOpened()
            writers.append(writer)
        trace = []
        initial_rel = local_obj[:, :3, 3].copy()
        initial_rot = Rotation.from_matrix(local_obj[:, :3, :3])
        for step in range(STEPS):
            env.step(native._absolute_targets_to_env_action(held_target, lower, upper, env.device))
            actual_wrist = cpu(env._base_state[:, :7])
            actual_obj = cpu(env._manip_obj_root_state[:, :7])
            actual_q = cpu(env._q)
            assert np.allclose(actual_wrist, wrist, atol=1e-5)
            wr = Rotation.from_quat(actual_wrist[:, 3:])
            ob = Rotation.from_quat(actual_obj[:, 3:])
            rel_pos = wr.inv().apply(actual_obj[:, :3] - actual_wrist[:, :3])
            rel_rot = wr.inv() * ob
            delta_m = np.linalg.norm(rel_pos - initial_rel, axis=-1)
            delta_deg = np.degrees((initial_rot.inv() * rel_rot).magnitude())
            failure = cpu(env.failure_buf).astype(bool)
            trace.append(dict(step=step + 1, time_s=(step + 1) / 30,
                              relative_position_m=rel_pos.tolist(),
                              object_translation_m=delta_m.tolist(),
                              object_rotation_deg=delta_deg.tolist(),
                              max_joint_change_rad=np.max(abs(actual_q - q0), axis=-1).tolist(),
                              native_failure=failure.tolist()))
            g.fetch_results(sim, True)
            g.step_graphics(sim)
            g.render_all_camera_sensors(sim)
            for i in range(n):
                panel = np.zeros((544, 1280, 3), dtype=np.uint8)
                for k, cam in enumerate(cameras[i]):
                    rgba = g.get_camera_image(sim, env.envs[i], cam, gymapi.IMAGE_COLOR).reshape(480, 640, 4)
                    panel[64:, k * 640:(k + 1) * 640] = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.putText(panel, f'episode {specs[i]["episode"]} frame {specs[i]["start"]} | no perturbation | 170g mu2.2',
                            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, f'physics step {step + 1} | displacement {delta_m[i] * 100:.1f} cm | rotation {delta_deg[i]:.1f} deg',
                            (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .62, (190, 230, 255), 1, cv2.LINE_AA)
                writers[i].write(panel)
                if step in [0, 29, 59]:
                    cv2.imwrite(str(OUT / f'episode{specs[i]["episode"]}_step{step + 1:02d}.jpg'), panel)
            if step in [0, 29, 59]:
                print(json.dumps(trace[-1]), flush=True)
        save_json(OUT / 'trace.json', trace)
        save_json(OUT / 'summary.json', dict(
            episodes=[dict(episode=specs[i]['episode'], start_frame=specs[i]['start'],
                           first_step_translation_m=trace[0]['object_translation_m'][i],
                           first_step_rotation_deg=trace[0]['object_rotation_deg'][i],
                           end_translation_m=trace[-1]['object_translation_m'][i],
                           end_rotation_deg=trace[-1]['object_rotation_deg'][i],
                           any_native_failure=any(t['native_failure'][i] for t in trace))
                      for i in range(n)]))
    finally:
        for writer in writers:
            writer.release()
        if asset is not None:
            native._restore_sharpa_asset_override(asset)
        native._destroy_environment(env)


if __name__ == '__main__':
    main()
