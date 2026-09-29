#!/usr/bin/env python3
"""Isolated real-reference experiment using the native bulb task and failures.

The placement fit uses a static hand only, never the rotation outcome.
Run in decv2; guided inference uses a separate dp process over existing IPC.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
REFERENCE = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/docs/experiment_reviews/20260924_horizontal_to_vertical/reference/reference.npz')
CONTROLLER = Path('/home/carus/Program/dex-controller')

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')

def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['fit','evaluate'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--placement',type=Path)
    p.add_argument('--mode',choices=['direct','guided'],default='guided')
    p.add_argument('--socket')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--settle-steps',type=int,default=60)
    p.add_argument('--hold-steps',type=int,default=30)
    return p.parse_args()

def main():
    args=parse();args.output.mkdir(parents=True,exist_ok=True)
    from isaacgym import gymapi, gymtorch  # must precede torch
    import torch
    import cv2
    from scipy.spatial.transform import Rotation as R
    sys.path.insert(0,str(ROOT/'eval'))
    import sim_eval as native
    from ipc import connect_unix,send_message,recv_message

    torch.set_num_threads(2)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);np.random.seed(args.seed)
    os.chdir(str(CONTROLLER))
    lib=native._import_local_maniptrans(CONTROLLER)
    count=144 if args.phase=='fit' else 3
    resets=dict(failureObjPosThres=.05,failureThumbTipPosThres=.1,
        failureIndexTipPosThres=.1,failureMiddleTipPosThres=.1,
        failurePinkyTipPosThres=.1,failureRingTipPosThres=.1,
        failureObjRotThres=180.,invalidObjPosThres=.15,
        FailureToleranceScale=10000.,fixedToleranceSteps=20000,
        trajStepsLimit=12000,resetOnReachGoal=False,
        enableCrossTrajectoryReset=True,crossTrajectoryGoalProb=.3)
    cfg=native._make_task_config(Path('/home/carus/Data/exp_data/hydra_config.yaml'),count,
        ['v3:bulb2@%03d'%i for i in range(150)],
        CONTROLLER/'data/NOKOV-v3',CONTROLLER/'data/retargeting/NOKOV-v3',resets)
    cfg.env.enableCameraSensors=True  # keep graphics device in headless mode
    override=native._install_sharpa_asset_override(native._resolve_sharpa_urdf(CONTROLLER/'maniptrans_envs/assets/sharpa_hand'))
    env=None;sock=None;writers=[]
    try:
        env=lib.make(sim_device='cuda:0',rl_device='cuda:0',graphics_device_id=0,
            multi_gpu=False,cfg=cfg,display=False,record=False,has_headless_arg=True,headless=True)
        native._restore_sharpa_asset_override(override);override=None
        env.compute_observations();env.reset()
        env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True)
        env.compute_observations()
        lower,upper=native._validate_environment(env,native._load_manifest(Path('/home/carus/Data/exp_data/hydra_config.yaml')))
        ref=np.load(REFERENCE,allow_pickle=False)
        q0=ref['hand_qpos_rad'][0]
        names=json.loads(REFERENCE.with_name('initial_state.json').read_text())['hand_joint_names']
        assert names==list(env.dexhand.dof_names), 'DOF mapping must be explicit'
        q0=np.clip(q0,lower.cpu().numpy(),upper.cpu().numpy())
        dt=float(env.dt*env.control_freq_inv)
        assert abs(dt-1/30)<1e-5,dt

        def arr(x):return x.detach().cpu().numpy().copy()
        def relative():
            base=arr(env._base_state[:,:7]);obj=arr(env._manip_obj_root_state[:,:7])
            bw=R.from_quat(base[:,3:]);ow=R.from_quat(obj[:,3:])
            return bw.inv().apply(obj[:,:3]-base[:,:3]),(bw.inv()*ow).as_quat()

        if args.phase=='fit':
            demos=arr(env.demo_data['opt_dof_pos']);lengths=arr(env.demo_data['seq_len']).astype(int)
            errors=((demos-q0)**2).mean(-1)
            for k,n in enumerate(lengths):errors[k,n:]=np.inf
            matches=[]
            for k in np.argsort(errors.min(1))[:3]:
                frame=int(errors[k].argmin())
                wr=R.from_rotvec(arr(env.demo_data['opt_wrist_rot'][k,frame]))
                wp=arr(env.demo_data['opt_wrist_pos'][k,frame]);obj=arr(env.demo_data['obj_trajectory'][k,frame])
                matches.append(dict(demo_index=int(k),demo_frame=frame,
                    initial_q_rmse_rad=float(np.sqrt(errors[k,frame])),
                    object_position_wrist=wr.inv().apply(obj[:3,3]-wp).tolist(),
                    object_quaternion_wrist_xyzw=(wr.inv()*R.from_matrix(obj[:3,:3])).as_quat().tolist()))
            offsets=[[x,y,z] for x in [0.,.02,.04,.06] for y in [0.,.02,.04,.06] for z in [-.02,0.,.02]]
            placements=[]
            for m in matches:
                for offset in offsets:
                    entry=dict(m);entry['offset_wrist_m']=offset
                    entry['object_position_wrist']=(np.array(m['object_position_wrist'])+offset).tolist()
                    placements.append(entry)
        else:
            selected=json.loads(args.placement.read_text())['selected_placement']
            placements=[selected.copy() for _ in range(count)]

        # Keep each environment's native randomized wrist and physical properties;
        # change only hand q0 and object placement, as required by this experiment.
        env._base_state[:,:3]=torch.tensor([0.,0.,.6],device=env.device)
        env._base_state[:,3:7]=torch.tensor([1.,0.,0.,0.],device=env.device)
        base=arr(env._base_state[:,:7]);brot=R.from_quat(base[:,3:])
        obj_pos=brot.apply(np.array([p['object_position_wrist'] for p in placements]))+base[:,:3]
        obj_rot=(brot*R.from_quat([p['object_quaternion_wrist_xyzw'] for p in placements])).as_quat()
        env._q[:]=torch.tensor(q0,device=env.device);env._qd.zero_()
        env.curr_targets[:]=env._q;env.prev_targets[:]=env._q;env._pos_control[:]=env._q
        env._base_state[:,7:]=0
        env._manip_obj_root_state[:,:3]=torch.tensor(obj_pos,device=env.device,dtype=torch.float32)
        env._manip_obj_root_state[:,3:7]=torch.tensor(obj_rot,device=env.device,dtype=torch.float32)
        env._manip_obj_root_state[:,7:]=0
        # Associate the native target stream with the source grasp, retaining
        # its own target advance and failure reference-frame implementation.
        env.envidx_to_demoidx[:]=torch.tensor([p['demo_index'] for p in placements],device=env.device)
        for attr in ['global_cur_idx','progress_buf']:
            getattr(env,attr)[:]=torch.tensor([p['demo_frame'] for p in placements],device=env.device)
        for attr in ['failure_progress_buf','reset_buf','failure_buf','success_buf','running_progress_buf','stable_frames_buf']:
            getattr(env,attr).zero_()
        hand_ids=env._global_dexhand_indices.flatten();obj_ids=env._global_manip_obj_indices.flatten()
        ids=torch.cat([hand_ids,obj_ids])
        env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env._dof_state),gymtorch.unwrap_tensor(hand_ids),len(hand_ids))
        env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env._root_state),gymtorch.unwrap_tensor(ids),len(ids))
        env.gym.set_dof_position_target_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env._pos_control),gymtorch.unwrap_tensor(hand_ids),len(hand_ids))
        env.compute_observations()
        initial_pos,initial_rot=relative()
        native._dump_initial_state(env,native._current_policy_observation(env,'qpos-target-residual'),args.output/'initial_state.npz')

        cameras=[]
        for i in range(count):
            prop=gymapi.CameraProperties();prop.width=640;prop.height=480;prop.horizontal_fov=55
            cam=env.gym.create_camera_sensor(env.envs[i],prop)
            # Position cameras in world coordinates around the actual hand.
            center=obj_pos[i]
            eye=center+np.array([.25,-.34,.12])
            env.gym.set_camera_location(cam,env.envs[i],gymapi.Vec3(*eye),gymapi.Vec3(*center))
            cameras.append(cam)
        def capture(label,step,save=False):
            env.gym.fetch_results(env.sim,True);env.gym.step_graphics(env.sim);env.gym.render_all_camera_sensors(env.sim)
            frames=[]
            for i,cam in enumerate(cameras):
                rgb=env.gym.get_camera_image(env.sim,env.envs[i],cam,gymapi.IMAGE_COLOR).reshape(480,640,4)[:,:,:3]
                bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                cv2.putText(bgr,'%s | env %d | step %d'%(label,i,step),(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                if save:cv2.imwrite(str(args.output/('%s_env%02d.jpg'%(label,i))),bgr)
                if writers:writers[i].write(bgr)
                frames.append(bgr)
            return frames
        # Camera FK is only refreshed after the first physics step. Do not
        # label stale pre-step rigid-body transforms as the initialized hand.
        if args.phase=='evaluate':
            for i in range(count):
                w=cv2.VideoWriter(str(args.output/('env%02d.mp4'%i)),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
                assert w.isOpened();writers.append(w)
        records=[];ever_failure=np.zeros(count,dtype=bool)
        def snapshot(phase,index,command):
            p,r=relative()
            forces=np.stack([arr(env.net_cf[:,env.dexhand_handles[k]]) for k in env.dexhand.contact_body_names],axis=1)
            records.append(dict(phase=phase,index=index,q=arr(env._q),target=arr(env.curr_targets),
                command=command.copy(),object_pose=arr(env._manip_obj_root_state[:,:7]),
                wrist_pose=arr(env._base_state[:,:7]),relative_position=p,relative_quaternion=r,
                contact_force_norm=np.linalg.norm(forces,axis=-1),failure=arr(env.failure_buf).astype(bool)))
            ever_failure[:]|=records[-1]['failure']
        def step(command,phase,index):
            env.step(native._absolute_targets_to_env_action(command,lower,upper,env.device))
            snapshot(phase,index,command)
            if args.phase=='evaluate':capture(phase,index)
            if index==0 and phase=='settle':
                assert np.max(np.abs(arr(env._base_state[:,:7])-base))<1e-5, 'fixed base snapped back'
        constant=np.repeat(q0[None],count,axis=0)
        for j in range(args.settle_steps):
            step(constant,'settle',j)
            if j==0:capture('settle_start',j,True)
        capture('settled',args.settle_steps,True)
        settled_pos,settled_rot=relative()
        fit_drift=np.linalg.norm(settled_pos-initial_pos,axis=1)
        q_rmse=np.sqrt(((arr(env._q)-q0)**2).mean(1))
        contacts=(records[-1]['contact_force_norm']>.05).sum(1)
        if args.phase=='fit':
            # This score contains NO reference motion or rotation outcome.
            axes=R.from_quat(settled_rot).apply([0,1,0])
            horizontal_error=np.abs(axes[:,2])
            score=fit_drift+.05*q_rmse+100*ever_failure+.05*(contacts<2)+.1*horizontal_error
            order=np.argsort(score);best=int(order[0])
            table=[dict(environment=i,placement=placements[i],position_drift_m=float(fit_drift[i]),
                final_q_rmse_rad=float(q_rmse[i]),contacting_tip_count=int(contacts[i]),
                native_failure=bool(ever_failure[i]),horizontal_tilt_deg=float(np.degrees(np.arcsin(horizontal_error[i]))),static_score=float(score[i])) for i in range(count)]
            dump(args.output/'placement.json',dict(selected_environment=best,selected_placement=placements[best],
                selection_basis='2 s static first-frame hand hold; no rotation outcome used',candidates=table,
                selected_settled_relative_pose_xyz_xyzw=np.r_[settled_pos[best],settled_rot[best]].tolist()))
            print('[fit]',json.dumps(table[best]),flush=True)
        else:
            settled_failure=ever_failure.copy()
            if args.mode=='guided':
                if not args.socket:raise ValueError('guided mode requires --socket')
                sock=connect_unix(args.socket);sock.settimeout(600)
                send_message(sock,{'type':'hello'})
                info,_=recv_message(sock);assert info['ok'],info
                print('[policy]',info,flush=True)
            history=np.repeat(native._current_policy_observation(env,'qpos-target-residual')[:,None],4,axis=1)
            # Nominal 30 Hz, one original action per control step. No interpolation.
            action_ref=ref['hand_target_rad']
            plan=None
            for j in range(len(action_ref)):
                if args.mode=='direct':command=np.repeat(action_ref[j][None],count,axis=0)
                else:
                    if j%2==0:
                        send_message(sock,{'type':'predict','reference_index':j},history.astype(np.float32))
                        response,plan=recv_message(sock)
                        if not response.get('ok'):raise RuntimeError(response)
                    command=plan[:,j%2]
                step(command,'reference',j)
                history=np.concatenate([history[:,1:],native._current_policy_observation(env,'qpos-target-residual')[:,None]],axis=1)
            capture('reference_end',len(action_ref),True)
            last=arr(env.curr_targets)
            for j in range(args.hold_steps):step(last,'post_hold',j)
            capture('final',args.hold_steps,True)
            dump(args.output/'run_metadata.json',dict(mode=args.mode,environment_seed=args.seed,
                prior_noise_seeds=[42,43,44] if args.mode=='guided' else None,native_failure_during_settle=settled_failure.tolist(),
                native_failure_anytime=ever_failure.tolist(),settle_drift_m=fit_drift.tolist(),
                settled_contacting_tip_count=contacts.tolist(),control_dt_s=dt,
                reference_actions=len(action_ref),reference_duration_sim_s=len(action_ref)*dt,
                original_reference_duration_s=float(ref['time_s'][-1]),
                time_mapping='one original action per nominal 30 Hz control step; no interpolation',
                rotation_axis_object_local=[0,1,0],dome_side='+Y',thread_side='-Y',
                right_rotation='negative body-Y spin, viewed from dome toward screw base',
                reference=str(REFERENCE),placement=placements[0],
                reset_protocol=resets,source_demo_range='000-149',
                deliberate_changes=['recorded real action sequence is the guidance reference','hand q0 set to real initial measured state','object placement fitted by static grasp only','fixed hand root; no real TCP replay because mount transform unavailable','free object; no socket','finite 90-action horizontal-to-vertical test','wrist pose fixed fingers downward at xyz 0,0,0.6 and xyzw 1,0,0,0; engineering approximation of source wrist'],
                unchanged=['native physical randomization and external forces','native target updates and failure logic','native 30 Hz control frequency','simulation paused during model inference'],
                post_failure_handling='No reset; first-episode failure is latched in analysis. Post-failure motion is diagnostic only.',
                prior=info['prior'] if args.mode=='guided' else None,
                ddim_steps=4 if args.mode=='guided' else None,
                execution_steps=2 if args.mode=='guided' else 1,
                guidance_steps=2 if args.mode=='guided' else None,
                guidance_scale=25 if args.mode=='guided' else None,
                guide_model=None,guide_model_note='recorded targets directly replace the guide proposal; no guide DDIM model used' if args.mode=='guided' else 'direct absolute position targets; no learned controller'))
        np.savez_compressed(args.output/'trajectory.npz',**{
            key:np.stack([r[key] for r in records]) for key in records[0]})
        print('[done]',args.output,flush=True)
    finally:
        for writer in writers:writer.release()
        if sock is not None:
            try:send_message(sock,{'type':'shutdown'});sock.close()
            except Exception:pass
        if override is not None:native._restore_sharpa_asset_override(override)
        native._destroy_environment(env)

if __name__=='__main__':main()
