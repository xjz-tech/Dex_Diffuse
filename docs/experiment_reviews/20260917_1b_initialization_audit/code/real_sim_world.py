"""Isolated Isaac Gym physics for reproducing archived SharpA experiments.

Use raw position drives, without task action clipping, observation noise,
random forces, reward termination, or automatic resets. Python 3.8 compatible.
"""
import os
from pathlib import Path
import numpy as np
import sim_eval as setup  # imports Isaac Gym before torch
from isaacgym import gymapi, gymtorch
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = Path('/home/carus/Program/dex-controller')


class World:
    def __init__(self, n=1, dt=1/600, demos=150, camera=True):
        os.chdir(str(CONTROLLER))
        lib = setup._import_local_maniptrans(CONTROLLER)
        cfg = setup._make_task_config(
            Path('/home/carus/Data/exp_data/hydra_config.yaml'), n,
            ['v3:bulb2@%03d' % i for i in range(demos)],
            CONTROLLER/'data/NOKOV-v3', CONTROLLER/'data/retargeting/NOKOV-v3')
        cfg.env.training = False
        cfg.env.simDataCollection = False
        cfg.env.randomWristOrientation = False
        cfg.env.randomObjectScales = [1., 1.]
        cfg.env.randomForceScale = 0.
        cfg.env.enableCameraSensors = camera
        cfg.task.randomize = False
        cfg.sim.dt = dt
        cfg.sim.substeps = int(os.environ.get('AUDIT_SUBSTEPS', '1'))
        cfg.env.controlFrequencyInv = 1
        override = setup._install_sharpa_asset_override(setup._resolve_sharpa_urdf(
            str(CONTROLLER/'maniptrans_envs/assets/sharpa_hand')))
        try:
            self.env = lib.make(sim_device='cuda:0', rl_device='cuda:0',
                graphics_device_id=0, multi_gpu=False, cfg=cfg, display=False,
                record=False, has_headless_arg=True, headless=True)
        finally:
            setup._restore_sharpa_asset_override(override)
        self.cfg = cfg
        self.dt = dt
        self.n = n
        self.e = self.env
        self.g = self.e.gym
        self.s = self.e.sim
        self.e.compute_observations()
        self.e.reset()
        self.camera = None

    def refresh(self):
        for name in ('dof_state', 'actor_root_state', 'rigid_body_state', 'net_contact_force'):
            getattr(self.g, 'refresh_'+name+'_tensor')(self.s)

    def target(self, q):
        q = np.broadcast_to(np.asarray(q, np.float32), (self.n,22)).copy()
        self.e._pos_control[:] = torch.as_tensor(q, device=self.e.device)
        self.g.set_dof_position_target_tensor(self.s, gymtorch.unwrap_tensor(self.e._pos_control))

    def step(self, count=1):
        for _ in range(count):
            self.g.simulate(self.s)
            self.g.fetch_results(self.s, True)
        self.refresh()

    def state(self):
        return self.e._q.cpu().numpy().copy()

    def objects(self):
        return self.e._manip_obj_root_state.cpu().numpy().copy()

    def reset(self, q, wrist, obj):
        self.e._q[:] = torch.as_tensor(np.broadcast_to(q, (self.n,22)).copy(), device=self.e.device)
        self.e._qd.zero_()
        self.e._base_state[:] = torch.as_tensor(np.broadcast_to(wrist, (self.n,13)).copy(), device=self.e.device)
        self.e._manip_obj_root_state[:] = torch.as_tensor(np.broadcast_to(obj, (self.n,13)).copy(), device=self.e.device)
        ids = torch.cat([self.e._global_dexhand_indices.flatten(),self.e._global_manip_obj_indices.flatten()])
        self.g.set_dof_state_tensor(self.s,gymtorch.unwrap_tensor(self.e._dof_state))
        self.g.set_actor_root_state_tensor_indexed(self.s,gymtorch.unwrap_tensor(self.e._root_state),gymtorch.unwrap_tensor(ids),len(ids))
        self.target(q)
        self.refresh()

    def candidates(self, q, count=64):
        d = self.e.demo_data
        candidates=[]
        for i in range(len(d['seq_len'])):
            length=int(d['seq_len'][i])
            poses=d['opt_dof_pos'][i,:length].cpu().numpy()
            dist=np.mean((poses-q)**2,axis=1)
            j=int(np.argmin(dist))
            w=np.zeros(13); o=np.zeros(13)
            w[:3]=d['opt_wrist_pos'][i,j].cpu().numpy()
            w[3:7]=Rotation.from_rotvec(d['opt_wrist_rot'][i,j].cpu().numpy()).as_quat()
            tr=d['obj_trajectory'][i,j].cpu().numpy()
            o[:3]=tr[:3,3];o[3:7]=Rotation.from_matrix(tr[:3,:3]).as_quat()
            candidates.append(dict(demo=i,frame=j,qpos_rms_rad=float(np.sqrt(dist[j])),wrist=w.tolist(),object=o.tolist()))
        return sorted(candidates,key=lambda x:x['qpos_rms_rad'])[:count]

    def image(self, env_id=0):
        if self.camera is None:
            props=gymapi.CameraProperties();props.width=800;props.height=600;props.horizontal_fov=55
            self.camera=self.g.create_camera_sensor(self.e.envs[env_id],props)
            self.camera_env=env_id
        w=self.e._base_state[env_id,:3].cpu().numpy()
        self.g.set_camera_location(self.camera,self.e.envs[env_id],
            gymapi.Vec3(*(w+np.array([.12,.55,.15]))),gymapi.Vec3(*(w+np.array([-.08,0,-.12]))))
        self.g.step_graphics(self.s);self.g.render_all_camera_sensors(self.s)
        a=self.g.get_camera_image(self.s,self.e.envs[env_id],self.camera,gymapi.IMAGE_COLOR)
        return np.asarray(a).reshape(600,800,4)[...,:3].copy()

    def close(self):
        self.g.destroy_sim(self.s)


def probe(out):
    import json
    from PIL import Image
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    q=np.array(json.loads((ROOT/'eval/real/sharpa_initial_pose.json').read_text())['policy_checkpoint_order']['qpos_rad'])
    w=World(n=64)
    candidates=w.candidates(q,64)
    wrists=np.array([x['wrist'] for x in candidates]);objects=np.array([x['object'] for x in candidates])
    w.reset(q,wrists,objects)
    w.step(600)
    final=w.objects();err=np.max(np.abs(w.state()-q),axis=1)
    for i,c in enumerate(candidates):
        c['displacement_m']=float(np.linalg.norm(final[i,:3]-objects[i,:3]))
        c['qpos_error_rad']=float(err[i]);c['final_object']=final[i].tolist()
        c['contact_force_N']=float(torch.linalg.norm(w.e._manip_obj_cf[i]).item())
        c['stable']=bool(c['displacement_m']<.025 and np.linalg.norm(final[i,7:10])<.05 and c['contact_force_N']>.01)
    stable=[i for i,c in enumerate(candidates) if c['stable']]
    chosen=min(stable,key=lambda i:candidates[i]['qpos_error_rad']) if stable else int(np.argmin([c['displacement_m'] for c in candidates]))
    (out/'placement_probe.json').write_text(json.dumps(dict(chosen=chosen,candidates=candidates),indent=2))
    Image.fromarray(w.image(chosen)).save(str(out/'placement_probe.png'))
    print('PLACEMENT',json.dumps(dict(stable_count=len(stable),chosen=candidates[chosen])),flush=True)
    w.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--probe',required=True);a=p.parse_args();probe(a.probe)
