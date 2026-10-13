import sys,json,socket,time,os
from pathlib import Path
import numpy as np
BASE=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/sim-real-holding-comparison')
sys.path.append(str(BASE/'eval'))
from real_sim_world import World
import torch
from ipc import send_message,recv_message
from omegaconf import OmegaConf

out=Path(sys.argv[1]).resolve(); n=16; groups=['native_bulb','native_empty','real_bulb','real_empty'];N=4*n
torch.manual_seed(42);np.random.seed(42)
w=World(n=N,demos=150,camera=False)
# Select 16 demonstrations across all 150, then use the ordinary reset sampler.
demoids=torch.as_tensor(np.linspace(0,149,n,dtype=int),device=w.e.device)
w.e.envidx_to_demoidx[:n]=demoids
w.e.reset_idx(torch.arange(n,device=w.e.device));w.refresh()
native_q=w.state()[:n];native_w=w.e._base_state[:n].cpu().numpy().copy();native_o=w.objects()[:n]
native_w[:,7:]=0;native_o[:,7:]=0
placements=json.loads((BASE/'reports/sim_real_holding_20260917/per_run_placements.json').read_text())
names=['bulb_nohold_1','bulb_nohold_2','bulb_hold_1','bulb_hold_2']
real=[placements['per_run'][names[i%4]] for i in range(n)]
rq=np.array([x['qpos'] for x in real]);rw=np.array([x['wrist'] for x in real]);ro=np.array([x['object'] for x in real]);ro[:,7:]=0
q=np.concatenate([native_q,native_q,rq,rq]);wr=np.concatenate([native_w,native_w,rw,rw]);obj=np.concatenate([native_o,native_o,ro,ro])
obj[n:2*n,:3]=[2,2,0];obj[3*n:,:3]=[2,2,0]
w.reset(q,wr,obj)
assert np.max(abs(w.state()-q))<1e-6
np.savez_compressed(out/'initial_state.npz',q=q,wrist=wr,object=obj,demo_ids=demoids.cpu().numpy())
(out/'placements.json').write_text(json.dumps(placements,indent=2))
(out/'simulation_config.yaml').write_text(OmegaConf.to_yaml(w.cfg))
cfg=dict(groups=groups,trials_per_group=n,seeds=list(range(50,66)),max_steps=12000,empty_steps=600,hz=30,ddim_steps=4,execute_steps=2,guide=False,holding=False,inference_delay=0,physics_dt=w.dt,step_clamp=None,joint_limits=True,drop_threshold_m=.05,init_seed=42,real_pose_sources=names,native_demo_ids=demoids.cpu().tolist())
(out/'config.json').write_text(json.dumps(cfg,indent=2))
# Check static retention in this same batched physics world without changing the inference starting state.
w.step(600)
static=np.linalg.norm(w.objects()[:,:3]-obj[:,:3],axis=1)
np.savez_compressed(out/'static_check.npz',displacement_m=static,q_error=np.max(abs(w.state()-q),axis=1))
w.reset(q,wr,obj)
hist=np.repeat(np.concatenate([q,q,np.zeros_like(q)],axis=1)[:,None,:],4,axis=1).astype(np.float32)
lower=w.e.dexhand_dof_lower_limits.cpu().numpy();upper=w.e.dexhand_dof_upper_limits.cpu().numpy()
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.connect(os.environ['AUDIT_SOCKET'])
bulb=np.isin(np.arange(N)//n,[0,2]);dropped=np.full(N,-1,int);active=np.ones(N,bool)
gens=[np.random.default_rng(x) for x in range(50,66)]
plans=np.zeros((N,5,22),np.float32);records={k:[] for k in ['raw','action','qpos','object','active']}
started=time.monotonic()
try:
    for step in range(12000):
        active[~bulb]=step<600;active[bulb]=dropped[bulb]<0
        if not np.any(active):break
        if step%2==0:
            noise=np.tile(np.stack([g.standard_normal((12,22)).astype(np.float32) for g in gens]),(4,1,1))
            ids=np.flatnonzero(active)
            send_message(sock,{'type':'predict','noise':noise[ids].tolist()},hist[ids])
            m,act=recv_message(sock);assert m['ok'] and act.shape==(len(ids),5,22)
            plans[ids]=act
        raw=plans[:,step%2].copy();target=np.clip(raw,lower,upper)
        # Freeze inactive hands; never reset a failed episode into the metrics.
        target[~active]=w.state()[~active]
        w.target(target);w.step(20)
        actual=w.state();objects=w.objects();assert np.isfinite(actual).all() and np.isfinite(raw).all()
        displacement=np.linalg.norm(objects[:,:3]-obj[:,:3],axis=1)
        new=bulb&(dropped<0)&(displacement>.05);dropped[new]=step
        for k,v in [('raw',raw),('action',target),('qpos',actual),('object',objects),('active',active.copy())]:records[k].append(v)
        obs=np.concatenate([actual,target,target-actual],axis=1)
        hist=np.concatenate([hist[:,1:],obs[:,None]],axis=1)
        if step%300==0:
            status=dict(step=step,seconds=(step+1)/30,elapsed_wall=time.monotonic()-started,alive={g:int(active[i*n:(i+1)*n].sum()) for i,g in enumerate(groups)},dropped_steps=dropped.tolist())
            (out/'progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    np.savez_compressed(out/'rollouts.npz',**{k:np.asarray(v) for k,v in records.items()},drop_steps=dropped)
    print('DONE',step,time.monotonic()-started,flush=True)
finally:
    send_message(sock,{'type':'stop'});sock.close();w.close()
