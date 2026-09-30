"""Matched 16 initial states across ordinary, scale0, and 10k-scale25 arms."""
import sys,os,json,socket,time,signal
from pathlib import Path
import numpy as np
PROJECT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
BASE=PROJECT/'.worktrees/sim-real-holding-comparison'
sys.path.append(str(BASE/'eval'))
from real_sim_world import World
from ipc import send_message,recv_message
import torch
from omegaconf import OmegaConf
out=Path(sys.argv[1]).resolve();old=PROJECT/'docs/experiment_reviews/20260917_1b_initialization_audit'
n=16;N=64;groups=['ordinary_1b','guided_scale0','guide10k_scale25']
torch.manual_seed(42);np.random.seed(42)
w=World(n=N,demos=1,camera=False)
initial=np.load(old/'initial_state.npz')
q=initial['q'].copy();wr=initial['wrist'].copy();obj=initial['object'].copy()
for start in [0,16,32]:
    q[start:start+n]=initial['q'][32:48];wr[start:start+n]=initial['wrist'][32:48];obj[start:start+n]=initial['object'][32:48]
w.reset(q,wr,obj);assert np.max(abs(w.state()-q))<1e-6
np.savez_compressed(out/'initial_state.npz',q=q,wrist=wr,object=obj)
(out/'simulation_config.yaml').write_text(OmegaConf.to_yaml(w.cfg))
cfg={'groups':groups,'trials_per_group':n,'seeds':list(range(50,66)),'guide_seeds':list(range(100050,100066)),'max_steps':12000,'hz':30,'physics_dt':w.dt,'physics_substeps':1,'execute_steps':2,'ddim_steps':4,'guide_scale':25,'guide_steps':2,'holding':False,'step_clamp':None,'joint_limits':True,'extra_inference_delay':0,'random_forces':False,'drop_threshold_m':.05,'initial_state_source':str(old/'initial_state.npz'),'source_env_indices':list(range(32,48)),'init_seed':42}
(out/'config.json').write_text(json.dumps(cfg,indent=2))
w.step(600)
np.savez_compressed(out/'static_check.npz',displacement_m=np.linalg.norm(w.objects()[:,:3]-obj[:,:3],axis=1),q_error=np.max(abs(w.state()-q),axis=1))
w.reset(q,wr,obj)
hist=np.repeat(np.concatenate([q,q,np.zeros_like(q)],axis=1)[:,None],4,axis=1).astype(np.float32)
lower=w.e.dexhand_dof_lower_limits.cpu().numpy();upper=w.e.dexhand_dof_upper_limits.cpu().numpy()
sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);sock.connect(os.environ['AUDIT_SOCKET'])
drop=np.full(N,-1,int);active=np.arange(N)<48;arms=np.arange(N)//16
gens=[np.random.default_rng(i) for i in range(50,66)];ggens=[np.random.default_rng(i) for i in range(100050,100066)]
plans=np.zeros((N,2,22),np.float32);rec={k:[] for k in ['raw','action','qpos','object','active']}
started=time.monotonic();interrupted=False
def stop(sig,frame):
    global interrupted
    interrupted=True
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
def save(partial):
    dest=out/('partial.npz' if partial else 'rollouts.npz');tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,**{k:np.asarray(v) for k,v in rec.items()},drop_steps=drop)
    tmp.replace(dest)
try:
    for step in range(12000):
        active=(np.arange(N)<48)&(drop<0)
        if not active.any() or interrupted:break
        if step%2==0:
            noise=np.tile(np.stack([g.standard_normal((12,22)).astype(np.float32) for g in gens]),(4,1,1))
            gn=np.tile(np.stack([g.standard_normal((12,22)).astype(np.float32) for g in ggens]),(4,1,1))
            ids=np.flatnonzero(active)
            send_message(sock,{'type':'predict','noise':noise[ids].tolist(),'guide_noise':gn[ids].tolist(),'arms':arms[ids].tolist()},hist[ids])
            m,pred=recv_message(sock);assert m['ok'] and pred.shape==(len(ids),2,22);plans[ids]=pred
        raw=plans[:,step%2].copy();target=np.clip(raw,lower,upper);target[~active]=w.state()[~active]
        w.target(target);w.step(20);actual=w.state();objects=w.objects()
        assert np.isfinite(actual).all() and np.isfinite(raw).all()
        displacement=np.linalg.norm(objects[:,:3]-obj[:,:3],axis=1)
        drop[active&(displacement>.05)]=step
        for k,v in [('raw',raw),('action',target),('qpos',actual),('object',objects),('active',active.copy())]:rec[k].append(v)
        hist=np.concatenate([hist[:,1:],np.concatenate([actual,target,target-actual],axis=1)[:,None]],axis=1)
        if (step+1)%300==0:
            status={'steps':step+1,'sim_seconds':(step+1)/30,'wall_seconds':time.monotonic()-started,'alive':{g:int((drop[i*16:(i+1)*16]<0).sum()) for i,g in enumerate(groups)},'drop_steps':drop[:48].tolist()}
            (out/'progress.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True);save(True)
    save(interrupted)
    (out/'completion.json').write_text(json.dumps({'complete':not interrupted,'steps':len(rec['raw']),'wall_seconds':time.monotonic()-started,'all_failed':bool((drop[:48]>=0).all())},indent=2))
    print('DONE',len(rec['raw']),time.monotonic()-started,flush=True)
finally:
    send_message(sock,{'type':'stop'});sock.close();w.close()
