import sys,os,time,json
from pathlib import Path
import numpy as np
sys.path.append('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/sim-real-holding-comparison/eval')
from real_sim_world import World
out=Path(sys.argv[1]).resolve();old=out.parent/'20260917_1b_initialization_audit'
ini=np.load(old/'initial_state.npz');z=np.load(old/'rollouts.npz');static=np.load(old/'static_check.npz')
os.environ['AUDIT_SUBSTEPS']='20'
w=World(n=64,dt=1/30,demos=1,camera=False);w.reset(ini['q'],ini['wrist'],ini['object'])
t=time.monotonic();w.step(30);static_error=float(np.max(abs(np.linalg.norm(w.objects()[:,:3]-ini['object'][:,:3],axis=1)-static['displacement_m'])))
w.reset(ini['q'],ini['wrist'],ini['object']);qs=[];objs=[]
for i in range(120):w.target(z['action'][i]);w.step(1);qs.append(w.state());objs.append(w.objects())
qerr=np.max(abs(np.asarray(qs)-z['qpos'][:120]));oerr=np.max(abs(np.asarray(objs)[:,:,:3]-z['object'][:120,:,:3]))
r={'static_displacement_max_error_m':static_error,'replay_qpos_max_error_rad':float(qerr),'replay_object_position_max_error_m':float(oerr),'steps':120,'wall_s':time.monotonic()-t,'outer_dt':1/30,'substeps':20,'effective_dt':1/600}
(out/'substep_validation.json').write_text(json.dumps(r,indent=2));print(r,flush=True);w.close()
