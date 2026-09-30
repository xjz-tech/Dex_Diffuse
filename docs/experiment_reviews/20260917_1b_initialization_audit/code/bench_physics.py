import sys,time,json
from pathlib import Path
import numpy as np
sys.path.append('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/sim-real-holding-comparison/eval')
from real_sim_world import World
out=Path(sys.argv[1]);dt=float(sys.argv[2]);z=np.load(out/'initial_state.npz')
w=World(n=64,dt=dt,demos=1,camera=False);w.reset(z['q'],z['wrist'],z['object'])
t=time.monotonic();w.step(round(1/dt));elapsed=time.monotonic()-t
np.savez_compressed(out/('physics_bench_'+sys.argv[3]+'.npz'),q=w.state(),obj=w.objects(),elapsed=elapsed)
print(json.dumps({'elapsed':elapsed,'dt':dt}),flush=True);w.close()
