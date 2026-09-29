import os,subprocess,json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'reference_turn_baseline_20260926';OUT=R/'episode2_prior_palm_direction_20260928';OUT.mkdir(exist_ok=True)
env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
for name,offset in [('toward_palm_10',[-.005,0.,-.01]),('toward_palm_20',[-.01,0.,-.02]),('toward_palm_30',[-.015,0.,-.03])]:
 folder=OUT/name/'static';folder.mkdir(parents=True,exist_ok=True)
 cmd=['/home/carus/miniforge3/envs/decv2/bin/python',str(P/'compare_prior_palm_rollouts.py'),'--mode','direct','--out',str(folder),'--reference',str(R/'qualified_comparison/episode_02/reference_full.npz'),'--reference-id','0','--source-episode','2','--front-only','--audit-recording','--grasp-evidence','--object-mass-kg','.044','--friction','1.1','--static-only','--object-wrist-offset',*map(str,offset)]
 with (folder/'sim.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
