from pathlib import Path
import json,os,sys,subprocess,concurrent.futures
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926'
env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
rows=json.load(open(R/'candidates.json'))['candidates']
def run(r):
 folder=Path(r['folder']);out=folder/'direct_probe';out.mkdir(exist_ok=True)
 if (out/'summary.json').exists():return
 cmd=['/home/carus/miniforge3/envs/decv2/bin/python',str(P/'compare_corrected_rollouts.py'),'--mode','direct','--out',str(out),'--reference',str(folder/'reference_full.npz'),'--reference-id','0','--source-episode',str(r['episode']),'--no-video','--grasp-evidence','--action-limit','180']
 subprocess.run(cmd,env=env,stdout=open(out/'sim.log','w'),stderr=subprocess.STDOUT,check=True)
 print('DONE',r['episode'],r['start'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(run,rows))
