import os,sys,json,time,subprocess,traceback
from pathlib import Path
ROOT=Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse')
R=Path(__file__).resolve().parents[1]
PY='/home/carus/miniforge3/envs/dp/bin/python'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
status={'started':time.strftime('%Y-%m-%d %H:%M:%S'),'pid':os.getpid(),'completed':[]}
def run(label,args):
 status['active']=label; (R/'status.json').write_text(json.dumps(status,indent=2))
 with (R/f'{label}.log').open('w') as f:
  subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 status['completed'].append(label)
try:
 for name,cap,seed in [('new10k_seed43',10000,43),('new30k_seed44',30000,44)]:
  run('build_'+name,[PY,'-u',str(R/'code/build_sim_hand_subset.py'),'--source','/home/carus/Data/exp_data','--output',str(ROOT/'data'/f'random_20260921_{name}'),'--max-transitions',str(cap),'--seed',str(seed)])
  run('train_'+name,[PY,'-u',str(R/'code/train_one.py'),name])
 for name in ['new10k_seed43','new30k_seed44']:
  run('eval_'+name,['bash',str(R/'code/eval_one.sh'),name])
 status['active']=None; status['complete']=True
except BaseException:
 status['error']=traceback.format_exc(); status['complete']=False
 raise
finally:
 status['updated']=time.strftime('%Y-%m-%d %H:%M:%S')
 (R/'status.json').write_text(json.dumps(status,indent=2))
