"""Resume all frozen physics/noise pairs using the same small-lift controller."""
import json,subprocess,time
from pathlib import Path
h=Path(__file__).resolve().parents[1];v=h/'gait_trials/v1'
# First exploration queue owns these runs; do not submit duplicates.
while not (v/'runs/guided50_e50_n0/summary.json').exists():time.sleep(1)
jobs=[]
for seed in [50,19,25]:
 jobs.append((seed,0,'astra_direct'))
 for noise in [0,1]:
  for method in ['prior','guided25','guided38','guided50']:jobs.append((seed,noise,method))
(v/'formal_jobs.json').write_text(json.dumps(jobs,indent=2))
errors=[]
for seed,noise,method in jobs:
 label=f'{method}_e{seed}_n{noise}';d=v/'runs'/label
 original=h/'runs'/label
 if method=='prior' and not d.exists() and (original/'summary.json').exists():d.symlink_to(original,target_is_directory=True)
 if (d/'summary.json').exists():continue
 with (v/(label+'.launcher.log')).open('w') as f:
  c=subprocess.run(['/home/carus/miniforge3/envs/dp/bin/python','-u',str(h/'code/gait_trial.py'),'--version','v1','--method',method,'--seed',str(seed),'--noise',str(noise),'--release-mm','1','--push-deg','4'],stdout=f,stderr=subprocess.STDOUT)
 print(label,c.returncode,flush=True)
 if c.returncode:errors.append(label)
(v/'batch_finished.json').write_text(json.dumps({'errors':errors,'complete':not errors},indent=2))
