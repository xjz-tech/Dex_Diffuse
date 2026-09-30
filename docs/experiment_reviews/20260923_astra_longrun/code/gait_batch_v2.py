"""Frozen measured-unload batch; preserve and retry harness failures, never discard controller failures."""
import json,subprocess,time,shutil
from pathlib import Path
h=Path(__file__).resolve().parents[1];v=h/'gait_trials/v2';py='/home/carus/miniforge3/envs/dp/bin/python'
while not (v/'runs/astra_direct_e50_n0/summary.json').exists():time.sleep(1)
jobs=[]
for seed in [50,19,25]:
 jobs.append((seed,0,'astra_direct'))
 for noise in [0,1]:
  for method in ['prior','guided25','guided38','guided50']:jobs.append((seed,noise,method))
(v/'formal_jobs.json').write_text(json.dumps(jobs,indent=2));errors=[]
for seed,noise,method in jobs:
 label=f'{method}_e{seed}_n{noise}';d=v/'runs'/label
 if method=='prior' and not d.exists():
  for old in [h/'runs'/label,h/'gait_trials/v1/runs'/label]:
   if (old/'summary.json').exists():
    d.symlink_to(old.resolve(),target_is_directory=True);break
 if (d/'summary.json').exists():continue
 for attempt in range(1,3):
  with (v/(label+f'.attempt{attempt}.launcher.log')).open('w') as f:
   c=subprocess.run([py,'-u',str(h/'code/gait_trial.py'),'--version','v2','--method',method,'--seed',str(seed),'--noise',str(noise),'--release-mm','1','--push-deg','4','--measured-unload'],stdout=f,stderr=subprocess.STDOUT)
  print(label,'attempt',attempt,c.returncode,flush=True)
  if c.returncode==0:break
  # Retry only pre-control setup failures; keep nonzero-step interruptions for explicit review.
  n=sum(1 for _ in (d/'trajectory.jsonl').open()) if (d/'trajectory.jsonl').exists() else 0
  if n or attempt==2:errors.append(label);break
  dest=v/'attempts'/f'{label}_setup{attempt}';dest.parent.mkdir(exist_ok=True)
  if d.exists():shutil.move(d,dest)
  else:dest.mkdir()
  if (v/'runs'/(label+'.log')).exists():shutil.copy2(v/'runs'/(label+'.log'),dest.with_suffix('.log'))
  (dest/'interruption.json').write_text(json.dumps({'steps':n,'counted_as_drop':False,'reason':'Nonzero subprocess exit before first control step. See retained launcher and simulator logs.'},indent=2))
(v/'batch_finished.json').write_text(json.dumps({'errors':errors,'complete':not errors},indent=2))
