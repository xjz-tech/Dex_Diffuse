import json,subprocess,time,shutil
from pathlib import Path
h=Path(__file__).resolve().parents[1];v=h/'gait_trials/v1';py='/home/carus/miniforge3/envs/dp/bin/python'
while not (v/'batch_finished.json').exists():time.sleep(1)
d=json.loads((v/'batch_finished.json').read_text())
if d['complete']:raise SystemExit(0)
(v/'batch_first_pass.json').write_text(json.dumps(d,indent=2))
remaining=[]
for label in d['errors']:
 r=v/'runs'/label
 if r.exists():
  n=sum(1 for _ in (r/'trajectory.jsonl').open()) if (r/'trajectory.jsonl').exists() else 0
  if n:remaining.append(label);print('Retain nonzero-step interruption for manual review:',label,n,flush=True);continue
  a=v/'attempts'/(label+'_zero_step_first_pass');a.parent.mkdir(exist_ok=True);assert not a.exists();shutil.move(r,a)
  if (v/'runs'/(label+'.log')).exists():shutil.copy2(v/'runs'/(label+'.log'),a.with_suffix('.log'))
 method,rest=label.rsplit('_e',1);seed,noise=rest.split('_n')
 with (v/(label+'.retry.log')).open('w') as f:
  c=subprocess.run([py,str(h/'code/gait_trial.py'),'--version','v1','--method',method,'--seed',seed,'--noise',noise,'--release-mm','1','--push-deg','4'],stdout=f,stderr=subprocess.STDOUT)
 print('Retry',label,c.returncode,flush=True)
 if c.returncode:remaining.append(label)
(v/'batch_finished.json').write_text(json.dumps({'errors':remaining,'complete':not remaining,'initial_infrastructure_errors':d['errors']},indent=2))
if not remaining:subprocess.run([py,str(h/'code/finish_gait.py')],check=True)
