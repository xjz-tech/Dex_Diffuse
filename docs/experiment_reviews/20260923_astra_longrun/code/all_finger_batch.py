from pathlib import Path
import json,subprocess,time,shutil
h=Path(__file__).resolve().parents[1];v=h/'gait_trials/v3_all_fingers';py='/home/carus/miniforge3/envs/dp/bin/python'
while not (v/'runs/astra_direct_e50_n0/summary.json').exists():time.sleep(1)
jobs=[(19,'astra_direct'),(25,'astra_direct'),(50,'guided25'),(50,'guided38'),(50,'guided50')]
errors=[]
for seed,method in jobs:
 label=f'{method}_e{seed}_n0'
 with (v/f'{label}.launcher.log').open('w') as log:
  c=subprocess.run([py,'-u',str(h/'code/all_finger_trial.py'),'--seed',str(seed),'--method',method],stdout=log,stderr=subprocess.STDOUT)
 if c.returncode:errors.append(label)
 print(label,c.returncode,flush=True)
p=v/'runs/prior_e50_n0'
if not p.exists():p.symlink_to(h/'runs/prior_e50_n0',target_is_directory=True)
(v/'batch_finished.json').write_text(json.dumps({'complete':not errors,'errors':errors,'scope':'Mechanism trial: direct three prescreened seeds; seed50 noise0 paired with Prior and scale25/38/50; not a full27-run comparison'},indent=2))
