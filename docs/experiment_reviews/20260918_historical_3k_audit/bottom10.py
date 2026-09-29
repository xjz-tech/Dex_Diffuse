import json,itertools
from pathlib import Path
import numpy as np
out=Path(__file__).resolve().parent
suite=out.parents[2]/'eval/hold_runs/20260912_fresh_noise_3000_seed42_8_19_25'
results=[]
for seed in [42,8,19,25]:
 sets={};rows={}
 for prior in ['1B','mixed']:
  for scale in [0,25]:
   key=f'{prior}_s{scale}';p=suite/f'seed{seed}_{prior}_scale{scale}'
   rec=[json.loads(x) for x in next(p.glob('*.jsonl')).read_text().splitlines() if x]
   rec.sort(key=lambda x:(x['length'],x['env']))
   cutoff=rec[299]['length'];bottom=[x for x in rec if x['length']<=cutoff]
   sets[key]={x['env'] for x in bottom}
   rows[key]={'cutoff_steps':cutoff,'cutoff_seconds':cutoff/30,'n_including_ties':len(bottom),'failure_count':sum(x['reason']=='failure' for x in bottom),'envs':sorted(sets[key])}
 pairs=[]
 for a,b in itertools.combinations(sets,2):
  both=len(sets[a]&sets[b]);pairs.append(dict(a=a,b=b,intersection=both,overlap_fraction_smaller=both/min(len(sets[a]),len(sets[b])),independent_expected=len(sets[a])*len(sets[b])/3000))
 result=dict(seed=seed,groups=rows,pairs=pairs,all_four_intersection=len(set.intersection(*sets.values())))
 results.append(result)
 print('seed',seed,'groups',[(k,v['cutoff_seconds'],v['n_including_ties'],v['failure_count']) for k,v in rows.items()],'all4',result['all_four_intersection'])
 print('1B overlap',pairs[0])
(out/'bottom10.json').write_text(json.dumps({'selection':'300th sorted length as cutoff; include all ties; within-seed env ID overlaps only, not proof of matched physical initial states','results':results},indent=2))
