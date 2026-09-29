"""Exploratory local sign-boundary refinement; does not assert global monotonicity."""
import json
from pathlib import Path
from run import run
P=Path(__file__).resolve().parent
left,right=-.75,-.5
history=[]
for i in range(4):
    mid=(left+right)/2
    result=run(mid,48,288)
    if result['native_failure']:raise RuntimeError('Failure cannot be used as sign-boundary data')
    y=result['twist_after_initial8_deg']
    history.append(dict(iteration=i,lambda_direction=mid,net_twist=y,prior_bracket=[left,right]))
    if y<0:left=mid
    else:right=mid
    (P/'local_refinement.json').write_text(json.dumps(dict(scope='Local finite-time net-angle sign bracket at noise48, adaptive post-grid exploration. Nonmonotonic response forbids interpreting as global minimum or stable direction threshold.',history=history,bracket=[left,right]),indent=2)+'\n')
print('LOCAL BRACKET',left,right,flush=True)
