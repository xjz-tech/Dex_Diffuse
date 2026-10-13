import os
import json
import time
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
def state(status, **kwargs):
    data = dict(status=status, updated_at=time.time(), **kwargs)
    tmp = ROOT/'state.tmp'
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(ROOT/'state.json')
    print(json.dumps(data),flush=True)

def idle():
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    return not gpu.stdout.strip()

def records(wait):
    path = ROOT/f'wait{wait}'/'episodes.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(rows)==3000 and len({r['env'] for r in rows})==3000
    assert all(r['episode']==0 and 0<r['length']<=12000 for r in rows)
    return rows

def summarize():
    initials=[np.load(ROOT/f'wait{w}'/'initial_state.npz') for w in (0,1)]
    matches={k:bool(np.array_equal(initials[0][k],initials[1][k])) for k in initials[0].files}
    if not all(matches.values()):
        raise RuntimeError('Initial states differ: '+str(matches))
    results={}
    for w in (0,1):
        rows=records(w)
        timing=json.loads((ROOT/f'wait{w}'/'timing.json').read_text())
        lengths=np.array([r['length'] for r in rows])/30.
        results[str(w)]=dict(environments=len(rows),failures=sum(r['reason']=='failure' for r in rows),
            survived_cap=sum(r['reason']=='timeout' and r['length']==12000 for r in rows),
            mean_capped_hold_seconds=float(lengths.mean()),median_capped_hold_seconds=float(np.median(lengths)),
            survival_at_seconds={str(t):float(np.mean(lengths>=t)) for t in (30,60,120,240,400)},
            action_steps=timing['action_steps'],hold_steps=timing['hold_steps'],
            wall_seconds=timing['wall_seconds'],mean_inference_seconds=timing['inference_seconds']/timing['calls'])
    data=dict(initial_states_equal=matches,results=results,
        caveat='Single seed, 3000 environments per arm. Native batched GPU inference latency; not a reproduction of single-hand hardware latency. Failure is the existing environment predicate, not an independent physical-drop detector. Capped hold includes right-censored survivors at 400 seconds.')
    (ROOT/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['# 10k guide → 1B: WAIT=0 vs WAIT=1, 3000 environments each','',
        '| Metric | WAIT=0 | WAIT=1 |','|---|---:|---:|']
    for key in ('environments','failures','survived_cap','mean_capped_hold_seconds','median_capped_hold_seconds','action_steps','hold_steps','mean_inference_seconds','wall_seconds'):
        values=[results[str(w)][key] for w in (0,1)]
        lines.append('| '+key+' | '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in values)+' |')
    lines += ['', 'All saved initial-state arrays are identical.', '', data['caveat']]
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    state('complete',comparison=str(ROOT/'comparison.md'),results=results)

try:
    if (ROOT/'state.json').exists():
        raise RuntimeError('Refusing to overwrite an existing run')
    for wait in (0,1):
        while not idle():
            state('waiting_gpu',wait=wait)
            time.sleep(30)
        folder=ROOT/f'wait{wait}'
        folder.mkdir(exist_ok=True)
        env=dict(os.environ,WAIT=str(wait),PYTHONDONTWRITEBYTECODE='1')
        with (folder/'console.log').open('w') as log:
            child=subprocess.Popen(['bash',str(ROOT/'run_arm.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
            state('running',wait=wait,pid=child.pid,log=str(folder/'console.log'))
            rc=child.wait()
        if rc not in (0,139):
            raise RuntimeError(f'WAIT={wait} exited {rc}')
        records(wait)
        timing=json.loads((folder/'timing.json').read_text())
        assert timing['episodes']==3000 and timing['control_steps']<=12000
        if wait==0: assert timing['hold_steps']>0
        else: assert timing['hold_steps']==0
        state('arm_complete',wait=wait,exit_code=rc)
    summarize()
except Exception as exc:
    state('failed',error=str(exc))
    raise
