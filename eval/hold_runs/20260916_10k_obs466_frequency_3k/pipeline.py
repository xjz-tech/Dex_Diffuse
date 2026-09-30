import os
import json
import time
import subprocess
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
HZ = (30, 45, 60, 90)
HISTORICAL = ROOT.parent/'20260915_10k_1b_wait_compare_3k'/'wait1'

def state(status, **kwargs):
    data = dict(status=status, updated_at=time.time(), **kwargs)
    tmp = ROOT/'state.tmp'
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(ROOT/'state.json')
    print(json.dumps(data), flush=True)

def idle():
    p = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], capture_output=True,text=True,check=True)
    return not p.stdout.strip()

def verify(folder, hz, n, cap):
    rows = [json.loads(x) for x in (folder/'episodes.jsonl').read_text().splitlines() if x.strip()]
    assert len(rows)==n and len({r['env'] for r in rows})==n
    assert all(r['episode']==0 and 0<r['length']<=cap and r['reason'] in ('failure','timeout') for r in rows)
    timing=json.loads((folder/'timing.json').read_text())
    schedule=json.loads((folder/'schedule.json').read_text())
    assert timing['episodes']==n and timing['hold_steps']==0 and timing['wait']==1
    assert timing['control_steps']<=cap and timing['action_steps']==timing['control_steps']
    assert timing['force_physics_ticks']==timing['control_steps']*(180//hz)
    assert schedule['control_hz']==hz and schedule['physics_hz']==180
    assert abs(schedule['physics_dt']*schedule['decimation']-1/hz)<1e-10
    assert schedule['substeps']==2
    assert (folder/'optimization_validation.json').is_file()
    return rows,timing

def compare_initial(folder, reference):
    with np.load(folder/'initial_state.npz') as a,np.load(reference/'initial_state.npz') as b:
        matches={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
    if not all(matches.values()):
        raise RuntimeError('Initial states differ: '+str(matches))
    return matches

def summarize(require_all=False):
    results={}
    for hz in HZ:
        folder=ROOT/f'hz{hz}'
        if not (folder/'timing.json').exists():
            if require_all: raise RuntimeError(f'Missing {hz} Hz results')
            continue
        rows,timing=verify(folder,hz,3000,hz*400)
        matches=compare_initial(folder,ROOT/'hz30')
        lengths=np.array([r['length']/hz for r in rows])
        survived=sum(r['reason']=='timeout' and r['length']==hz*400 for r in rows)
        results[str(hz)]=dict(environments=3000,failures=sum(r['reason']=='failure' for r in rows),
            survived_cap=survived,completion_percent=100*survived/3000,
            mean_capped_hold_seconds=float(lengths.mean()),median_capped_hold_seconds=float(np.median(lengths)),
            survival_at_seconds={str(t):float(np.mean(lengths>=t)) for t in (30,60,120,240,400)},
            initial_states_equal=matches,control_steps=timing['control_steps'],
            simulated_seconds=timing['control_steps']/hz,wall_seconds=timing['wall_seconds'],
            mean_inference_seconds=timing['inference_seconds']/timing['calls'])
    historical=json.loads((HISTORICAL.parent/'comparison.json').read_text())['results']['1']
    data=dict(results=results,historical_30hz=historical,
        caveat='Single seed8, 3000 first episodes, 400 simulated seconds. WAIT=1 freezes physics during inference. Same 180-Hz outer physics and 2 substeps for all new arms. Policy history4 and execute2 remain in action steps, intentionally changing their physical duration. Force updates and loaded-force duty cycle retain the old 30-Hz world-time schedule; stability/failure timers use common 180-Hz ticks. Random realizations and event-detection granularity can diverge. Historical 30-Hz physics was 60 Hz with 2 substeps, so compare primarily against the new 30-Hz control; historical result is context. No true-hardware throughput/latency claim. Failure is the existing environment predicate, not an independent drop detector.')
    (ROOT/'comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['# 10k guide + obs_4-66: control-frequency comparison','',
        '| Control Hz | Finished | 400-s survivors | Survival | Mean capped hold (s) | Median capped hold (s) |',
        '|---|---|---:|---:|---:|---:|']
    for hz in HZ:
        if str(hz) not in results:
            lines.append(f'| {hz} | pending | — | — | — | — |');continue
        r=results[str(hz)]
        lines.append(f"| {hz} | yes | {r['survived_cap']}/3000 | {r['completion_percent']:.2f}% | {r['mean_capped_hold_seconds']:.3f} | {r['median_capped_hold_seconds']:.3f} |")
    lines += ['',data['caveat'],'',f"Historical optimized WAIT=1 30 Hz: {historical['survived_cap']}/3000 survivors, mean {historical['mean_capped_hold_seconds']:.3f}s, median {historical['median_capped_hold_seconds']:.3f}s."]
    (ROOT/'comparison.md').write_text('\n'.join(lines)+'\n')
    return results

def run(folder,hz,n,cap,smoke=False):
    if folder.exists(): raise RuntimeError('Refusing overwrite: '+str(folder))
    while not idle():
        state('waiting_gpu',control_hz=hz,phase='smoke' if smoke else 'full')
        time.sleep(30)
    folder.mkdir()
    env=dict(os.environ,CONTROL_HZ=str(hz),NUM_ENV=str(n),MAX_STEPS=str(cap),ARM_DIR=str(folder),PYTHONDONTWRITEBYTECODE='1')
    if smoke: env['DATA_INDICES']='000-002'
    else: env['DATA_INDICES']='000-149'
    with (folder/'console.log').open('w') as out:
        child=subprocess.Popen(['bash',str(ROOT/'run_arm.sh')],env=env,stdout=out,stderr=subprocess.STDOUT)
        state('running',control_hz=hz,phase='smoke' if smoke else 'full',pid=child.pid,log=str(folder/'console.log'))
        rc=child.wait()
    if rc not in (0,139): raise RuntimeError(f'{folder.name} exited {rc}')
    rows,timing=verify(folder,hz,n,cap)
    state('arm_complete',control_hz=hz,phase='smoke' if smoke else 'full',exit_code=rc)
    return timing

if __name__=='__main__':
    try:
        if (ROOT/'state.json').exists(): raise RuntimeError('Existing state; refusing duplicate pipeline')
        for hz in HZ:
            run(ROOT/f'check_hz{hz}',hz,4,hz*2,True)
            compare_initial(ROOT/f'check_hz{hz}',ROOT/'check_hz30')
        (ROOT/'smoke_validation.json').write_text(json.dumps(dict(passed=True,frequencies=HZ,initial_states_identical=True),indent=2)+'\n')
        for hz in HZ:
            run(ROOT/f'hz{hz}',hz,3000,hz*400)
            summarize()
        results=summarize(True)
        state('complete',comparison=str(ROOT/'comparison.md'),results=results)
    except Exception as exc:
        state('failed',error=repr(exc))
        raise
