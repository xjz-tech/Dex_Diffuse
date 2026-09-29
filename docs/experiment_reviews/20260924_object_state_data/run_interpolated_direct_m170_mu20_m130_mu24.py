"""Run matched >0.1 rad interpolated direct controls for two physics settings."""
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

P=Path(__file__).resolve().parent
R=P/'reference_turn_baseline_20260926'
C=R/'qualified_comparison'
EPISODES=(76,34,54,2)
PHYSICS=(('m170_mu20',.170,2.0,R/'m170_mu20_guidance4_vs_edit015_20260928'),
         ('m130_mu24',.130,2.4,R/'m130_mu24_guidance4_vs_edit015_20260928'))


def folder(output,ep):
    return output/f'episode_{ep:02d}'/'direct_interp'


def run(job):
    tag,mass,friction,output,ep=job
    f=folder(output,ep);f.mkdir(parents=True,exist_ok=True)
    if (f/'summary.json').exists():
        return dict(tag=tag,episode=ep,status='existing')
    case=C/f'episode_{ep:02d}'
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],
        PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    cmd=['/home/carus/miniforge3/envs/decv2/bin/python',str(P/'compare_reference_edit_rollouts.py'),
        '--mode','direct','--out',str(f),'--reference',str(case/'reference_full.npz'),
        '--reference-id','0','--source-episode',str(ep),'--front-only','--audit-recording',
        '--grasp-evidence','--no-video','--object-mass-kg',str(mass),'--friction',str(friction),
        '--reference-interpolation-threshold','.1','--guidance-steps','9','--guidance-scale','0',
        '--execution-steps','1','--prior-noise-seed','44']
    with (f/'sim.log').open('w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',tag,ep,flush=True)
    return dict(tag=tag,episode=ep,status='complete')


def main():
    jobs=[(*physics,ep) for physics in PHYSICS for ep in EPISODES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,jobs))
    for tag,mass,friction,output in PHYSICS:
        selected=[x for x in results if x['tag']==tag]
        (output/'direct_interp_run_status.json').write_text(json.dumps(dict(
            mass_kg=mass,friction=friction,interpolation_threshold_rad=.1,
            episodes=EPISODES,results=selected,video_recorded=False),indent=2)+'\n')
    print('ALL DIRECT RUNS COMPLETE',len(results),flush=True)


if __name__=='__main__':
    main()
