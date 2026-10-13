"""Change only guidance scale on the four already-qualified, frozen references."""
from pathlib import Path
import json,sys,subprocess,concurrent.futures,hashlib
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';C=R/'qualified_comparison';OUT=R/'scale45_55';OUT.mkdir(exist_ok=True)
EPS=[76,34,54,2]
jobs=[dict(episode=ep,execution_steps=exe,scale=sc) for ep in EPS for exe in [2,1] for sc in [45,55]]
manifest=dict(episodes=EPS,jobs=jobs,recording='user cancelled video; existing completed runs reused; remaining runs no-video; no composite video',reference_sha256={str(ep):hashlib.sha256((C/f'episode_{ep:02d}/reference_full.npz').read_bytes()).hexdigest() for ep in EPS},changed_variable='guidance_scale only:45 or55 vs existing50',fixed=dict(checkpoint='/home/carus/data_usb/10B_obs_4-66.ckpt',ema=True,ddim=4,guide_horizon=2,interpolation=1,mass_kg=.17,friction=2.2,environment_seed=42,prior_noise_seed=44,settle_steps=60,terminal_hold_steps=60,reference_advance='actual executed steps',stop_on_failure=False),baseline='existing direct and scale50 reused; no new episode/start selection')
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
def run(job):
 ep=job['episode'];exe=job['execution_steps'];sc=job['scale'];assert json.load(open(C/f'episode_{ep:02d}/baseline_verified.json'))['baseline_verified']
 subprocess.run(['/home/carus/miniforge3/envs/dp/bin/python',str(P/'run_turn_qualified.py'),str(ep),f'guide{exe}',str(sc),'--no-video'],check=True)
 return job
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 for job in ex.map(run,jobs):print('SWEEP COMPLETE',job,flush=True)
