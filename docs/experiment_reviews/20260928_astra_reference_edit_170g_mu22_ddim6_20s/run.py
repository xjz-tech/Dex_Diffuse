"""One 20-second capped Astra fixed-reference rollout at 170g/mu2.2, DDIM6."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=HERE/'source'
OLD=ROOT/'.worktrees/Astra-controller/outputs'
REFERENCES={'left':OLD/'astra_left_sequence/v2_s5_pause100',
            'right':OLD/'astra_halfturn/seed42_10b_right_gait_s100_v2'}
METHODS={'direct':None,'zero':0.,'edit010':.1,'edit020':.2,'edit035':.35}


def main():
    p=argparse.ArgumentParser();p.add_argument('direction',choices=REFERENCES)
    p.add_argument('method',choices=METHODS);a=p.parse_args()
    ref=REFERENCES[a.direction];ratio=METHODS[a.method]
    dest=HERE/'runs'/(a.direction+'_'+a.method)
    if (dest/'summary.json').exists():
        print('Existing completed run:',dest,flush=True);return
    if dest.exists():raise RuntimeError('Refusing to overwrite '+str(dest))
    dest.parent.mkdir(exist_ok=True)
    source_manifest=json.loads((ref/'manifest.json').read_text())
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTRA_')}
    env.update(ASTRA_SESSION=str(dest), ASTRA_MODE='direct' if a.method=='direct' else 'guided',
        ASTRA_BULB_MASS_KG='.17',ASTRA_BULB_FRICTION='2.2',
        ASTRA_PHYSICAL_REVIEW='1',ASTRA_REPLAY_TAIL_HOLD='1',
        GUIDANCE_SCALE='0',ASTRA_DDIM_STEPS='6',SEED='42',MAX_STEPS='600',
        TRAJ_STEPS_LIMIT='12000',CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',
        ASTRA_TARGET_TWIST_DEG='540' if a.direction=='left' else '-180',
        ASTRA_INSTRUCTION=source_manifest['instruction'],ASTRA_REPLAY=str(ref),
        ASTRA_MODEL_NOISE_SEED='44',ASTRA_FIXED_NOISE='1',
        ASTRA_DEMO_CACHE=str(OLD/'astra_noise_direction/demo_cache'),
        MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',
        SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python',CUDA_VISIBLE_DEVICES='0',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        ASTRA_VIDEO_LABEL=a.direction+' '+a.method+' DDIM6 170g mu2.2')
    if ratio is not None:env['ASTRA_EDIT_NOISE_RATIO']=str(ratio)
    env['ASTRA_RESOURCE_LOCK']='/tmp/codex-astra-reference-edit-170g-ddim6.lock'
    started=time.monotonic()
    with dest.with_suffix('.log').open('w') as log:
        subprocess.run(['bash','eval/astra_halfturn_16.sh'],cwd=SOURCE,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
    z=np.load(dest/'initial_state.npz');b=np.load(ref/'initial_state.npz')
    changed={'cached_object_mass','object_mass','object_inertia','object_friction'}
    assert set(z.files)==set(b.files)
    for key in z.files:
        if key not in changed:np.testing.assert_array_equal(z[key],b[key],err_msg=key)
    assert np.isclose(z['object_mass'][0],.17,atol=1e-6)
    assert np.allclose(z['object_friction'],2.2,atol=1e-6)
    rows=[json.loads(x) for x in (dest/'trajectory.jsonl').read_text().splitlines()]
    native=next((r['step'] for r in rows if r['failure']),None)
    review=(json.loads((dest/'physical_drop_review.json').read_text())
            if (dest/'physical_drop_review.json').exists() else None)
    if review is not None:
        assert review['action']=='confirmed_drop'
    held_step=review['last_confirmed_held_step'] if review else len(rows)
    sign=1 if a.direction=='left' else -1
    summary=dict(direction=a.direction,method=a.method,noise_ratio=ratio,mass_kg=float(z['object_mass'][0]),
        object_friction=float(z['object_friction'][0,0]),physics_seed=42,noise_seed=44 if ratio is not None else None,
        ddim_steps=6,max_observation_steps=600,steps=len(rows),physical_drop_review=review,physical_drop_confirmed=review is not None,
        observation_censored_without_confirmed_drop=review is None,first_native_failure_step=native,
        held_net_requested_deg=sign*rows[held_step-1]['twist_degrees'],
        held_max_requested_deg=max(sign*r['twist_degrees'] for r in rows[:held_step]),
        wall_seconds=time.monotonic()-started,initial_unmodified_fields_verified=len(z.files)-len(changed))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
