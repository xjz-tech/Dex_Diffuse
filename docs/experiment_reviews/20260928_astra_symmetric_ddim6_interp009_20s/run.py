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
METHODS={'direct':None,'zero':0.,'edit010':.1,'edit020':.2,'edit035':.35,'edit015_ddim4':.15,
         'guidance25_ddim4':None,'guidance100_ddim4':None,'guidance50_g4_ddim4':None}


def main():
    p=argparse.ArgumentParser();p.add_argument('direction',choices=REFERENCES)
    p.add_argument('method',choices=METHODS);a=p.parse_args()
    ref=REFERENCES[a.direction];ratio=METHODS[a.method]
    is_guidance=a.method.startswith('guidance')
    guidance_scale=float(a.method.split('_')[0].replace('guidance','')) if is_guidance else 0.
    ddim_steps=4 if a.method=='edit015_ddim4' or is_guidance else 6
    guide_steps=4 if a.method=='guidance50_g4_ddim4' else 9
    dest=HERE/'runs'/(a.direction+'_'+a.method)
    if (dest/'summary.json').exists():
        print('Existing completed run:',dest,flush=True);return
    if dest.exists():raise RuntimeError('Refusing to overwrite '+str(dest))
    dest.parent.mkdir(exist_ok=True)
    source_manifest=json.loads((ref/'manifest.json').read_text())
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTRA_')}
    env.update(ASTRA_SESSION=str(dest), ASTRA_MODE='direct' if a.method=='direct' else 'guided',
        ASTRA_BULB_MASS_KG='.17',ASTRA_BULB_FRICTION='2.2',
        ASTRA_PHYSICAL_REVIEW='1',ASTRA_CONTINUOUS_20S='1',
        ASTRA_FIXED_REFERENCE_NPZ=str(HERE/(a.direction+'_reference.npz')),
        GUIDANCE_SCALE=str(guidance_scale),ASTRA_DDIM_STEPS=str(ddim_steps),SEED='42',MAX_STEPS='600',
        TRAJ_STEPS_LIMIT='12000',CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',
        ASTRA_TARGET_TWIST_DEG='180' if a.direction=='left' else '-180',
        ASTRA_INSTRUCTION='同一换指循环持续'+('左转' if a.direction=='left' else '右转')+'，完整执行20秒',ASTRA_REPLAY=str(ref),
        ASTRA_MODEL_NOISE_SEED='44',ASTRA_FIXED_NOISE='1',
        ASTRA_DEMO_CACHE=str(OLD/'astra_noise_direction/demo_cache'),
        MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',
        SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python',CUDA_VISIBLE_DEVICES='0',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        ASTRA_VIDEO_LABEL=a.direction+' '+a.method+' cyclic DDIM'+str(ddim_steps)+' 170g mu2.2')
    if ratio is not None:env['ASTRA_EDIT_NOISE_RATIO']=str(ratio)
    env['ASTRA_RESOURCE_LOCK']='/tmp/codex-astra-symmetric-ddim6.lock'
    entry='eval/astra_halfturn_16.sh'
    if guide_steps==4:
        env.update(ASTRA_PLAN_STEPS='16',ASTRA_GUIDANCE_STEPS='4',ASTRA_REFERENCE_TAIL_POLICY='replan',ASTRA_REPLAN_STEPS='0',ASTRA_TASK_FEEDBACK='1')
        entry='eval/astra_controller.sh'
    (HERE/(a.direction+'_'+a.method+'_launch.json')).write_text(json.dumps(dict(entry=entry,environment={k:v for k,v in env.items() if k.startswith('ASTRA_') or k in ['GUIDANCE_SCALE','SEED','MAX_STEPS','CKPT_PATH']}),indent=2)+'\n')
    started=time.monotonic()
    with dest.with_suffix('.log').open('w') as log:
        subprocess.run(['bash',entry],cwd=SOURCE,env=env,
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
    assert len(rows)==600
    model=json.loads((dest/'model.json').read_text())
    assert model['prior_ddim_steps']==ddim_steps
    if ratio is not None: assert model['reference_editor']['inference_steps']==ddim_steps
    if is_guidance:
        assert model['reference_editor'] is None and model['guidance_scale']==guidance_scale
        assert model['guidance_steps']==guide_steps and model['fixed_noise'] and model['noise_seed']==44
    review=(json.loads((dest/'physical_drop_review.json').read_text())
            if (dest/'physical_drop_review.json').exists() else None)
    if review is not None:
        assert review['action']=='confirmed_drop'
    held_step=review['last_confirmed_held_step'] if review else len(rows)
    sign=1 if a.direction=='left' else -1
    summary=dict(direction=a.direction,method=a.method,noise_ratio=ratio,mass_kg=float(z['object_mass'][0]),
        object_friction=float(z['object_friction'][0,0]),physics_seed=42,noise_seed=44 if a.method!='direct' else None,
        guidance_scale=guidance_scale,guidance_steps=guide_steps,
        ddim_steps=ddim_steps,max_observation_steps=600,steps=len(rows),physical_drop_review=review,physical_drop_confirmed=review is not None,
        physical_review_pending=review is None,
        observation_censored_without_confirmed_drop=None,first_native_failure_step=native,
        held_net_requested_deg=sign*rows[held_step-1]['twist_degrees'] if review else None,
        held_max_requested_deg=max(sign*r['twist_degrees'] for r in rows[:held_step]) if review else None,
        wall_seconds=time.monotonic()-started,initial_unmodified_fields_verified=len(z.files)-len(changed))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
