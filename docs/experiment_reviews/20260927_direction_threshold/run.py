"""Paired native closed-loop sweep of signed reference contrast, fixed scale25."""
import argparse,json,os,sys,subprocess,time,signal
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[2];SRC=P/'source'
sys.path[:0]=[str(SRC/'eval')]
from astra_kinematic_step import propose
from astra_bridge import atomic_json,validate_response
HISTORY=ROOT/'.worktrees/Astra-controller/outputs/astra_gait_pause9/no_gait_e3577_n48'

def run(lam,noise,steps,label=None):
    label=label or f'native_n{noise}_lambda{lam:+.5f}'.replace('+','p').replace('-','m').replace('.','p')
    dest=P/'runs'/label;dest.parent.mkdir(exist_ok=True)
    if (dest/'trial_summary.json').exists():return json.loads((dest/'trial_summary.json').read_text())
    assert not dest.exists(),dest
    cfg=dict(lambda_direction=lam,noise_seed=noise,physical_seed=3577,scale=25,guide=9,exec=2,ddim=4,
        steps_cap=steps,reference='r_mid + lambda*(r_left-r_right)/2; same current state, original left target-base/right measured-base rule; recompute every8steps',
        initial_hold_steps=8,force_sensor='Existing object net-contact force and total joint torques; no new physics sensors')
    env=os.environ.copy();env.update(ASTRA_SESSION=str(dest),ASTRA_MODE='guided',GUIDANCE_SCALE='25',ASTRA_DDIM_STEPS='4',
        SEED='3577',MAX_STEPS=str(steps),CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',ASTRA_TARGET_TWIST_DEG='180',
        ASTRA_INSTRUCTION='保持抓握，执行左右参考的固定比例插值，记录有符号转动。',
        ASTRA_GRASP_PROBE_CONFIG='',ASTRA_TASK_LIMIT_FILE='',ASTRA_REPLAY='',ASTRA_PRIOR_MAX_DELTA_RAD='',
        ASTRA_GUIDANCE_SCHEDULE_FILE='',ASTRA_MODEL_NOISE_SEED=str(noise),ASTRA_FIXED_NOISE='0',ASTRA_GUIDANCE_OBJECTIVE='joint_mse',
        ASTRA_FK_CONFIG='',ASTRA_POSTHOC_MATCH='0',ASTRA_DIRECT_FROM_STEP='',ASTRA_CAPTURE_STRIDE='1',
        ASTRA_DEMO_CACHE=str(ROOT/'.worktrees/Astra-controller/outputs/astra_noise_direction/demo_cache'),
        MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python',CUDA_VISIBLE_DEVICES='0',
        PYTHONDONTWRITEBYTECODE='1',PRINT_EVERY='100')
    with (P/'runs'/f'{label}.log').open('w') as log:
        started=time.monotonic()
        proc=subprocess.Popen(['bash','eval/astra_halfturn_16.sh'],cwd=SRC,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic()-started>1500:raise TimeoutError(label)
                pending=dest/'pending.json'
                if pending.exists():
                    p=json.loads(pending.read_text());out=Path(p['response'])
                    if not out.exists():
                        req=json.loads(Path(p['request']).read_text())
                        if req['step']==0:
                            left=right=np.repeat(np.asarray(req['state']['target_before'])[None],16,axis=0)
                        else:
                            left=propose(req,15.,16,0.,max_delta=.15,measured_base=False)[0]
                            right=propose(req,-15.,16,0.,max_delta=.15,measured_base=True)[0]
                        actions=(left+right)/2+lam*(left-right)/2
                        ans=dict(session_id=req['session_id'],request_id=req['request_id'],joint_names=req['joint_names'],units='absolute_joint_radians',actions=actions.tolist(),
                            rationale='Precommitted signed reference contrast sweep, scale fixed25.',lambda_direction=lam,
                            endpoint_left=left.tolist(),endpoint_right=right.tolist())
                        validate_response(ans,req);atomic_json(out,ans)
                        if not (dest/'config.json').exists():atomic_json(dest/'config.json',cfg)
                time.sleep(.06)
            if proc.returncode!=0:raise RuntimeError(f'{label} exit {proc.returncode}')
        except BaseException:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
            raise
    rows=[json.loads(s) for s in (dest/'trajectory.jsonl').read_text().splitlines()]
    initial=np.load(dest/'initial_state.npz'); original=np.load(HISTORY/'initial_state.npz')
    for key in original.files:np.testing.assert_array_equal(initial[key],original[key])
    angle=np.array([r['twist_degrees'] for r in rows]);base=angle[7]
    summary=dict(label=label,lambda_direction=lam,noise_seed=noise,steps=len(rows),native_failure=rows[-1]['failure'],
        twist_after_initial8_deg=float(angle[-1]-base),final_twist_deg=float(angle[-1]),
        final_1s_net_deg=float(angle[-1]-angle[max(7,len(angle)-31)]),wall_seconds=time.monotonic()-started,
        initial_27_fields_equal=True,video=str(dest/'rollout.mp4'))
    atomic_json(dest/'trial_summary.json',summary);print(json.dumps(summary),flush=True);return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lambdas',type=float,nargs='+',default=[1,0,-1]);ap.add_argument('--noise',type=int,default=48);ap.add_argument('--steps',type=int,default=288);ap.add_argument('--label');a=ap.parse_args()
    for lam in a.lambdas:run(lam,a.noise,a.steps,a.label)
