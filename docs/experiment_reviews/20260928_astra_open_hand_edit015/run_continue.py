"""Continue direct and edit through the full open, including after native failure.

Native failure is still recorded. The environment is not reset, and the hand
keeps receiving targets until step 128, when the reference reaches full open.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/Astra-controller')
sys.path.insert(0, str(ROOT / 'eval'))
from astra_bridge import atomic_json, validate_response

PLAN = json.loads((ROOT / 'docs/astra_controller/20260922_open_hand/plan_spec.json').read_text())
DIRECT = ROOT / 'outputs/astra_open_hand/seed42_direct'
BASE = ROOT / 'outputs/astra_open_hand'
INSTRUCTION = '逐渐张开整个手掌：从初始抓握姿态，用约4秒将五指伸直到中性关节位置，然后保持张开。'
CAP = 128
Q0 = np.asarray(PLAN['initial_target'], dtype=np.float64)


def actions_for(step):
    alpha = np.clip((np.arange(step + 1, step + 17) - 8) / 120.0, 0, 1)
    return (Q0[None, :] * (1 - alpha[:, None])).tolist()


def run(label, mode):
    session = BASE / label
    if session.exists():
        raise FileExistsError(session)
    env = os.environ.copy()
    env.update(ASTRA_SESSION=str(session), ASTRA_INSTRUCTION=INSTRUCTION,
        ASTRA_MODE='guided', ASTRA_PLAN_STEPS='16', ASTRA_GUIDANCE_STEPS='9',
        ASTRA_REFERENCE_TAIL_POLICY='replan', ASTRA_REPLAN_STEPS='0',
        CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',
        ASTRA_DDIM_STEPS='4', SEED='42', MAX_STEPS=str(CAP),
        ASTRA_TASK_FEEDBACK='1', ASTRA_TARGET_TWIST_DEG='', ASTRA_REPLAY='',
        ASTRA_PRIOR_MAX_DELTA_RAD='', ASTRA_CONTINUE_AFTER_FAILURE='1',
        CUDA_VISIBLE_DEVICES='0',
        MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',
        SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python',
        PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    env.pop('ASTRA_EDIT_NOISE_RATIO', None)
    env.pop('ASTRA_MODEL_NOISE_SEED', None)
    env.pop('ASTRA_FIXED_NOISE', None)
    if mode == 'direct':
        env['ASTRA_MODE'] = 'direct'
        env['GUIDANCE_SCALE'] = '25'
        env.pop('ASTRA_EDIT_NOISE_RATIO', None)
        env.pop('ASTRA_MODEL_NOISE_SEED', None)
        env.pop('ASTRA_FIXED_NOISE', None)
    elif mode == 'edit':
        env['ASTRA_MODE'] = 'guided'
        env['GUIDANCE_SCALE'] = '0'
        env['ASTRA_EDIT_NOISE_RATIO'] = '0.15'
        env['ASTRA_MODEL_NOISE_SEED'] = '44'
        env['ASTRA_FIXED_NOISE'] = '1'
    else:
        raise ValueError(mode)
    print('START', label, mode, flush=True)
    checked = False
    log_path = BASE / (label + '.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(['bash', 'eval/astra_controller.sh'], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 2400
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(label)
                pending = session / 'pending.json'
                if pending.exists():
                    ticket = json.loads(pending.read_text())
                    response_path = Path(ticket['response'])
                    if not response_path.exists():
                        request = json.loads(Path(ticket['request']).read_text())
                        if not checked:
                            assert request['step'] == 0
                            a, b = np.load(session / 'initial_state.npz'), np.load(DIRECT / 'initial_state.npz')
                            for key in a.files:
                                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                            print('CHECKED', label, flush=True)
                            checked = True
                        step = request['step']
                        response = dict(session_id=request['session_id'], request_id=request['request_id'],
                            joint_names=PLAN['joint_names'], units='absolute_joint_radians',
                            actions=actions_for(step),
                            rationale='Same fixed open-hand timetable; continue after native failure without reset.',
                            helper='Fixed endpoint interpolation only; no independent action selection',
                            plan_source='docs/astra_controller/20260922_open_hand/plan_spec.json',
                            global_start_step=step)
                        validate_response(response, request)
                        atomic_json(response_path, response)
                        print(label, 'step', step, flush=True)
                time.sleep(0.08)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        if process.returncode:
            raise RuntimeError(f'{label} exit {process.returncode}; see {log_path}')
    rows = [json.loads(s) for s in (session / 'trajectory.jsonl').read_text().splitlines()]
    fail = next(r['step'] for r in rows if r['failure'])
    print('END', label, 'steps', len(rows), 'first_failure', fail, flush=True)


if __name__ == '__main__':
    run('seed42_direct_open', 'direct')
