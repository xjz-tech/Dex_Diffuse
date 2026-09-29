"""Scale 50 on the same open-hand schedule, with guidance horizon 4 instead of 9.

Exec 2, DDIM 4, fresh prior noise, physics seed 42. The global opening
targets stay on the 2026-09-22 timetable; only the guidance window length changes.
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

PLAN = ROOT / 'docs/astra_controller/20260922_open_hand/plan_spec.json'
DIRECT = ROOT / 'outputs/astra_open_hand/seed42_direct'
BASE = ROOT / 'outputs/astra_open_hand'
SESSION = BASE / 'seed42_guided50_guide4'
INSTRUCTION = '逐渐张开整个手掌：从初始抓握姿态，用约4秒将五指伸直到中性关节位置，然后保持张开。'
CAP = 240


def actions_for(step, q0):
    """16 targets for trajectory steps step+1 .. step+16 of the fixed timetable."""
    alpha = np.clip((np.arange(step + 1, step + 17) - 8) / 120.0, 0, 1)
    return (q0[None, :] * (1 - alpha[:, None])).tolist()


def main():
    spec = json.loads(PLAN.read_text())
    q0 = np.asarray(spec['initial_target'], dtype=np.float64)
    # The stored step-0 chunk is 16 holds; guide 9 discarded the last 8 before opening.
    # Executed references follow holds for steps 1-8, then this same ramp.
    np.testing.assert_allclose(actions_for(0, q0)[:8], np.repeat(q0[None, :], 8, axis=0))
    for key, stored in spec['plans'].items():
        if int(key) == 0:
            continue
        np.testing.assert_allclose(actions_for(int(key), q0), stored, atol=1e-12, rtol=0)
    if SESSION.exists():
        raise FileExistsError(SESSION)
    env = os.environ.copy()
    env.update(ASTRA_SESSION=str(SESSION), ASTRA_INSTRUCTION=INSTRUCTION,
        ASTRA_MODE='guided', ASTRA_PLAN_STEPS='16', ASTRA_GUIDANCE_STEPS='4',
        ASTRA_REFERENCE_TAIL_POLICY='replan', ASTRA_REPLAN_STEPS='0',
        CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',
        ASTRA_DDIM_STEPS='4', GUIDANCE_SCALE='50', SEED='42', MAX_STEPS=str(CAP),
        ASTRA_TASK_FEEDBACK='1', ASTRA_TARGET_TWIST_DEG='', ASTRA_REPLAY='',
        ASTRA_PRIOR_MAX_DELTA_RAD='', CUDA_VISIBLE_DEVICES='0',
        MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',
        SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python',
        PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    env.pop('ASTRA_EDIT_NOISE_RATIO', None)
    env.pop('ASTRA_MODEL_NOISE_SEED', None)
    env.pop('ASTRA_FIXED_NOISE', None)
    print('START guided50 guide4', flush=True)
    checked = False
    log_path = BASE / 'guided50_guide4.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(['bash', 'eval/astra_controller.sh'], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 2400
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('guided50 guide4')
                pending = SESSION / 'pending.json'
                if pending.exists():
                    ticket = json.loads(pending.read_text())
                    response_path = Path(ticket['response'])
                    if not response_path.exists():
                        request = json.loads(Path(ticket['request']).read_text())
                        if not checked:
                            assert request['step'] == 0 and request['guidance_steps'] == 4
                            assert request['execution_steps'] == 2
                            np.testing.assert_array_equal(request['state']['qpos'], q0)
                            a, b = np.load(SESSION / 'initial_state.npz'), np.load(DIRECT / 'initial_state.npz')
                            assert set(a.files) == set(b.files)
                            for key in a.files:
                                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                            model = json.loads((SESSION / 'model.json').read_text())
                            assert model['guidance_scale'] == 50 and model['guidance_steps'] == 4
                            assert model['prior_ddim_steps'] == 4 and model['reference_editor'] is None
                            assert not model['fixed_noise']
                            manifest = json.loads((SESSION / 'manifest.json').read_text())
                            direct = json.loads((DIRECT / 'manifest.json').read_text())
                            assert manifest['protocol'] == direct['protocol'] and manifest['seed'] == 42
                            print('CHECKED', flush=True)
                            checked = True
                        step = request['step']
                        assert request['guidance_steps'] == 4 and request['required_plan_steps'] == 16
                        response = dict(session_id=request['session_id'], request_id=request['request_id'],
                            joint_names=spec['joint_names'], units='absolute_joint_radians',
                            actions=actions_for(step, q0),
                            rationale='Same fixed open-hand timetable as 2026-09-22; guidance horizon is 4.',
                            helper='Fixed endpoint interpolation only; no independent action selection',
                            plan_source='docs/astra_controller/20260922_open_hand/plan_spec.json',
                            global_start_step=step)
                        validate_response(response, request)
                        atomic_json(response_path, response)
                        print('guided50 guide4 step', step, flush=True)
                time.sleep(0.08)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        if process.returncode:
            raise RuntimeError(f'guided50 guide4 exit {process.returncode}; see {log_path}')
    assert checked and (SESSION / 'rollout.mp4').exists()
    rows = [json.loads(s) for s in (SESSION / 'trajectory.jsonl').read_text().splitlines()]
    guided9 = [json.loads(s) for s in (BASE / 'seed42_guided50' / 'trajectory.jsonl').read_text().splitlines()]
    overlap = min(len(rows), len(guided9))
    for new, old in zip(rows[:overlap], guided9[:overlap]):
        np.testing.assert_allclose(new['astra_reference'], old['astra_reference'], atol=1e-5, rtol=0)
    print('END', 'step', rows[-1]['step'], 'failure', rows[-1]['failure'],
          'reference_matched_steps', overlap, flush=True)


if __name__ == '__main__':
    main()
