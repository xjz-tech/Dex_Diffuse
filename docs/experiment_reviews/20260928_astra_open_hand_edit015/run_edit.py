"""One extra open-hand arm: reference edit, noise ratio 0.15, DDIM 4.

Same seed, initial state, and pre-authored opening schedule as
.worktrees/Astra-controller/docs/astra_controller/20260922_open_hand.
Does not rerun or overwrite the existing direct / scale25 / scale50 / prior-only arms.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/.worktrees/Astra-controller')
sys.path.insert(0, str(ROOT / 'eval'))
from astra_bridge import atomic_json, validate_response

PLAN = ROOT / 'docs/astra_controller/20260922_open_hand/plan_spec.json'
SOURCE = ROOT / 'outputs/astra_halfturn/seed42_right_gait_direct_replay'
DIRECT = ROOT / 'outputs/astra_open_hand/seed42_direct'
BASE = ROOT / 'outputs/astra_open_hand'
SESSION = BASE / 'seed42_edit015_ddim4'
INSTRUCTION = '逐渐张开整个手掌：从初始抓握姿态，用约4秒将五指伸直到中性关节位置，然后保持张开。'
CAP = 240


def main():
    spec = json.loads(PLAN.read_text())
    if SESSION.exists():
        raise FileExistsError(SESSION)
    BASE.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(ASTRA_SESSION=str(SESSION), ASTRA_INSTRUCTION=INSTRUCTION,
        ASTRA_MODE='guided', ASTRA_PLAN_STEPS='16', ASTRA_GUIDANCE_STEPS='9',
        ASTRA_REFERENCE_TAIL_POLICY='replan', ASTRA_REPLAN_STEPS='0',
        CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',
        ASTRA_DDIM_STEPS='4', GUIDANCE_SCALE='0', SEED='42', MAX_STEPS=str(CAP),
        ASTRA_TASK_FEEDBACK='1', ASTRA_TARGET_TWIST_DEG='', ASTRA_REPLAY='',
        ASTRA_PRIOR_MAX_DELTA_RAD='', ASTRA_EDIT_NOISE_RATIO='0.15',
        ASTRA_MODEL_NOISE_SEED='44', ASTRA_FIXED_NOISE='1',
        CUDA_VISIBLE_DEVICES='0',
        MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',
        SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python', PYTHONIOENCODING='utf-8',
        PYTHONUNBUFFERED='1')
    print('START edit015', flush=True)
    checked = False
    log_path = BASE / 'edit015_ddim4.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(['bash', 'eval/astra_controller.sh'], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 2400
            while process.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('edit015')
                pending = SESSION / 'pending.json'
                if pending.exists():
                    p = json.loads(pending.read_text())
                    response_path = Path(p['response'])
                    if not response_path.exists():
                        request = json.loads(Path(p['request']).read_text())
                        if not checked:
                            assert request['step'] == 0
                            np.testing.assert_array_equal(request['state']['qpos'], spec['initial_target'])
                            a, b = np.load(SESSION / 'initial_state.npz'), np.load(DIRECT / 'initial_state.npz')
                            assert set(a.files) == set(b.files)
                            for key in a.files:
                                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
                            direct_manifest = json.loads((DIRECT / 'manifest.json').read_text())
                            manifest = json.loads((SESSION / 'manifest.json').read_text())
                            assert manifest['protocol'] == direct_manifest['protocol']
                            assert manifest['seed'] == 42 and manifest['data_indices'] == '000-149'
                            model = json.loads((SESSION / 'model.json').read_text())
                            editor = model['reference_editor']
                            assert editor['requested_noise_ratio'] == 0.15
                            assert editor['inference_steps'] == 4 and editor['execution_steps'] == 2
                            assert editor['guidance_scale'] == 0 and model['guidance_scale'] == 0
                            assert model['prior_ddim_steps'] == 4 and model['guidance_steps'] == 9
                            print('CHECKED', json.dumps(dict(actual_noise_ratio=editor['actual_noise_ratio'],
                                                             timesteps=editor['timesteps']), ensure_ascii=False), flush=True)
                            checked = True
                        assert request['joint_names'] == spec['joint_names']
                        step = request['step']
                        response = dict(session_id=request['session_id'], request_id=request['request_id'],
                            joint_names=spec['joint_names'], units='absolute_joint_radians',
                            actions=spec['plans'][str(step)],
                            rationale='Codex Astra: gradually extend all five fingers to zero joint angles; same fixed schedule as the 2026-09-22 open-hand arms.',
                            helper='Fixed endpoint interpolation only; no independent action selection',
                            plan_source='docs/astra_controller/20260922_open_hand/plan_spec.json',
                            global_start_step=step)
                        validate_response(response, request)
                        atomic_json(response_path, response)
                        if step % 32 == 0:
                            print('edit015 step', step, flush=True)
                time.sleep(0.08)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        if process.returncode:
            raise RuntimeError(f'edit015 exit {process.returncode}; see {log_path}')
    assert checked and (SESSION / 'rollout.mp4').exists()
    rows = [json.loads(s) for s in (SESSION / 'trajectory.jsonl').read_text().splitlines()]
    print('END', 'step', rows[-1]['step'], 'failure', rows[-1]['failure'], flush=True)


if __name__ == '__main__':
    main()
