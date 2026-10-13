"""Prespecified native-physics screening and paired long-horizon control."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE / 'source'
sys.path.insert(0, str(ROOT / 'eval'))
from astra_bridge import atomic_json, validate_response
from astra_kinematic_step import propose

PROTOCOL = json.loads((HERE / 'protocol.json').read_text())


def plan(request, pure=False):
    start = np.asarray(request['state']['target_before'])
    diagnostics = []
    if request['step'] == 0 or pure:
        actions = np.repeat(start[None, :], 16, axis=0)
    else:
        actions, diagnostics = propose(request, -8, 16, 0, max_delta=.12)
        endpoint = actions[-1].copy()
        floors = {'right_thumb_IP': .3, 'right_thumb_MCP_FE': .55}
        for finger in ('index', 'middle', 'ring'):
            floors['right_' + finger + '_DIP'] = .65
            floors['right_' + finger + '_PIP'] = .4
        for name, floor in floors.items():
            idx = request['joint_names'].index(name)
            endpoint[idx] = max(endpoint[idx], floor)
        endpoint = np.clip(endpoint, request['joint_lower'], request['joint_upper'])
        actions = np.asarray([start + (endpoint-start)*(i+1)/16 for i in range(16)])
    return dict(session_id=request['session_id'], request_id=request['request_id'],
        joint_names=request['joint_names'], units='absolute_joint_radians',
        actions=actions.tolist(), diagnostics=diagnostics,
        rationale='Astra-authored continuous rightward pad sweep with retained finger flexion; recompute from this arm current measured state. No stop at 180 degrees.',
        helper='Analytic URDF Jacobian feedback controller selected by current Astra; no per-window language-model calls',
        pure_prior_reference_ignored=pure, controller_parameters=PROTOCOL['controller_parameters'])


def summarize(dest, config):
    m = json.loads((dest / 'manifest.json').read_text())
    z = np.load(dest / 'initial_state.npz')
    mass = float(z['object_mass'][0])
    mu = float(z['object_friction'].mean())
    # Stream rows: large runs contain full link poses and contact diagnostics.
    from scipy.spatial.transform import Rotation
    axis = np.asarray(m['axis_world'])
    first180 = None
    min_twist = max_twist = 0.
    max_tilt = 0.
    right_path = reverse_path = 0.
    previous = 0.
    first_failure = None
    last = None
    review_file = dest/'physical_drop_review.json'
    review = json.loads(review_file.read_text()) if review_file.exists() else None
    held_step = review.get('last_confirmed_held_step', 0) if review else 0
    held_angle = held_max_right = 0.
    held_right_path = held_reverse_path = 0.
    for line in (dest / 'trajectory.jsonl').open():
        row = json.loads(line)
        angle = row['twist_degrees']
        if angle is None:
            raise ValueError('undefined twist: ' + str(dest))
        min_twist, max_twist = min(min_twist, angle), max(max_twist, angle)
        right_path += max(0, previous-angle)
        reverse_path += max(0, angle-previous)
        if row['step'] <= held_step:
            held_right_path += max(0, previous-angle)
            held_reverse_path += max(0, angle-previous)
            held_angle = -angle
            held_max_right = max(held_max_right, -angle)
        previous = angle
        tilt = np.rad2deg(np.arccos(np.clip(np.dot(Rotation.from_quat(row['state']['object_xyzw']).apply([0,1,0]), axis), -1, 1)))
        max_tilt = max(max_tilt, float(tilt))
        if angle <= -180 and first180 is None:
            first180 = row['step']*m['control_dt']
        if row['failure'] and first_failure is None:
            first_failure = row['step']*m['control_dt']
        last = row
    result = dict(**config, steps=last['step'], seconds=last['step']*m['control_dt'],
        native_failure=first_failure is not None, first_native_failure_seconds=first_failure,
        net_right_deg=-last['twist_degrees'], max_net_right_deg=-min_twist,
        cumulative_right_path_deg=right_path, reverse_path_deg=reverse_path,
        max_tilt_deg=max_tilt, first_180_seconds=first180,
        mass_g=1000*mass, object_friction=mu, mass_friction_ratio_g=1000*mass/mu,
        hand_friction_min=float(z['hand_friction'].min()), hand_friction_max=float(z['hand_friction'].max()),
        object_scale=float(z['object_scale'][0]), initial_fields=len(z.files),
        physical_drop=review if review else 'not_yet_visually_verified', run=str(dest),
        terminal_kind='visually_confirmed_drop' if review else 'native_failure' if last['failure'] else 'censored_or_native_timeout')
    if review:
        result.update(drop_time_bracket_seconds=[review['last_confirmed_held_step']*m['control_dt'],review['first_confirmed_separated_step']*m['control_dt']],
            held_cutoff_step=held_step, held_net_right_deg=held_angle, held_max_net_right_deg=held_max_right,
            held_right_path_deg=held_right_path, held_reverse_path_deg=held_reverse_path,
            first_180_confirmed_while_held=first180 is not None and first180 <= held_step*m['control_dt'])
    if config['phase'] == 'formal':
        screen = HERE/'runs'/f"screen_e{config['environment_seed']}"/'initial_state.npz'
        baseline = np.load(screen)
        for key in z.files:
            np.testing.assert_array_equal(z[key], baseline[key], err_msg=key)
        result['paired_initial_fields_verified'] = len(z.files)
    atomic_json(dest/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def run(phase, env_seed, noise_seed, method):
    label = f'screen_e{env_seed}' if phase == 'screen' else f'{method}_e{env_seed}_n{noise_seed}'
    dest = HERE/'runs'/label
    config = dict(label=label, phase=phase, environment_seed=env_seed, noise_seed=noise_seed,
                  method=method, scale=0 if method == 'prior' else None if method == 'astra_direct' else int(method.removeprefix('guided')))
    if (dest/'summary.json').exists():
        result = json.loads((dest/'summary.json').read_text())
        assert all(result[k] == v for k,v in config.items())
        return result
    if dest.exists():
        raise FileExistsError('Incomplete trial is retained; refusing overwrite: ' + str(dest))
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASTRA_')}
    env.update(ASTRA_SESSION=str(dest), ASTRA_MODE='direct' if method == 'astra_direct' else 'guided',
        GUIDANCE_SCALE=str(config['scale'] or 0), ASTRA_DDIM_STEPS='4', SEED=str(env_seed),
        MAX_STEPS='150' if phase == 'screen' else str(PROTOCOL['max_steps']),
        TRAJ_STEPS_LIMIT=str(PROTOCOL['traj_steps_limit']),
        CKPT_PATH=PROTOCOL['checkpoint'], ASTRA_TARGET_TWIST_DEG='-180',
        ASTRA_INSTRUCTION='保持抓握，持续向右转动灯泡；达到180度后继续转动，不暂停，直到本次原生环境失败。',
        ASTRA_MODEL_NOISE_SEED=str(noise_seed), ASTRA_FIXED_NOISE='0',
        ASTRA_PHYSICAL_REVIEW='1' if phase == 'formal' else '0',
        ASTRA_DEMO_CACHE=str(HERE/'cache'/f'env{env_seed}'),
        MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',
        SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python', CUDA_VISIBLE_DEVICES='0')
    began = time.monotonic()
    with (HERE/'runs'/(label+'.log')).open('w') as log:
        proc = subprocess.Popen(['bash', 'eval/astra_halfturn_16.sh'], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(HERE/(label+'_process.json'), dict(pid=proc.pid, config=config, began=time.time()))
        try:
            while proc.poll() is None:
                if time.monotonic()-began > PROTOCOL['operational_wall_limit_seconds']:
                    raise TimeoutError('Operational wall limit, not a drop')
                pending = dest/'pending.json'
                if pending.exists():
                    data = json.loads(pending.read_text())
                    response_path = Path(data['response'])
                    if not response_path.exists():
                        request = json.loads(Path(data['request']).read_text())
                        response = plan(request, pure=method == 'prior')
                        validate_response(response, request)
                        atomic_json(response_path, response)
                time.sleep(.025)
        except BaseException as exc:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
            atomic_json(dest/'interrupted.json', dict(reason=repr(exc), counted_as_drop=False))
            raise
        if proc.returncode:
            raise RuntimeError(f'{label} simulator exit {proc.returncode}; inspect retained log')
    return summarize(dest, config)


def screen():
    results = []
    selected = []
    for seed in PROTOCOL['candidate_environment_seeds']:
        row = run('screen', seed, PROTOCOL['screen_noise_seed'], 'prior')
        row['selection_pass'] = not row['native_failure'] and row['seconds'] >= 5-1e-5 and row['mass_friction_ratio_g'] <= 100
        row['selection_reason'] = 'passed' if row['selection_pass'] else 'native failure within 5 seconds' if row['native_failure'] else 'mass/friction ratio above prespecified 100 g threshold'
        results.append(row)
        if row['selection_pass']:
            selected.append(seed)
        atomic_json(HERE/'screening.json', dict(results=results, selected_environment_seeds=selected, frozen=False))
        if len(selected) == 3:
            break
    atomic_json(HERE/'screening.json', dict(results=results, selected_environment_seeds=selected, frozen=True))
    if len(selected) < 3:
        raise RuntimeError('Fewer than three eligible candidates; review screening before expanding candidate set')


def formal(workers):
    selection = json.loads((HERE/'screening.json').read_text())
    assert selection['frozen']
    jobs = []
    for seed in selection['selected_environment_seeds']:
        # Direct is deterministic, so run once per physical configuration.
        jobs.append(('formal', seed, 0, 'astra_direct'))
        for noise in PROTOCOL['formal_noise_seeds']:
            for method in ('prior', 'guided5', 'guided25', 'guided100'):
                jobs.append(('formal', seed, noise, method))
    atomic_json(HERE/'formal_jobs.json', jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        results = [f.result() for f in futures]
    atomic_json(HERE/'results.json', results)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['screen', 'formal', 'all'])
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.phase in ('screen', 'all'):
        screen()
    if args.phase in ('formal', 'all'):
        formal(args.workers)
