"""Matched bulb-turn guidance/edit runs for the 1B, 10B, and h8 priors."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time

P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
C = R / 'qualified_comparison'
O = P.parent / '20260929_h8_object_state_data'
CHECKPOINTS = {
    '1b': Path('/home/carus/data_usb/aggresive_random_ckpt/obs_4-66.ckpt'),
    '10b': Path('/home/carus/data_usb/aggresive_random_ckpt/10B_obs_4-66.ckpt'),
    'h8': Path('/home/carus/data_usb/aggresive_random_ckpt/h8.ckpt'),
}
PHYSICS = {'m044_mu11': (.044, 1.1), 'm170_mu20': (.170, 2.0), 'm130_mu24': (.130, 2.4)}
EPISODES = (76, 34, 54, 2)
METHODS = ('guidance4', 'edit015')


def destination(seed, model, physics, episode, method):
    return O / f'seed{seed}' / model / physics / f'episode_{episode:02d}' / method


def run_one(job):
    seed, model, physics, episode, method = job
    ckpt = CHECKPOINTS[model]
    mass, friction = PHYSICS[physics]
    case = C / f'episode_{episode:02d}'
    out = destination(*job)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists() and (out / 'predictions.json').exists():
        return dict(job=job, status='existing')
    sim_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
        PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    server_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    sock = Path(f'/tmp/compare_h8_{os.getpid()}_{model}_{seed}_{physics}_{episode}_{method}.sock')
    if sock.exists():
        sock.unlink()
    if method == 'guidance4':
        command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P / 'server.py'),
            '--socket', str(sock), '--log', str(out / 'predictions.json'),
            '--checkpoint', str(ckpt), '--reference', str(case / 'reference_full.npz'),
            '--ddim-steps', '4', '--guidance-steps', '4', '--guidance-scale', '50',
            '--execution-steps', '2', '--reference-interpolation-threshold', '.1']
        reference_steps = 4
        guidance_scale = 50
    else:
        command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P / 'server_reference_action_editor.py'),
            '--socket', str(sock), '--log', str(out / 'predictions.json'),
            '--checkpoint', str(ckpt), '--reference', str(case / 'reference_full.npz'),
            '--noise-ratio', '.15', '--ddim-steps', '4', '--execution-steps', '2',
            '--reference-interpolation-threshold', '.1']
        reference_steps = 5 if model == 'h8' else 9
        guidance_scale = 0
    server = None
    try:
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(command, env=server_env, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if sock.exists():
                break
            if server.poll() is not None:
                raise RuntimeError(f'server exited early: {out}')
            time.sleep(.2)
        else:
            raise TimeoutError(out)
        sim = ['/home/carus/miniforge3/envs/decv2/bin/python', str(P / 'compare_reference_edit_rollouts.py'),
            '--mode', 'guided', '--out', str(out), '--reference', str(case / 'reference_full.npz'),
            '--reference-id', '0', '--source-episode', str(episode), '--front-only',
            '--audit-recording', '--grasp-evidence', '--no-video', '--object-mass-kg', str(mass),
            '--friction', str(friction), '--guidance-steps', str(reference_steps),
            '--guidance-scale', str(guidance_scale), '--execution-steps', '2',
            '--prior-noise-seed', str(seed), '--reference-interpolation-threshold', '.1',
            '--socket', str(sock)]
        with (out / 'sim.log').open('w') as log:
            subprocess.run(sim, env=sim_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if server.wait(timeout=60) != 0:
            raise RuntimeError(f'server exit failed: {out}')
        print('DONE', job, flush=True)
        return dict(job=job, status='complete')
    except Exception as exc:
        print('FAILED', job, repr(exc), flush=True)
        return dict(job=job, status='failed', error=repr(exc))
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait(timeout=30)
        sock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[44, 45, 46])
    parser.add_argument('--models', nargs='+', choices=CHECKPOINTS, default=list(CHECKPOINTS))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    O.mkdir(parents=True, exist_ok=True)
    manifest = dict(checkpoints={k: str(v) for k, v in CHECKPOINTS.items()},
        physics=PHYSICS, episodes=EPISODES, methods=METHODS, planned_seeds=[44, 45, 46],
        run_seeds=args.seeds, run_models=args.models,
        ddpm_training_steps=100, ddim_steps=4, execution_steps=2,
        guidance=dict(scale=50, future_reference_steps=4),
        edit=dict(requested_noise_ratio=.15, history_steps=3,
            future_reference_steps={'h8': 5, '1b': 9, '10b': 9}),
        interpolation='insert one midpoint when max 22-joint adjacent target jump > 0.1 rad',
        initial_state='per-episode qualified lateral grasp; fixed wrist; native seed 42',
        settle_steps=60, terminal_hold_steps=60, video=False,
        native_protocol='eval/xjz_test.sh thresholds preserved; physical geometry analyzed separately')
    (O / ('manifest_seeds_' + '_'.join(map(str, args.seeds)) + '.json')).write_text(json.dumps(manifest, indent=2) + '\n')
    (O / 'manifest.json').write_text(json.dumps({k: v for k, v in manifest.items()
        if k not in ('run_seeds', 'run_models')}, indent=2) + '\n')
    jobs = [(s, m, p, e, method) for s in args.seeds for m in args.models
            for p in PHYSICS for e in EPISODES for method in METHODS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_one, jobs):
            status = O / 'run_status.jsonl'
            with status.open('a') as file:
                file.write(json.dumps(row) + '\n')
    print('BATCH DONE', len(jobs), flush=True)


if __name__ == '__main__':
    main()
