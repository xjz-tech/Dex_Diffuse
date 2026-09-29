"""Paired adaptive010 direct and DDIM4 scale50 at 44 g / mu1.1."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from reference_resampling import interpolate_large_jumps


P = Path(__file__).resolve().parent
ROOT = P / 'reference_turn_baseline_20260926'
CASES = ROOT / 'qualified_comparison'
OUT = ROOT / 'm044_mu11_adaptive010_ddim4_scale50_g4e2_20260928'
OUT.mkdir(exist_ok=True)
EPISODES = (76, 34, 54, 2)
CONFIGS = (('guided', 4, 50),)
JOBS = [(ep, mode, ddim, scale) for ep in EPISODES
        for mode, ddim, scale in CONFIGS]
MASS_KG = .044
FRICTION = 1.1
THRESHOLD = .10


def name(mode, ddim, scale):
    return ('direct_adaptive010_m044_mu11' if mode == 'direct' else
            f'guide4exec2_adaptive010_ddim{ddim}_scale{scale}_m044_mu11')


manifest = dict(
    episodes=EPISODES,
    jobs=[dict(episode=ep, mode=mode, ddim=ddim, scale=scale,
               folder=name(mode, ddim, scale)) for ep, mode, ddim, scale in JOBS],
    raw_direct_baseline='direct_m044_mu11/baseline_verified.json',
    object_mass_kg=MASS_KG, friction=FRICTION,
    reference_interpolation_threshold=THRESHOLD,
    selective_rule='max absolute adjacent target jump across 22 joints >0.10 rad: insert one midpoint',
    checkpoint='/home/carus/data_usb/10B_obs_4-66.ckpt',
    guidance_steps=4, execution_steps=2, fixed_guidance=True,
    environment_seed=42, prior_noise_seed=44,
    settle_steps=60, terminal_hold_steps=60,
    stop_on_native_failure=False, video=False,
    reference_sha256={str(ep): hashlib.sha256(
        (CASES / f'episode_{ep:02d}' / 'reference_full.npz').read_bytes()).hexdigest()
        for ep in EPISODES},
)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(job):
    ep, mode, ddim, scale = job
    case = CASES / f'episode_{ep:02d}'
    assert json.loads((case / 'direct_m044_mu11/baseline_verified.json').read_text())['baseline_verified']
    reference = np.load(case / 'reference_full.npz')['hand_target_rad']
    expanded, progress = interpolate_large_jumps(reference, THRESHOLD)
    folder = case / name(mode, ddim, scale)
    folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists():
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['steps'] == summary['intended_steps']
        assert summary['steps']['action'] == expanded.shape[1]
        assert summary['reference_interpolation_threshold'] == THRESHOLD
        assert summary['mass_kg'] == MASS_KG and summary['friction'] == FRICTION
        assert summary['vertical_scale_switch_angle_deg'] is None
        assert summary['mode'] == mode
        assert np.array_equal(np.load(folder / 'reference_progress.npy'), progress)
        if mode == 'guided':
            assert summary['prior']['ddim'] == ddim and summary['guidance_scale'] == scale
        return job, 'existing'

    sock = Path(f'/tmp/turn_g4e2_adaptive010_m044_mu11_{ep}_{ddim}_{scale}_{os.getpid()}.sock')
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                  PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
                  PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
                  LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    serverenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                     LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    server = None
    try:
        if mode == 'guided':
            with (folder / 'server.log').open('w') as log:
                server = subprocess.Popen([
                    '/home/carus/miniforge3/envs/dp/bin/python', str(P / 'server.py'),
                    '--socket', str(sock), '--log', str(folder / 'predictions.json'),
                    '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
                    '--reference', str(case / 'reference_full.npz'),
                    '--ddim-steps', str(ddim), '--guidance-steps', '4',
                    '--guidance-scale', str(scale), '--execution-steps', '2',
                    '--reference-interpolation-threshold', str(THRESHOLD)],
                    env=serverenv, stdout=log, stderr=subprocess.STDOUT)
            for _ in range(600):
                if sock.exists():
                    break
                if server.poll() is not None:
                    raise RuntimeError(f'prior server exited for {job}')
                time.sleep(.2)
            else:
                raise TimeoutError(f'prior server did not start for {job}')

        cmd = ['/home/carus/miniforge3/envs/decv2/bin/python',
               str(P / 'compare_corrected_rollouts.py'),
               '--mode', mode, '--out', str(folder),
               '--reference', str(case / 'reference_full.npz'),
               '--reference-id', '0', '--source-episode', str(ep),
               '--front-only', '--audit-recording', '--grasp-evidence', '--no-video',
               '--object-mass-kg', str(MASS_KG), '--friction', str(FRICTION),
               '--guidance-steps', '4', '--guidance-scale', str(scale or 50),
               '--execution-steps', '2',
               '--reference-interpolation-threshold', str(THRESHOLD)]
        if mode == 'guided':
            cmd.extend(['--socket', str(sock)])
        with (folder / 'sim.log').open('w') as log:
            subprocess.run(cmd, env=simenv, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        if server and server.wait(timeout=30):
            raise RuntimeError(f'prior server failed for {job}')
        return job, 'complete'
    finally:
        if server and server.poll() is None:
            server.terminate()
            server.wait()
        sock.unlink(missing_ok=True)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for future in concurrent.futures.as_completed(
                [pool.submit(run, job) for job in JOBS]):
            print(future.result(), flush=True)
