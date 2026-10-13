"""Record actual DDIM16/scale50 adaptive012 rollouts for the four episodes."""

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
R = P / 'reference_turn_baseline_20260926'
C = R / 'qualified_comparison'
OUT = R / 'adaptive012_ddim16_scale50_video_20260927'
OUT.mkdir(exist_ok=True)
EPISODES = (76, 34, 54, 2)
NEW_CONFIGS = ((16, 50),)
JOBS = [(ep, ddim, scale) for ep in EPISODES for ddim, scale in NEW_CONFIGS]
THRESHOLD = .12

manifest = dict(
    episodes=EPISODES, ddim_steps=(16,), fixed_guidance_scales=(50,),
    new_jobs=[dict(episode=ep, ddim=ddim, scale=scale) for ep, ddim, scale in JOBS],
    paired_no_video_config=dict(ddim=16, scale=50,
                                folder='guide1_adaptive012_ddim16_scale50'),
    guidance_steps=2, execution_steps=1,
    selective_rule='insert one midpoint iff max absolute 22-joint target jump >0.12 rad',
    threshold_rad=THRESHOLD, checkpoint='/home/carus/data_usb/10B_obs_4-66.ckpt',
    mass_kg=.17, friction=2.2, environment_seed=42, prior_noise_seed=44,
    settle_steps=60, terminal_hold_steps=60, stop_on_native_failure=False,
    video=True,
    reference_sha256={str(ep): hashlib.sha256(
        (C / f'episode_{ep:02d}' / 'reference_full.npz').read_bytes()).hexdigest()
        for ep in EPISODES},
)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(job):
    ep, ddim, scale = job
    case = C / f'episode_{ep:02d}'
    assert json.loads((case / 'baseline_verified.json').read_text())['baseline_verified']
    reference = np.load(case / 'reference_full.npz')['hand_target_rad']
    expanded, progress = interpolate_large_jumps(reference, THRESHOLD)
    folder = case / f'guide1_adaptive012_ddim{ddim}_scale{scale}_video'
    folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists():
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['steps'] == summary['intended_steps']
        assert summary['steps']['action'] == expanded.shape[1]
        assert summary['reference_interpolation_threshold'] == THRESHOLD
        assert summary['guidance_scale'] == scale
        assert summary['prior']['ddim'] == ddim
        assert summary['vertical_scale_switch_angle_deg'] is None
        assert summary['video_recorded']
        assert np.array_equal(np.load(folder / 'reference_progress.npy'), progress)
        return job, 'existing'

    sock = Path(f'/tmp/turn_ddim_scale_{ep}_{ddim}_{scale}_{os.getpid()}.sock')
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                  PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
                  PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
                  LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    serverenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                     LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    server = None
    try:
        with (folder / 'server.log').open('w') as log:
            server = subprocess.Popen([
                '/home/carus/miniforge3/envs/dp/bin/python', str(P / 'server.py'),
                '--socket', str(sock), '--log', str(folder / 'predictions.json'),
                '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
                '--reference', str(case / 'reference_full.npz'),
                '--ddim-steps', str(ddim), '--guidance-steps', '2',
                '--guidance-scale', str(scale), '--execution-steps', '1',
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
        cmd = [
            '/home/carus/miniforge3/envs/decv2/bin/python',
            str(P / 'compare_corrected_rollouts.py'),
            '--mode', 'guided', '--out', str(folder),
            '--reference', str(case / 'reference_full.npz'),
            '--reference-id', '0', '--source-episode', str(ep),
            '--front-only', '--audit-recording', '--grasp-evidence',
            '--guidance-steps', '2', '--guidance-scale', str(scale),
            '--execution-steps', '1',
            '--reference-interpolation-threshold', str(THRESHOLD),
            '--socket', str(sock)]
        with (folder / 'sim.log').open('w') as simlog:
            subprocess.run(cmd, env=simenv, stdout=simlog,
                           stderr=subprocess.STDOUT, check=True)
        if server.wait(timeout=30):
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
