"""Run scale 50-to-25 switch after the simulated bulb reaches near vertical."""

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
THRESHOLD = .12
TAG = 'adaptive012_switch25'
OUT = R / f'{TAG}_20260927'
OUT.mkdir(exist_ok=True)
EPISODES = (76, 34, 54, 2)
MODES = ('guide1',)
JOBS = [(ep, mode) for ep in EPISODES for mode in MODES]

manifest = dict(
    episodes=EPISODES,
    jobs=[dict(episode=ep, mode=mode) for ep, mode in JOBS],
    selective_rule=f'max(abs(a_next - a_current)) across 22 joints > {THRESHOLD:g} rad; insert one linear midpoint',
    threshold_rad=THRESHOLD,
    changed_variable='guided scale 50 until first simulated bulb angle <=30 deg, then scale 25 for all later predictions',
    fixed=dict(checkpoint='/home/carus/data_usb/10B_obs_4-66.ckpt',
               ddim=4, guidance_steps=2, guidance_scale_initial=50,
               guidance_scale_after=25, switch_angle_deg=30,
               mass_kg=.17, friction=2.2, environment_seed=42,
               prior_noise_seed=44, settle_steps=60, terminal_hold_steps=60,
               stop_on_native_failure=False, video=False),
    reference_sha256={str(ep): hashlib.sha256(
        (C / f'episode_{ep:02d}' / 'reference_full.npz').read_bytes()).hexdigest()
        for ep in EPISODES},
    insertion_counts={},
)
for ep in EPISODES:
    reference = np.load(C / f'episode_{ep:02d}' / 'reference_full.npz')['hand_target_rad']
    expanded, _ = interpolate_large_jumps(reference, THRESHOLD)
    manifest['insertion_counts'][str(ep)] = expanded.shape[1] - reference.shape[1]
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(job):
    ep, mode = job
    case = C / f'episode_{ep:02d}'
    assert json.loads((case / 'baseline_verified.json').read_text())['baseline_verified']
    folder = case / f'{mode}_{TAG}'
    folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists():
        summary = json.loads((folder / 'summary.json').read_text())
        reference = np.load(case / 'reference_full.npz')['hand_target_rad']
        expanded, progress = interpolate_large_jumps(reference, THRESHOLD)
        assert summary['reference_interpolation_threshold'] == THRESHOLD
        assert summary['guidance_scale'] == 50
        assert summary['vertical_scale_switch_angle_deg'] == 30
        assert summary['vertical_scale_after'] == 25
        assert summary['steps'] == summary['intended_steps']
        assert summary['steps']['action'] == expanded.shape[1]
        assert np.array_equal(np.load(folder / 'reference_progress.npy'), progress)
        return job, 'existing'

    exe = int(mode[-1]) if mode.startswith('guide') else 2
    guided = mode.startswith('guide')
    sock = Path(f'/tmp/turn_adaptive_{ep}_{mode}_{os.getpid()}.sock')
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                  PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
                  PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
                  LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    serverenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                     LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    server = None
    try:
        if guided:
            with (folder / 'server.log').open('w') as log:
                server = subprocess.Popen([
                    '/home/carus/miniforge3/envs/dp/bin/python', str(P / 'server.py'),
                    '--socket', str(sock), '--log', str(folder / 'predictions.json'),
                    '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
                    '--reference', str(case / 'reference_full.npz'),
                    '--guidance-steps', '2', '--guidance-scale', '50',
                    '--execution-steps', str(exe),
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
            '--mode', 'guided' if guided else 'direct',
            '--out', str(folder), '--reference', str(case / 'reference_full.npz'),
            '--reference-id', '0', '--source-episode', str(ep),
            '--front-only', '--audit-recording', '--grasp-evidence', '--no-video',
            '--guidance-steps', '2', '--guidance-scale', '50',
            '--vertical-scale-switch-angle-deg', '30',
            '--vertical-scale-after', '25',
            '--execution-steps', str(exe),
            '--reference-interpolation-threshold', str(THRESHOLD)]
        if guided:
            cmd.extend(['--socket', str(sock)])
        with (folder / 'sim.log').open('w') as simlog:
            subprocess.run(cmd, env=simenv, stdout=simlog,
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
