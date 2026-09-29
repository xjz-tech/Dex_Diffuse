"""Scale-50, guide-2 experiment using the original 30-Hz reference actions."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
C = R / 'qualified_comparison'
OUT = R / 'nointerp_scale50_20260926'
OUT.mkdir(exist_ok=True)
EPISODES = (76, 34, 54, 2)
JOBS = [(ep, exe) for ep in EPISODES for exe in (1, 2)]

manifest = dict(
    episodes=EPISODES,
    jobs=[dict(episode=ep, execution_steps=exe) for ep, exe in JOBS],
    changed_variable='reference_interpolation 1 -> 0; original 30-Hz actions',
    fixed=dict(checkpoint='/home/carus/data_usb/10B_obs_4-66.ckpt',
               ddim=4, guidance_steps=2, guidance_scale=50,
               mass_kg=.17, friction=2.2, environment_seed=42,
               prior_noise_seed=44, settle_steps=60, terminal_hold_steps=60,
               stop_on_native_failure=False, video=False),
    reference_sha256={str(ep): hashlib.sha256(
        (C / f'episode_{ep:02d}' / 'reference_full.npz').read_bytes()).hexdigest()
        for ep in EPISODES},
)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def run(job):
    ep, exe = job
    case = C / f'episode_{ep:02d}'
    assert json.loads((case / 'baseline_verified.json').read_text())['baseline_verified']
    folder = case / f'guide{exe}_nointerp'
    folder.mkdir(exist_ok=True)
    if (folder / 'summary.json').exists():
        summary = json.loads((folder / 'summary.json').read_text())
        assert summary['reference_interpolation'] == 0
        assert summary['execution_steps'] == exe
        assert summary['guidance_scale'] == 50
        assert summary['steps'] == summary['intended_steps']
        return job, 'existing'

    sock = Path(f'/tmp/turn_nointerp_{ep}_{exe}_{os.getpid()}.sock')
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
                '--guidance-steps', '2', '--guidance-scale', '50',
                '--execution-steps', str(exe), '--reference-interpolation', '0'],
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
                str(P / 'compare_corrected_rollouts.py'), '--mode', 'guided',
                '--out', str(folder), '--reference', str(case / 'reference_full.npz'),
                '--reference-id', '0', '--source-episode', str(ep),
                '--front-only', '--audit-recording', '--grasp-evidence', '--no-video',
                '--guidance-steps', '2', '--guidance-scale', '50',
                '--execution-steps', str(exe), '--reference-interpolation', '0',
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
