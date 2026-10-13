"""Run raw qualification, guide4/exec2 and reference editing at 170 g, mu=2.0."""
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
C = R / 'qualified_comparison'
O = R / 'm170_mu20_guidance4_vs_edit015_20260928'
EPISODES = (76, 34, 54, 2)
METHODS = ('raw', 'guidance4', 'edit015')
MASS = .170
FRICTION = 2.0
THRESHOLD = .1
SEED = 44


def folder(ep, method):
    return O / f'episode_{ep:02d}' / method


def run(job):
    ep, method = job
    case = C / f'episode_{ep:02d}'
    out = folder(ep, method)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'summary.json').exists() and (method == 'raw' or (out / 'predictions.json').exists()):
        return dict(episode=ep, method=method, status='existing')
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
        PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    serverenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    sock = Path(f'/tmp/m170mu20_{method}_{ep}_{os.getpid()}.sock')
    server = None
    try:
        if method == 'guidance4':
            command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P/'server.py'),
                '--socket', str(sock), '--log', str(out/'predictions.json'),
                '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
                '--reference', str(case/'reference_full.npz'), '--ddim-steps', '4',
                '--guidance-steps', '4', '--guidance-scale', '50', '--execution-steps', '2',
                '--reference-interpolation-threshold', str(THRESHOLD)]
        elif method == 'edit015':
            command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P/'server_reference_action_editor.py'),
                '--socket', str(sock), '--log', str(out/'predictions.json'),
                '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
                '--reference', str(case/'reference_full.npz'), '--noise-ratio', '.15',
                '--ddim-steps', '4', '--execution-steps', '2',
                '--reference-interpolation-threshold', str(THRESHOLD)]
        else:
            command = None
        if command:
            with (out/'server.log').open('w') as log:
                server = subprocess.Popen(command, env=serverenv, stdout=log, stderr=subprocess.STDOUT)
            for _ in range(600):
                if sock.exists():
                    break
                if server.poll() is not None:
                    raise RuntimeError(f'server exited: {out}')
                time.sleep(.2)
            else:
                raise TimeoutError(out)
        sim = ['/home/carus/miniforge3/envs/decv2/bin/python', str(P/'compare_reference_edit_rollouts.py'),
            '--mode', 'direct' if method == 'raw' else 'guided', '--out', str(out),
            '--reference', str(case/'reference_full.npz'), '--reference-id', '0',
            '--source-episode', str(ep), '--front-only', '--audit-recording', '--grasp-evidence',
            '--no-video', '--object-mass-kg', str(MASS), '--friction', str(FRICTION),
            '--guidance-steps', '4' if method == 'guidance4' else '9',
            '--guidance-scale', '50' if method == 'guidance4' else '0',
            '--execution-steps', '1' if method == 'raw' else '2',
            '--prior-noise-seed', str(SEED)]
        if method != 'raw':
            sim += ['--reference-interpolation-threshold', str(THRESHOLD), '--socket', str(sock)]
        with (out/'sim.log').open('w') as log:
            subprocess.run(sim, env=simenv, stdout=log, stderr=subprocess.STDOUT, check=True)
        if server is not None:
            assert server.wait(timeout=30) == 0
        print('DONE', ep, method, flush=True)
        return dict(episode=ep, method=method, status='complete')
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait()
        sock.unlink(missing_ok=True)


def main():
    O.mkdir(parents=True, exist_ok=True)
    manifest = dict(episodes=EPISODES, methods=METHODS, mass_kg=MASS, friction=FRICTION,
        object_size_multiplier=1, interpolation_threshold_rad=THRESHOLD,
        guidance=dict(ddim=4, scale=50, guide=4, exec=2),
        edit=dict(noise_ratio=.15, actual_ratio_expected=.1533970386, ddim=4, exec=2,
                  history=3, future_reference=9),
        raw_qualification='original reference at 30 Hz without interpolation',
        source_frames={76:114,34:79,54:110,2:103}, environment_seed=42,
        prior_seed=SEED, settle_steps=60, terminal_hold_steps=60,
        native_protocol='unchanged', stop_on_native_failure=False, video_recorded=False)
    (O/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [(ep,m) for ep in EPISODES for m in METHODS]))
    (O/'run_status.json').write_text(json.dumps(results, indent=2)+'\n')
    print('ALL RUNS COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
