"""Record Episode 54 at 130 g / mu=2.4 for interpolated direct, guidance and SDEdit."""
import json
import os
import subprocess
import time
from pathlib import Path

P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
CASE = R / 'qualified_comparison/episode_54'
BASE = R / 'm130_mu24_guidance4_vs_edit015_20260928' / 'episode_54'
OUT = R / 'm130_mu24_episode54_threeway_video_20260928'
METHODS = ('direct_interp', 'guidance4', 'edit015')
MASS = .130
FRICTION = 2.4
THRESHOLD = .1
SEED = 44


def run(method):
    out = OUT / method
    out.mkdir(parents=True, exist_ok=True)
    video = out / ('direct_front.mp4' if method == 'direct_interp' else 'guided_front.mp4')
    if (out / 'summary.json').exists() and video.exists() and (
            method == 'direct_interp' or (out / 'predictions.json').exists()):
        print('EXISTING', method, flush=True)
        return dict(method=method, status='existing')
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
        PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    serverenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    sock = Path(f'/tmp/m130mu24_ep54_video_{method}_{os.getpid()}.sock')
    server = None
    try:
        if method == 'guidance4':
            command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P/'server.py'),
                '--socket', str(sock), '--log', str(out/'predictions.json'),
                '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
                '--reference', str(CASE/'reference_full.npz'), '--ddim-steps', '4',
                '--guidance-steps', '4', '--guidance-scale', '50', '--execution-steps', '2',
                '--reference-interpolation-threshold', str(THRESHOLD)]
        elif method == 'edit015':
            command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P/'server_reference_action_editor.py'),
                '--socket', str(sock), '--log', str(out/'predictions.json'),
                '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
                '--reference', str(CASE/'reference_full.npz'), '--noise-ratio', '.15',
                '--ddim-steps', '4', '--execution-steps', '2',
                '--reference-interpolation-threshold', str(THRESHOLD)]
        else:
            command = None
        if command:
            with (out/'server.log').open('w') as log:
                server = subprocess.Popen(command, env=serverenv, stdout=log,
                                          stderr=subprocess.STDOUT)
            for _ in range(600):
                if sock.exists():
                    break
                if server.poll() is not None:
                    raise RuntimeError(f'server exited: {method}')
                time.sleep(.2)
            else:
                raise TimeoutError(method)

        sim = ['/home/carus/miniforge3/envs/decv2/bin/python', str(P/'compare_reference_edit_rollouts.py'),
            '--mode', 'direct' if method == 'direct_interp' else 'guided', '--out', str(out),
            '--reference', str(CASE/'reference_full.npz'), '--reference-id', '0',
            '--source-episode', '54', '--front-only', '--audit-recording', '--grasp-evidence',
            '--object-mass-kg', str(MASS), '--friction', str(FRICTION),
            '--reference-interpolation-threshold', str(THRESHOLD),
            '--guidance-steps', '4' if method == 'guidance4' else '9',
            '--guidance-scale', '50' if method == 'guidance4' else '0',
            '--execution-steps', '1' if method == 'direct_interp' else '2',
            '--prior-noise-seed', str(SEED)]
        if method != 'direct_interp':
            sim += ['--socket', str(sock)]
        with (out/'sim.log').open('w') as log:
            subprocess.run(sim, env=simenv, stdout=log, stderr=subprocess.STDOUT, check=True)
        if server is not None:
            assert server.wait(timeout=30) == 0
        print('DONE', method, flush=True)
        return dict(method=method, status='complete')
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait()
        sock.unlink(missing_ok=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(episode=54, source_frame=110, methods=METHODS, mass_kg=MASS,
        friction=FRICTION, object_size_multiplier=1,
        reference_interpolation_threshold_rad=THRESHOLD,
        direct=dict(execution_steps=1),
        guidance=dict(ddim=4, scale=50, guidance_steps=4, execution_steps=2),
        edit=dict(noise_ratio=.15, ddim=4, execution_steps=2, history=3,
                  future_reference=9),
        environment_seed=42, prior_seed=SEED, settle_steps=60,
        terminal_hold_steps=60, native_protocol='unchanged',
        baseline_folder=str(BASE), video_recorded=True)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    results = [run(method) for method in METHODS]
    (OUT/'run_status.json').write_text(json.dumps(results, indent=2)+'\n')
    print('ALL VIDEO RUNS COMPLETE', len(results), flush=True)


if __name__ == '__main__':
    main()
