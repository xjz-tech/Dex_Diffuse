#!/usr/bin/env python3
"""Run ep54 with edit through action 102, then autonomous 10B prior."""
import json
import os
import subprocess
import time
from pathlib import Path

P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
CASE = R / 'qualified_comparison/episode_54'
OUT = R / 'm170_mu20_episode54_edit_history_then_prior_20260928'
SOCK = Path(f'/tmp/ep54_edit_then_prior_{os.getpid()}.sock')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    server_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                      LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    sim_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
                   PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:'
                              '/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
                   LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    server_cmd = [
        '/home/carus/miniforge3/envs/dp/bin/python',
        str(P/'server_edit_then_autonomous_prior.py'),
        '--socket', str(SOCK), '--log', str(OUT/'predictions.json'),
        '--reference', str(CASE/'reference_full.npz'),
        '--checkpoint', '/home/carus/data_usb/10B_obs_4-66.ckpt',
        '--noise-ratio', '.15', '--ddim-steps', '4', '--execution-steps', '2',
        '--reference-interpolation-threshold', '.1', '--switch-action-index', '102']
    sim_cmd = [
        '/home/carus/miniforge3/envs/decv2/bin/python',
        str(P/'compare_reference_edit_rollouts.py'), '--mode', 'guided',
        '--out', str(OUT), '--reference', str(CASE/'reference_full.npz'),
        '--reference-id', '0', '--source-episode', '54', '--socket', str(SOCK),
        '--guidance-steps', '9', '--guidance-scale', '0', '--execution-steps', '2',
        '--reference-interpolation-threshold', '.1', '--prior-noise-seed', '44',
        '--object-mass-kg', '.170', '--friction', '2.0',
        '--front-only', '--audit-recording', '--grasp-evidence']
    server = None
    try:
        with (OUT/'server.log').open('w') as log:
            server = subprocess.Popen(server_cmd, env=server_env, stdout=log,
                                      stderr=subprocess.STDOUT)
        for _ in range(600):
            if SOCK.exists():
                break
            if server.poll() is not None:
                raise RuntimeError((OUT/'server.log').read_text())
            time.sleep(.2)
        else:
            raise TimeoutError('server socket')
        with (OUT/'sim.log').open('w') as log:
            subprocess.run(sim_cmd, env=sim_env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        assert server.wait(timeout=30) == 0
        manifest = dict(
            episode=54, mass_kg=.170, friction=2.0,
            reference_interpolation_threshold_rad=.1,
            prior='/home/carus/data_usb/10B_obs_4-66.ckpt', seed=44,
            edit=dict(noise_ratio=.15, actual_noise_ratio=.1533970386,
                      ddim_steps=4, execution_steps=2,
                      future_reference_steps=9),
            branch=dict(first_autonomous_zero_based_action_index=102,
                        common_executed_action_steps=102,
                        observation_history_frames=4,
                        previous_issued_action_targets_in_history=3,
                        future_reference_after_branch=False),
            autonomous_prior=dict(ddim_steps=4, execution_steps=2,
                                  guidance_scale=0),
            native_protocol='unchanged', video_recorded=True,
            downloads_written=False)
        (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('COMPLETE', OUT, flush=True)
    finally:
        if server is not None and server.poll() is None:
            server.terminate(); server.wait()
        SOCK.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
