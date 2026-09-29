"""Verify raw-reference turn baselines at 44 g and friction 1.1."""

import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


P = Path(__file__).resolve().parent
CASES = P / 'reference_turn_baseline_20260926/qualified_comparison'
EPISODES = (76, 34, 54, 2)


def run(ep):
    case = CASES / f'episode_{ep:02d}'
    assert json.loads((case / 'baseline_verified.json').read_text())['baseline_verified']
    out = case / 'direct_m044_mu11'
    out.mkdir(exist_ok=True)
    if (out / 'summary.json').exists():
        summary = json.loads((out / 'summary.json').read_text())
        assert summary['mass_kg'] == .044 and summary['friction'] == 1.1
        assert summary['steps'] == summary['intended_steps']
        assert summary['video_recorded']
        return ep, 'existing'
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                  PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
                  PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
                  LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    cmd = ['/home/carus/miniforge3/envs/decv2/bin/python',
           str(P / 'compare_corrected_rollouts.py'), '--mode', 'direct',
           '--out', str(out), '--reference', str(case / 'reference_full.npz'),
           '--reference-id', '0', '--source-episode', str(ep),
           '--front-only', '--audit-recording', '--grasp-evidence',
           '--object-mass-kg', '.044', '--friction', '1.1',
           '--guidance-steps', '2', '--guidance-scale', '50',
           '--execution-steps', '1']
    with (out / 'sim.log').open('w') as log:
        subprocess.run(cmd, env=simenv, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return ep, 'complete'


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run, EPISODES):
            print(result, flush=True)
