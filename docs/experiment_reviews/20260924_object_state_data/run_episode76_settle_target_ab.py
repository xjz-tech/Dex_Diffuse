"""Paired 60-step fixed-target diagnostic; no prior or reference progression."""

import argparse
import json
import os
from pathlib import Path
import subprocess


P = Path(__file__).resolve().parent
ROOT = P / 'reference_turn_baseline_20260926'
OUT = ROOT / 'episode76_settle_target_ab_20260927'
REFERENCE = ROOT / 'qualified_comparison/episode_76/reference_full.npz'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
               PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
               LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    manifest = dict(episode=76, source_frame=114, mass_kg=.044, friction=1.1,
                    environment_seed=42, control_hz=30, control_steps=60,
                    target_sources=['qpos', 'reference_action'],
                    reference=str(REFERENCE), prior=False, video=args.record,
                    note='Targets are fixed for all 60 steps; reference_action is action.npy[114,9:], not a time-varying reference replay.')
    manifest_name = 'video_recording_manifest.json' if args.record else 'manifest.json'
    (OUT / manifest_name).write_text(json.dumps(manifest, indent=2) + '\n')
    for source in manifest['target_sources']:
        folder = OUT / (source + '_video' if args.record else source)
        folder.mkdir(exist_ok=True)
        command = ['/home/carus/miniforge3/envs/decv2/bin/python',
                   str(P / 'compare_corrected_rollouts.py'), '--mode', 'direct',
                   '--out', str(folder), '--reference', str(REFERENCE),
                   '--reference-id', '0', '--source-episode', '76',
                   '--static-only', '--grasp-evidence',
                   '--object-mass-kg', '.044', '--friction', '1.1',
                   '--settle-target-source', source]
        command += (['--front-only', '--audit-recording'] if args.record
                    else ['--no-video'])
        with (folder / 'sim.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        print('COMPLETE', source, flush=True)


if __name__ == '__main__':
    main()
