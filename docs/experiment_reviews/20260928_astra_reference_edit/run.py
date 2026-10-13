"""Frozen Astra reference replay, paired direct and reference-initialized DDIM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / 'source'
OLD = ROOT / '.worktrees/Astra-controller/outputs'
REFERENCES = {
    'left': OLD / 'astra_left_sequence/v2_s5_pause100',
    'right': OLD / 'astra_halfturn/seed42_10b_right_gait_s100_v2',
}
METHODS = [('direct', None), ('zero', 0.), ('edit010', .1), ('edit020', .2), ('edit035', .35)]


def rows(path):
    return [json.loads(line) for line in (path / 'trajectory.jsonl').read_text().splitlines()]


def run(direction, method, ratio):
    ref = REFERENCES[direction]
    dest = HERE / 'runs' / (direction + '_' + method)
    summary = dest / 'summary.json'
    if summary.exists():
        return json.loads(summary.read_text())
    if dest.exists():
        raise RuntimeError('Incomplete attempt retained: ' + str(dest))
    manifest = json.loads((ref / 'manifest.json').read_text())
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTRA_')}
    env.update(ASTRA_SESSION=str(dest), ASTRA_MODE='direct' if method == 'direct' else 'guided',
               GUIDANCE_SCALE='0', ASTRA_DDIM_STEPS='4', SEED='42', MAX_STEPS='300',
               TRAJ_STEPS_LIMIT='12000', CKPT_PATH='/home/carus/data_usb/10B_obs_4-66.ckpt',
               ASTRA_TARGET_TWIST_DEG='540' if direction == 'left' else '-180',
               ASTRA_INSTRUCTION=manifest['instruction'], ASTRA_REPLAY=str(ref),
               ASTRA_MODEL_NOISE_SEED='44', ASTRA_FIXED_NOISE='1',
               ASTRA_DEMO_CACHE=str(OLD / 'astra_noise_direction/demo_cache'),
               MODEL_PYTHON='/home/carus/miniforge3/envs/dp/bin/python',
               SIM_PYTHON='/home/carus/miniforge3/envs/decv2/bin/python',
               CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               ASTRA_VIDEO_LABEL=direction + ' ' + method)
    if ratio is not None:
        env['ASTRA_EDIT_NOISE_RATIO'] = str(ratio)
    began = time.monotonic()
    with dest.with_suffix('.log').open('w') as log:
        subprocess.run(['bash', 'eval/astra_halfturn_16.sh'], cwd=SOURCE, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    a, b = np.load(ref / 'initial_state.npz'), np.load(dest / 'initial_state.npz')
    assert set(a.files) == set(b.files)
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    actual = json.loads((dest / 'manifest.json').read_text())
    assert actual['protocol'] == manifest['protocol']
    for prediction in dest.glob('prediction_*.npz'):
        src = np.load(ref / prediction.name)
        new = np.load(prediction)
        np.testing.assert_array_equal(new['reference'], src['reference'])
        if method in ('direct', 'zero'):
            np.testing.assert_array_equal(new['prediction'], new['reference'][:, :2])
    rs = rows(dest)
    sign = 1 if direction == 'left' else -1
    result = dict(direction=direction, method=method, requested_noise_ratio=ratio,
                  physics_seed=42, noise_seed=44 if ratio is not None else None,
                  steps=len(rs), seconds=len(rs) * actual['control_dt'],
                  net_requested_direction_deg=sign * rs[-1]['twist_degrees'],
                  max_requested_direction_deg=max(sign * r['twist_degrees'] for r in rs),
                  first_native_failure_step=next((r['step'] for r in rs if r['failure']), None),
                  paired_initial_fields=len(a.files), reference_exact=True,
                  wall_seconds=time.monotonic()-began,
                  physical_drop='requires visual/contact review; native failure is not a drop label')
    summary.write_text(json.dumps(result, indent=2) + '\n')
    print('DONE', json.dumps(result), flush=True)
    return result


def validate_zero(direction):
    direct = rows(HERE / 'runs' / (direction + '_direct'))
    zero = rows(HERE / 'runs' / (direction + '_zero'))
    assert len(direct) == len(zero)
    for a, b in zip(direct, zero):
        for key in ('step', 'twist_degrees', 'failure', 'done', 'applied_target',
                    'proposed_target', 'astra_reference', 'state'):
            assert a[key] == b[key], (direction, a['step'], key)
    result = dict(direction=direction, full_physics_trace_exact=True, steps=len(direct))
    (HERE / (direction + '_zero_verification.json')).write_text(json.dumps(result, indent=2) + '\n')
    print('ZERO VERIFIED', result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['zero', 'main', 'all'], default='all')
    args = parser.parse_args()
    (HERE / 'runs').mkdir(exist_ok=True)
    provenance = {}
    for direction, ref in REFERENCES.items():
        provenance[direction] = dict(path=str(ref), files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ref.glob('response_*.json')) + [ref/'initial_state.npz', ref/'manifest.json']})
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if args.phase in ('zero', 'all'):
        for direction in REFERENCES:
            for method, ratio in METHODS[:2]:
                run(direction, method, ratio)
            validate_zero(direction)
    if args.phase in ('main', 'all'):
        for direction in REFERENCES:
            assert json.loads((HERE / (direction+'_zero_verification.json')).read_text())['full_physics_trace_exact']
            for method, ratio in METHODS[2:]:
                run(direction, method, ratio)
    results = [json.loads(p.read_text()) for p in sorted((HERE/'runs').glob('*/summary.json'))]
    (HERE / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
