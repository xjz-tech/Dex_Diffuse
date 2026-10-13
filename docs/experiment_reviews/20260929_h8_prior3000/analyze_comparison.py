"""Compare paired 900-step first episodes without treating censored runs as failures."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD_10B = ROOT.parent / '20260927_prior3000_joint_turn' / 'seed42'
RUNS = {'1B': ROOT / 'seed42_1b', '10B': OLD_10B, 'h8': ROOT / 'seed42_h8'}
CAP = 900


def read_run(folder):
    assert (folder / 'RUN_COMPLETE').exists(), folder
    records = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines()]
    assert len(records) == 3000 and {r['env'] for r in records} == set(range(3000))
    records.sort(key=lambda r: r['env'])
    lengths = np.array([r['length'] for r in records], dtype=int)
    reasons = [r['reason'] for r in records]
    assert lengths.min() >= 1 and lengths.max() <= CAP
    net = np.zeros(3000, np.float64)
    moving_right = moving_left = 0
    total_valid = 0
    files = sorted(folder.glob('trajectory_*.npz'))
    assert len(files) == 3, files
    for path in files:
        with np.load(path) as z:
            assert z['episode'].shape[1] == 3000
            valid = (z['episode'] == 0) & (z['episode_step'] <= CAP)
            net += np.where(valid, z['right_increment_deg'], 0).sum(axis=0)
            speed = z['right_speed_deg_s']
            moving_right += int((valid & (speed > 5)).sum())
            moving_left += int((valid & (speed < -5)).sum())
            total_valid += int(valid.sum())
    return dict(records=records, lengths=lengths, reasons=reasons, net_right_deg=net,
        motion_frames=dict(right=moving_right, left=moving_left, total=total_valid))


def compare_initial(a, b):
    with np.load(a / 'initial_state.npz') as x, np.load(b / 'initial_state.npz') as y:
        return dict(same_keys=set(x.files) == set(y.files),
            mismatched=[k for k in x.files if k not in y.files or not np.array_equal(x[k], y[k])])


def main():
    data = {name: read_run(path) for name, path in RUNS.items()}
    paired = {name: compare_initial(OLD_10B, path) for name, path in RUNS.items() if name != '10B'}
    if any(not value['same_keys'] or value['mismatched'] for value in paired.values()):
        raise RuntimeError(f'initial states differ: {paired}')
    result = dict(protocol='DDIM4/exec2/EMA, seed42, 3000 native initial states, zero external force, 900-step first-episode cap',
        initial_state_checks=paired, models={})
    for name, run in data.items():
        lengths = run['lengths']
        net = run['net_right_deg']
        reasons = run['reasons']
        result['models'][name] = dict(
            n=len(lengths), mean_truncated_steps=float(lengths.mean()),
            median_steps=float(np.median(lengths)),
            step_quantiles={str(q): float(np.percentile(lengths, q)) for q in (10, 25, 50, 75, 90)},
            failed_by_150=int(sum(r == 'failure' and n <= 150 for r, n in zip(reasons, lengths))),
            failed_by_300=int(sum(r == 'failure' and n <= 300 for r, n in zip(reasons, lengths))),
            native_failure=int(sum(r == 'failure' for r in reasons)),
            censored_at_cap=int(sum(r == 'timeout' and n == CAP for r, n in zip(reasons, lengths))),
            other_end_reasons={r: reasons.count(r) for r in set(reasons) if r not in ('failure', 'timeout')},
            net_right_over_30=int((net > 30).sum()), net_left_over_30=int((net < -30).sum()),
            net_within_30=int((np.abs(net) <= 30).sum()),
            median_net_right_deg=float(np.median(net)), motion_frames=run['motion_frames'])
    for model, baseline in (('h8', '1B'), ('h8', '10B'), ('1B', '10B')):
        delta = data[model]['lengths'] - data[baseline]['lengths']
        right = data[model]['net_right_deg'] > 30
        baseline_right = data[baseline]['net_right_deg'] > 30
        cap = np.array([reason == 'timeout' and length == CAP for reason, length in zip(data[model]['reasons'], data[model]['lengths'])])
        baseline_cap = np.array([reason == 'timeout' and length == CAP for reason, length in zip(data[baseline]['reasons'], data[baseline]['lengths'])])
        paired_effects = dict(mean_truncated_steps=delta.astype(float),
                              cap_probability=cap.astype(float) - baseline_cap.astype(float),
                              right_turn_probability=right.astype(float) - baseline_right.astype(float))
        rng = np.random.default_rng(42)
        boot = {name: np.empty(2000, float) for name in paired_effects}
        for i in range(2000):
            sample = rng.integers(0, 3000, 3000)
            for name, values in paired_effects.items():
                boot[name][i] = values[sample].mean()
        result[f'{model}_minus_{baseline}'] = dict(mean_truncated_step_difference=float(delta.mean()),
            longer=int((delta > 0).sum()), shorter=int((delta < 0).sum()), equal=int((delta == 0).sum()),
            right_turn_both=int((right & baseline_right).sum()),
            right_turn_only_model=int((right & ~baseline_right).sum()),
            right_turn_only_baseline=int((~right & baseline_right).sum()),
            paired_effect_95pct_bootstrap={name: [float(v) for v in np.percentile(values, [2.5, 97.5])]
                                          for name, values in boot.items()})
    (ROOT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
