"""Audit each seed45/46 rollout as soon as its server and simulator finish."""
import concurrent.futures
import json
from pathlib import Path
import time

from analyze_h8_cross_model_comparison import audit
from run_h8_cross_model_comparison import EPISODES, METHODS, O, PHYSICS, destination


def wait_and_audit(job):
    path = destination(*job)
    for _ in range(2880):
        if (path / 'summary.json').exists() and (path / 'predictions.json').exists():
            return audit(*job)
        time.sleep(5)
    raise TimeoutError(path)


def main():
    jobs = [(seed, model, physics, episode, method)
            for seed in (45, 46) for model in ('h8', '1b', '10b')
            for physics in PHYSICS for episode in EPISODES for method in METHODS]
    output = O / 'audited_seed45_46.jsonl'
    if output.exists():
        raise FileExistsError(output)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool, output.open('w') as file:
        for row in pool.map(wait_and_audit, jobs):
            file.write(json.dumps(row) + '\n')
            file.flush()
            print(row['seed'], row['model'], row['physics'], row['episode'], row['method'],
                  row['stable_turn'], row['first_separation_action_step'], flush=True)
    assert len(output.read_text().splitlines()) == len(jobs) == 144
    print('ONLINE AUDIT COMPLETE', len(jobs), flush=True)


if __name__ == '__main__':
    main()
