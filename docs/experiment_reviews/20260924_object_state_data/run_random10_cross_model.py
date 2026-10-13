"""Run ten preselected noise seeds on the matched bulb-turn cases."""
import concurrent.futures
import fcntl
import json
from pathlib import Path
import random
import sys
import time

import analyze_h8_cross_model_comparison as analysis
import run_h8_cross_model_comparison as run

OUT = Path(__file__).resolve().parent.parent / '20260930_h8_object_state_random10'
SEEDS = random.Random(20260930).sample(range(1000, 1000000), 10)
assert len(set(SEEDS)) == 10 and not set(SEEDS) & {42, 44, 45, 46}


def work(job):
    folder = run.destination(*job)
    audit_file = folder / 'audit.json'
    if audit_file.exists():
        row = json.loads(audit_file.read_text())
        assert tuple(row[k] for k in ('seed', 'model', 'physics', 'episode', 'method')) == job
        return row
    result = run.run_one(job)
    if result['status'] == 'failed':
        raise RuntimeError(f'rollout failed: {job}: {result["error"]}')
    row = analysis.audit(*job)
    audit_file.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    run.O = OUT
    analysis.O = OUT
    with (OUT / 'batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = dict(
            seed_selection='random.Random(20260930).sample(range(1000, 1000000), 10)',
            prior_noise_seeds=SEEDS, environment_seed=42,
            models={k: str(v) for k, v in run.CHECKPOINTS.items()},
            episodes=list(run.EPISODES), physics=run.PHYSICS,
            methods=list(run.METHODS),
            guidance=dict(ddim_steps=4, scale=50, guide_steps=4, exec_steps=2),
            replay_edit=dict(ddim_steps=4, requested_noise_ratio=.15, exec_steps=2),
            interpolation='one midpoint iff max adjacent 22-joint jump >0.1 rad',
            static_only=False, settle_steps=60, terminal_hold_steps=60,
            video=False, workers=4,
        )
        (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        jobs = [(seed, model, physics, episode, method)
                for seed in SEEDS for model in ('h8', '1b', '10b')
                for physics in run.PHYSICS for episode in run.EPISODES
                for method in run.METHODS]
        assert len(jobs) == 720
        print('RANDOM10 SEEDS', SEEDS, 'JOBS', len(jobs), flush=True)
        start = time.monotonic()
        failed = []
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(work, job): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                    completed += 1
                    print('AUDITED', completed, '/', len(jobs), job,
                          'ordinary', row['ordinary_turn'], 'stable', row['stable_turn'],
                          'steps', row['first_separation_action_step'], flush=True)
                except Exception as exc:
                    failed.append(job)
                    print('FAILED', job, repr(exc), flush=True)
        (OUT / 'batch_status.json').write_text(json.dumps(dict(
            expected=len(jobs), audited=completed, failed=failed,
            elapsed_seconds=time.monotonic()-start), ensure_ascii=False, indent=2) + '\n')
        if failed:
            raise RuntimeError(f'{len(failed)} of {len(jobs)} jobs failed')
        assert completed == len(jobs)
        print('BATCH COMPLETE', completed, flush=True)


if __name__ == '__main__':
    main()
