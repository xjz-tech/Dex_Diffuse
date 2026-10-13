# Retired 1000-env evaluation data

Removed from active eval/hold_runs on 2026-09-13 at the user's request.
These are recoverable archived results, not current comparison inputs.
Do not scan this directory for benchmark selection or aggregate its results.

Original parent: /home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs

Exact targets:

- 20260911_mixed_vs_1b_3seeds_1000: 10 run folders (including interrupted runs).
- 20260911_10kguide_1B_vs_mixed_3seeds_1000: 6 run folders.

Every run's console.log explicitly reports num_envs=1000, and episode IDs stay
within 0..999. No matching live evaluation processes were found before moving.
Training datasets, checkpoints, 3000-env results, and 1024-env runs are untouched.
To recover, move the required directory back to its original parent only after
checking that no directory already occupies that destination.
