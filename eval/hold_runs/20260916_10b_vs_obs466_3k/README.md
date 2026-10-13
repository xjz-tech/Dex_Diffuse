# 10B_obs_4-66 versus obs_4-66

This queue adds only the missing `10B_obs_4-66.ckpt` arms. The matching
`obs_4-66.ckpt` baseline and 10k-guided results already exist under
`20260912_fresh_noise_3000_seed42_8_19_25` and are reused read-only.

Protocol: seeds 42, 8, 19, and 25; 3000 first episodes per arm; 12000-step
(400 simulated seconds at 30 Hz) cap; DDIM4 for prior and guide; execute2;
guidance_steps2; fresh prior/guide noise; guide seed = prior seed + 100000.
Scale 0 is the 10B baseline and scale 25 is the original 10k guidance. The
failure thresholds, trajectory selection, and censoring settings match the
existing obs_4-66 experiment. Each checkpoint retains its own normalizer.

The checkpoints passed the serial compatibility validator before this queue
was created. The queue waits for the already-scheduled frequency experiments
to finish and for the GPU to become idle. It never reruns or overwrites the
obs_4-66 results. State is in `state.json`; final aggregate and per-seed
comparisons are written to `comparison.json` and `comparison.md`.

The historical protocol did not save complete initial-state arrays, so exact
bitwise initial-state equality cannot be retroactively checked. Matching seeds,
simulator settings, data indices, and evaluation code are used.
