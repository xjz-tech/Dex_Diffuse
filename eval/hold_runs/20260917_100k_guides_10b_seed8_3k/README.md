# 100k guide → 10B_obs_4-66, seed 8

One matched arm: 100k own-normalizer epoch-200/latest guide, scale 25, guidance
actions 2, DDIM4/4, execute 2, fresh noise, seed 8 / guide seed 100008,
3000 first episodes and 12000-step / 400-second cap. Uses the historical
hold-evaluation thresholds, not RL defaults. Neither checkpoint is modified;
each retains its own normalizer. No real hardware commands.

Read-only comparisons: existing 100k guide → obs_4-66 seed-8 result in
`20260914_serial_data_scaling_guide_scale_seed8_3k`, and 10B baseline / 10k
guide seed-8 results in `20260916_10b_vs_obs466_3k`. This is a single-seed
comparison, not a four-seed claim. Existing runs did not save enough complete
initial-state arrays for bitwise paired-initialization verification.
