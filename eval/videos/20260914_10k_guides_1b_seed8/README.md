# Original 10k guiding 1B — simulation recording

Visualization only: one environment, seed 8, 1800 control steps. Not a new 3000-environment benchmark or a selected successful episode. Resets, if any, remain in the uncut recording. MP4 is wall-clock paced; rendering affects speed.

Original checkpoints and their original normalizers: obs_4-66.ckpt base and sim_hand_10k_seed42/checkpoints/latest.ckpt guide. DDIM 4 for both; predict 9, guide 2, execute 2; serial guidance scale 25; fresh noise, guide seed 100008. Historical hold thresholds explicitly retained, not RL defaults. Exact launch settings are in record.sh.

The preceding late_x0_w30 evaluation completed normally (3000 environments, completion 30.00%, median hold 155.2 s). Follow-on experiment queue and watchdog are paused during recording and will be restored afterwards.
