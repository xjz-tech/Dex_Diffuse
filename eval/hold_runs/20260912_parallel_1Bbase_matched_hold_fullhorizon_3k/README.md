# Full-horizon guidance with historical hold benchmark settings

Reuse the completed 200-epoch shared-normalizer 10k checkpoint.
All runs use trajectories 000-149, object/tip failure thresholds 0.05/0.1 m,
tolerance scale 10000, fixed tolerance 20000, cross-trajectory probability
0.3, 12000-step cap, and first-episode censoring, matching xjz_test.sh.
Each method uses 3000 environments for each of seeds 8, 19, 25.
DDIM steps: 4. Execute: 2 actions. Guide: all 9 predicted actions.
Methods: serial scale 25; fixed base weights .95/.90/.70/.60; dynamic scale 25.

Historical guidance used the original 10k checkpoint and 2 guided actions.
The new experiment intentionally uses the retrained checkpoint and 9 actions;
it is not an exact replication of that historical algorithm configuration.
