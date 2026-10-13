# Control frequency and 0.03-rad clamp: final verified results

All arms: 3000 identical initial environments, first episodes only, 400 simulated seconds, seed8/100008, 10k guide + obs_4-66, DDIM4/4, guide2/execute2, scale25, original normalizers, TensorRT FP16/fused updates, WAIT=1. Failure is the configured simulator predicate, not a separately verified physical drop. Means and medians are right-censored at400s.

| Group | Policy Hz / clamp | Survived400s | Survival | Mean hold(s) | Median hold(s) |
|---|---|---:|---:|---:|---:|
| A | 30 | 1514/3000 | 50.47% | 269.33 | 400.00 |
| A | 45 | 928/3000 | 30.93% | 212.48 | 197.11 |
| A | 60 | 519/3000 | 17.30% | 156.64 | 103.39 |
| A | 90 | 147/3000 | 4.90% | 88.59 | 43.79 |
| B | 30 | 1489/3000 | 49.63% | 266.10 | 394.97 |
| B | 40 | 1120/3000 | 37.33% | 232.84 | 250.74 |
| C | 30 | 1669/3000 | 55.63% | 284.55 | 400.00 |
| C | 25 | 1797/3000 | 59.90% | 295.29 | 400.00 |
| C | 20 | 1978/3000 | 65.93% | 306.34 | 400.00 |
| D | unclamped | 1514/3000 | 50.47% | 269.33 | 400.00 |
| D | clamp003 | 1054/3000 | 35.13% | 194.74 | 158.12 |

A/D:180Hz outer physics,2substeps. B:360Hz,1substep. C:300Hz,1substep. A/B integration step1/360s; C1/300s. Compare each group with its own30Hz bridge; do not merge the30Hz controls or claim identical physics across groups. Original pre-sweep30Hz result46.27% used60Hz outer physics/2substeps and is contextual only.

## Equal-action-count survival

| Group | Setting | Common action count | Survival |
|---|---|---:|---:|
| A | 30 | 12000 | 50.47% |
| A | 45 | 12000 | 41.67% |
| A | 60 | 12000 | 33.93% |
| A | 90 | 12000 | 22.70% |
| B | 30 | 12000 | 49.63% |
| B | 40 | 12000 | 45.47% |
| C | 30 | 8000 | 65.30% |
| C | 25 | 8000 | 65.00% |
| C | 20 | 8000 | 65.93% |
| D | unclamped | 12000 | 50.47% |
| D | clamp003 | 12000 | 35.13% |

Action-matched endpoints have different physical duration; equal-time and equal-action-count results answer different questions. Sampling of observations, replanning, and terminal checks also changes with action frequency. Single-seed results are not proof of an optimal real-hardware frequency or of the bulb pull-in mechanism.

## Step-clamp result

At30Hz,0.03rad clamp changes survival from50.47% to35.13% (460 fewer survivors), mean hold269.33s to194.74s. Clamp activated for13.20% of joint commands and61.95% of environment-action samples (at least one joint clipped). It is a target-to-target clamp after URDF bounds; the next observation uses the clipped target. No extra inference holds are present. This demonstrates a negative effect in this matched simulation; it does not authorize removing hardware protection or prove causation in the real video.

Training-data exceedance fractions from the earlier audit are not these observed closed-loop clip fractions. Mean raw-to-sent absolute difference0.002969rad; measured residual0.034815rad in clamped arm only, without matching baseline trace. The0.9rad/s nominal command-increment bound is not a measured joint-velocity bound.

## Validation

All10 full arms have3000 unique episode0 records, the expected action/physics tick counts, equal saved observations/demo/frame/root states, zero WAIT holds, matching checkpoint metadata and passing numerical gates. Clamp sample denominators and per-joint counts reconcile exactly; maximum target increment0.030000000000000027rad. See results.json for verification details.

Source reports:
- [20260916_10k_obs466_frequency_3k](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260916_10k_obs466_frequency_3k/comparison.md)
- [20260916_10k_obs466_40hz_3k](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260916_10k_obs466_40hz_3k/comparison.md)
- [20260916_10k_obs466_20_25hz_3k](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260916_10k_obs466_20_25hz_3k/comparison.md)
- [30Hz clamp comparison](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval/hold_runs/20260916_10k_obs466_30hz_clamp003_3k/comparison.md)
