# 30 Hz: no step clamp vs 0.03 rad step clamp

| Setting | 400-s survivors | Survival | Mean capped hold(s) | Median capped hold(s) |
|---|---:|---:|---:|---:|
| unclamped | 1514/3000 | 50.47% | 269.330 | 400.000 |
| clamp003 | 1054/3000 | 35.13% | 194.741 | 158.117 |

Step clamp active on 13.20% of joint commands, 61.95% of environment-action samples. Mean raw-to-sent absolute change: 0.002969 rad. Mean measured tracking residual in clamp arm: 0.034815 rad (no corresponding baseline telemetry was saved).

Single seed, 3000 first episodes, both30Hz/180Hz outer physics/2substeps/WAIT1. Only additional target-to-target step clamp0.03rad. Absolute limits precede clamp; observations contain the applied clipped target. No hardware-delay emulation; failure is existing simulator predicate. Telemetry covers first episodes only and includes their terminal action.
