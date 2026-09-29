# Reference tracking diagnosis

The older `comparison_slow.mp4` is episode37/case110. Slow means video playback at 15 fps instead of 30 fps; references were executed at 30 Hz. It used guide2/exec2, DDIM4, scale25, 10B EMA, noise44. The bulb ended 22.48 degrees from vertical and remained held, but command/reference RMSE was 0.18280 rad: object rotation did not imply accurate joint-reference tracking.

Archived comparisons (rad):

| Run | Command/reference RMSE | Actual joint/command RMSE | Maximum adjacent command change |
|---|---:|---:|---:|
| Episode37 case110 guide2/exec2 | 0.18280 | 0.07797 | 0.20120 |
| Episode53 guide2/exec2 | 0.22704 | 0.06798 | 0.12716 |
| Episode53 guide9/exec2 | 0.16778 | 0.07380 | 0.09126 |
| Episode53 guide9/exec4 | 0.19113 | 0.08075 | 0.05301 |

Both raw references have maximum adjacent joint changes of approximately 0.18 rad. These maxima are observed statistics, not controller speed limits. Episode53 guide9/exec2 has lower overall tracking RMSE than the older case110 but substantially worse object rotation. The trajectories, initial grasp, contact configuration and randomized environments differ, so the comparison does not isolate a physical cause.

For the latest guide9/exec4, ring PIP has RMSE 0.47052 rad; at action index20 the reference is 1.64546 rad, generated command 1.04819 rad. The plotted actual joint follows the generated command closely. This is a large target bias/amplitude difference, not merely evidence of a temporal delay. The older case110 also had large errors: its pinky DIP reference was nearly constant 1.396 rad while the generated command decreased toward 0.8 rad.

## Mechanism established by code

`diffusion_policy/guidance/guided_ddim.py` computes normalized action MSE averaged over horizon and 22 joints. With a frozen epsilon prediction, each element's direct loss-gradient coefficient is proportional to `scale/(horizon*22)`. Moving from horizon2 to horizon9 at scale25 reduces that coefficient to 2/9. Scale112.5 restores the coefficient of horizon2/scale25, but does not make the full denoising process equivalent because additional actions are constrained and the model is reevaluated each reverse step. Guidance is a soft correction during four DDIM steps, not a terminal action-equality constraint.

Reference indices advance by the executed chunk length without checking reference tracking error. Observations contain current joint positions, preceding executed targets, and their residual; neither object pose nor an object-rotation objective enters this prior/guidance. Consequently the method does not explicitly close the loop on bulb rotation. Differences in contact geometry can change the physical effect of similar joint errors.

## Fixed-observation inference intervention

Reconstructed all 19 guide9/exec4 observations from the archived actual joints and executed targets. Reused noise44 and DDIM4. Scale25 reproduced every archived output exactly (max absolute difference 0). Only the guidance scale changed in the other probes.

| Scale | Command/reference RMSE (rad) |
|---|---:|
| 0 | 0.20585 |
| 25 | 0.19113 |
| 50 | 0.16884 |
| 112.5 | 0.10682 |

Increasing scale from25 to112.5 reduced fixed-observation generation error by44.1%. This directly supports insufficient reference influence at scale25 for these observations. It does not establish a safe/optimal scale or successful closed-loop rotation: stronger guidance will alter future contact states and observations. These are offline inference probes, not additional simulation evaluation runs.

Recommended next causal comparison: same episode53 initial state, guide9/exec4 and original reference timing, scales25/50/112.5, using the existing native failure protocol. Measure both per-joint tracking and bulb rotation; do not infer successful manipulation from tracking RMSE alone.

Artifacts: `metrics.json`, `offline_predictions.npz`, `tracking.png`; reproducible script: `../../diagnose_reference_tracking.py`.
