# No interpolation: scale25 vs scale50

Episode53 recorded actions90–164 were used directly as the 75-step hand reference at 30Hz. This tests scale50 with no interpolated targets. Guide2/exec2, guide9/exec2 and guide9/exec4 use the 10B EMA prior, DDIM4 and fixed noise44; the reference index advances by exactly the executed two or four steps. All other physical and evaluation settings match the previous corrected episode53 experiment: native wrist, imported hand/object initial pose, 170g bulb, friction2.2, 60-step static grasp, 75 action steps and 60-step final-target hold, with the native `xjz_test.sh` failure protocol. All 27 archived initial fields and every one of the 60 static trace rows are identical between scale25 and scale50 for each configuration.

| Guidance/execution | Scale | Command/reference RMSE | Actual joint/reference RMSE | World angle from vertical at action end | After 2s hold | Native failure |
|---|---:|---:|---:|---:|---:|---|
| guide2/exec2 | 25 | 0.22705rad | 0.23541rad | 115.78° | 115.67° | no |
| guide2/exec2 | 50 | 0.07892rad | 0.10827rad | 50.35° | 50.24° | no |
| guide9/exec2 | 25 | 0.16778rad | 0.17852rad | 78.98° | 77.86° | no |
| guide9/exec2 | 50 | **0.09102rad** | **0.11375rad** | **39.42°** | **39.67°** | no |
| guide9/exec4 | 25 | 0.19113rad | 0.19615rad | 97.97° | 98.97° | no |
| guide9/exec4 | 50 | 0.10099rad | 0.11933rad | 78.79° | 87.26° | no |

The original recorded actions directly executed in simulation (no prior) reached 21.46° at action end and 18.85° after hold, with no native failure. No guided run here reaches the 20° vertical criterion. With no interpolation, scale50 improves command tracking in all configurations, and guide9/exec2 gives the closest vertical result. This differs from the insert1/scale50 sweep, where guide2/exec2 gave the best vertical result (24.63°) and guide9/exec2 ended at 73.35°. Interpolation, guidance horizon, scale and physical contact interact; a lower joint RMSE does not guarantee a better object outcome. Native nonfailure is an evaluation proxy rather than independent proof of physical grasp.

All panels in `nointerp_scale25_vs50_front_progress.mp4` are simulation front-camera views, including the top-left original recorded actions directly executed in simulation. Frames align by original reference progress, and each panel labels its own elapsed simulation time. The file has 135 fully decoded H.264 frames at 30fps. Still images: `nointerp_scale25_vs50_action_end.png` and `nointerp_scale25_vs50_hold_end.png`. Each scale50 run directory contains its raw front video, full trace, model predictions, summary and initial state. Run with `../../run_insert1_scale50.sh 0`; render with `../../render_insert1_scale50_comparison.py --interpolation 0`.
