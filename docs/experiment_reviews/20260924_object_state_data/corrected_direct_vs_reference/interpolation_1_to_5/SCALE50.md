# Insert 1: scale25 vs scale50

One linearly interpolated 22-joint target was inserted between each adjacent original episode53 action, yielding 149 control steps at 30Hz (4.97s). Both scales use the 10B EMA prior, DDIM4, fixed noise44, native wrist, 170g bulb, friction2.2, identical reference index advancement by the number of executed steps, 60-step static grasp, and 60-step final-target hold. The native `xjz_test.sh` failure configuration is used. All 27 initial-state fields and the full 60-step static trace match exactly between scale25 and scale50 for each guidance/execution configuration.

| Guidance/execution | Scale | Command/reference RMSE | Actual joint/reference RMSE | Angle from vertical at action end | After 2s hold | Native failure |
|---|---:|---:|---:|---:|---:|---|
| guide2/exec2 | 25 | 0.22715rad | 0.23698rad | 64.47° | 66.02° | no |
| guide2/exec2 | 50 | **0.08787rad** | **0.11263rad** | **24.63°** | **24.99°** | no |
| guide9/exec2 | 25 | 0.16619rad | 0.18014rad | 48.87° | 51.25° | no |
| guide9/exec2 | 50 | 0.09381rad | 0.11242rad | 73.35° | 75.04° | no |
| guide9/exec4 | 25 | 0.20107rad | 0.20650rad | 66.13° | 66.04° | no |
| guide9/exec4 | 50 | 0.10837rad | 0.12597rad | 80.20° | 80.51° | no |

The original 75 recorded actions directly executed in the same simulation ended at 21.46° from vertical and 18.85° after a two-second hold. Raising the guidance scale improves joint-action RMSE in every guided configuration, but only guide2/exec2 improves bulb rotation here. This shows that joint-reference RMSE cannot be used as a surrogate for the physical object outcome. Scale50 guide2/exec2 remains outside a 20° vertical completion criterion. Native nonfailure is a proxy, not independent physical-grasp confirmation.

Visuals: `insert1_scale25_vs50_action_end.png`, `insert1_scale25_vs50_hold_end.png`, `insert1_scale25_vs50_front_progress.mp4` (135 fully decoded H.264 frames at 30fps). The video aligns each simulation by original reference progress and labels each panel's actual elapsed simulation time. Inputs, traces, summaries and predictions are under `insert1_guide2_exec2_scale50/`, `insert1_guide9_exec2_scale50/`, and `insert1_guide9_exec4_scale50/`. Run script: `../../run_insert1_scale50.sh`; renderer: `../../render_insert1_scale50_comparison.py`.
