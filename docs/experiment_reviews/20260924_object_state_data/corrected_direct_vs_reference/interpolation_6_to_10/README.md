# Episode53: 6–10 interpolated hand targets per reference interval

This extends the earlier 1–5 target sweep. The source is episode53 actions90–164. Linear interpolation keeps all 75 original 22-joint absolute targets exactly. Inserting 6–10 targets produces 519, 593, 667, 741, 815 action steps. At 30Hz the action phase lasts 17.30, 19.77, 22.23, 24.70, 27.17 seconds respectively, versus 2.50 seconds without interpolation. Every rollout then holds its final commanded joint target for 60 steps (2 seconds).

At each model call, the next nine expanded actions guide the 10B EMA prior. The first two or four generated actions are applied, and the expanded reference index advances by exactly the number applied. Configuration: guide9, exec2 or exec4, DDIM4, scale25, fixed noise44, native wrist, 170g object, friction2.2, episode53 initial hand/object import, and the native `xjz_test.sh` failure protocol. All 27 archived initial fields and all 60 static settle trace rows match the noninterpolated baseline exactly. Only the front camera was recorded for new runs.

Command RMSE is computed over every time-matched expanded reference action and all 22 joints. A dagger marks a native failure during the action phase; its ending angle is diagnostic only. All other runs had no native failure through the final 2-second hold.

| Inserted | Action length | Exec2 command RMSE | Exec2 angle from vertical: action / hold | Exec4 command RMSE | Exec4 angle from vertical: action / hold |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.50 s | 0.1678 rad | 79.0° / 77.9° | 0.1911 rad | 98.0° / 99.0° |
| 6 | 17.30 s | 0.1639 rad | 58.3° / 61.1° | 0.2126 rad | 49.5° / 50.5° |
| 7 | 19.77 s | 0.1641 rad | †77.2° / 90.1° | 0.2150 rad | **26.4° / 26.9°** |
| 8 | 22.23 s | 0.1637 rad | 30.0° / 28.4° | 0.2161 rad | 37.3° / 37.9° |
| 9 | 24.70 s | **0.1628 rad** | **24.1° / 25.5°** | 0.2177 rad | 32.5° / 33.0° |
| 10 | 27.17 s | 0.1684 rad | 25.3° / 25.2° | 0.2202 rad | †76.1° / 76.1° |

First native failures: insert7/exec2 at action index583 of593; insert10/exec4 at index743 of815. Their end displacements from imported bulb pose were 26.5cm and 27.8cm, respectively. Native failure is a proxy; other no-failure outcomes do not independently prove a stable physical grasp.

The best surviving vertical angles are insert9/exec2 (24.05° action end, 25.54° hold end) and insert7/exec4 (26.40° action end, 26.85° hold end). Neither meets a 20° vertical criterion. Insert7/exec4 has 1.54° wrist-relative bulb-axis error to the recorded real final axis, but real and simulation wrist world trajectories differ, and the tracked object mesh/origin were not calibrated to the simulated object. This axis comparison is geometric evidence only.

Increasing interpolation density makes each reference step smaller (maximum adjacent single-joint change drops from 0.18rad at zero insertion to 0.0164rad at insertion10). It does not make prior action tracking reliably closer: exec2 RMSE changes modestly while exec4 RMSE rises to 0.2202rad. Longer action duration and changed replanning windows are coupled to interpolation in this sweep, so improved object rotation should not be attributed to smaller action jumps alone.

Two front-only videos put the **original recorded joint actions directly executed in simulation, without prior**, at top left. The other five panels show insert6–10 guided simulations. All six panels use the same simulation camera view. The direct-action control ended at 21.46° from world vertical and 18.85° after its two-second final-target hold, with no native failure. The numeric table's row 0 is a separate no-interpolation **guided** baseline; the video top-left direct control must not be confused with that row.

Frames are aligned by original action progress, with each panel's actual simulation time shown. Following the action phase, each simulation holds its last target for two seconds. The H.264 videos each have 135 fully decoded frames at 30fps; action-end and hold-end PNGs are also provided. Files: `guide9_exec2_front_direct_original_insert6_10.mp4` and `guide9_exec4_front_direct_original_insert6_10.mp4`.

Data and code: `comparison.json`, `comparison.png`, each `insertN_execM/` summary, trace, predictions and front video; `../../run_interpolation_sweep.sh`, `../../analyze_interpolation_sweep.py`, `../../render_interpolation_6_to_10_front.py`.
