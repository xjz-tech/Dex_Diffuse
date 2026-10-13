# Insert1, guide4/exec2, scales25 and50

Episode53 original75 absolute hand targets were expanded to149 at30Hz by inserting one linear target between adjacent actions. Each model call guides the next four expanded targets and executes the first two generated actions; the next call advances the reference index by two. Configuration: 10B EMA, DDIM4, fixed noise44, same native wrist and imported hand/object state, 170g bulb, friction2.2, 60-step static grasp, 149-step action phase, 60-step final-target hold, and native `xjz_test.sh` failure protocol. All27 archived initial fields and every static trace row match guide2/exec2 scale50 exactly for both guide4 scales.

| Variant | Command/reference RMSE | Actual joint/reference RMSE | End angle from world vertical | After2s hold | Native failure |
|---|---:|---:|---:|---:|---|
| guide2/exec2 scale50 | 0.08787rad | 0.11263rad | 24.63° | 24.99° | no |
| guide4/exec2 scale25 | 0.13518rad | 0.14975rad | 85.99° | 85.63° | no |
| guide4/exec2 scale50 | **0.07205rad** | **0.09863rad** | 59.13° | 59.75° | no |
| guide9/exec2 scale50 | 0.09381rad | 0.11242rad | 73.35° | 75.04° | no |

The original actions directly executed in simulation ended 21.46° from vertical and 18.85° after hold. Guide4 scale50 has the lowest joint-action tracking error of these guided variants, but guide2 scale50 rotated the bulb closer to vertical. Guide4 did not reach the 20° vertical criterion. Native nonfailure is an evaluation proxy, not independent physical-grasp confirmation.

Front-view comparison artifacts in the parent folder: `insert1_guide4_exec2_action_end.png`, `insert1_guide4_exec2_hold_end.png`, and `insert1_guide4_exec2_front_comparison.mp4`. The video aligns original reference progress and labels each panel's elapsed simulation time; it was fully decoded at 135 H.264 frames,30fps. Run script: `../../run_insert1_guide4.sh`; renderer: `../../render_insert1_guide4_comparison.py`.
