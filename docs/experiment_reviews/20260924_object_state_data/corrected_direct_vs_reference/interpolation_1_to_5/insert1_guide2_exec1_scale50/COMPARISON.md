# Insert1, guide2 scale50: execute1 versus execute2

Episode53 original75 absolute hand targets were expanded to149 actions at30Hz by inserting one linear target between each adjacent pair. Both variants use the 10B EMA prior, DDIM4, guide2, scale50, fixed noise44, native wrist, imported hand/object pose,170g bulb, friction2.2, 60-step static grasp and 60-step final-target hold under the native `xjz_test.sh` failure protocol. All27 stored initial-state fields and all60 static trace rows match exactly. Execute1 applies the first generated action and advances the expanded reference by one action; execute2 applies the first two and advances by two. Saved model-call indices confirm 149 calls at0,1,...,148 versus75 calls at0,2,...,148.

| Execution | Command/reference RMSE | Actual joint/reference RMSE | Mean absolute adjacent command change per joint | Action-end angle from vertical | After2s hold | Native failure |
|---:|---:|---:|---:|---:|---:|---|
| 2 | 0.08787rad | 0.11263rad | 0.01589rad | **24.63°** | **24.99°** | no |
| 1 | **0.04265rad** | **0.07885rad** | **0.01166rad** | 26.05° | 25.03° | no |

Execute1 follows the expanded joint reference substantially more closely and emits smoother adjacent commands. Both variants rotate the bulb to about25° after the two-second hold, with execute2 slightly closer to vertical at action end. Neither reaches a20° completion criterion. Original recorded actions directly executed in simulation reached21.46° action-end and18.85° hold-end. Native nonfailure is an evaluation proxy, not independent physical-grasp confirmation.

Front-view artifacts in the parent folder: `insert1_guide2_exec1_vs_exec2_action_end.png`, `insert1_guide2_exec1_vs_exec2_hold_end.png`, and `insert1_guide2_exec1_vs_exec2_front_progress.mp4` (135 fully decoded H.264 frames at30fps). The video aligns original reference progress and labels actual simulation time per panel. Run script: `../../run_insert1_guide2.sh 1 50`; renderer: `../../render_insert1_guide2_comparison.py --exec1`.
