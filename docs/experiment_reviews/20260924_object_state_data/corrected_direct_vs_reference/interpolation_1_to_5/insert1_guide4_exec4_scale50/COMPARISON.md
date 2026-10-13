# Insert1, guide4: execute2 versus execute4

Episode53 original75 absolute hand targets are expanded to149 by one linear target between each adjacent pair, at30Hz. The 10B EMA prior uses DDIM4, fixed noise44, guidance horizon4 and scale25 or50. Execute2 applies two generated actions and advances the expanded reference index by two; execute4 applies four and advances by four (final call applies one remaining action). Saved prediction indices confirm 75 model calls for execute2 (0,2,...,148) and38 for execute4 (0,4,...,148). All27 archived initial-state fields and the full60-step static trace are exactly equal across the four runs. Native wrist,170g bulb,friction2.2 and native `xjz_test.sh` failure protocol are unchanged.

| Scale | Execution | Command/reference RMSE | Actual joint/reference RMSE | End world angle from vertical | After2s hold | Native failure |
|---:|---:|---:|---:|---:|---|
| 25 | 2 | 0.13518rad | 0.14975rad | 85.99° | 85.63° | no |
| 25 | 4 | 0.29777rad | 0.28807rad | 129.03° | 137.77° | no |
| 50 | 2 | 0.07205rad | 0.09863rad | 59.13° | 59.75° | no |
| 50 | 4 | 0.15821rad | 0.16379rad | 99.79° | 99.69° | no |

Guide4/exec4 worsens both joint-reference tracking and bulb rotation for both scales in this initial state. Execute2 replans after two actions and its four-step reference windows overlap; execute4 replans after four and its windows do not overlap. The experiment identifies a difference in outcome, but does not independently isolate overlapping windows from changed feedback frequency or physical contact evolution. None reaches a20° vertical completion criterion. Native nonfailure is an evaluation proxy, not independent proof of physical grasp.

Front-view artifacts in the parent folder: `insert1_guide4_exec4_action_end.png`, `insert1_guide4_exec4_hold_end.png`, and `insert1_guide4_exec4_front_comparison.mp4` (135 fully decoded H.264 frames at30fps). The video aligns original reference progress and labels actual simulation time per panel. Run: `../../run_insert1_guide4.sh 4`; render: `../../render_insert1_guide4_comparison.py --execution 4`.
