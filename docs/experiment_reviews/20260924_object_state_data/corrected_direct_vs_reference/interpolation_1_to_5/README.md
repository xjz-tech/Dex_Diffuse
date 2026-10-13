# Episode53 reference interpolation: guide9, execute2 and execute4

Adjacent source actions from episode53/frame90–164 were linearly interpolated in the 22 absolute joint targets. Between each original pair, 1–5 new targets were inserted; every original target is retained exactly. The expanded references contain 149, 223, 297, 371 and 445 actions (original 75). Simulation stays at 30 Hz, so this stretches the action phase from 2.5 seconds to 4.97, 7.43, 9.90, 12.37 and 14.83 seconds. Every run then holds its final target for 60 steps (2 seconds). These different action durations matter when comparing object rotation.

At each inference call, guide9 uses the next nine expanded reference actions. Execute2 applies the first two model actions, then the reference index advances by two. Execute4 applies the first four, then advances by four; on the final partial chunk only the remaining actions are applied. Reference progress is not tied to measured joint or object error.

All runs use the same 10B EMA checkpoint, DDIM4, scale25, fixed noise44, native wrist, episode53 initial hand/object state, 170 g object and friction2.2. They use the native `xjz_test.sh` failure configuration. All 27 stored initial fields and the 60-step static trajectories were checked equal to the noninterpolated baseline. No native failure was recorded in these 12 runs. Native failure remains an evaluation proxy, not proof of a physical grasp.

| Inserted per pair | Action steps | Guide9/exec2 command RMSE (rad) | Exec2 end angle from vertical | Guide9/exec4 command RMSE (rad) | Exec4 end angle from vertical |
|---:|---:|---:|---:|---:|---:|
| 0 | 75 | 0.1678 | 79.0° | 0.1911 | 98.0° |
| 1 | 149 | 0.1662 | **48.9°** | 0.2011 | 66.1° |
| 2 | 223 | 0.1616 | 58.4° | 0.2097 | 54.4° |
| 3 | 297 | 0.1606 | 62.3° | 0.2104 | 48.5° |
| 4 | 371 | 0.1609 | 54.5° | 0.2107 | 49.1° |
| 5 | 445 | 0.1651 | 79.9° | 0.2121 | **29.7°** |

Command RMSE compares every generated/executed joint target to its time-matched expanded reference target over all actions and all 22 joints. Evaluating only at original-action anchor steps gives nearly identical values; see `comparison.json`. The actual-joint/reference RMSE values are also there. The reference's maximum adjacent single-joint change falls from 0.18 rad at insertion0 to 0.03 rad at insertion5.

The lowest final angle is insert5/exec4: 29.74° at the end of the 14.83-second action phase and 29.84° after 2 seconds of hold. Its wrist-relative bulb axis is 5.3° from the recorded real final bulb axis, though its world-vertical angle is 29.7° because the real and simulated wrists do not follow the same world trajectory. End displacement from imported object pose is 4.6 cm. The action-end and hold-end renderings show the bulb by the fingers; do not infer independent grasp confirmation from angle and native failure alone. This result is near, but does not meet, a 20° vertical completion criterion.

Interpolation did not consistently improve **joint** tracking: exec2 RMSE varies modestly and exec4 RMSE rises with insertion count. Object rotation is not monotonic either. The old video was a different initial grasp/reference, and these results do not isolate interpolation from the longer physical execution time. A command RMSE improvement by itself is not evidence of task success.

The two progress-aligned videos show insertion0–5 side by side for each execution length. Each panel is sampled at the same *original reference action index*; its caption also reports that panel's actual elapsed simulation time. The final 2 seconds are each variant's 60 physical hold steps. The files `guide9_exec2_progress_aligned.mp4` and `guide9_exec4_progress_aligned.mp4` have 135 H.264 frames each and were fully decoded for verification.

Data: `comparison.json`, `comparison.png`, and each `insertN_execM/` trace, summary, predictions and two-view video. Scripts: `../../reference_resampling.py`, `../../run_interpolation_sweep.sh`, `../../analyze_interpolation_sweep.py`, `../../render_interpolation_sweep.py`.
