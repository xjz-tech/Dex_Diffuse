# Object_state_data: all 80 horizontal-grasp starts

> **有效性复审：** [PROCESS_AUDIT.md](../../visibility_audit_20260926/PROCESS_AUDIT.md) 证实旧录像未展示静置期间的失稳，近桌起点的支撑环境未匹配，固定 wrist/无灯座场景也不支持真实完整尾段的成功率评价。全程批次数据保留，但不再作为 prior 任务成功率证据。

> **当前全程结果：** [FULL_RUN_RESULTS.md](FULL_RUN_RESULTS.md) 记录了 80 条各自横抓起点到末帧的 240 条完整仿真，以及四条不同 episode 的合成视频。以下内容保留为旧版“首次原生 failure 即停止”批次的归档，不能用于判断是否跑到放手末帧。

> **2026-09-26 correction after comparing the previous episode-53 video:** This batch used `--stop-on-native-failure`; the previous video continued to the end of the reference after the same native failure signal. All 27 initial fields and the shared trace prefix match exactly for episode 53. Guided old runs execute all 713 actions; this batch stops after 604 actions at failure index 603. Therefore `0/80 full_no_failure` is only a native-signal endpoint and **must not be reported as 0/80 reaching the intended hand release**. Reorientation-and-release completion remains unmeasured for the truncated episodes. Pairwise initial equality also does not establish real-to-simulation initialization accuracy.

This experiment replays every `Object_state_data/episode_0..79` from one selected frame where the bulb is approximately horizontal. Each episode contributes one reference from that frame to its last recorded state. The same imported hand joints and bulb-in-wrist pose are used for three independent, single-environment rollouts. The native simulation wrist remains fixed, as in the earlier episode-53 comparison.

**Per-episode initialization:** Each of the 80 episodes has its own selected hand joints and `obj_state`/bulb-in-wrist pose. The statement that 27 initial snapshot fields match means only that the three comparison methods start identically *within one episode*. It does not mean all episodes share one object pose or that the imported pose agrees perfectly with the real camera image.

## Start selection

The local bulb `+Y` axis must be 70–110° from world vertical, the object center must be at least 0.22 m high, and at least 80 recorded actions must remain. Among eligible frames, select the closest match to episode 51 frame 105 or episode 53 frame 90 by hand-joint RMSE and bulb-in-wrist position error. Prefer starts with at least 1 cm estimated mesh clearance above the approximate real table and within 20 cm of either template's world object position. This yields 43 preferred lifted starts; the other 37 use the best eligible horizontal frame and are marked `fallback`. The mesh clearance is approximate because the object marker-to-mesh transform and real table height are not independently calibrated. Of the 43 preferred starts, 41 also have the bulb axis within 30° of a template in wrist coordinates.

Selected images: [pages 1](selection_page_1.png), [2](selection_page_2.png), [3](selection_page_3.png), [4](selection_page_4.png). Frame choices and start diagnostics are in [selection.json](selection.json).

## Rollouts and endpoints

Each selected episode is run three times: direct recorded joint targets, 10B prior guided by two interpolated reference targets with `exec2`, and the same prior with `exec1`. Guidance scale is 50; one midpoint target is inserted between each pair of original targets. Guided targets therefore take nearly twice as many 30 Hz control steps as direct replay. All runs use prior DDIM 4, guide DDIM 4, fixed seed 44, 170 g bulb mass, and friction coefficient 2.2. Simulations stop at the first native failure; otherwise they continue through the complete reference and a 60-step final hold. Video recording is disabled for this batch to avoid altering simulation throughput.

The native evaluation configuration follows `eval/xjz_test.sh`: object position threshold 0.05 m, fingertip thresholds 0.1 m, object rotation threshold 180°, immediate-invalid position threshold 0.15 m, `FailureToleranceScale=10000`, `fixedToleranceSteps=20000`, `resetOnReachGoal=false`, and cross-trajectory goal probability 0.3 over demonstrations 000–149. Position error is the difference between the current bulb-in-wrist position and the current native demonstration goal. Native failure is an evaluation proxy, not an independently verified physical drop.

The primary reorientation endpoint is the bulb's local `+Y` axis within 45° of **either** world-vertical direction for 15 consecutive action steps *before* native failure, after the 60-step settle ended in a horizontal orientation. This sign-invariant angle is `min(angle, 180°-angle)`: some recorded object poses flip axis sign even though the final camera images show similar socket placements. In the recorded endpoints, 79 of 80 are within 45° of either vertical and all 80 within 50°. The stricter ≤30° version is also reported. These are orientation proxies, not confirmed maintained grasps. Completing the action sequence and final hold without native failure is a separate endpoint. No run can establish real socket insertion because the simulation wrist is fixed while recorded wrists travel roughly 10–24 cm.

The analysis verifies that all 27 initial snapshot fields and every static-step state agree exactly across the three methods within each episode. Runs are grouped into all 80, the 43 lifted starts, the 41 lifted starts with a similar wrist-frame bulb axis, and the 37 fallback starts. Because some imports fail before control begins, it also reports the shared static-survivor subset and the subset that is still horizontal after the settle. Results are in `single_aggregate_results.json`, `single_per_episode_results.csv`, and `single_failure_free_progress.png` after batch completion.

## Results

All 80 × 3 runs completed. All 27 initial snapshot fields and all recorded settle-step states match exactly within each episode's three methods. Twenty-three of 80 episodes trigger native failure during the initial 60-step settle, before any reference action, identically in all three methods. Of the 57 that survive the settle, 47 still have a horizontal bulb axis (60–120° from directed vertical). Median first-physics-step bulb-in-wrist displacement is 1.41 cm across all 80; this import drift should be considered when interpreting apparent reorientation.

| Method | Axis ≤45° for 15 pre-failure steps among eligible 47 | Axis ≤30° among eligible 47 | Reach 25% / 50% of source actions without failure among all 80 | Median source progress among 57 settle survivors | Complete actions + 60-step hold without native failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct recorded targets | 33/47 | 25/47 | 24/80 / 8/80 | 16.5% | 0/80 |
| 10B guide2/exec2, scale 50 | 30/47 | 24/47 | 33/80 / 14/80 | 34.1% | 0/80 |
| 10B guide2/exec1, scale 50 | 33/47 | 23/47 | 22/80 / 7/80 | 14.4% | 0/80 |

For the 43 preferred lifted starts, the ≤45° orientation proxy is 21/43 direct, 20/43 exec2, and 19/43 exec1. Eight of the 43 trigger failure during settle, so the corresponding rates among lifted starts surviving settle are 21/35, 20/35, and 19/35. Every remaining episode subsequently triggers native failure during actions; none completes the entire selected tail. In paired source-progress comparisons, exec2 is ahead of exec1 in 39 episodes, behind in 11, and tied in 30 (including 23 shared settle failures). Against direct replay, exec2 is ahead in 38, behind in 19, and tied in 23.

The orientation proxy does not establish that the bulb was still firmly held; its pose may change while slipping or contacting a surface. The full-tail failure result is also constrained by the fixed simulation wrist, whereas the real wrist moves a median 17.7 cm after the selected frame. Midpoint interpolation changes the physical duration, so source-progress improvements cannot be attributed to the prior alone. A native failure is the task's evaluation signal, not an independently confirmed drop.

Outputs: [per-episode CSV](single_per_episode_results.csv), [aggregate JSON](single_aggregate_results.json), [failure-free progress plot](single_failure_free_progress.png). The [previous episode-53 four-panel example](/home/carus/Downloads/episode53_full_real_direct_guide2_exec2_exec1_xyz_axes.mp4) shows the visual comparison; this statistical batch was run without per-episode video recording.
