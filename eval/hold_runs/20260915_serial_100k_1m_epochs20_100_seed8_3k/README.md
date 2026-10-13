# Earlier guide checkpoint comparison

User requested earlier-epoch 100k/1M guides at scale25. Initial screen selects
epoch20 and epoch100 from existing periodic checkpoints, reusing epoch200
results. No retraining and no checkpoint overwrites. The selected checkpoints
are from the original 200-epoch cosine-schedule training runs, NOT separate
20/100-epoch trainings with rescaled learning-rate schedules.

Order after all four scale10/50 evaluations finish: 100k epoch20, 100k epoch100,
1M epoch20, 1M epoch100. Wait also for idle GPU before every job. FP32 serial,
DDIM4/4, predict9 guide2 execute2, scale25, seed8 / guide seed100008, fresh noise,
3000 first episodes, 12000-step cap, original hold thresholds and normalizers.
Each checkpoint epoch is checked against embedded metadata before evaluation.

Service: serial-guide-epochs-seed8-20260915.service. Current PIDs/progress:
state.json. Results including reused epoch200 controls: comparison.md/json.
Known zero-sample startup segfaults have at most two verified retries per stage.
Other errors halt for review; do not blindly rerun completed evaluations.
Frozen source, config and protected-input fingerprints: manifest.json.
The older watchdog does not supervise this new queue; it has its own queue logic.
