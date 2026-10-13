# Additional serial guide scale sweep

User-requested order: 100k scale10, 100k scale50, 1M scale10, 1M scale50.
Reuse existing 200-epoch checkpoints with their own normalizers; no training.
All new runs: FP32 serial, guide2 / predict9 / execute2, DDIM4 per model,
seed8 and guide seed100008, fresh noise, 3000 first episodes, 12000-step cap.
Same explicit historical DP hold thresholds and 000-149 trajectory range.
Scale25 controls are verified and reused from the prior run without alteration.

Service: serial-guide-scale-seed8-20260915.service. State/PIDs: state.json.
Results update after each completed evaluation: comparison.md / comparison.json.
The frozen queue preserves completed stages and archives interrupted attempts
on restart. Known zero-sample Isaac Gym startup segfaults get at most two
verified retries per stage; other errors halt for review. GPU work is serialized.
The older watchdog is not the supervisor for this new queue.

Sources are frozen in .worktrees/serial-guide-scale-run-20260915. Source hashes,
config, checkpoint fingerprints and prior-control fingerprints are captured in
manifest.json. Formal evaluation does not use the modified recording entry point.
