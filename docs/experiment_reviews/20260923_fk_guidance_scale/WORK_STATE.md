# Work state

User approved original 10k guide + 1B prior, FK fingertip scales 25/50/100/200/500/1000 and joint scale25 control; seeds42/8/19; native random initial states.

Implementation and validation completed: differentiable SharpA FK; optional loss callback in production DDIM; metric selector. 19 tests passed. 64 Isaac Gym states matched FK within 1.069e-6 m after fixed root/palm transform. Scale1000, 8 environments, 20-step headless-camera smoke passed with readable video and exactly20 frames. Native viewer path crashed, so production uses camera-only wrapper around unchanged sim_eval.main, one frame per actual control step of env0's first episode.

Production queue launched 2026-09-23 11:50:29 Asia/Shanghai; PID1261355, detached process group. First joint_scale25_seed42 run passed300 control steps at6.49 actual steps/wall-second, no model/runtime error. 21 runs total, 1024 first episodes each, capped12000 steps/400 simseconds. Fresh prior seed=simseed, guide seed=simseed+100000. This is explicitly recorded in manifest and launch script.

Read status.json, queue.log and runs/<current>/run.log for progress. `code/run_queue.py` automatically validates episode coverage, exact initial/physical pairing, converts actual video to H264, validates frames against env0 first-episode length, and writes report.md/results.json/pairing_validation.json. Do not infer completion from process disappearance; require state=complete and all21 validated runs. Runs are long (initial baseline throughput implies ~31min/arm before additional FK overhead). Partial tables must not be reported as final scale ranking.

Resume: inspect incomplete outputs before restart; completed runs are reused only with complete.json and valid episode coverage. Do not delete or overwrite partial experiment records blindly. The queue intentionally stops on failure or pairing mismatch. All source hashes and snapshots are in manifest.json/source/. No changes to RL failure defaults.
