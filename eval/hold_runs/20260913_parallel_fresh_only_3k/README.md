# Active replacement: parallel experiments only

LATEST USER OVERRIDE: seed8 ONLY. The already running fixed_c0.95_seed8 is
uninterrupted. Subsequent seed19/25 calls return immediately without GPU work;
four fixed coefficients + one dynamic = five actual simulations. A seed8
dispatcher was installed by renaming the original launcher to
eval/run_parallel_guidance_eval.seed8_target.sh and atomically moving a new
entrypoint into place. The running shell retains its original open inode:
no in-place edits were made to the script it is reading. Underlying inference
code and evaluation parameters are unchanged. Old multi-seed text below is
superseded. Reports use the existing seed8 serial reference, not a new control.

No new serial simulation: explicitly launch with --skip-training --skip-serial.
Reuse completed user reference in 20260912_fresh_noise_3000_seed42_8_19_25.
Frozen execution worktree: .worktrees/parallel-fresh-only-run-20260913.

Order: numerical check; fixed base coefficients .95/.90/.70/.60, each with seeds
8/19/25 (12 runs); dynamic scale25, seeds 8/19/25 (3 runs); summary.
Every run: FIXED_NOISE=0, 3000 environments, full nine-action guidance,
two executed actions, four DDIM steps, shared-normalizer retrained 10k guide,
explicit historical hold thresholds. No original training is repeated.

Service: parallel-guidance-fresh-only.service. Status: state.env; log: pipeline.log.
The stopped serial attempt in 20260913_parallel_fresh_fullhorizon_3k is excluded.
