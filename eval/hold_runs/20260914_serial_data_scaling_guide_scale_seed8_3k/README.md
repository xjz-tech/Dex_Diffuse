# Queued serial experiments — final approved matrix

Service: serial-guidance-data-scaling-seed8.service (initial supervisor PID2050013).
Frozen worktree: .worktrees/serial-data-scaling-run-20260914, commit fd3f327.
Waits for the existing parallel candidates' state to become complete AND an idle
GPU. Its waiting supervisor uses no GPU. No active experiment code was modified.

After the remaining late_x0_w10 and late_x0_w30 finish:

1. Nested100k complete-episode subset -> train200 epochs -> serial guide2/scale25 eval.
2. Nested1M complete-episode subset -> train200 epochs -> serial guide2/scale25 eval.
3. Original10k guide -> guide5 at scale25,50,75, then guide9 at scale50,75,100.
4. Original10k guide2/scale25 is reused (46.50%,332.6s), not rerun or retrained.

Eight new simulations in total after two training jobs (two data-size evaluations
and six guidance-window/scale evaluations). All simulations: seed8,3000 first
episodes,fresh noise,predict9,execute2,DDIM4,FP32,unchanged historical hold criteria.
Training seed42, batch512, architecture/optimizer/EMA unchanged from original10k.
Each dataset supplies its own normalizer, not the shared-1B normalizer experiment.
The base checkpoint remains /home/carus/data_usb/obs_4-66.ckpt.

For each guidance window, rank completed results by completion rate then median
hold time; exact ties prefer lower scale. Guide2 has only scale25, not a search.
comparison.json/md update after each evaluation. Until all seven 10k combinations
are available, best-scale entries are explicitly provisional. Single seed only.

All stages record command, PID and logs in state.json and per-stage log files.
Source/config/input changes block launches. Previous complete runs are validated
and skipped; partial eval directories are archived under interrupted/. Verified
zero-sample simulator startup segfaults retry at most twice per stage. Other errors
stop for review and best-effort desktop notification. Training can resume from
its checkpoint on manual service restart. The old Codex watchdog is not claimed
to supervise this new queue. No auto-reboot or GPU-reset behavior is configured.

Inspect: systemctl --user status serial-guidance-data-scaling-seed8.service
Stop: systemctl --user stop serial-guidance-data-scaling-seed8.service
Resume unchanged: systemctl --user start serial-guidance-data-scaling-seed8.service
Do not modify frozen source/config while running; create a new reviewed snapshot
for changes. The unit is persistent but not enabled for automatic login startup.
