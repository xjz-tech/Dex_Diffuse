# ~1k-transition guide training and serial scale25 evaluation

Queued after all four 100k/1M epoch20/100 scale25 evaluations complete. The
epoch queue had received a systemd stop at 13:55 on September 15; it was resumed
at the user's request, preserving and archiving its incomplete prior evaluation.

Training: same source, whole-episode selection rule and seed42 as original10k.
Preflight selects 993 transitions in 6 episodes, not 1000 environments. This is
not guaranteed nested within the original10k selection. 10% episode validation,
batch512, own normalizer, EMA, 200 epochs; expected 400 optimizer updates.
Important intentional difference: warmup56 instead of500, approximately matching
10k's 500/3600 warmup fraction. Keeping500 would leave all 400 updates in warmup.
The warmup change is a comparison caveat, not an unreported default override.
No additional epoch/scale sweep is scheduled for 1k in this initial screen.

Evaluation: trained 1k EMA checkpoint guides the unchanged base; serial guide2,
scale25, predict9 execute2, DDIM4/4, seed8/guide seed100008, 3000 first episodes,
fresh noise, same historical DP hold thresholds and 12000-step cap, FP32.
Service: serial-small-guide-seed8-20260915.service. State/PIDs: state.json.
Checkpoints: runs/sim_hand_1k_ownnorm_seed42_equal_epochs/checkpoints.
Results: comparison.json/md after completion, including reused 10k/100k/1M controls.
All GPU stages wait for idle GPU; no concurrent evaluations or retraining of
existing models. Known zero-sample Isaac Gym startup segfaults have up to two
verified retries; other errors halt for review. Frozen source/config/input
fingerprints are recorded in manifest.json.
