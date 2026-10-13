# 45/60 Hz failure increase: evidence and remaining causal uncertainty

Analysis of completed 30/45/60-Hz arms from `eval/hold_runs/20260916_10k_obs466_frequency_3k`. All have 3000 identical saved initial states, seed8/100008, WAIT=1, guide2/execute2, original checkpoints and normalization, shared 180-Hz outer physics with 2 substeps. No new GPU experiments were started or existing queues changed for this analysis.

## Equal-action-count comparison

A failure at action K counts as not surviving after K completed actions; capped survivors count as alive. All arms have follow-up through 12000 actions.

| Completed actions | 30 Hz | 45 Hz | 60 Hz |
|---|---:|---:|---:|
| 100 | 86.77% | 87.13% | 86.80% |
| 1000 | 84.03% | 82.83% | 79.80% |
| 3000 | 76.93% | 72.73% | 65.23% |
| 6000 | 66.37% | 60.10% | 50.53% |
| 12000 | 50.47% | 41.67% | 33.93% |

At 12000 actions, 45 Hz is 8.80 percentage points below 30 Hz; 60 Hz is 16.53 points below. The extra actions at a fixed 400-second horizon therefore do not fully explain the decline. Action-matched runs have different physical exposure times (400/266.67/200s), so this is a complementary endpoint, not a control of every variable.

Among matched initial environments at 12000 actions: 30 survives/45 fails =709 versus 30 fails/45 survives=445; corresponding counts for 60 Hz are 851 versus355. Of 150 initial demonstration groups, 45 Hz is worse in98, better in34 and tied in18; 60 Hz is worse in128, better in10 and tied in12. These are descriptive single-seed findings, not independent causal trials.

Failure within the first physical second: 344/349/372 out of3000 for30/45/60Hz. Large differences develop later. Failures within two action steps of the last logged goal arrival in the first episode: 0/1486,0/2072,0/2481. This does not exclude all reference-goal effects, but there is no observed immediate post-arrival failure concentration.

## Confirmed changes in temporal semantics

| Quantity | 30 Hz | 45 Hz | 60 Hz |
|---|---:|---:|---:|
| One target duration |33.33ms|22.22ms|16.67ms|
| Four-observation oldest-to-newest span (3 intervals)|100ms|66.67ms|50ms|
| Two-action execution duration|66.67ms|44.44ms|33.33ms|
| Model requests per simulated second|15|22.5|30|

Training sequences use consecutive transitions from the approximately30-Hz collection configuration. `sim_eval.py` shifts history at every action; `FastPair.trajectory` flattens the4x66 normalized history without an explicit dt/control-Hz input. The inference argument named diffusion timestep is a denoising index, not the action interval. Both guide and prior receive the compressed history.

Targets are absolute joint angles, not guaranteed achieved joint states. The same plausible target angle may produce a different transition when applied for less time. Higher target-update frequency does not by itself increase position-drive gains or ensure reaching each target. It is plausible that remaining tracking error, contact timing, and the next policy prediction then differ from training. This is a hypothesis supported by the changed interface, not a measured increase in rollout residual: per-step qpos, command, velocity and contact traces were not saved.

Each new inference draws fresh noise for guide and prior. At60Hz there are twice as many independent replans per physical second as at30Hz, so action continuity across chunk boundaries is another candidate. It cannot by itself explain why equal numbers of actions/requests have different survival; it may interact with dynamics. No measured jitter claim is warranted.

## What the existing failures do and do not establish

WAIT hold_steps=0 for all three arms. The real-hand0.03-rad clamp is absent. Same physics resolution and saved initial states eliminate those obvious differences within this group.

The active latency-tracking failure predicate is the OR of: accumulated tracking failure over its tolerance, extreme-velocity sanity checks, and relative object-to-reference position distance>0.15m. Configured cross-target tolerance20000 original30-Hz steps equals666.67s; non-cross minimum skip40 times10000 gives13333.33s. The400s run cannot accumulate enough for these tolerance branches. Thus, under this configuration, a logged terminal failure must come from the0.15m invalid condition or extreme-velocity checks. Logs do not separate these two branches. Relative reference separation is not necessarily a physical object drop.

Failures are checked once per action.60Hz observes twice as often as30Hz and can detect a transient excursion between30-Hz checkpoints. Therefore the result measures deployment frequency plus associated observation/replanning/termination sampling changes, not an isolated actuator-bandwidth effect. Equal physics timesteps do not remove this detection confound. Subsequent RNG consumption and perturbation realizations also diverge even though time-based disturbance rates match.

## Literature context, not proof about this hand

- [Diffusion Policy, original authors](https://diffusion-policy.cs.columbia.edu/): conditions on an observation history, predicts action sequences, and executes a prefix before replanning. This supports treating physical observation/execution horizons as part of the policy interface.
- [SPACE, section5.3](https://arxiv.org/html/2606.24049v1#S5.SS3): authors report a command-predicting policy degrading when execution frequency increases from its data-collection frequency15Hz to30Hz, attributed to under-reaching. This uses a different robot/action representation and is only analogous evidence.
- [RTR, original authors](https://sjtu-zhao-lab.github.io/RTR/): studies representations and training for high-frequency continuous action chunks. It illustrates why native high-frequency policies and simply accelerating a lower-frequency action sequence are different interventions.

## Discriminating follow-up, not yet run

1. Instrument30/45/60Hz with per-joint raw/sent targets,qpos,dq,residuals,contact magnitudes,object pose relative to palm and reference,chunk-boundary target jumps,and separate terminal flags. Sample physical excursions on a common clock and report both common-clock and native termination criteria.
2. At60Hz keep target timing/replanning unchanged but resample the four observation frames at30-Hz spacing. Recovery would support observation-time mismatch; failure to recover would not exclude output-timing mismatch.
3. Compare30-Hz policy semantics executed via a faster command/servo loop against genuinely consuming new policy actions at60Hz. Preserve original action timestamps and replanning cadence in the first arm. This distinguishes output-interface frequency from accelerating the learned behavior, rather than assuming high-rate control is intrinsically harmful.

Current conclusion: higher action consumption frequency reduces equal-step survival as well as equal-time survival. The strongest code-supported mechanism is temporal mismatch of the learned observation-to-action transition, with tracking/contact and chunk continuity as unmeasured intermediate mechanisms. A unique physical root cause cannot be identified from aggregate failure logs.
