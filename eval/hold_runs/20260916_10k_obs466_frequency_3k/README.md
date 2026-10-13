# Frequency experiment: 10k guide + obs_4-66

User confirmed that “0.1B” means the same `/home/carus/data_usb/obs_4-66.ckpt` previously called 1B. Guide: `runs/sim_hand_10k_seed42/checkpoints/latest.ckpt`. Request: add 45, 60, 90 Hz to existing 30-Hz results.

All arms: 3000 first episodes, 400 simulated seconds, seed8 / guide seed100008, DDIM4/4, fresh noise, serial guide2 / execute2, scale25, original normalizers, existing TensorRT FP16 / fused updates / microbatch256. WAIT=1 freezes physics during model requests. No added real-hand 0.03-rad step clamp. Runs are sequential and wait for unrelated GPU compute jobs to finish. This does not command hardware.

## Physics and time

Old 30-Hz baseline used 60-Hz outer physics with 2 substeps. 45 and 90 Hz are not integer decimations of that clock. All new arms therefore use 180-Hz outer physics with unchanged 2 substeps, decimations 6 / 4 / 3 / 2. A new 30-Hz bridge control is required; the old 30-Hz result is contextual and is not the sole comparator. Caps: 12000 / 18000 / 24000 / 36000 policy steps.

Failure accumulation and stable-goal dwell counts use integer 180-Hz ticks; thresholds retain the old 30-Hz duration. Random external forces update every 6 physics ticks (=30 Hz). Isaac Gym applies force tensors for only the immediate simulation timestep (local API docs); the original force was active during the first of two 60-Hz physics steps. The new fixed-clock implementation applies it during the first three of six 180-Hz physics steps, preserving force amplitude, pulse duration and duty cycle across all arms. The original force decay and random-event probability per 30-Hz update are retained. The original dt was rounded 0.0166667; new clock uses exact 1/180. Fixed-clock force scheduling is independent of policy frequency. Noise samples and subsequent resets may diverge; identical random realizations are not promised.

History remains 4 observations and execution remains 2 actions, so both physical-time horizons change with control frequency as intended for the requested deployment-rate experiment. Threshold crossings are observed at the selected control frequency. These results do not establish feasible real-hand throughput or eliminate real inference delay.

## Validation and outputs

Independent copied code; no shared evaluator or real defaults edited. Short 4-environment / 2-second checks at each frequency precede full evaluation. Checks enforce 3000 unique first-episode records for full runs, exact equal saved initial observations/demo/frame/root-state arrays, correct simulated duration and physics-step counts, WAIT=1 zero holds, and existing model numerical gates. Legacy Isaac Gym teardown exit139 is accepted only with all required complete outputs. No automatic retry or overwrite.

`pipeline.py` runs the entire sequence, with `state.json`, per-arm console/episode/timing/schedule/optimization logs, and incremental `comparison.json` / `comparison.md`. Main results should be the new matched-physics 30/45/60/90-Hz table. Earlier 30-Hz reference: `../20260915_10k_1b_wait_compare_3k/wait1`, 1388/3000 survivors at 400 s (46.27%). Preliminary `smoke45` was deliberately stopped before collecting episodes while fixing force scheduling; it is excluded.
