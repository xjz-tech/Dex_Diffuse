# Empty-hand holding comparison

**Goal:** Run three paired real SharpA trials with and without inference hold, document conditions and average jumps, and retain all launcher logs.

**Architecture:** Keep the current launcher and control code. Extend numbered TXT/JSONL rotation without deletion; use existing full command vectors for offline analysis.

**Protocol:** User reports no bulb. Six sequential runs, 600 actions each, chunk=2, 30 Hz, ROTATE reset before each trial. Paired seeds 50/51/52, order off/on, on/off, off/on. TensorRT FP32, fused DDIM, four sampling steps, existing software-limit settings. Exclude initialization, first policy action and exit hold from jump comparisons. Report per-run statistics and equal-weight averages across three trials.

- [x] Change logging regression test to require preservation beyond ten runs, including paired sidecars and original history; observe failure.
- [x] Extend rotation to highest existing numeric archive plus one, document unlimited retention, include optional experiment description in TXT. Run logging regression tests and shell syntax validation.
- [x] Execute six bounded runs with unique output paths and full JSONL recording. Verify 600 acknowledged policy commands, 300 chunks, session completion and condition-specific holds for every run.
- [x] Calculate max absolute joint differences for adjacent policy actions (within/between chunks), actual command transitions and separate hold transitions. Write conditions, per-trial means and aggregate means to TXT plus machine-readable results.
- [x] Verify all artifacts and report comparison with limits of three empty-hand trials.
