# Real debug recording implementation plan

**Goal:** Record the seven user-approved categories on the real machine for transfer to simulation: configuration/checkpoint hashes, full observations, predictions and guide references, actual commands, measured state/tracking, timestamps, and sampling noise.

**Architecture:** A portable JSONL sidecar accompanies each TXT run. A bounded background writer serializes CPU snapshots. A hand-client wrapper records existing reads/sends without extra hardware requests. Optional policy capture retains actual sampled tensors without drawing any additional randomness. Rotate TXT and JSONL together, keeping ten runs.

**Constraints:** Preserve current control/observation semantics, user edits, RNG sequences and acceleration modes. No live hardware tests. Use existing checkout because these changes extend the user's uncommitted real-runner work. First state before a command means latest available sample, not an extra synchronous read. Host timestamps do not imply hardware acquisition timestamps. The position API does not expose velocity.

## Tasks

- [x] Add failing offline tests for JSONL snapshots, timestamps, command/state wrapper, and paired ten-run retention. Implement `eval/real/debug_recording.py` and launcher sidecar rotation. Verify with stdlib unittest.
- [x] Add parity tests for ordinary/fused/guided sampling capture. Capture full trajectories, normalized observations, guide reference, initial noise, timesteps and scheduler stochastic noise. Recording must not change outputs or RNG state. Update the real policy implementations and fused sampler only at their optional capture points.
- [x] Integrate recorder into startup initialization and runtime: metadata once per process, model metadata after loading, complete inference input/output events, selected chunk, command and state events including cleanup; always close/drain on exit. Record checkpoint SHA256 and versions, normalizer parameters, scheduler config and exact selected weights. Verify with mocked hardware and toy CPU policies.
- [x] Document field conventions and transfer/replay boundaries. Run focused tests, check shell syntax/diff, obtain a read-only review, and report paths and limitations.

## Verification

Use `/home/frankagvl/anaconda3/envs/dexIL/bin/python` with `PYTHONDONTWRITEBYTECODE=1` for Python tests. Existing suites: `tests/test_real_inference_hold.py`, `tests/test_real_launcher_logging.py`, `tests/test_real_fused_guidance.py`. New tests must verify actual serialized arrays and no extra commands/reads, capture-on/off numerical and RNG parity, and preservation of paired historical files. All launcher tests use stub interpreters and `CHECK_ONLY=1`.

## Completion evidence

35 focused tests passed (recording 5, sampling 5, guidance 4, control 19, launcher 2). Both actual checkpoint CUDA fused-PyTorch CHECK_ONLY runs passed in `/tmp/real-debug-smoke-cn_k0ro9`, producing parseable complete 1x12x22 predictions and actual prior/guide noise. No hardware connected and no real logs rotated by tests. Shell syntax and git diff whitespace checks passed. Read-only review found recorder-drain and missing source-hash concerns; both fixed. No GPU TensorRT execution test was performed.
