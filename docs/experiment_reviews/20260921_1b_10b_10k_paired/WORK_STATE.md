# Work state

- Goal: compare ordinary 1B and ordinary 10B on 10,000 paired native randomized initial states.
- Design: seeds 42, 8, 19, 25; 2,500 environments each; DDIM4; exec2; native xjz failure protocol; first episode; 400-second cap.
- Pairing evidence: each arm archives post-reset state, CUDA simulator RNG, cached disturbance parameters, and actual Isaac Gym object/hand physical properties to NPZ for exact comparison.
- Current state: complete; all eight batches valid, exact pairing verified, analysis and report generated.

