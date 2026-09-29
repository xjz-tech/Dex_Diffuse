# Ordinary 1B versus ordinary 10B on 10,000 paired initial states

Four simulator seeds (42, 8, 19, 25), 2,500 environments per seed. Both checkpoints run the same post-reset simulator state and actual randomized actor parameters for each pair. The comparison uses native `eval/xjz_test.sh` failure semantics, demonstrations 000–149, ordinary DDIM with 4 denoising steps, 2 executed actions per inference, first episode only, and a 12,000-step (400-second) cap.

This experiment uses the real ordinary sampler for both checkpoints. It does not use a guided sampler with scale zero.

