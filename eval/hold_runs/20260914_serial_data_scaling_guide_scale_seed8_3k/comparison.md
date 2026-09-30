# Serial data-size and guidance-window comparison

FP32 serial predict9 execute2 DDIM4 seed8 3000 first episodes fresh noise; own dataset normalizers

| Guide model / window / scale | Completion | Median hold (s) |
|---|---:|---:|
| 10k / guide2 / scale25 (reused historical) | 46.50% | 332.6 |
| 100k / guide2 / scale25 | 20.63% | 114.0 |
| 1m / guide2 / scale25 | 19.70% | 100.0 |
| 10k_g5_s25 / guide5 / scale25 | 32.90% | 201.4 |
| 10k_g5_s50 / guide5 / scale50 | 36.83% | 229.9 |
| 10k_g5_s75 / guide5 / scale75 | 24.80% | 120.5 |
| 10k_g9_s50 / guide9 / scale50 | 46.57% | 325.0 |
| 10k_g9_s75 / guide9 / scale75 | 38.10% | 209.0 |
| 10k_g9_s100 / guide9 / scale100 | 25.80% | 118.4 |

Data scaling uses guide2/scale25. Original-10k sweep: guide2=25 (reused), guide5=25/50/75, guide9=50/75/100; best by completion, then median hold. Rankings are provisional until all combinations complete.

Single-seed screening; data-size experiments fix epochs, not total optimizer updates.

## Best scale per guidance length so far

- guide2: scale25, completion 46.50%, median hold 332.6s
- guide5: scale50, completion 36.83%, median hold 229.9s
- guide9: scale50, completion 46.57%, median hold 325.0s
