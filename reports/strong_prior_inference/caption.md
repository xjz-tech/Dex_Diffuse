**Figure. Dual-guided strong-prior inference for real-world manipulation.** A visual diffusion policy produces a 16-step arm–hand proposal. At each hand replanning call, a frozen strong prior performs eight DDIM iterations, guided jointly by a nine-step window of the cached DP hand proposal and a freshly predicted two-step weak-policy reference. The two MSE losses are evaluated in the strong prior’s normalized action space and weighted independently (100 and 25). Guidance differentiates with respect to the noisy trajectory while detaching the predicted noise. The first two generated hand actions are combined with the corresponding DP arm actions and executed with state feedback. Four controller calls execute eight actions before the visual DP replans. Dashed arrows denote feedback or denoising recurrence.

图中动作索引相对于当前可用的未来动作窗口；实现中完整 diffusion horizon 的可用动作从索引 3 开始。k = 0, 1, 2, 3 表示一轮 DP 内的 controller 调用。参数对应脚本当前默认值。初始化、模型加载与耗时日志为简洁起见未画出；30 Hz 为执行阶段的目标频率。

重新生成：
```bash
dot -Tsvg reports/strong_prior_inference/pipeline.dot -o reports/strong_prior_inference/pipeline.svg
dot -Tpdf reports/strong_prior_inference/pipeline.dot -o reports/strong_prior_inference/pipeline.pdf
dot -Tpng -Gdpi=160 reports/strong_prior_inference/pipeline.dot -o reports/strong_prior_inference/pipeline.png
```
