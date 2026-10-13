# 已作废的首次静置运行

本目录 `summary.json` 中的 10.06/15.64 cm 首步位移，以及本目录视频，记录的是**错误的 GPU PhysX 属性/状态设置顺序**。`initial_state.npz` 仅证明位姿写入了观测张量；它没有成为下一物理步的实际起点。把物体临时放在世界上方 3 m，仍在一步内跳回手附近约 3 m，直接证明这不是手—物体碰撞造成的位移。

有效设置顺序为：修改质量与摩擦 → 推进并同步一帧 → 写入手与物体初态 → 开始静置。修正后数据和视频在 [`../static_unperturbed_flushprops/`](../static_unperturbed_flushprops/)，完整分析见 [`../initial_pose_diagnostic/README.md`](../initial_pose_diagnostic/README.md)。
