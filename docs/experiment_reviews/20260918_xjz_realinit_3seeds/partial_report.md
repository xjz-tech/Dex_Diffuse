# xjz_test原生协议：三个seed实录对照

DDIM4/4，执行2步，guide scale25/引导2步；相同真机初态和原生随机化，约30Hz，最多400秒。只比较普通1B与1B+10k guide。时间为仿真时间、原生环境首轮结束时间；failure包含相对当前目标的位置误差超过15cm、累计跟踪失败或数值异常。

| seed | 普通1B | 1B+10k guide |
|---:|---:|---:|
| 58 | 0.63s (failure) | 运行中 |
| 63 | 236.13s (failure) | 0.30s (failure) |
| 65 | 0.83s (failure) | 0.83s (failure) |

## 实录视频

每个视频均来自实际运行中的摄像头，逐控制步采集，未重新推理或离线重建。

- [seed=58 ordinary_1b](seed58_ordinary_1b/live.mp4)
- [seed=63 ordinary_1b](seed63_ordinary_1b/live.mp4)
- [seed=63 guide10k](seed63_guide10k/live.mp4)
- [seed=65 ordinary_1b](seed65_ordinary_1b/live.mp4)
- [seed=65 guide10k](seed65_guide10k/live.mp4)

三个seed是诊断案例，不足以估计总体成功率。它们与旧5cm初始化偏移诊断的判据、物理设置和随机化不同，不能把跨协议的时长差异归因于guide。每对初始q、qd、腕部、物体、目标、质量和GPU随机数状态逐元素核对一致。
