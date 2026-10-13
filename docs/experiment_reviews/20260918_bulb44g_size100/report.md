# 44克、原始尺寸灯泡视频实验

质量44g，scale=1，对应模型横向包围盒7.4663cm（非理想球体直径）；所有物理质量从引擎读回验证。原生xjz判据；prior/guide DDIM4、执行2步，guide scale25引导前2步，约30Hz，最多400秒。普通1B对照10k guide，无scale0。

默认初始化：前述demo079/082/094的历史快速失败案例，继承q、腕部、物体位姿、目标帧以及手部质量/摩擦/Kp/Kd、物体表面属性、外力概率。仅物体质量和尺度指定为44g/1，重新计算物体惯量，并将外力质量缓存同步44g。使用新的单环境随机轨迹，不能直接与原3000环境旧scale0时长作因果比较。

| demo | seed | 方法 | 结束时间/s | 原因 | 摩擦 |
|---:|---:|---|---:|---|---:|
| 79 | 25 | ordinary_1b | 4.70 | failure | 1.704 |
| 79 | 25 | guide10k | ≥400.00 | timeout | 1.704 |
| 82 | 19 | ordinary_1b | 11.70 | failure | 1.312 |
| 82 | 19 | guide10k | ≥400.00 | timeout | 1.312 |
| 94 | 42 | ordinary_1b | 0.93 | failure | 1.774 |
| 94 | 42 | guide10k | 0.87 | failure | 1.774 |
| 131 | 50 | ordinary_1b | 84.30 | failure | 2.572 |
| 131 | 50 | guide10k | ≥400.00 | timeout | 2.572 |

## 实际运行视频

- [25 ordinary_1b](seed25_ordinary_1b/live.mp4)
- [25 guide10k](seed25_guide10k/live.mp4)
- [19 ordinary_1b](seed19_ordinary_1b/live.mp4)
- [19 guide10k](seed19_guide10k/live.mp4)
- [42 ordinary_1b](seed42_ordinary_1b/live.mp4)
- [42 guide10k](seed42_guide10k/live.mp4)
- [50 ordinary_1b](seed50_ordinary_1b/live.mp4)
- [50 guide10k](seed50_guide10k/live.mp4)
- [seed25 并排视频](paired_seed25/comparison.mp4)
- [seed19 并排视频](paired_seed19/comparison.mp4)
- [seed42 并排视频](paired_seed42/comparison.mp4)
- [seed50 并排视频](paired_seed50/comparison.mp4)

短于10秒的并排视频标注0.25倍慢放，其余为正常速度；较短一侧结束后停在已标注的末帧。原生failure为失败代理，未独立自动标注接触丢失。

## 新增 seed 50 与核对说明

seed 50 使用之前真机初始手型存档的 env32（bulb_nohold_1，demo131/frame469），保留本次 seed50 原生物理随机化，物体质量设为44g、scale=1。引擎读回物体摩擦系数2.572，滚动/扭转摩擦均为0。

全部4组配对初始状态（包括随机数状态）与实际物理参数逐项一致。写入物理参数后先做1步不计分的引擎同步，再完整恢复初始状态；8次运行首步手腕旋转检查均通过。计时从恢复后开始。排除了初始化调试期间的无效试跑。全部8个单独录像及4个并排录像已核对帧数、30fps及首/中/末帧解码。

400秒为观察上限，表示没有触发原生失败判定；不能由此断言始终准确跟踪目标。demo082 guide 在结束时 failure_progress=11779，仍低于固定容忍步数20000。当前样本量不足以推断总体成功率，也不能将与历史3000环境的差异全部归因于减重。
