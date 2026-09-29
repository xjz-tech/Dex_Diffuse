# 原生 xjz_test：真机初态，三个 seed 的 1B / 10k guide 对照

用户指定统一使用原 `eval/xjz_test.sh` 评估口径。本次只比较普通1B和1B+10k guide，不运行guided scale=0控制组。

- prior与guide都是DDIM4；每次执行2步；guide scale25、引导前2步；fresh prior噪声按seed配对，guide另用seed+100000。
- seed58、63、65，各方法每seed一轮。这三个seed按旧实验诊断案例选取，不是随机抽样，不能据三个样本估计总体成功率。
- 保留对应实测手型和假定物体摆放，示范目标设置为该摆放对应的原始demo/frame。之后目标更新完全使用原生任务逻辑，示范范围000–149。
- 使用 `sim_eval._make_task_config` 和 `_reset_overrides_from_args` 构建原协议配置，直接调用原生 `env.step()`、reward、failure、goal transition；没有自定义掉落停止规则。
- 物体误差相对于当前示范目标在手腕坐标系中的位置：5cm误差/10cm指尖误差累计异常步；普通目标容忍10000×abs(skipSteps)，跨轨迹目标20000步；15cm位置误差或数值异常立即失败。
- 原生物理配置（约30Hz控制）、物体尺寸/质量/摩擦随机化及随机外力均保留。仅初始q、腕部/物体姿态和初始目标demo/frame按旧实测手型实验设置。每对保存初态、质量和GPU随机数状态以核对一致性。
- 同步推理，推理等待期间不推进物理；最长12000控制步，约400秒；首轮失败/成功后停止，不重置继续计数。
- 摄像头在每次真实 `env.step()` 后采集图像，30fps。`live_raw.mp4` 是原始录制，`live.mp4` 是H.264转码；不是离线状态回放。

初始化改变和原生协议恢复意味着本次绝对时长不能与旧5cm诊断实验直接归因比较。原生failure仍是环境失败代理，不是独立接触式掉落检测。

运行入口：`code/launch.sh SEED ARM real`，ARM=0代表普通1B、ARM=2代表1B+10k guide。模型实现保留上次已验证的生产采样公式，环境任务源码与9月15日归档哈希一致。`source_manifest.json`保存本次实际调用源文件的哈希。
