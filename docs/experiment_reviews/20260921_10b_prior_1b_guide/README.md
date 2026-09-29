# 1B guide → 10B prior

2026-09-21。用户要求“用1B guide 10B”。方向固定为10B主策略、1B作guide。10B checkpoint `/home/carus/data_usb/10B_obs_4-66.ckpt`，1B checkpoint `/home/carus/data_usb/obs_4-66.ckpt`；第三组旧10k guide仅用于同一10B主策略的对照，不替代用户要求的1B guide组。

三组均运行同样48个case：普通10B、10B+1B guide、10B+旧10k guide。4初始化×2质量×2摩擦×3推理噪声seed，400s上限。评估代码直接拷贝20260921_random_10k_30k的原生xjz条件版本，只替换prior checkpoint路径；prior/guide DDIM4，exec2，guide scale25及可执行动作前2步，普通10B用真正普通DDIM而非scale0。每组录制预先固定的case3/16/29实际运行视频。记录完整初态、物理参数、模型hash及逐case保持时间。

10B作为prior的历史12000组结果使用过不同复现条件/基线入口，不能将绝对时长直接与当前48组对比。先验证两个检查点的66维观测、22维动作、时间序列约定一致，再跑评估。原生failure仍是评估代理。
