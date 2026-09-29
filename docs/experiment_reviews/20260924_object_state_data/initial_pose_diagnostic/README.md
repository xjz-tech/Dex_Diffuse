# case467 真机—仿真初态诊断

**2026-09-25 再更正：前述 10.06/15.64 cm“第一物理步位移”主要是仿真状态设置顺序的伪迹，不能解释成纯粹的碰撞弹开。** 旧实验与第一次无扰动重测都在运行中的 GPU PhysX 场景内设置灯泡质量后、尚未推进物理的同一时段写入物体位姿。质量属性更新使该位姿写入虽然在张量快照中可见，却没有成为下一物理步的实际起点。先在设置质量与摩擦后推进并同步一帧，再写位姿，才得到有效导入。旧 case467 还主动加入 `[+2,+2,-2]` cm、局部 X 轴 15° 的扰动，视频从静置 2 秒后才开始；直接回放与 prior 引导组只构成**同一错误初始化下**的内部对比，不能作为真实抓握迁移成功的证据。

[真机第 90 帧、旧代码写入的张量位姿、静置末态](case467_source_import_settled.png)并排显示旧实验记录的状态。[按列 / 按行 / 扰动写入 / 静置末态](case467_rotation_parser_comparison.png)把不同 6D 旋转解释置于同一原生 wrist、同一虚拟相机下。旧图中“导入瞬间”仅表示写入张量的候选姿态，不能证明 PhysX 已应用。仿真图由原生手 URDF 网格、灯泡网格及存档的关节/根位姿离线重建；不是真实物理帧。源图与仿真相机外参不同。复现脚本是[几何图生成](../make_initial_pose_comparison.py)和[碰撞探针](../probe_initial_collision.py)，数值见[位姿指标](metrics.json)及[探针结果](collision_probe.json)。

## 碰撞几何问题

使用原生手 URDF 的 collision 网格顶点与封闭的 `bulb1_col.obj` 做离线包含测试。旧代码写入张量的候选姿态中有 10,920 个手网格顶点落入灯泡碰撞网格，最深约 **3.24 cm**；静置结束后最深约 2.3 mm。该候选姿态有几何问题，但**不能**用它直接解释旧实验的第一步大位移，因为下面的对照证明物理场景未真正从候选位姿开始。顶点测试也不是 PhysX 接触深度的直接读数。

## 6D 旋转约定和外参

这份数据的 `state.npy` 前 9 维是 TCP 位置与 6D 旋转，`obj_state.npy` 是物体 4×4 位姿；但本地代码有两种互相冲突的解析：灯泡[训练数据读取器](/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/diffusion_policy/dataset/bulb_image_dataset.py)把两组三维向量作为旋转矩阵的前两**列**，而 [TacMP 真机控制代码](/home/carus/Program/TacMP/third_party/diffusion_policy/direct_robot_env.py)及这次实验的 `prepare.py` 把它们作为前两**行**。episode53 第 90 帧两种解释使 TCP 朝向相差 **90.63°**，物体相对 wrist 的位置相差 **19.99 cm**。当前 episode53 的 `state.npy`、`obj_state.npy` 与 TacMP `data/bulb_tac_80/episode_53` 哈希逐字节一致；TacMP 的真机 TCP 9D 转换函数明确写入矩阵前两行。这强烈支持按行解析，但目录拷贝哈希不能单独证明当时采集程序的版本，仍需源采集记录来最终核定。

几何检查也没有给出单一答案：按行解释的源位姿在第 90 帧有约 **3.23 cm** 的手/灯泡互穿；按列解释没有检测到互穿，但最近手/灯泡碰撞网格**顶点间**距约 **2.01 cm**，第 120/165 帧间距增至 4.75/8.79 cm。按行解释时，手指在世界系主要向下，与真机前视图方向较接近；按列解释时手指主要朝侧方。因此不能简单把旧解析改成按列就宣布问题解决。`obj_state` 与 TCP 的共同坐标系、TCP→手根外参、真实灯泡与仿真资产网格的对应关系都需要独立核实。

原始 episode 目录没有这些标定元数据。原生仿真手与现有真机回放手的 `base_joint` 固定安装参数相同，灯泡资产文件哈希相同，见[资产核验](../reference/verification.json)；这些也不足以验证源 TCP 与仿真手根重合，或动捕灯泡坐标原点与网格原点重合。源图中的灯泡有灰色外覆层，而仿真碰撞网格不含该层，实际表面尺寸仍需测量。

## 首步跳变的定位与修复

按行解析 episode51 第105帧与 episode53 第90帧；初始手关节与源数据逐值相同，写入张量的物体相对 wrist 位姿误差低于 1e−6，见[导入校验](../static_unperturbed/validation.json)。旧设置顺序下，第一控制步出现 10.06/15.64 cm 跳变。但把灯泡临时放到**世界系上方 3 m**（远离任何手接触），再不发手指动作、只直接推进一次物理，仍跳回手附近约 **3.08/3.05 m**；既没有手接触，也不可能是重力导致。更换为全量根状态 setter、只设置物体根状态、跳过额外观测刷新，结果相同。

单因素测试锁定触发项：仅设置手摩擦或仅设置物体摩擦时，远离手的灯泡单次物理更新只下落约 **2.04 mm**；仅在运行中的 GPU PhysX 场景内调用 `set_actor_rigid_body_properties` 改灯泡质量到 170 g 后，位姿写入会在下一物理步被旧场景状态覆盖，即使 setter 返回 `True`。在修改质量/摩擦后**先推进并同步一帧，再写入手和物体的初态**，远离手的同样测试只下落 2.04 mm。代码和逐次直接物理结果见[静置测试脚本](../static_unperturbed.py)与[单因素审计](root_state_setter_audit.json)；这是本环境中的实验现象，尚未进一步定位 Isaac Gym 内部机制。先前 10.06/15.64 cm 和原视频的 7.54 cm 不能再被解释为实际导入位姿的碰撞解算结果。

修正设置顺序后，仍按原生 wrist、170 g、手/灯泡滑动摩擦 2.2、原生 failure 参数保持源首帧手指目标 60 步（2 秒）：episode51/53 第一控制步相对位移为 **0.59/2.41 cm**，旋转 **4.72°/21.04°**；第60步累计位移 **1.46/2.78 cm**，旋转 **11.34°/22.40°**，两段都**未触发原生 failure**。见[修正后摘要](../static_unperturbed_flushprops/summary.json)、[逐步轨迹](../static_unperturbed_flushprops/trace.json)和[episode51](../static_unperturbed_flushprops/episode51_source_vs_static.mp4)、[episode53](../static_unperturbed_flushprops/episode53_source_vs_static.mp4)真机起始帧与物理视频。旧目录 `static_unperturbed/` 保留作为错误设置顺序的证据，**不得**作为有效抓握静置结果使用。

离线网格检查仍显示无扰动源位姿有互穿：episode51 最大约 1.47 cm，episode53 最大约 3.23 cm；它们是顶点包含测试，非 PhysX 深度。修正设置顺序后 episode53 的 2.41 cm/21° 首步变化仍说明抓握几何不精确。沿 wrist 局部 X 轴试移 −4 cm 后，episode53 第一控制步平移降至 0.21 cm、60步末 0.44 cm，但旋转仍有 12.58°/16.26°，且 episode51 更差；[试探结果](../static_exploratory_x_-0.040m_z_+0.000m_worldz_+0.000m_flushprops/summary.json)只用于定位，不能当作标定值。尺寸、TCP→手根与动捕物体原点仍需独立核实。

所需的固定变换和进入策略对比前的验收门槛见[标定下一步](../static_unperturbed/calibration_next_step.md)。当前两段没有通过预设的首步/末步姿态稳定门槛，因此尚未以修正后初态重跑原始动作与 10B prior。
