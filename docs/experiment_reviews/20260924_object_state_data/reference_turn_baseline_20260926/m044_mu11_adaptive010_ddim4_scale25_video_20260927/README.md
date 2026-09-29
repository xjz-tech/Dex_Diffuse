# Episode54 >0.1插值录像

视频 episode54_adaptive010_raw_guide_real.mp4：27.4秒，30fps，822帧，H.264，全部帧已解码验证。左：原速reference仿真正面；中：>0.1插值、DDIM4/scale25、guide2/exec1；右：原数据集front。按原始reference进度对齐，插入中点时原速reference与真机图重复前帧；不是相同物理时间。真机相机曝光与状态不保证精确同步。

沿用ep54源帧110、44g、摩擦1.1、原尺寸。包含零物理步FK重建、60步静置、完整动作及末尾2秒保持。零步由同URDF的精确FK离线绘制，并验证首物理步的FK与仿真手刚体位置误差<1e-6m；避免使用可能过期的导入渲染。动作阶段为实际仿真录制。录像与无录像实验的27项初态全部逐值一致；全部732个控制步的command、q、object_pose、vertical_error_deg、native_failure、hand_body_pose逐值一致。

guide在首次持续分离之前，只有1个控制步满足竖直误差≤30°：原始进度113，guide实际动作控制步143，角度29.9922°。视频约9.73秒。之后继续倾斜，原始进度154、guide动作步202进入此前相同的持续几何分离判据，视频约11.70秒。最长竖直接触保持1步，未达到30步标准；不是先稳定完成翻转再掉落。此前>0.12版本最长42步，原速reference为61步。不能把进度154解释成保持竖直154步，也不能把该几何指标当native failure。

这说明未通过的具体轨迹形态；尚未独立定位导致轨迹差异的关节/接触机制。中间多出的插值目标同时改变执行时序、观测及prior的后续输出，不能仅由视频认定单一原因。

diagnostic.json保存分离和角度信息；real_front_video_verification.json保存视频映射与解码验证。未写Downloads。
