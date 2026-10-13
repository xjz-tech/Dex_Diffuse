# 四条episode：>0.1插值与原速reference、真机front对比

视频长度 99.83秒，2995帧，30fps，H.264；全部帧解码验证。

顺序76、34、54、2。左：原速reference仿真正面；中：>0.1 rad插1中点、10B EMA/DDIM4/scale25、guide2/exec1；右：各episode自己的数据集front。质量44g、手与灯泡滑动摩擦1.1、原尺寸。环境seed42，prior固定噪声44。未写Downloads。

同一episode内按原始reference动作进度对齐；插入中点时原速reference和真机重复前帧，非相同物理时间。真机图像曝光与状态并未独立时间标定。每条包括精确零物理步FK重建、60步静置、完整reference尾段、60步末目标保持；只展示仿真正面和真机front，不显示仿真宽景。

所有录制的导入初态与既有无录像guide和原速reference的27项字段逐值一致。全部控制步的command、q、object_pose、vertical_error_deg、native_failure、hand_body_pose与对应无录像guide结果逐值一致；重采样进度表与原始动作数、native协议、guide2/exec1和DDIM4/scale25已核验。

| Episode | 起始秒 | 结束秒 | 原始动作数 | guide实际动作步数 |
|---|---:|---:|---:|---:|
| 76 | 0.00 | 17.50 | 239 | 315 |
| 34 | 17.50 | 43.17 | 441 | 560 |
| 54 | 43.17 | 70.57 | 462 | 612 |
| 2 | 70.57 | 99.83 | 492 | 668 |
