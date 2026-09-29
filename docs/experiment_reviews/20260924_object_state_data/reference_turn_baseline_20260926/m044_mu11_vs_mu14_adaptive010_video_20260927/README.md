# 摩擦1.1与1.4录像对比

four_episodes_mu11_vs_mu14_raw_guide_real.mp4，按76、34、54、2顺序播放。上排原速reference，下排>0.1插值guide；左列摩擦1.1，中列摩擦1.4，右上为同episode原始front。下排guide固定10B EMA、DDIM4/scale25、guide2/exec1。全部44g、原尺寸。

四条采用各自相同的导入手关节/物体位姿。相同摩擦下guide录像与先前无录像运行的全部command、q、object_pose、vertical_error_deg、native_failure、hand_body_pose逐值相同，导入快照所有字段相同。跨摩擦初始快照仅hand_friction和object_friction变化。

按原始reference进度对齐，原速与真机在guide插入中点时重复前帧；不是相同物理时间，也不保证真机曝光与状态精确同步。包含零物理步精确FK重建、60步静置、完整动作、末目标保持2秒。原速reference为实际运行录像，guide也为实际重新运行录像。只显示仿真正面，不显示桌面宽景。

Episode54的摩擦1.4原速reference基线未通过，故guide没有运行；相应面板明确留空说明，不以其他轨迹替代。约50.67秒，上排可以看到摩擦1.4原速reference在原始进度54达到持续几何分离判据。标记为此前几何/接触判据，不是native failure。

完整视频99.83秒/2995帧/30fps/H.264。verification.json记录逐段映射、录像一致性与全帧解码核验。分段起点：76在0秒，34在17.50秒，54在43.17秒，2在70.57秒。未写Downloads。
