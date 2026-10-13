from pathlib import Path
import json,pickle
import h5py,numpy as np
from scipy.spatial.transform import Rotation as R
OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'summary.json').read_text());ep=json.loads((OUT/'episodes.json').read_text())
assert s['counts']['all']['rows']==s['manifest_transitions']
assert s['pair_coverage']['actual']==s['pair_coverage']['expected'],s['pair_coverage']
# Recompute small reference audit with the native optimized wrist convention.
refs=[]
for i in range(150):
 with h5py.File(f'/home/carus/Program/dex-controller/data/NOKOV-v3/data/bulb2/{i:03d}.h5') as f:obj=R.from_matrix(f['object/6dpose'][:,:3,:3])
 with open(f'/home/carus/Program/dex-controller/data/retargeting/NOKOV-v3/mano2sharpa_rh/bulb2/{i:03d}.pkl','rb') as f:wrist=R.from_rotvec(pickle.load(f)['opt_wrist_rot'][::2])
 assert len(obj)==len(wrist)
 rel=wrist.inv()*obj;a=-np.degrees((rel[:-1].inv()*rel[1:]).as_rotvec()[:,1])
 refs.append(dict(demo=i,frames=len(rel),net_right_deg=float(a.sum()),right_deg=float(np.maximum(a,0).sum()),left_deg=float(np.maximum(-a,0).sum())))
s['references_150']=dict(episodes=150,convention='Native object trajectory relative to optimized wrist, original 30Hz frames; not raw recorded wrist.',net_counts={str(t):{'right':sum(r['net_right_deg']>t for r in refs),'left':sum(r['net_right_deg'] < -t for r in refs),'small':sum(abs(r['net_right_deg'])<=t for r in refs)} for t in [0,10,30,90,180]},right_deg=sum(r['right_deg'] for r in refs),left_deg=sum(r['left_deg'] for r in refs))
s['counts']['all']['episode_net_counts']['180']={'right':sum(r['net_right_deg']>180 for r in ep),'left':sum(r['net_right_deg'] < -180 for r in ep),'small':sum(abs(r['net_right_deg'])<=180 for r in ep)}
s['validation']={'all_manifest_rows_scanned':True,'all_within_episode_adjacent_pairs_covered':True,'positive_right_sign_sanity_test':True}
for angles,expected in [([0,-10,-20],[10,10]),([0,10,20],[-10,-10])]:
 r=R.from_euler('y',angles,degrees=True);assert np.allclose(-np.degrees((r[:-1].inv()*r[1:]).as_rotvec()[:,1]),expected)
(OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n');(OUT/'references.json').write_text(json.dumps(refs,indent=2)+'\n')
lines=['# exp_data 灯泡旋转方向全量审计','',f"扫描全部 {s['shards']} 个 HDF5 shard，共 {s['counts']['all']['rows']:,} 条 transition、{s['counts']['all']['episodes']:,} 条 rollout。",'',
'方向约定：从灯泡圆顶朝螺纹端看，顺时针为右转。模型 +Y 朝圆顶，因此负局部 Y 轴转动为右转，与现有 `eval/analyze_real_reference_sim.py` 一致。', '',
'帧统计使用记录角速度在灯泡自身轴上的分量；绝对轴向速度 ≤5°/s 单列为慢速。角度统计按 episode/step 重组后，累加物体相对实际手腕的相邻旋转向量局部 Y 分量；连接所有跨 shard 的相邻帧，不连接不同 rollout。', '',
'## 实际转动帧数','', '| 范围 | 右转帧 | 左转帧 | 慢速帧 | 运动帧中右/左占比 |','|---|---:|---:|---:|---:|']
for key,label in [('all','exp_data 全量'),('mmap_train','mmap 训练 split'),('guide_10k','sim_hand_10k_seed42 全子集')]:
 c=s['counts'][key];r,l,z=c['speed_counts']['5'];lines.append(f'| {label} | {r:,} | {l:,} | {z:,} | {100*r/(r+l):.2f}% / {100*l/(r+l):.2f}% |')
lines+=['','## 按整条 rollout 净角度分类','', '不是纯单向轨迹标签：同一 rollout 可多次左右换向。净角度右转超过30°归右，左转超过30°归左，其余单列。','','| 范围 | 净右转 >30° | 净左转 >30° | 净转动 ≤30° |','|---|---:|---:|---:|']
for key,label in [('all','exp_data 全量'),('mmap_train','mmap 训练 split'),('guide_10k','sim_hand_10k_seed42 全子集')]:
 c=s['counts'][key]['episode_net_counts']['30'];lines.append(f"| {label} | {c['right']:,} | {c['left']:,} | {c['small']:,} |")
c=s['counts']['all'];r=s['references_150']['net_counts']['30']
lines+=['','## 正放/倒放与参考示范','',f"`traj_direction=+1` 共 {c['forward']:,} 帧，`-1` 共 {c['reverse']:,} 帧；{c['episodes_with_both_playback_directions']:,} 条 rollout 包含两种播放方向。此字段是示范索引推进方向，不等同物体左右转。重置为 +1，达到示范边界后切换；采集配置 reverseTargetProb=0，crossTrajectoryGoalProb=0.3。",'',f"原生 000–149 参考按净角度30°分组：右 {r['right']} 条、左 {r['left']} 条、小净转动 {r['small']} 条。使用原生优化手腕姿态，未把原始动捕 wrist_pose 误当优化手腕。",'',
'## 解释与限制','',
'全量计数接近均衡但略偏左，不能用“右转数据几乎没有”解释推理几乎总向左。全局帧比例不等于同一抓姿下的方向覆盖，更不等于持续旋转能力。66维 Prior 输入为关节位置、上一目标、二者残差，没有显式物体位姿或左右目标；这提供了后续排查方向，但本统计没有确定因果。', '',
'本报告统计训练数据中的全部运动，未按接触或成功筛选，因此包括回摆、换向和失败附近转动，不将这些计数表述为成功旋转次数。慢速阈值1/5/10/30°/s、净角度阈值及原始累计量保存在 summary.json。', '',
'此前 20260923_astra_longrun/report.md 中的两个 Prior 30秒抓握审阅截止净右转为 −681.32°、−864.88°，确有明显左转；这些是特定初态的运行结果，不能直接推成全状态成功率。', '',
'复现：使用 dp Python 执行 audit.py，再执行 finalize.py。未启动新的仿真、修改训练数据或控制器。','']
(OUT/'report.md').write_text('\n'.join(lines))
print(json.dumps(s,indent=2))
