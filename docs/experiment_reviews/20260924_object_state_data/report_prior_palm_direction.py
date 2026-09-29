from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
P=Path(__file__).resolve().parent;O=P/'reference_turn_baseline_20260926/episode2_prior_palm_direction_20260928';data=json.loads((O/'RESULTS.json').read_text());rows=[]
for seed in [44,45,46]:
 traces=[];degrees=[]
 for name in ['original','toward_palm_20']:
  f=O/name/f'prior_seed{seed}';t=json.loads((f/'trace.json').read_text());r=json.loads((f/'direction_result.json').read_text());assert r['first_separation'] is None or r['first_separation']['trace_index']>69
  axis=np.asarray(r['rotation_axis_world']);rot=Rotation.from_quat(np.asarray([x['object_pose'][3:] for x in t[59:70]]));degrees.append(float(np.degrees(((rot[1:]*rot[:-1].inv()).as_rotvec()@axis).sum())));traces.append(t)
 a,b=traces;qa=np.asarray([x['command'] for x in a[60:70]])-a[59]['command'];qb=np.asarray([x['command'] for x in b[60:70]])-b[59]['command'];cos=float((qa*qb).sum()/np.linalg.norm(qa)/np.linalg.norm(qb));rows.append(dict(seed=seed,original_rotation_deg=degrees[0],toward_palm_rotation_deg=degrees[1],first10_cmd_displacement_cosine=cos))
(O/'common_pre_separation_direction.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# Episode2 自主prior：物体位置与转动方向','', '先固定手型与灯泡朝向，只移动灯泡。真正移到掌心中心附近的两组在第一物理步发生大位移（7.57cm与52.68cm），均拒绝，不进入prior对照。更小偏移的三个静置候选中，选择沿wrist坐标[-1,0,-2]cm平移的可保持组；更远[-1.5,0,-3]cm静置掉落。该选择发生在任何prior rollout之前。**这不是完整掌心包握，只能检验向掌心的位置扰动。**','', '向掌心平移长度2.24cm；该组导入后第一物理步移动2.25cm、转11.75°，因此不声称导入位置就是策略起点。60静置后末30步位移范围0.019mm、转角范围0.012°，至少3个接近接触link，悬空23cm；通过静置数值验收。两组策略开始时q的RMSE为0.02328rad，初始张量仅object字段不同，native demo索引及其他字段相同。','', '固定44g、摩擦1.1、DDIM4、exec2、scale0、自主动作不插值。配对seed44/45/46，各自主运行180控制步（6秒），随后60步固定末目标保持。沿用原生协议，独立分离指标不改变终止。原位置seed44与前次完整rollout逐值复现。','', '## 共同未分离时段：前10控制步（0.333秒）','', '正方向定义为：原位置动作起点的灯泡+Y长轴向世界竖直转动的轴 cross(u0,z)，在世界坐标固定。数值为每步SO(3)旋转增量在该轴上的有符号投影累加，不是欧拉角或纯长轴倾角变化。使用同一轴比较两组；两组全部seed前10步均未出现持续几何/接触分离。','', '| seed | 原位置旋转投影 | 向掌心旋转投影 | 22关节目标变化方向余弦 |','|---|---:|---:|---:|']
for row in rows:lines.append(f"| {row['seed']} | {row['original_rotation_deg']:+.2f}° | {row['toward_palm_rotation_deg']:+.2f}° | {row['first10_cmd_displacement_cosine']:.3f} |")
lines+=['','目标变化方向余弦：前10个cmd分别减去各自静置末目标，展平后计算余弦；1为相同方向，0为正交，-1为相反。它衡量关节空间动作形态，不能等同物体旋转方向。','', '| seed | 原位置首次持续分离控制步 | 向掌心首次持续分离控制步 | 稳定翻转（原/移） |','|---|---:|---:|---|']
for seed in [44,45,46]:
 rr=[next(x for x in data['results'] if x['condition']==name and x['seed']==seed) for name in ['original','toward_palm_20']];loss=[str(x['first_separation']['control_step']) if x['first_separation'] else '>180（动作6秒内未分离）' for x in rr];lines.append(f'| {seed} | {loss[0]} | {loss[1]} | 未通过 / 未通过 |')
lines+=['','三个配对seed的早期方向符号一致，但跨seed有正有负。支持在本次小范围物体位移下，同一噪声种子的动作倾向得以保留；不支持prior无论状态都固定朝同一个方向转，也不证明已经知道物体位姿。三个seed不是足够的总体统计证据，尤其seed45小角度易受扰动影响。完整掌心包握尚未测试，需要另外的有效手型/物体构型。','', 'RESULTS.json中的first15/first30与旧pairs仅为原始诊断字段，seed44移位组这些时窗包含分离后动作；**方向结论仅用common_pre_separation_direction.json中的前10步**。视频按相同物理时间对齐，先显示60步静置，再6秒自主动作，最后2秒保持；三个seed顺序各10秒。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
