from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import report
h=Path(__file__).resolve().parents[1];v=h/'gait_trials/v3_all_fingers';report.HERE=v
rows=[json.loads(p.read_text()) for p in sorted((v/'runs').glob('*/summary.json')) if not p.parent.name.startswith('screen')]
assert len(rows)==7
(v/'audit.json').write_text(json.dumps([report.audit(r) for r in rows],indent=2))
source=h/'code/all_finger_trial.py';assert source.read_bytes()==(v/'all_finger_trial_frozen.py').read_bytes()
(v/'source_hashes.json').write_text(json.dumps({str(source):hashlib.sha256(source.read_bytes()).hexdigest()},indent=2))
report.TITLES=['Astra direct / all fingers','Prior only','All-finger reference / 25','All-finger reference / 38','All-finger reference / 50']
report.DIRECT_REUSE_NOTE='One matched noise seed in this trial.'
report.render(rows,50,0)
fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
for i,method in enumerate(report.METHODS):
 r=next(r for r in rows if r['environment_seed']==50 and r['method']==method)
 y=report.curves(r)[:r['held_cutoff_step']];t=np.arange(1,len(y)+1)/30
 axes[0].plot(t,y,label=method,color=report.COLORS[i])
 if method!='prior':axes[1].plot(t,y,label=method,color=report.COLORS[i])
for ax,title in zip(axes,['All five arms (full range)','Direct and guided arms (detail)']):
 ax.set(ylabel='Net right rotation (deg)',title=title);ax.axhline(0,color='gray',lw=.5);ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[1].set_xlabel('Executed simulation time (s)');fig.suptitle('Seed50 / noise0: stop at last visible contact or 30s cap');fig.tight_layout();fig.savefig(v/'held_rotation.png',dpi=160);plt.close(fig)
lines=['# 五指同步小幅抬起、左回位、闭合后右转：实测','',
'按用户的新顺序实现并测试：右转24步 → 五指同步抬起8步 → 沿灯泡周向左回位8步 → 闭合16步 → 重复。五指包括拇指，不保留单独的固定支撑指。每步1/30秒，每次最多900步（30秒），提前脱手则结束。', '',
'本轮是动作机制尝试：Astra direct在原先独立筛选出的seed50、19、25上各一次；seed50/模型噪声0增加Prior和scale25、38、50对照，共7条。没有追加scale100。Prior使用已有连续实录的前900步，其余6条新跑；未把这7条称作完整27次批量结果。此前v2单指批次按用户改动停止扩展，已完成数据保留。', '',
'这里的Astra direct是当前Astra编写并依据实时状态更新的解析控制器，直接下发关节目标；不是每个窗口调用语言模型。', '',
'## 实际结果','',
'右转为正。提前脱手角度只计至保守的最后可见接触帧；脱手后的旋转不算有效右转。可见接触不等于稳定力闭合。达到30秒属于观察截止，不视为掉落或成功。', '',
'| seed/噪声 | 方法 | 脱手区间或观察上限/s | 截止净右转/° | 截止最大净右转/° |', '|---|---|---:|---:|---:|']
for r in rows:
 b=r.get('drop_time_bracket_seconds');t=f'{b[0]:.2f}–{b[1]:.2f}' if b else '30（观察截止）';lines.append(f"| {r['environment_seed']}/{r['noise_seed']} | [{r['method']}](runs/{r['label']}/comparison_clip.mp4) | {t} | {r['held_net_right_deg']:.2f} | {r['held_max_net_right_deg']:.2f} |")
lines+=['','本版scale25、38在30秒末仍有接触，净右转分别为−5.45°和+3.05°；scale50在17.00–17.20秒脱手。全部没有形成稳定连续右转。仅为seed50/噪声0的单次配对，不能据此排序普遍性能。', '', '## 视频','', '[五组同步实录：seed50/噪声0](comparison_env50_noise0.mp4)', '', '[Astra direct近景、阶段标注](runs/astra_direct_e50_n0/gait_closeup.mp4)', '', '![抓握审阅截止曲线](held_rotation.png)', '',
'## 动作参数和实现限制','',
'上方解释为世界坐标+Z；左移解释为沿固定灯泡轴向左回转4°。抬起阶段以实测关节姿态为基准，指垫目标向上1mm并径向向外0.5mm；闭合向下1mm并向内0.75mm。IK增量限0.06rad，推转每份16步计划目标为向右4°，执行8步后按各组自己的新状态重算。抬起/回位计划的后8步有意保持阶段端点，执行器没有隐式尾部补齐。', '',
'与v2的单指方案相比，本版推进没有附加PIP/DIP/拇指屈曲下限，闭合改为统一笛卡尔下压和内收，因此不能把v2/v3的差异解释为只有“同时/依次抬指”一个变量改变。', '',
'目标位移不等于实际位移。seed50首个抬起阶段，扣除手腕刚体运动后，五个指垫代理中心的上移分量约0.14–0.80mm；拇指/食指仍有0.96/2.74N净接触力。seed19同期部分指尖几乎不升甚至略向下，五个监测指垫净力均为0。故尚未实现“每个指尖都抬起同样距离并及时重抓”的可靠执行。这些代理中心位移不是指垫与物体的表面间隙。详细量测见[measured_phase_diagnostics.json](measured_phase_diagnostics.json)。', '',
'三个direct在第一轮回位/闭合附近均掉落，没有建立重复右转循环；这些结果只能说明当前具体实现不稳定，不能证明该动作思路本身不可行。', '',
'![Direct首轮指垫净接触力](direct_contact_forces.png)', '', '图中的力只来自五个监测指垫刚体，零值不能独立证明全手与物体都无接触；物理脱手仍按实际图像确认。', '', '## 协议与物理条件','',
'沿用10B EMA模型，Prior DDIM4、数值reference guide9、每次执行2步、16步计划每8步更新。这里没有1B+10k学习式guide网络，不能写成那套实验的guide DDIM4。原生示范000–149；位置误差0.05m、指尖误差0.1m、旋转180°、立即失效位置0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、resetOnReachGoal=false、跨轨迹概率0.3。原生failure和视觉确认脱手分开记录。轨迹时限10000000步未在30秒内触发。', '',
'物理seed沿用正式结果出现前的独立Prior5秒预筛和质量/摩擦比例筛选；不是根据本版结果挑好seed。质量、摩擦、重力、物体自由度和手腕均无人工固定/替换。27个初态字段逐元素匹配原筛选初态。', '',
'| seed | 质量/g | 物体摩擦 |','|---:|---:|---:|']
for seed in [50,19,25]:
 r=next(r for r in rows if r['environment_seed']==seed);lines.append(f"| {seed} | {r['mass_g']:.3f} | {r['object_friction']:.3f} |")
lines+=['', '[冻结协议](protocol.json) · [执行与reference审计](audit.json) · [冻结控制代码](all_finger_trial_frozen.py)', '']
(v/'report.md').write_text('\n'.join(lines))
(v/'results.json').write_text(json.dumps(rows,indent=2));print(v/'report.md')
