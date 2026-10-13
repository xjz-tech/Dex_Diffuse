import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;s=json.loads((p/'segments.json').read_text());m=json.loads((p/'initial_scan.json').read_text());v=json.loads((p/'validation.json').read_text());a=np.load(p/'angles.npz')
positive=[r for r in s['rows'] if r['events']];events=[e for r in positive for e in r['events']]
body='''# bulb2 全部150条：横向持握→手内转正的占比

2026-09-12。根据用户对084、090回放的解释，专门区分“灯泡长轴从横向转回常见手内旋转方向”和普通抬指、换指。

## 结论

- 全量读取000–149，共89,927个原始H5帧，按原始时间戳合计3,002.59秒（50.04分钟）。全部帧进行了长轴角度扫描。
- 主标准检出43/150条（28.67%）包含完整明显转正过程，共58段。
- 从首次持续横向姿态开始，到首次持续摆正姿态开始，共6,289帧，占6.99%；按真实时间戳累计210.41秒，占7.01%。可概括为：约三成示范有这种过程，相关阶段约占全部原始数据7%。
- 这7%包含横向持握、等待、反复微调和最后转正，不能称为“连续转正运动占7%”。单段中位时长3.00秒，范围0.60–9.73秒。
- 横向阈值50–70度、摆正阈值25–35度组合的结果为41–50条（27.33%–33.33%），阶段帧占比6.10%–8.39%。这是规则敏感性范围，不是统计置信区间。
- 主标准另有9条出现片尾未完成的横向阶段，合计1,341帧；其中109、126、144整条没有完整转正事件，其他6条此前已有完整事件。这些尾段没有计入上述7%。完整与未完整阶段合计8.48%，不能把这个数当完整成功转正占比。

## 口径与限制

1. 对同一bulb2网格做PCA，提取灯泡长轴。物体局部长轴约为[-0.00347, 0.99999, -0.00332]；使用有方向的轴，灯头与灯座不会被当成相同朝向。
2. 原始H5的物体姿态和手部点已在手腕相对坐标中；以000–059的长轴方向中位数作为常见旋转姿态参考，归一化后为[0.22787, 0.93496, -0.27188]。这个参考代表常见持握方向，不是世界竖直方向。
3. 逐帧计算灯泡长轴与参考方向的夹角，做9帧中位数平滑。绕灯泡自身长轴的旋转不改变该指标，因此不会把持续拧转的圈数当成横向调整。
4. 主标准：偏离至少60度、连续至少9帧（约0.3秒）作为明显横向；此后回到30度以内、连续至少15帧（约0.5秒）作为摆正。计时从第一次满足横向条件的区间起点，到后续摆正区间起点，确认用的0.5秒保持不额外计入。
5. 没有稳定摆正就再次横向的区间合并；片尾未见摆正的区间单列。门槛以前的向外倾转部分没有计入；仅普通换指、轻微倾斜不计入。
6. 核看了58个候选事件各4帧（232个关键帧），以及13个边界/片尾候选与090各4帧（56个关键帧），并检查全150条角度时间图。不是逐帧人工观看150段完整视频。
7. “摆正/可旋转”是几何姿态代理标签，没有用动力学、接触力或后续持续旋转成功验证。该规则可能漏掉小幅但对控制很重要的调整，也没有覆盖位置移动为主的重新持握。
8. 更换为全150条已对齐帧的中位参考方向（相差4.90度），得到42条、58段、6.51%。改变平滑与保持时间，结果约41–43条、6.53%–7.03%；主体结论稳定。

## 084和090重新核对

- 084：明确两段，原始时间0.00–1.77秒、8.40–11.07秒，合计4.43秒，约占本条22.2%。60Hz插值viewer约Frame 0–106、504–664。前文强调的13.63–14.43秒是另一段抬指/调整，并不是这两段最明显的横向转正。
- 090：全条平滑长轴偏离最大38.38度；没有达到本次50–70度的明显横向门槛。可作为轻微倾斜/换指的对照，不列为084级别的大幅转正。不能因为两条一起打开就把它们默认归为同一类。
- 所有主标准阳性都在061–148，000–059没有完整明显事件；在060–149这90条中为43/90（47.8%）。这是当前编号分布观察，不代表官方任务标签或随机抽样分布。

## 对prior和DP的意义

这批参考示范确实包含“横向持握→转正”的技能；但完整相关阶段只有约7%的帧，不能用“约三成示范含有”代替训练时的帧占比。按帧均匀采样时，这类阶段可能被更长的常规旋转片段稀释；这只是值得验证的假设，不能据此认定当前DP失败的原因。

本统计针对150条参考H5，不是RL实际采集后供DP训练的transition分布。RL的目标选择、随机起点、正反向推进、跨示范跳转和成功率都会改变分布，因此不能声称DP训练集里也正好有7%。

建议下一步先用下面的58段索引核对RL采集数据：实际遇到多少对应状态、完成多少次转正、是否衔接后续旋转；随后给“横向起点→转正→继续旋转”建立独立评测，再比较原采样与增加该阶段采样的prior。应按示范分离训练/测试，避免同一段相邻帧落入两侧。

## 全量清单

时间为原始30Hz名义秒；区间右端为首次稳定摆正的帧。原始frame乘2可近似映射到60Hz插值viewer；网页播放墙钟时间可能更慢。

| 编号 | 完整段数 | 完整阶段时间/秒 | 原始帧区间[start,end) | 本条阶段占比 | 片尾未完成 |
|---|---:|---|---|---:|---|
'''
for r in s['rows']:
 i=r['index'];ev=r['events'];n=m['rows'][i]['frames'];frames=sum(e['end']-e['start'] for e in ev)
 times='；'.join(f"{e['start']/30:.2f}–{e['end']/30:.2f}" for e in ev) or '—'
 ranges='；'.join(f"[{e['start']},{e['end']})" for e in ev) or '—'
 body+=f"| {i:03d} | {len(ev)} | {times} | {ranges} | {frames/n*100:.2f}% | {'有' if r['incomplete'] else '—'} |\n"
body+='\n## 可复核文件\n\n- [全150条角度时间图](all150_angles.png)\n- [简要图](summary.png)\n- [逐段标签与阈值敏感性](segments.json)\n- [输入文件SHA256与帧数](sources.json)\n- [参考方向/平滑/时间戳核验](validation.json)\n'
for k in range(1,9):body+=f'- [完整事件关键帧，第{k}页](events_{k}.png)\n'
for k in range(1,3):body+=f'- [边界与未完成案例，第{k}页](borderline_{k}.png)\n'
body+='\n复现顺序：使用含numpy/scipy/h5py/trimesh/matplotlib的环境依次执行scan.py、segment.py、render_review.py、validate.py、write_report.py。仅CPU单线程离线读取；不运行RL、DP或GPU仿真。\n'
(p/'review.md').write_text(body)
fig,axs=plt.subplots(3,1,figsize=(11,8),gridspec_kw={'height_ratios':[1,2,2]},layout='constrained')
ax=axs[0];ax.barh([1,0],[43/150*100,6289/89927*100],color=['#336caa','#de8c32']);ax.set_yticks([1,0],['Demos with complete event','Frames in complete episode']);ax.set_xlim(0,100);ax.set_xlabel('Percent of all 150 reference demonstrations / all frames');ax.text(29.8,1,'43 / 150 = 28.7%',va='center');ax.text(8.2,0,'6,289 / 89,927 = 7.0%',va='center');ax.spines[['top','right']].set_visible(False)
for ax,i in zip(axs[1:],[84,90]):
 y=a[f'{i:03d}'];ax.plot(np.arange(len(y))/30,y,color='#336caa',lw=1.5);ax.axhline(60,c='#b64237',ls='--',label='Sideways threshold: 60 deg');ax.axhline(30,c='#438853',ls='--',label='Aligned threshold: 30 deg')
 for e in s['rows'][i]['events']:ax.axvspan(e['start']/30,e['end']/30,color='#de8c32',alpha=.3)
 ax.set_xlim(0,20);ax.set_ylim(0,125);ax.set_ylabel('Axis deviation (degrees)');ax.set_xlabel('Original recording time (nominal seconds)');ax.set_title(f'Demo {i:03d}: '+('two clear sideways-to-aligned episodes' if i==84 else 'smaller adjustments; below sideways threshold'));ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
axs[1].legend(loc='upper right',fontsize=8)
fig.suptitle('Bulb2: sideways-to-spin-ready geometric audit\nEpisode duration includes sideways holding and adjustment; no physical success label.',fontsize=14);fig.savefig(p/'summary.png',dpi=150);plt.close(fig)
print(p/'review.md')
