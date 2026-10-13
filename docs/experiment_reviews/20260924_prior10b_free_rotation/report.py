from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
s=json.loads((OUT/'summary.json').read_text());protocol=json.loads((OUT/'protocol.json').read_text())
assert s['complete'],'Final report requires all three seeds complete'
for seed in [42,123,2026]:
 p=OUT/f'seed{seed}';c=json.loads((p/'recorder_closed.json').read_text());assert c['steps']==12000 and c['max_applied_external_force']==0
f=s['frame_metrics']['all'];speed=f['speed_thresholds']['5'];first=s['first_30s']['30'];all_ep=s['all_episodes']['30']
lines=['# 10B Prior 无外力多环境旋转方向统计','',f"完成3个seed（42、123、2026），每个32个环境、12,000控制步。共96条环境流、{f['frames']:,}条物体运动记录，累计约{f['frames']/30/3600:.2f}小时仿真时间。每条环境流连续运行约400秒，原生失败/结束后重置继续；没有筛选初态或剔除失败运行。",'',
'## 主要结果','',f"全部运动帧（沿灯泡长轴速度绝对值>5°/s）中，左转占 **{100*speed['left_fraction_of_moving']:.2f}%**，右转占 **{100*(1-speed['left_fraction_of_moving']):.2f}%**。另有 {speed['slow']:,} 帧为慢速，未强行分方向。",'',
f"96个环境各只取首轮，观察前30秒或更早原生终止时的净角度：**左转>30°：{first['left']}个；右转>30°：{first['right']}个；净转动≤30°：{first['small']}个**。方向明确的首轮中左转比例为 {100*first['left_fraction_of_directional']:.2f}%。早期终止保留，不筛选存活30秒的环境。",'',
'| seed | 已运行控制步 | 运动帧左转比例 | 首轮前30秒净左转>30° | 净右转>30° | 小净转动 |','|---|---:|---:|---:|---:|---:|']
for row in s['seeds']:
 q=row['first_30s']['30'];v=row['all_frames']['speed_thresholds']['5'];lines.append(f"| {row['seed']} | {row['steps']:,} | {100*v['left_fraction_of_moving']:.2f}% | {q['left']} | {q['right']} | {q['small']} |")
survivors=s['first_episode_duration_strata']['400'];sv=survivors['30']
lines+=['',f"96个首轮中，{survivors['episodes']}个实际运行到400秒观察上限；其中净左转>30° {sv['left']}个、净右转>30° {sv['right']}个、小净转动{sv['small']}个。这是按运行时长条件化的补充结果，不替代上述包含早期失败的首轮统计。", '']
lines+=['',f"全程含重置后共记录{s['all_episodes']['episodes']}轮，其中原生failure {s['total_native_failures']}轮，观察截止仍运行{s['censored_episodes']}轮。按各轮实际记录区间的净角度分类：左>{30}° {all_ep['left']}轮，右>{30}° {all_ep['right']}轮，小净转动{all_ep['small']}轮。该口径包含观察截止的未结束轮次，且频繁失败环境贡献更多轮次，因此与96个首轮结果分开报告。",'',
f"累计左向轴转角 {f['left_angle_deg']:,.1f}°，累计右向轴转角 {f['right_angle_deg']:,.1f}°；左/右累计角度比 {f['left_angle_deg']/f['right_angle_deg']:.3f}。累计量保留回摆，不能视为独立完成的整圈次数。",'',
'![方向统计](direction_summary.png)','', '![分时段转向](temporal_bias.png)', '',
'## 协议与方向算法','',
'使用 /home/carus/data_usb/10B_obs_4-66.ckpt 的EMA权重（step91600，epoch50）；DDIM4，每次执行2步，纯model_server，无guide或右转reference。推理期间物理暂停，沿用原生串行IPC。', '',
'按本次用户“不受干扰”要求，将原生 randomForceScale 从1改为0，逐控制步断言实际外力张量为零。保留原生随机初态、示范000–149、质量/摩擦/尺度/手腕随机化。未筛选“容易转”的质量、摩擦或初始抓姿。', '',
'原生评估阈值显式为物体位置0.05m、指尖0.1m、物体旋转180°、立即失效位置0.15m、FailureToleranceScale=10000、fixedToleranceSteps=20000、trajStepsLimit=12000、resetOnReachGoal=false、跨轨迹目标概率0.3。原生目标更新与错误参考系不变。未把“偏离初始位置5cm”作为失败。', '',
'右转为从圆顶向螺纹看顺时针，即负灯泡局部Y轴旋转。帧方向由世界角速度转至灯泡自身坐标系后取-Y分量；净角度由灯泡相对实际手腕的相邻四元数变化计算，累计-Y轴旋转向量分量。原生reset_done后更新基准，不跨重置累计姿态跳变。每步保存位姿、角速度、episode编号、原生done/failure及转角增量。', '',
'## 解释与限制','',
'运动帧比例与净方向比例回答不同问题：向左的持续漂移可夹杂很多短暂右摆。当前结果统计轴向转动，不等同于成功完成稳定拧入或保持接触。原生failure只是失败代理；没有逐帧视觉确认所有96个环境的物理掉落。指尖末端link记录的净接触力全零，不能用它做接触筛选，未据此改变终止规则。', '',
'用户中断过首次尝试，部分数据完整保存在 interrupted_attempts/turn_interrupted_20260924；本表来自按同样预先指定seed重新开始并完成的三组运行，没有拼接中断轨迹或按结果选择重跑。', '',
'同一环境的连续帧及重置后轮次不是独立试验。报告按seed分列，并给出各环境仅首轮的固定30秒口径，避免把大量帧当作大量独立成功样本。3个seed不能覆盖全部可能初态。观察截止仍运行的轮次标为截尾，未记作成功。', '',
'## 实录与复现','',
'每个seed事先固定记录env0–3，未按转向挑选视频；10fps对应每3控制步取一张实际仿真相机帧。视频中时间与净角度按各轮重置。','']
for seed in [42,123,2026]:lines.append(f'- [seed{seed}四环境原始实录](seed{seed}/actual_env0-3.mp4)')
lines+=['','[12环境前30秒同步视频](first30_comparison.mp4)','', '配置 protocol.json；checkpoint哈希 checkpoint.json；冻结源码 source/；仪表化差异 instrumentation.diff；原始逐步记录 seed*/trajectory_*.npz；原生结束日志 seed*/episodes.jsonl；转角逐轮表 rotation_episodes.json；统计 summary.json。运行 run.sh，完成后执行 analyze.py、report.py。', '']
(OUT/'report.md').write_text('\n'.join(lines))
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
labels=['Training data']+[f'Seed {r["seed"]}' for r in s['seeds']]+['Pooled sim']
left=[50.90]+[100*r['all_frames']['speed_thresholds']['5']['left_fraction_of_moving'] for r in s['seeds']]+[100*speed['left_fraction_of_moving']]
y=np.arange(len(left));axes[0].barh(y,left,color='#b84a42',label='Left');axes[0].barh(y,100-np.array(left),left=left,color='#397ca8',label='Right')
for i,l in enumerate(left):axes[0].text(l/2,i,f'{l:.1f}%',ha='center',va='center',color='white')
axes[0].set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='Share of moving frames (%)',title='Axial speed >5 deg/s');axes[0].invert_yaxis();axes[0].legend(loc='lower center',bbox_to_anchor=(.5,-.35),ncol=2)
rs=[r['first_30s']['30'] for r in s['seeds']];l=np.array([r['left'] for r in rs]);r=np.array([q['right'] for q in rs]);z=np.array([q['small'] for q in rs]);x=np.arange(3)
axes[1].bar(x,l,color='#b84a42',label='Net left >30 deg');axes[1].bar(x,r,bottom=l,color='#397ca8',label='Net right >30 deg');axes[1].bar(x,z,bottom=l+r,color='#aaaaaa',label='Small net angle')
for i in x:
 for count,bottom in [(l[i],0),(r[i],l[i]),(z[i],l[i]+r[i])]:
  if count:axes[1].text(i,bottom+count/2,str(count),ha='center',va='center',color='white')
axes[1].set(xticks=x,xticklabels=[f'Seed {q["seed"]}' for q in s['seeds']],ylabel='First episodes / 32 per seed',ylim=(0,34),title='Net angle: first 30 s or earlier native end');axes[1].legend(loc='lower center',bbox_to_anchor=(.5,-.35),ncol=1,fontsize=8)
fig.suptitle('Pure 10B Prior | DDIM4 / exec2 | no applied random force');fig.tight_layout();fig.savefig(OUT/'direction_summary.png',dpi=160,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for row in s['seeds']:
    bins=row['time_bins'];x=[(b['start_s']+b['end_s'])/2 for b in bins];y=[100*b['speed_thresholds']['5']['left_fraction_of_moving'] for b in bins]
    ax.plot(x,y,'o-',label=f"Seed {row['seed']}")
ax.axhline(50,color='gray',ls='--',lw=1);ax.set(xlabel='Simulation time (s), 40 s bins',ylabel='Left share of moving frames (%)',title='Direction bias over time; includes native resets');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'temporal_bias.png',dpi=150);plt.close(fig)
# Build a fixed, preselected first-30-second comparison from actual cameras.
import cv2
caps=[cv2.VideoCapture(str(OUT/f'seed{seed}/actual_env0-3.mp4')) for seed in [42,123,2026]]
assert all(cap.isOpened() for cap in caps)
writer=cv2.VideoWriter(str(OUT/'first30_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1920,480))
assert writer.isOpened()
for frame in range(300):
    tiles=[]
    for seed,cap in zip([42,123,2026],caps):
        ok,img=cap.read();assert ok,(seed,frame)
        img=cv2.resize(img,(640,480))
        cv2.putText(img,f'SEED {seed}',(250,465),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
        tiles.append(img)
    canvas=np.concatenate(tiles,axis=1);writer.write(canvas)
    if frame==99:cv2.imwrite(str(OUT/'first30_preview.jpg'),canvas)
writer.release()
for cap in caps:cap.release()
import subprocess,imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y','-i',str(OUT/'first30_comparison.mp4'),'-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'first30_comparison_h264.mp4')],check=True)
(OUT/'first30_comparison_h264.mp4').replace(OUT/'first30_comparison.mp4')
print(OUT/'report.md')
