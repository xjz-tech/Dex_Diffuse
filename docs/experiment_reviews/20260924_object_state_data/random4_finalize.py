from pathlib import Path
import json,sys,subprocess,time,shutil
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
P=Path(__file__).resolve().parent;R=P/'random4_ep53_standard_20260926';D=Path('/home/carus/Downloads')
EPS=[50,76,70,40];METHODS=['direct','guide2','guide1'];rows=[]
for ep in EPS:
 f=R/f'episode_{ep:02d}';row=dict(episode=ep)
 for m in METHODS:
  d=json.load(open(f/m/'retention.json'));loss=d['first_separation'];s=json.load(open(f/m/'summary.json'));t=json.load(open(f/m/'trace.json'));sen={}
  for gap in [.003,.005,.008,.01]:
   flags=[(r['mesh_vertex_gap_m']>gap and r['object_contact_force_norm_N']<.05) or r['mesh_vertex_gap_m']>.02 for r in d['frames']]
   idx=next(i for i in range(len(flags)-2) if all(flags[i:i+3]));r=t[idx];sen[str(gap)]=1+r['index']/(1+s['reference_interpolation'])
  row[m]=dict(reference_action_number=loss['reference_action_number'],control_step=loss['control_step'],source_action_frame=loss['source_action_frame'],first_native_failure=d['first_native_failure'],sensitivity_reference_action_number=sen,video_review='manual triptych reviewed at marker-4, marker, marker+8 control steps; visible separation and subsequent falling',control_hz=30)
 rows.append(row)
win={m:sum(r[m]['reference_action_number']>r['direct']['reference_action_number'] for r in rows) for m in ['guide2','guide1']}
report=dict(episodes=rows,wins_of_4=win,step_convention='1-based original action number: guided j zero-based expanded index maps to 1+j/2; half means inserted midpoint',selection=json.load(open(R/'selection_final.json')),pair_audit=json.load(open(R/'paired_audit.json')),caution='comparison is raw direct vs guidance+one midpoint at30Hz; effect cannot be attributed to prior alone; fixed wrist without socket; all full tails run even after native failure')
(R/'RESULTS.json').write_text(json.dumps(report,indent=2))
lines=['# 4个随机合格横抓起点：原始reference与10B guidance保持能力\n','随机种子20260926。先按源数据几何选帧，再在冻结的随机episode顺序中只检查静置；取前4个通过者：50、76、70、40。没有根据控制结果挑选。每episode使用自身完整手/物体初态，禁止近桌fallback。\n','**结果：guide2/exec2在4/4条上比原始动作更晚脱手；guide2/exec1为3/4条。** 这是本次4条、固定seed下的结果，不是整个数据集成功率。全部12次完整reference加60步末尾保持均执行完毕。\n','数值为**脱手标记时对应的原始reference动作编号，1开始，越大表示跟随reference走得越远**。小数.5表示相邻两个原始动作之间的插值目标。脱手时刻是可复核的操作定义，不能精确到连续物理接触瞬间。\n','| episode | 源起点 | 原始动作direct | guide2/exec2 | guide2/exec1 |','|---:|---:|---:|---:|---:|']
sel={r['episode']:r for r in report['selection']['reviewed']}
for r in rows:lines.append(f'| {r["episode"]} | {sel[r["episode"]]["start"]} | {r["direct"]["reference_action_number"]:g} | {r["guide2"]["reference_action_number"]:g} | {r["guide1"]["reference_action_number"]:g} |')
lines+=['\n例如ep50：direct第53个原始动作附近脱手，exec2到第128个附近、exec1到第131个附近。ep70的exec2到177和178之间，direct在第30个附近。ep76的exec1较差，保留原样。\n','## 物理保持分析与原生failure分开\n','只作离线分析，不改环境控制/目标/失败逻辑：手全部collision网格顶点与灯泡collision顶点最近距离超过5mm，同时物体合接触力小于0.05N；或最近顶点距离超过2cm。连续3个控制步满足时，以第1步标记分离。距离用AABB排除加速，超过3cm封顶；这是网格顶点距离而非PhysX接触深度。净接触力也不是手物体pair专属，因此逐一检查了12组 marker前4步、marker、后8步录像，均能看到分离及后续掉落。\n','阈值3/5/8/10mm敏感性结果见RESULTS.json；四条exec2相对direct、exec1的3胜1负排序未受这些阈值影响。native failure另存，不能直接称脱手。\n','## 起点与公平性核对\n','源手状态与动作逐值核对；同episode的27项初态完全相同，60步静置位姿、关节、指令及手接触力完全相同。物体合接触力有小于1e-5N的浮点归约差异。新增记录代码回归episode53：27项初态及旧60步trace字段全部逐值一致。\n','| episode | 末1秒位置变化mm | 末1秒旋转变化° | 静置累计位置漂移cm | 累计旋转° | 末1秒最少近物接触link |','|---:|---:|---:|---:|---:|---:|']
for ep in EPS:
 m=sel[ep]['static'];lines.append(f'| {ep} | {m["last30_translation_range_m"]*1000:.3f} | {m["last30_rotation_range_deg"]:.3f} | {m["final_drift_m"]*100:.2f} | {m["final_rotation_deg"]:.2f} | {m["minimum_near_contact_links"]} |')
lines+=['\n前9个随机候选中，49和75姿态/漂移不合格，65、36、71没有保持住；50、76、70、40通过。77随并行静置探针额外检查，也通过，但排在第5个合格者，未进入本轮控制比较。筛选及排除记录全保留。\n','## 固定配置和解释边界\n','170g，摩擦2.2，环境seed42、prior噪声44，10B EMA、DDIM4、scale50、guide2、插入1个线性中点，分别exec1/2，每次reference前进实际执行数；全部末端padding与模型调用索引已核对。direct不插值。\n','原生eval/xjz_test.sh阈值/目标流/随机化保留，demo000–149；未使用新分析指标停止或重置。仿真固定原生wrist，没有插座。这比较的是**guidance+插值放慢整套方案**与原始直接动作，不能单独归因prior，也不是完整真机放置成功率。TCP/动捕/网格外参未独立标定，因此只能确认稳定且近似源姿态的仿真起点，不能宣称真实姿态完全标定准确。\n','## 视频\n','Downloads/Object_state_data_random4_ep53_standard_comparison.mp4：4条顺序放在同一视频。每条左上原始动作仿真、右上真机、左下exec2、右下exec1；显示导入暂停1秒、静置2秒、共同抓稳起点暂停1秒、完整尾段、末尾保持2秒。主画面正面，宽景小窗显示桌面；仿真XYZ轴红绿蓝。按原始reference进度对齐，direct放慢2倍，真实状态显示相应目标之后的记录帧。\n','经验文档：../EPISODE53_REPLAY_LESSONS.md；项目AGENTS.md已链接并记录关键规则。']
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
for name in ['RESULTS.md','RESULTS.json']:
 shutil.copy2(R/name,D/('Object_state_data_random4_'+name))
# Intro summary card, encoded identically to the episode videos.
font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',34);small=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',25)
card=Image.new('RGB',(1280,1152),(17,24,33));dr=ImageDraw.Draw(card)
for i,line in enumerate(['4个随机合格横抓起点 · 完整对比','按修正后的 episode53 标准：170g / 摩擦2.2','10B DDIM4 · scale50 · guide2 · 插值1次','数值：脱手时对应的原始reference动作编号（1开始）']):dr.text((50,65+i*60),line,font=font if i==0 else small,fill='white')
for j,line in enumerate(['episode       原始直接        guide2/exec2        guide2/exec1']+[f'{r["episode"]:02d}                 {r["direct"]["reference_action_number"]:g}                    {r["guide2"]["reference_action_number"]:g}                       {r["guide1"]["reference_action_number"]:g}' for r in rows]):dr.text((70,360+j*72),line,font=font if j else small,fill=(184,239,211))
for i,line in enumerate(['exec2：4/4保持更久；exec1：3/4保持更久','顺序：episode50 → 76 → 70 → 40','左上原始仿真 / 右上真机 / 左下exec2 / 右下exec1','每条包含导入、静置、完整动作、末尾2秒','这是guidance+插值方案对比；仿真wrist固定。']):dr.text((50,800+i*55),line,font=small,fill=(203,218,238))
card.save(R/'overview.png')
with imageio.get_writer(str(R/'intro.mp4'),fps=30,codec='libx264',quality=7,macro_block_size=16,ffmpeg_params=['-preset','fast','-threads','2','-movflags','+faststart']) as writer:
 for _ in range(120):writer.append_data(np.asarray(card))
paths=[R/'intro.mp4']+[R/f'episode_{ep:02d}/episode_{ep:02d}_comparison.mp4' for ep in EPS]
for p in paths:
 deadline=time.monotonic()+1200
 while not p.with_name('video_manifest.json').exists() and p.name!='intro.mp4':
  if time.monotonic()>deadline:raise TimeoutError(str(p))
  time.sleep(2)
(R/'concat.txt').write_text(''.join("file '"+str(p)+"'\n" for p in paths))
ff='/home/carus/miniforge3/envs/dp/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
out=D/'Object_state_data_random4_ep53_standard_comparison.mp4'
subprocess.run([ff,'-y','-v','error','-f','concat','-safe','0','-i',str(R/'concat.txt'),'-c','copy','-movflags','+faststart',str(out)],check=True)
cap=cv2.VideoCapture(str(out));count=0
while True:
 ok,fr=cap.read()
 if not ok:break
 assert fr.shape==(1152,1280,3);count+=1
expected=120+sum(json.load(open(p.with_name('video_manifest.json')))['frames'] for p in paths[1:]);assert count==expected,(count,expected)
verify=dict(file=str(out),fully_decoded_frames=count,fps=cap.get(cv2.CAP_PROP_FPS),duration_s=count/30,episode_order=EPS)
cap.release();(R/'final_video_verification.json').write_text(json.dumps(verify,indent=2));print(json.dumps(verify),flush=True)
