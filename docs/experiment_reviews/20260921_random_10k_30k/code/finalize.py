from pathlib import Path
import json,time,hashlib
R=Path(__file__).resolve().parents[1]
summary=json.loads((R/'evaluation_summary.json').read_text())['summary']
verification=json.loads((R/'verification.json').read_text())
assert verification['prior_unchanged']
assert all(all(v.values()) for v in verification['pairing'].values())
videos=json.loads((R/'video_verification.json').read_text());assert len(videos)==3 and all(v['verified'] for v in videos)
p=R/'report.md';text=p.read_text()
conclusion='''## 本次结论

**没有复现旧10k那样的大幅提升。新10k接近普通1B；新30k恢复了部分长时收益，但快速失败更多。**

- 新10k平均85.75秒，相对1B仅增加5.70秒；中位数从32.20秒降至3.13秒。成对案例中12组增加超过20秒、14组减少超过20秒，不能据此声称稳定改善。
- 新30k平均107.38秒，相对1B增加27.33秒，低于旧10k的87.54秒增益；中位数仅2.07秒。其13组增加超过20秒、14组减少超过20秒，收益集中在少数长时案例。
- 两个新模型达到20秒的比例都是35.4%，低于普通1B/旧10k的54.2%。新30k达到400秒有9组，比1B的3组、新10k的4组多，但仍少于旧10k的16组。
- 真机初始手型的均值依次为1B54.17秒、旧10k241.56秒、新10k61.06秒、新30k106.35秒：旧guide最突出的这部分优势，也只恢复了一部分。

这轮说明，不能把旧10k的结果当作重新随机选一份10k/30k就能保证获得的收益。两份新数据更广的目标demo覆盖和初始化帧覆盖，没有自动带来更好的闭环效果。**数据内容、训练得到的guide以及初始化的相互作用值得继续检验；本次没有证明质量/摩擦构成就是原因。** 原始库没有物理标签，且每份数据只训练一个模型，评估还存在重复运行波动，因果解释应止于这里。

'''
if '## 本次结论' not in text:text=text.replace('## 48个相同配置的结果',conclusion+'## 48个相同配置的结果')
text=text.replace('![保持分布](comparison.png)',f'![保持分布]({R}/comparison.png)')
if '## 四路对比录像' not in text:
 text+='\n## 四路对比录像\n\n真实运行录像，10倍播放，面板时钟显示仿真时间。左上普通1B、右上旧10k、左下新10k、右下新30k；前两路来自上轮相同配置。已结束面板标注并冻结最后一帧。三个录制case预先固定，非按结果挑选。\n\n'
 labels={3:'demo079 / noise8',16:'demo082 / noise19',29:'demo094 / noise25'}
 for v in videos:text+=f"- [{labels[v['case']]}]({v['path']})\n"
 text+='\n三段均为44g、摩擦2.572。特别注意：demo094/noise25中新30k达到400秒，但该初始化的全部12个配置均值只有35.25秒，低于1B的55.28秒；不能把这段单例当作总体胜出。\n\n新模型6段完整原始视频保留在evaluation/对应模型的case03_raw.mp4、case16_raw.mp4、case29_raw.mp4，以上四路对比是加速预览。\n'
 text+='\n## 新模型归档\n\n'
 for row in json.loads((R/'guide_registry.json').read_text()):
  text+=f"- [{row['name']} checkpoint]({row['checkpoint']})：抽样seed={row['sampling_seed']}，训练seed=42，200 epochs，{row['global_step']}次更新；SHA256 `{row['sha256']}`。\n"
 text+=f'\n[数据与训练登记]({R}/guide_registry.json)；[每条轨迹构成]({R}/data_audit.json)；[逐case结果]({R}/evaluation_summary.json)。\n'
p.write_text(text)
(R/'visual_qa.json').write_text(json.dumps({'reviewed_by':'assistant using view_image','reviewed':['comparison.png','videos/case03_frame11999.png','videos/case16_frame01255.png','videos/case29_frame11999.png'],'checks':['chart labels and values readable','four-way video panels correctly labelled','terminal freeze labels visible','simulation time and 10x playback labels visible','both successful holding and failed cases visible'],'passed':True},indent=2))
hashes=json.loads((R/'source_hashes.json').read_text())
for pth in (R/'code').glob('*'):
 if pth.is_file():hashes[str(pth)]=hashlib.sha256(pth.read_bytes()).hexdigest()
(R/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
done={'status':'complete','completed_local_time':time.strftime('%Y-%m-%d %H:%M:%S'),'timezone':'Asia/Shanghai','trained_new_guides':2,'new_evaluation_cases':96,'reused_reference_cases':96,'case_count_per_method':48,'new_actual_recordings':6,'verified_comparison_videos':3,'report':str(R/'report.md'),'summary':summary,'visual_qa_passed':True}
(R/'completion.json').write_text(json.dumps(done,indent=2))
(R/'WORK_STATE.md').write_text('# COMPLETE\n\n'+time.strftime('%Y-%m-%d %H:%M:%S')+' Asia/Shanghai. Both random resampled guides trained and all96 new eval cases complete;96 prior reference cases verified comparable. Report,3 comparison videos, data/checkpoint provenance and visual review complete. No pending runs.\n\nNew10k mean85.75s/median3.13s; new30k107.38s/median2.07s; ordinary1B80.05s/median32.20s; old10k167.59s/median86.15s. Both new guides>=20s17/48 vs26/48 ordinary/old. New30k9/48 cap400, new10k4/48, ordinary3/48,old16/48. No reproduction of old10k magnitude;30k partial long-duration benefit with more fast failures.\n\nSee report.md,completion.json,verification.json,data_audit.json,guide_registry.json,video_verification.json,visual_qa.json. Training/data seeds and unknown source physical labels clearly recorded.\n')
print('COMPLETE',R/'report.md')
