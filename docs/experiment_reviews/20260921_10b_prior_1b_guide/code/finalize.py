from pathlib import Path
import json,time,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1]
summary=json.loads((R/'evaluation_summary.json').read_text())['summary'];verify=json.loads((R/'verification.json').read_text());videos=json.loads((R/'video_verification.json').read_text());sampler=json.loads((R/'guided_sampler_verification.json').read_text())
assert sampler['passed'] and all(all(v.values()) for v in verify['case_pairing'].values())
assert len(videos)==3 and all(v['verified'] for v in videos)
b,g,k=[summary[n] for n in ['ordinary_10b','guide_1b','guide_old10k']]
actions={};base=np.load(R/'evaluation/ordinary_10b/first_actions.npz')['raw']
for n in ['ordinary_10b','guide_1b','guide_old10k']:
 a=np.load(R/'evaluation'/n/'first_actions.npz')['raw'];q=np.load(R/'evaluation'/n/'initial_state.npz')['q'];actions[n]={'mean_abs_first_action_minus_ordinary10b_rad':float(np.abs(a-base).mean()),'max_abs_first_action_minus_ordinary10b_rad':float(np.abs(a-base).max()),'mean_abs_first_target_minus_initial_q_rad':float(np.abs(a-q).mean())}
(R/'first_action_diagnostics.json').write_text(json.dumps({'scope':'first commanded action only; no causal inference for long rollouts','values':actions},indent=2))
intro=f'''## 本次结论

在相同48个case、DDIM4/4、exec2、scale25下，**10B+1B guide平均保持{g['mean_capped_s']:.2f}秒，普通10B为{b['mean_capped_s']:.2f}秒，相差{g['mean_paired_gain_vs_ordinary10b_s']:+.2f}秒；旧10k guide同样引导10B时为{k['mean_capped_s']:.2f}秒。** 1B guide的中位保持{g['median_s']:.2f}秒，普通10B为{b['median_s']:.2f}秒；20秒通过数分别是{g['ge20_count']}/48与{b['ge20_count']}/48。

成对案例中，1B guide救回{g['rescued20_count']}组原本不到20秒的案例，却让{g['harmed20_count']}组原本超过20秒的案例提前失败。归档真机手型12组均值：普通10B {b['by_initialization']['real_pose50']['mean_s']:.2f}秒，1B guide {g['by_initialization']['real_pose50']['mean_s']:.2f}秒，旧10k guide {k['by_initialization']['real_pose50']['mean_s']:.2f}秒。

首次动作相对普通10B的平均变化，1B guide为{actions['guide_1b']['mean_abs_first_action_minus_ordinary10b_rad']:.4f}rad，旧10k guide为{actions['guide_old10k']['mean_abs_first_action_minus_ordinary10b_rad']:.4f}rad。首步变化大小不能单独解释闭环保持结果。这里的比较限定在当前scale与四种指定初始化；每个case只运行一轮。

'''
p=R/'report.md';text=p.read_text()
if '## 本次结论' not in text:text=text.replace('## 结果',intro+'## 结果')
if '## 四路对比录像' not in text:
 text+='\n## 四路对比录像\n\n三个录制case预先固定，均为44g、摩擦2.572；10倍播放。左上为上轮1B+旧10k，右上为普通10B，左下为10B+1B guide，右下为10B+旧10k guide。面板时钟为仿真时间，已结束的画面冻结并标注。\n\n'
 label={3:'demo079 / noise8',16:'demo082 / noise19',29:'demo094 / noise25'}
 for row in videos:text+=f"- [{label[row['case']]}]({row['path']})\n"
 text+='\n完整原速录制保留于本实验的evaluation/ordinary_10b、evaluation/guide_1b、evaluation/guide_old10k目录中，每种方法均有case03_raw.mp4、case16_raw.mp4、case29_raw.mp4。单个录像不能替代48组统计。\n'
 text+=f'\n[采样数值核对]({R}/guided_sampler_verification.json)；[首动作差异]({R}/first_action_diagnostics.json)；[逐case结果]({R}/evaluation_summary.json)。\n'
p.write_text(text)
hashes=json.loads((R/'source_hashes.json').read_text())
for f in (R/'code').glob('*'):
 if f.is_file():hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
(R/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
(R/'visual_qa.json').write_text(json.dumps({'reviewed_by':'assistant using view_image','reviewed':['comparison.png','videos/case03 terminal PNG','videos/case16 terminal PNG','videos/case29 terminal PNG'],'checks':['chart values readable','four method panels labelled correctly','failure and terminal frames visible','10x and simulation clock labels visible'],'passed':True},indent=2))
(R/'completion.json').write_text(json.dumps({'status':'complete','completed_local_time':time.strftime('%Y-%m-%d %H:%M:%S'),'timezone':'Asia/Shanghai','new_evaluation_cases':144,'reused_reference_cases':96,'case_count_per_method':48,'recorded_new_raw_videos':9,'comparison_videos':3,'report':str(p),'summary':summary,'visual_qa_passed':True},indent=2))
(R/'WORK_STATE.md').write_text(f"# COMPLETE\n\n{time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai. 10B prior + 1B guide, ordinary10B, and10B prior +old10k guide: all48 cases each complete. Previous1B baseline and1B+old10k48 cases each verified paired. See report.md, evaluation_summary.json, verification.json, guided_sampler_verification.json, video_verification.json, visual_qa.json, completion.json. No pending work.\n")
print('COMPLETE',p)
