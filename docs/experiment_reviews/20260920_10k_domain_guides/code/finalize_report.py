from pathlib import Path
import json, hashlib, ast

r = Path(__file__).resolve().parents[1]
s = json.loads((r / 'evaluation_summary.json').read_text())
t = json.loads((r / 'transfer_analysis.json').read_text())
assert s['complete'] and len(s['all_results']) == 336
assert len({(x['method'], x['case']) for x in s['all_results']}) == 336
assert all(x['seconds'] <= 400.001 and not x['error_flag'] for x in s['all_results'])
models = {x['method']: x for x in s['models']}
labels = {'ordinary_1b': '普通1B', 'old10k': '1B + 旧10k', 'light_low': '1B + 轻/低',
          'light_high': '1B + 轻/高', 'heavy_low': '1B + 重/低', 'heavy_high': '1B + 重/高', 'balanced': '1B + 四域混合'}
registry = json.loads((r / 'guide_registry.json').read_text())
for x in registry:
    assert x['checkpoint_status'] == 'complete'
    assert hashlib.sha256(Path(x['checkpoint_expected']).read_bytes()).hexdigest() == x['checkpoint_sha256']
    assert hashlib.sha256(Path(x['distribution_manifest']).read_bytes()).hexdigest() == x['distribution_manifest_sha256']
    mm = json.loads((r / 'evaluation' / x['name'] / 'model_manifest.json').read_text())
    assert mm['guide_sha256'] == x['checkpoint_sha256']
for name in models:
    mm = json.loads((r / 'evaluation' / name / 'model_manifest.json').read_text())
    assert mm['prior_sha256'] == 'ad0bf60d9fe743161916c55fee36a1b6b13840757baaca2136dcfc3b5b597302'
for x in json.loads((r / 'source_manifest.json').read_text()):
    assert hashlib.sha256(Path(x['source']).read_bytes()).hexdigest() == x['sha256'], x['source']
integrity = json.loads((r / 'dataset_integrity.json').read_text())
for name, entry in integrity.items():
    for rel, expected in entry['files'].items():
        assert hashlib.sha256((Path(entry['root']) / rel).read_bytes()).hexdigest() == expected
for p in (r / 'code').glob('*.py'):
    ast.parse(p.read_text())
(r / 'code_manifest.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((r / 'code').iterdir()) if p.is_file()}, indent=2))

mix = models['balanced']; base = models['ordinary_1b']; old = models['old10k']
lines = ['# 固定1B、五种10k数据分布：完成报告', '',
         '已完成5个新guide的训练，以及7种方法×48个配对案例，共336次闭环评估。1B权重固定。', '',
         '训练：每组40个250步片段，总10k=9k梯度训练+1k验证；seed42，200epoch/3600次更新。采集设定为轻40–80g、重180–300g，物体摩擦低0.5–1.5、高2.5–4.0；混合组四域各25%。实际保留分布和每段标签均已存档。', '',
         '评估：4初态（demo079、082、094、真机pose50）×44/240g×物体摩擦1.312/2.572×策略seed8/19/25。xjz原生判据，DDIM4/4、exec2、scale25引导前2个动作，观察上限400秒。物体尺寸scale1。真机pose50使用同一历史手型，但本轮手部物理随机参数重新固定采样，不能把时长直接与过去单独seed50的结果混在一起。', '',
         '| 方法 | 截断均值/s | 中位数/s | ≥20s | ≥80s | ≥400s |', '|---|---:|---:|---:|---:|---:|']
for name, x in models.items():
    median = '≥400' if x['median_seconds'] >= 400 else f"{x['median_seconds']:.2f}"
    lines.append(f"| {labels[name]} | {x['mean_capped_seconds']:.2f} | {median} | {x['survival20_pct']:.1f}% | {x['survival80_pct']:.1f}% | {x['survival400_pct']:.1f}% |")
lines += ['', '400秒是观察截断，不是失败时长；原生failure是评估失败代理，并非独立的物理掉落检测。', '',
          f"混合组相比普通1B，截断均值变化{mix['mean_capped_seconds']-base['mean_capped_seconds']:+.2f}秒，20秒存活率变化{mix['survival20_pct']-base['survival20_pct']:+.1f}个百分点；相比旧10k，截断均值变化{mix['mean_capped_seconds']-old['mean_capped_seconds']:+.2f}秒。",
          '', '| 测试质量/摩擦 | 普通1B | 旧10k | 轻/低 | 轻/高 | 重/低 | 重/高 | 混合 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in t['factorial']:
    lines.append(f"| {row['mass_g']:.0f}g / {row['mu']} | " + ' | '.join(f"{row['means'][n]:.2f}" for n in models) + ' |')
lines += ['', '每格12例，均为400秒截断均值。它们是特定诊断初态上的结果，不能直接当作总体成功率或质量/摩擦的通用阈值。没有等比例但不同质量/摩擦的扫描，也没有接触法向力记录，本轮不足以确认一个通用质量/摩擦比定律。', '',
          '| 方法 | 原本<20s、现≥20s | 原本≥20s、现<20s | 相比1B提升>20s | 下降>20s |', '|---|---:|---:|---:|---:|']
for name, row in t['paired_outcomes'].items():
    lines.append(f"| {labels[name]} | {row['rescued_baseline_under20_to_atleast20_count']} | {row['harmed_baseline_atleast20_to_under20_count']} | {row['gain_over20_count']} | {row['loss_over20_count']} |")
lines += ['', '这些是同一批案例的描述性计数，不是统计显著性检验。', '', '## 对原因的证据与限制', '',
          '新五组使用共同初始化模板、相同训练预算，能更直接比较数据分配的影响。它们使用四域都成功的共同模板，且只保存连续250步片段，因此是经过成功筛选的保留分布，不能把设定的均匀采样区间当成实际数据分布。', '',
          '新旧10k还存在片段长度、成功筛选、质量缓存处理和起始观察覆盖差异，不能把新旧性能差全部归因于质量/摩擦。旧训练集有18个目标残差RMS<0.001rad的起始观察；新五组因跳过4步预热而均无此观察，但评估从目标=q、残差0开始。这是候选因素，尚未做消融证明。混合组共享这一限制却改善早期表现，也说明它不能单独解释所有结果。', '',
          '在同一批留出专家观察上，新guide动作MAE约0.041–0.045rad，旧guide约0.049–0.052rad；这项开环误差不能预测闭环guidance的排名。留出指不参与梯度更新；归一化沿用项目实现，统计整个replay buffer（包含验证片段）。', '',
          '模型输入只有关节角、上一目标和目标残差，不直接输入质量、摩擦或物体位姿。各guide的初始目标改变量接近，单看动作幅度也不足以解释结果；目前没有接触力证据来确认某个抓握机制。', '',
          '只有一个训练seed和4个诊断初态，所有guide固定scale25，未单独调优各模型。所有方法初态、实际物理参数、初始RNG及prior噪声配对，原生目标更新仍依赖各自状态；同输入基线重跑也存在后续轨迹变化。']
rep = t.get('baseline_repeatability_at_common_censoring')
if rep:
    lines += ['', f"中断前后的两次普通1B运行，初态、物理和第一步动作相同；按共同240秒窗口截断，逐案例时长绝对差平均{rep['mean_absolute_duration_difference_s']:.2f}秒，{rep['duration_difference_over20_count']}/48例相差超过20秒。这提示单条轨迹的改善需要重复验证；它不是置信区间，也不能确定波动的来源。"]
lines += ['', '## 文件与录像', '',
          f"- [训练分布与checkpoint登记]({r / 'guide_registry.md'})",
          f"- [逐案例与分组结果]({r / 'evaluation_summary.json'})",
          f"- [逐初态/seed、早期动作和迁移分析]({r / 'transfer_analysis.json'})",
          f"- [留出专家观察分析]({r / 'offline_transfer_report.md'})",
          f"- [实际训练分布图]({r / 'training_distributions.png'})",
          f"- [结果对比图]({r / 'comparison.png'})"]
for i, init in [(3, 'demo079 / noise8'), (16, 'demo082 / noise19'), (29, 'demo094 / noise25')]:
    lines.append(f"- {init}，44g/μ2.572：[原速并排录像]({r / 'videos' / f'case{i:02d}_all_guides.mp4'})、[10倍速预览]({r / 'videos' / f'case{i:02d}_overview_10x.mp4'})。结束一侧冻结并标注最后一帧。")
lines += ['', '| 录像方法 | demo079 / seed8 | demo082 / seed19 | demo094 / seed25 |', '|---|---:|---:|---:|']
for name in models:
    case_rows = {x['case']: x for x in s['all_results'] if x['method'] == name}
    values = ['≥400' if case_rows[i]['seconds'] >= 400 else f"{case_rows[i]['seconds']:.2f}" for i in [3, 16, 29]]
    lines.append(f"| {labels[name]} | " + ' | '.join(values) + ' |')
lines += ['', '录像说明了单例与总体排名不同：重/高guide在demo094的seed25案例达到400秒，但同物理条件另外两个seed案例仅1.03秒、2.40秒。这一条录像不能当作对该手型稳定成功的保证。']
lines += ['', '模型加载、EMA/训练计数、数据哈希、初始化与物理参数配对、DDIM加速实现对原始实现的一致性均已检查。三个录像案例在运行前固定选择；录像不替代336例统计。']
new_best = max([x['name'] for x in registry], key=lambda n: models[n]['mean_capped_seconds'])
positive_cells = sum(x['means']['balanced'] > x['means']['ordinary_1b'] for x in t['factorial'])
headlines = [f"本轮新guide中，{labels[new_best]}的整体截断均值最高，为{models[new_best]['mean_capped_seconds']:.2f}秒。混合组在{positive_cells}/4个测试物理点的均值高于普通1B；四个单域guide在240g的两个测试点均低于1B。此结果不支持简单地认为训练质量或摩擦范围越窄、越对应测试域，就一定更好。", '',
             '旧10k在低摩擦两个测试点明显领先混合组；在高摩擦两个点，旧10k与混合组的截断均值差均不足0.5秒，不能据此区分优劣。旧10k缺少逐片段质量/摩擦标签，因此无法确认它的优势来自哪种实际物理比例。', '',
             '混合组相对1B的均值增益，在三个策略seed分组中分别为：' + '、'.join(f"seed{k} {v['paired_mean_gain_s']:+.2f}秒" for k, v in t['paired_outcomes']['balanced']['by_policy_seed'].items()) + '。这仍是一个训练seed上的探索性结果。', '']
lines[4:4] = headlines
(r / 'final_report.md').write_text('\n'.join(lines))
(r / 'final_data_verification.json').write_text(json.dumps({'complete_evaluations': 7, 'cases': 336,
    'checkpoint_and_dataset_hashes_valid': True, 'prior_unchanged': True,
    'source_dependencies_unchanged': True, 'script_syntax_valid': True}, indent=2))
print(r / 'final_report.md')
