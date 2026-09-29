from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;R=P/'reference_turn_baseline_20260926';O=R/'episode54_reference_edit_20260928';C=R/'qualified_comparison/episode_54';METHODS=['standard_g4_s50','edit010','edit020','edit035'];LABELS=['原有 guidance：scale50 guide4','reference编辑：弱','reference编辑：中','reference编辑：较强']
def main():
 result=json.loads((O/'RESULTS.json').read_text());raw=json.loads((C/'direct_m044_mu11/m044_mu11_result.json').read_text());base_loss=raw['first_separation']['control_step'];lines=['# Episode54：reference初始化的动作编辑实验','', '同一已验收初态episode54/f110，44g、手和物体摩擦1.1、原尺寸、固定wrist、30Hz。使用原速reference，不插值；462步完整动作，另加60步静置和60步末目标保持。原速reference能稳定翻转，为本轮功能基线。','', '新方法把12步动作窗口构造为实际历史3步控制目标+未来9步reference，按训练噪声日程加噪后进行4次DDIM更新。历史3步在每个噪声层固定为对应加噪值，未来9步由prior去噪；执行前2步并推进reference2步。新方法scale=0，不叠加关节MSE guidance，不设额外动作修正上限或后处理平滑。使用10B EMA。','', '| 方法 | seed44 分离步 | seed45 分离步 | seed46 分离步 | 平均分离步 | 稳定翻转 |','|---|---:|---:|---:|---:|---:|',f'| 原速 reference（确定性） | {base_loss} | — | — | {base_loss} | 通过 |']
 for m,label in zip(METHODS,LABELS):
  rr=[next(x for x in result if x['method']==m and x['seed']==s) for s in [44,45,46]];vals=[x['first_separation']['control_step'] if x['first_separation'] is not None and x['first_separation']['phase']=='action' else None for x in rr]
  cells=[('>462' if v is None else f"{v}{' ✓' if x['stable_turn'] else ' ✗'}") for v,x in zip(vals,rr)];mean='—' if any(v is None for v in vals) else f'{np.mean(vals):.1f}';lines.append('| '+' | '.join([label]+cells+[mean,f"{sum(x['stable_turn'] for x in rr)}/3"])+ ' |')
 lines+=['','分离步为首次连续3个控制步满足几何/接触分离条件的起始步，使用1起始物理控制步编号；本轮无插值，因此它也等于原始reference进度。它独立于native failure，不用于改变环境终止。稳定翻转要求首次分离前连续至少30个控制步处于距竖直≤30°、悬空且保持接触状态。','', '| 方法 / seed | 首段稳定竖直接触起点 | 最长竖直接触步数 | 分离前动作RMSE(rad) | 分离前最大单关节修改(rad) |','|---|---:|---:|---:|---:|']
 for m,label in zip(METHODS,LABELS):
  for x in [z for z in result if z['method']==m]:
   start=x['first_stable_turn_start_step'];lines.append(f"| {label} / {x['seed']} | {start if start is not None else '未达到'} | {x['longest_vertical_contact_steps']} | {x['command_reference_rmse_before_separation_rad']:.4f} | {x['command_max_abs_edit_before_separation_rad']:.4f} |")
 # Compare errors on one identical, entirely pre-separation prefix across methods.
 end=min([base_loss]+[x['first_separation']['control_step'] for x in result if x['first_separation'] and x['first_separation']['phase']=='action'])-1;ref=np.load(C/'reference_full.npz')['hand_target_rad'][0];common=[]
 for x in result:
  f=O/f"{x['method']}_seed{x['seed']}";trace=json.loads((f/'trace.json').read_text());cmd=np.asarray([z['command'] for z in trace if z['phase']=='action'])[:end];d=cmd-ref[:end];common.append(dict(method=x['method'],seed=x['seed'],common_prefix_steps=end,rmse_rad=float(np.sqrt(np.mean(d*d))),max_abs_rad=float(np.max(np.abs(d)))))
 (O/'common_prefix_metrics.json').write_text(json.dumps(common,indent=2)+'\n')
 lines+=['',f'上表“分离前RMSE”各组覆盖的时间不同，不能当作等窗口误差排名。另在所有组都未分离的前{end}步，统一比较修改量：','', '| 方法 | 同窗口动作RMSE，三seed平均 |','|---|---:|']
 for m,label in zip(METHODS,LABELS):lines.append(f"| {label} | {np.mean([x['rmse_rad'] for x in common if x['method']==m]):.4f} rad |")
 lines+=['','## 噪声及数值核验','', '| 档位 | 实际归一化噪声/信号比 | DDIM训练时间步（均4次更新） |','|---|---:|---|']
 for m,label in zip(METHODS[1:],LABELS[1:]):
  meta=next(x['editor'] for x in result if x['method']==m);lines.append(f"| {label} | {meta['actual_noise_ratio']:.6f} | {meta['timesteps']} |")
 lines+=['','噪声比为sqrt((1-alpha_bar)/alpha_bar)，不等于弧度；各关节由checkpoint normalizer映射回实际单位。数值检查见validation.json：任意时间跨度的DDIM更新与官方均匀日程核对，已知干净轨迹的oracle epsilon在非均匀日程上准确重建，零编辑逐值返回reference。zero_verification.json确认零编辑闭环整段582物理步与原始direct一致（物体净接触力允许<1e-5N浮点差）。','', '所有新增运行的初态张量逐字段等同原速reference；60步静置与首物理步已逐字段核验（同一接触力容差）。每次执行2步后reference前进2，尾段只padding不增加实际动作数。所有控制步均完成；原生demo范围、目标更新和failure阈值沿用项目协议。首次物理分离与native failure分别保存。','', '本轮新编辑同时采用reference初始化、较低噪声起点和已知历史固定，因此与旧guidance的差异不能单独归因于某一项。结果仅限episode54的这个初态及3个采样seed，不能推广到全部episode。若保持更久但未稳定翻转，不记为成功优化。未新增录像或写入Downloads。']
 lines+=['','## 起始动作诊断','', '弱编辑seed44/45/46的第一段（实际执行2步）对reference的RMSE分别为0.08794、0.08329、0.08151rad；最大单关节修改分别为0.24027、0.24293、0.24786rad，均为第一步食指MCP_FE目标角被减小。seed44该关节静置末目标0.13365rad，原reference第一目标0.41361rad，弱编辑输出0.17335rad：原本+0.27997rad的目标变化变为+0.03970rad。不能据此认定单关节修改是唯一失败原因，尚未做单关节替换干预。','', 'reference_clipping_audit.json排查了原reference本身超出归一化[-1,1]范围的影响：前50步单纯归一化裁剪回原单位的最大变化约1.2e-7rad，不能解释上述0.24rad修改。first_chunk_diagnostic.json保存逐seed数据。']
 (O/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
