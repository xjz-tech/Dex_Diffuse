"""Build static figures and an offline episode browser from audit outputs."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent


def main():
    a=np.load(OUT/'audit_arrays.npz')
    de=np.load(OUT/'deform_arrays.npz')
    summary=json.loads((OUT/'summary.json').read_text())
    media=json.loads((OUT/'media_summary.json').read_text())
    findings=json.loads((OUT/'additional_findings.json').read_text())
    rows=list(csv.DictReader((OUT/'episodes.csv').open()))
    ids=[int(r['id'][3:]) for r in rows]
    colors=['#8c78a8','#2d8a92','#d28b35','#5579bc','#bd6464']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
    fig,axs=plt.subplots(3,2,figsize=(13,12),layout='constrained')
    ax=axs[0,0]
    ax.bar(ids,[float(r['nominal_seconds']) for r in rows],color='#2d8a92')
    ax.set(title='75 complete episodes / 48,894 frames',xlabel='Source episode ID',ylabel='Duration at 30 Hz (s)')
    ax.axvline(36,color='#bd6464',ls='--',lw=1,label='id_36 incomplete');ax.legend(fontsize=8)
    ax=axs[0,1]
    ax.hist(a['dt_ms'],bins=np.arange(10,201,1),color='#5579bc')
    ax.axvline(1000/30,color='#2d8a92',ls='--',label='33.33 ms')
    ax.axvline(50,color='#bd6464',ls='--',label='50 ms')
    ax.set(yscale='log',xlabel='Actual sampling interval (ms)',ylabel='Count (log)',title='17 intervals > 50 ms; max 195.8 ms');ax.legend(fontsize=8)
    ax=axs[1,0]
    for e in ids:
        xyz=a['state'][a['episode_id']==e,:3]
        ax.plot(xyz[:,0],xyz[:,1],alpha=.18,lw=.7,color='#2d8a92')
    initial=a['state'][a['frame']==0]
    ax.scatter(initial[:,0],initial[:,1],s=9,c='#bd6464',label='Initial state')
    ax.set(xlabel='state[0]',ylabel='state[1]',title='Position trajectories / limited scene diversity');ax.legend(fontsize=8)
    ax=axs[1,1]
    force=np.linalg.norm(a['tactile_f6'][:,:,:3],axis=2)
    boxes=ax.boxplot([force[:,i] for i in range(5)],showfliers=False,patch_artist=True,labels=[f'S{i}' for i in range(5)])
    for box,c in zip(boxes['boxes'],colors):box.set_facecolor(c)
    ax.set(yscale='log',ylabel='Norm of first 3 tactile components (raw units)',title='Sensor 0 is much less active')
    ax=axs[2,0]
    ax.hist(a['hand_action_step'],bins=100,color='#d28b35')
    ax.axvline(.5,color='#bd6464',ls='--',label='0.5 threshold')
    ax.set(yscale='log',xlabel='Largest absolute hand-joint command change per frame',ylabel='Count (log)',title='One large hand-command transition per episode');ax.legend(fontsize=8)
    ax=axs[2,1]
    x=np.arange(5)
    ax.bar(x-.18,np.asarray(media['deform_nonzero_frame_fraction'])*100,.36,color='#2d8a92',label='Nonzero deformation')
    contact_active=np.zeros(5)
    for k,v in summary['contact_shapes'].items():contact_active+=(np.array(json.loads(k))>0)*v
    contact_active/=summary['frames']
    ax.bar(x+.18,contact_active*100,.36,color='#d28b35',label='Nonempty contact points')
    ax.set(xticks=x,xticklabels=[f'S{i}' for i in x],ylabel='Frames (%)',title='Tactile contact coverage');ax.legend(fontsize=8)
    fig.suptitle('realworld_bulb_sft_260909 | Dataset audit',fontsize=17)
    fig.savefig(OUT/'overview.png',dpi=160);fig.savefig(OUT/'overview.pdf');plt.close(fig)

    e=0;m=a['episode_id']==e;t=a['timestamp'][m];st=a['state'][m];ac=a['actions'][m]
    jump=next(x['frame'] for x in findings['hand_jumps'] if x['id']==e)
    fig,axs=plt.subplots(4,1,figsize=(13,10),sharex=True,layout='constrained')
    for j,c in enumerate(colors[:3]):
        axs[0].plot(t,st[:,j],color=c,label=f'state[{j}]')
        axs[0].plot(t,ac[:,j],color=c,ls='--',alpha=.8,label=f'actions[{j}]')
    axs[0].set(ylabel='Position (project units)',title='id_0: state / command / tactile alignment');axs[0].legend(ncol=3,fontsize=8)
    for j,c in zip([9,12,18,27],colors):
        axs[1].plot(t,st[:,j],color=c,label=f'state[{j}]')
        axs[1].plot(t,ac[:,j],color=c,ls='--',alpha=.8,label=f'actions[{j}]')
    axs[1].set(ylabel='Selected hand joints');axs[1].legend(ncol=4,fontsize=8)
    for s,c in enumerate(colors):
        axs[2].plot(t,force[m,s],color=c,label=f'S{s}')
        axs[3].plot(t,de['max'][de['episode_id']==e,s],color=c,label=f'S{s}')
    axs[2].set(ylabel='Force norm (raw units)');axs[2].legend(ncol=5,fontsize=8)
    axs[3].set(ylabel='Peak deformation (0-255)',xlabel='Nominal timestamp (s)')
    for ax in axs:ax.axvline(jump/30,color='#333333',ls=':',alpha=.7);ax.grid(alpha=.15)
    fig.savefig(OUT/'example_id_0.png',dpi=160);plt.close(fig)

    # Candidate boundaries are review aids, not ground-truth segment labels.
    with (OUT/'candidate_phase_boundaries.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['id','candidate_release_frame','nominal_seconds','max_joint_step','frames_from_transition_to_end'])
        w.writeheader()
        for x in findings['hand_jumps']:
            w.writerow({'id':f"id_{x['id']}",'candidate_release_frame':x['frame'],'nominal_seconds':x['frame']/30,
                        'max_joint_step':x['max_joint_step'],'frames_from_transition_to_end':x['remaining']+1})
    complete=[r['id'] for r in rows]
    (OUT/'complete_episode_manifest.json').write_text(json.dumps({'source':summary['source'],'included':[f'rank_0/{e}' for e in complete],
        'excluded':[{'path':'rank_0/id_36','reason':'No committed data parquet or episode metadata'}],
        'meaning':'Structurally complete episodes, not success labels'},indent=2))

    opts=''.join(f'<option value="{r["id"]}">{r["id"]} · {r["frames"]} 帧</option>' for r in rows)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>灯泡数据集分析</title><style>body{font:16px/1.7 system-ui,sans-serif;background:#f1f5f8;color:#193447;margin:0}main{max-width:1280px;margin:auto;padding:28px}h1{line-height:1.25}section{background:white;padding:24px;border-radius:12px;margin:20px 0}img{width:100%;height:auto}select,button{font:inherit;padding:7px 12px;margin:6px;border:1px solid #b7c8d3;border-radius:6px}a{color:#176c7a}small{color:#526a7a}.cards{display:flex;gap:20px;flex-wrap:wrap}.cards b{font-size:28px}.cards div{flex:1;min-width:130px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #e2e9ed;text-align:left}</style>
<main><h1>realworld_bulb_sft_260909</h1><p>真实机器人灯泡拧入任务 · 离线数据检查与轨迹浏览</p>
<section class="cards"><div><b>75</b><br>完整轨迹</div><div><b>48,894</b><br>有效帧</div><div><b>27.16 分钟</b><br>标称总时长</div><div><b>33.19 GiB</b><br>目录文件总量</div></section>
<section><p>主要发现：id_36 保存不完整；触觉 S0 接触覆盖很低；每条轨迹有一次明显手部动作切换；少量实际采样间隔过长。</p>
<p>没有成功/失败字段，done 只标识结束。完整轨迹清单不等于成功示范清单。图中的触觉数值使用原始单位，传感器到手指的映射未在元数据中给出。</p>
<p><a href="report.md">详细中文报告</a> · <a href="episodes.csv">逐轨迹统计 CSV</a> · <a href="dimensions.csv">各维度统计 CSV</a> · <a href="candidate_phase_boundaries.csv">候选阶段切换点</a> · <a href="overview.pdf">统计图 PDF</a></p></section>
<section><img src="overview.png" alt="数据集统计图"></section>
<section><h2>逐轨迹画面</h2><p>每路相机均匀抽取 5 帧。可以切换轨迹检查起始、操作过程与结束状态。</p>
<button id="prev">上一条</button><select id="episode">OPTIONS</select><button id="next">下一条</button><p id="detail"></p>
<img id="front" alt="image 相机"><img id="extra" alt="extra_view_image 相机"><small>image 从画面看为外部视角，extra_view_image 为随手臂运动的近景视角。</small></section>
<section><h2>动作切换复核示例</h2><p>id_18 的手部动作在第 491 帧发生明显变化。画面与随后的松手、撤离过程一致；该推断仍需结合采集端逻辑确认。</p><img src="samples/id_18_jump.jpg" alt="切换前后画面"><img src="example_id_0.png" alt="状态动作触觉轨迹"></section>
<section><h2>全部终帧</h2><p>用于人工检查末态，不能替代任务成功判定。</p><img loading="lazy" src="final_frames_1.jpg"><img loading="lazy" src="final_frames_2.jpg"></section>
<section><p>检查范围：75 条完整轨迹的低维数据、触觉形变和 timing 全量扫描；97,788 张图像检查非空、PNG 首尾及相邻字节重复；750 张均匀抽样图像完整解码，另读取动作切换附近画面。原始数据未修改。</p></section></main>
<script>const rows=ROWS;const sel=document.getElementById('episode');function show(){const r=rows[sel.selectedIndex];document.getElementById('front').src='samples/'+r.id+'_image.jpg';document.getElementById('extra').src='samples/'+r.id+'_extra_view_image.jpg';document.getElementById('detail').textContent=r.id+' · '+r.frames+' 帧 · '+Number(r.nominal_seconds).toFixed(2)+' 秒 · 实际 '+Number(r.effective_hz).toFixed(3)+' Hz · 最大采样间隔 '+Number(r.dt_max_ms).toFixed(1)+' ms';}sel.onchange=show;document.getElementById('prev').onclick=()=>{sel.selectedIndex=Math.max(0,sel.selectedIndex-1);show()};document.getElementById('next').onclick=()=>{sel.selectedIndex=Math.min(rows.length-1,sel.selectedIndex+1);show()};show();</script></html>'''
    (OUT/'index.html').write_text(html.replace('OPTIONS',opts).replace('ROWS',json.dumps(rows,ensure_ascii=False)))
    print('Wrote overview.png/pdf, example_id_0.png, index.html, manifests and candidate boundaries')


if __name__=='__main__':main()
