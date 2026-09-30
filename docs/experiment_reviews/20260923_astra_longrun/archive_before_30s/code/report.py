"""Audit and render only actual completed simulations; never synthesize motion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
METHODS = ['astra_direct', 'prior', 'guided5', 'guided25', 'guided100']
TITLES = ['Astra direct', 'Prior only', 'Astra + Prior / 5', 'Astra + Prior / 25', 'Astra + Prior / 100']
COLORS = ['#e879f9', '#94a3b8', '#fbbf24', '#34d399', '#60a5fa']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def audit(r):
    run = Path(r['run'])
    m = json.loads((run/'manifest.json').read_text())
    model = json.loads((run/'model.json').read_text())
    expected = dict(failure_obj_pos_thres_m=.05, failure_tip_pos_thres_m=.1,
        failure_obj_rot_thres_deg=180, invalid_obj_pos_thres_m=.15,
        failure_tolerance_scale=10000, fixed_tolerance_steps=20000,
        traj_steps_limit=10000000, reset_on_reach_goal=False, cross_trajectory_goal_prob=.3)
    assert m['protocol'] == expected, m['protocol']
    assert m['execution_steps'] == 2 and m['guidance_steps'] == 9 and m['required_plan_steps'] == 16
    assert m['data_indices'] == '000-149' and m['replay'] is None
    assert model['prior_ddim_steps'] == 4 and not model['fixed_noise']
    assert model['guidance_steps'] == 9
    assert model['prior_residual_bound_rad'] is None and model['guidance_schedule_file'] is None
    assert model['noise_seed'] == r['noise_seed']
    assert model['mode'] == ('direct' if r['method'] == 'astra_direct' else 'guided')
    reqs = sorted(run.glob('request_*.json'))
    windows = 0
    for req_path in reqs:
        request = json.loads(req_path.read_text())
        response = json.loads((run/req_path.name.replace('request_', 'response_')).read_text())
        assert request['session_id'] == response['session_id']
        assert len(response['actions']) == 16
        actions = np.asarray(response['actions'], dtype=np.float32)
        start = request['step']
        for step in range(start, min(start+8, r['steps']), 2):
            path = run/f'prediction_{step:06d}.npz'
            values = np.load(path)
            np.testing.assert_array_equal(values['reference'][0], actions[step-start:step-start+9])
            if r['method'] == 'astra_direct':
                np.testing.assert_array_equal(values['prediction'], values['reference'][:, :2])
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['reference_hold_padding_steps'] == 0
            windows += 1
    frames = sorted((run/'frames').glob('*.png'))
    assert len(frames) == r['steps']+1
    result = dict(label=r['label'], references=windows, no_tail_padding=True,
        direct_bypasses_prior=r['method'] == 'astra_direct',
        recorded_frames=len(frames), paired_initial_fields=r['paired_initial_fields_verified'])
    (run/'audit.json').write_text(json.dumps(result, indent=2))
    return result


def curves(r):
    rows = [json.loads(x) for x in (Path(r['run'])/'trajectory.jsonl').open()]
    assert [x['step'] for x in rows] == list(range(1, r['steps']+1))
    return np.array([-x['twist_degrees'] for x in rows])


def render(rows, seed, noise):
    selected = [next(r for r in rows if r['environment_seed'] == seed and r['method'] == method
        and r['noise_seed'] == (0 if method == 'astra_direct' else noise)) for method in METHODS]
    angles = [curves(r) for r in selected]
    max_steps = max(r['steps'] for r in selected)
    W, H = 1440, 940
    out = HERE/f'comparison_env{seed}_noise{noise}.mp4'
    proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '10', '-i', '-',
        '-an', '-c:v', 'libx264', '-crf', '21', '-preset', 'veryfast', '-threads', '2',
        '-pix_fmt', 'yuv420p', str(out)], stdin=subprocess.PIPE)
    font = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 17)
    last_images = {}
    # 10 fps samples of original 30 Hz camera: synchronized, normal speed.
    steps = list(range(1, max_steps+1, 3))
    if steps[-1] != max_steps:
        steps.append(max_steps)
    steps += [max_steps]*20
    for frame, step in enumerate(steps):
        canvas = Image.new('RGB', (W,H), '#0f172a')
        draw = ImageDraw.Draw(canvas)
        draw.text((16,8), f'Continuous right-turn feedback | env {seed} | Prior noise {noise} | t={min(step,max_steps)/30:.2f}s', font=font, fill='white')
        draw.text((16,36), f"Native physics: {selected[0]['mass_g']:.2f}g, object friction {selected[0]['object_friction']:.3f} | 10B DDIM4 / guide9 / exec2", font=small, fill='#cbd5e1')
        for i,r in enumerate(selected):
            x, top = (i%3)*480, 66+(i//3)*426
            actual = min(step,r['steps'])
            draw.text((x+8,top), TITLES[i], font=font, fill=COLORS[i])
            key=(i,actual)
            if key not in last_images:
                im=Image.open(Path(r['run'])/'frames'/f'{actual:06d}.png').convert('RGB').resize((480,360))
                last_images={k:v for k,v in last_images.items() if k[0]!=i}
                last_images[key]=im
            canvas.paste(last_images[key],(x,top+28))
            draw.rectangle((x,top+388,x+479,top+425),fill='#0f172a')
            draw.text((x+8,top+388), f"right={angles[i][actual-1]:+.1f} deg | t={actual/30:.2f}s",font=small,fill='white')
            status = ('DROP %.2f-%.2fs | FROZEN' % tuple(r['drop_time_bracket_seconds'])) if step>=r['steps'] and isinstance(r.get('physical_drop'),dict) else 'CENSORED - frozen' if step>=r['steps'] else 'running'
            draw.text((x+8,top+408),status,font=small,fill='#fb923c' if 'FROZEN' in status or 'frozen' in status else '#cbd5e1')
        x,top=978,500
        for j,line in enumerate(['Actual simulation recordings.', 'Each arm uses its own feedback.', 'Right-positive angle in overlays.', 'Astra direct: numeric feedback rule;', 'no per-window language-model calls.', 'Direct is shared across paired noises.', 'First native failure is logged separately.', 'Frozen panels do not imply survival.', 'Last 2 seconds: final-frame hold.']):
            draw.text((x,top+j*29),line,font=small,fill='#cbd5e1')
        proc.stdin.write(canvas.tobytes())
        if frame == min(20,len(steps)-1):
            canvas.save(out.with_suffix('.jpg'),quality=92)
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError('video encoder failed')
    out.with_suffix('.json').write_text(json.dumps(dict(environment_seed=seed, noise_seed=noise,
        methods=METHODS, source_runs=[r['run'] for r in selected], fps=10, source_fps=30,
        synchronized=True, frame_zero_skipped=True, end_freeze_seconds=2,
        maximum_timing_error_seconds=1/30),indent=2))
    print(out,flush=True)


def report(rows):
    screening=json.loads((HERE/'screening.json').read_text())
    lines=['# Astra / Prior 长程右转对照','',
        '本轮是经过物理参数与独立 Prior 预筛的条件性比较。与旧10秒固定参考重放不同：各组基于自身最新状态运行同一 Astra 编写的几何反馈规则；没有每窗口调用语言模型，未使用旧数值动作重放。',
        '',f"完成 {len(rows)}/27 次正式试验。所有提前失败保留；Astra direct 每个环境1次，Prior与每个scale每个环境配对2个噪声seed。",'',
        '取消10秒观察截止，最长步数与轨迹时限均预设为10000000步；6小时墙钟保护只算中断。原生失效判定不变；首次failure暂停查看实录，若仍抓握则仅抑制自动终止/重置并继续同一轨迹。记录首次原生failure与视觉确认的脱手时间区间。',
        '', '## 预筛（正式结果出现前冻结）','',
        '按固定顺序选最先通过的3个环境seed。独立噪声seed101，Prior先跑5秒；5秒内原生失败或质量(g)/物体摩擦>100的配置排除。100是参考历史关联预设的筛选阈值，不是普遍物理定律。', '',
        '| 环境seed | 质量/g | 物体摩擦 | 5秒内失败 | 入选 |', '|---:|---:|---:|---|---|']
    for r in screening['results']:
        lines.append(f"| {r['environment_seed']} | {r['mass_g']:.3f} | {r['object_friction']:.3f} | {r['native_failure']} | {r['selection_pass']} |")
    lines += ['', '## 正式逐次结果','',
        '右转为正；角度保守截取至末个确认仍抓握的审阅帧，脱手时刻用“最后确认抓握—最早确认分离”的区间表示。最大轴倾斜统计含终止前全程，另在分析数据保存截至抓握审阅截止帧的倾斜。可见接触并不证明稳定力闭合。原始终角、累计正反向路程及180°首达时间另存逐次summary.json，不能将脱手后的旋转当成抓握内有效转动。', '',
        '| 环境/噪声 | 方法 | 原生首次失败/s | 脱手区间/s | 抓握审阅截止净右转/° | 抓握审阅截止最大净右转/° | 确认抓握内达180° | 最大倾斜/° |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        hit='—' if r['first_180_seconds'] is None else f"{r['first_180_seconds']:.2f}"
        bracket=r.get('drop_time_bracket_seconds')
        drop='未确认' if bracket is None else f"{bracket[0]:.2f}–{bracket[1]:.2f}"
        native='—' if r['first_native_failure_seconds'] is None else f"{r['first_native_failure_seconds']:.2f}"
        lines.append(f"| {r['environment_seed']}/{r['noise_seed'] if r['method']!='astra_direct' else '—'} | [{r['method']}](runs/{r['label']}/rollout.mp4) | {native} | {drop} | {r.get('held_net_right_deg',float('nan')):.2f} | {r.get('held_max_net_right_deg',float('nan')):.2f} | {r.get('first_180_confirmed_while_held',False)} | {r['max_tilt_deg']:.2f} |")
    if (HERE/'statistics.json').exists():
        stats=json.loads((HERE/'statistics.json').read_text())
        lines += ['', '## 分组汇总','', '| 方法 | 次数 | 掉落时间中位数区间/s | 抓握审阅截止净右转中位数/° | 最大净右转中位数/° | 达右转180°的轨迹 |', '|---|---:|---:|---:|---:|---:|']
        for method in METHODS:
            g=stats['groups'][method];lo,hi=g['median_drop_interval_seconds']
            lines.append(f"| {method} | {g['n']} | {lo:.2f}–{hi:.2f} | {g['median_held_net_right_deg']:.2f} | {g['median_held_max_net_right_deg']:.2f} | {g['reached_right_180_before_cutoff']}/{g['n']} |")
        lines += ['', '中位数区间由各次视觉审阅的上下界计算，表示时刻定位的不确定性，不是统计置信区间。达到180°只表示展开角越过阈值，不能忽略反转或倾斜而称为稳定成功。', '', '| 对比Prior | 确定保持更久/6对 | 确定更短/6对 |', '|---|---:|---:|']
        for method,g in stats['paired_against_prior'].items():
            lines.append(f"| {method} | {g['definitely_longer']} | {g['definitely_shorter']} |")
        lines += ['', '![全部逐次保持与转角](outcomes.png)', '', '![截至审阅抓握截止帧的转角曲线](angle_curves.png)', '']
    lines += ['', '## 五组同步实录','']
    for p in sorted(HERE.glob('comparison_*.mp4')):
        lines.append(f'- [{p.stem}]({p.name})')
    lines += ['', '## 复现与限制','',
        'CUDA初始化错误保留在attempts/，全部发生在首个控制步之前，不计为掉落；按相同配置重跑。为避免与其他任务争用显存，后续试验加入显存检查与进程锁串行运行，未中断其他任务。', '',
        '入口 code/run.py，冻结协议 protocol.json，筛选 screening.json，源文件校验 source_hashes.json；每条 runs/ 内含完整初态、请求/响应、逐窗口采样、逐步轨迹、原视频及审计。10B EMA、DDIM4、guide9、exec2，每份16步计划、每8步更新、零尾部补齐。原生示范000–149、位置0.05m、指尖0.1m、旋转180°、立即位置阈值0.15m、FailureToleranceScale10000、fixedToleranceSteps20000、resetOnReachGoal=false、跨轨迹目标概率0.3。', '',
        '未改变质量、摩擦、重力、物体自由度或手腕状态；质量与摩擦来自真实原生随机化，27个初态字段与该环境预筛逐元素核对。固定物理参数、不同模型噪声不可解释为不同物理环境。新反馈规则也不能与旧固定数值参考的时长直接作单变量比较。']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--videos',action='store_true')
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    rows=[json.loads(p.read_text()) for p in sorted((HERE/'runs').glob('*/summary.json')) if json.loads(p.read_text())['phase']=='formal']
    if args.audit:
        (HERE/'audit.json').write_text(json.dumps([audit(r) for r in rows],indent=2))
    if args.videos:
        selection=json.loads((HERE/'screening.json').read_text())
        for seed in selection['selected_environment_seeds']:
            for noise in (0,1):
                render(rows,seed,noise)
    report(rows)
