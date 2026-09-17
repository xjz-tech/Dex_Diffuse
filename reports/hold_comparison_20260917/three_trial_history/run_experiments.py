"""Authorized empty-hand comparison; six sequential, bounded hardware runs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
runs = []
for pair, seed in enumerate((50, 51, 52), 1):
    for hold in ((0, 1) if pair % 2 else (1, 0)):
        name = f'pair{pair}_seed{seed}_hold{hold}'
        log = OUT / f'{name}.txt'
        if log.exists():
            raise SystemExit(f'Refusing to repeat existing trial: {log}')
        description = (f'空手无灯泡 holding 对照实验; 配对轮次={pair}/3; '
                       f'HOLD_DURING_INFERENCE={hold}; SEED={seed}; '
                       '每轮600动作/300chunk; chunk=2; 30Hz; 每轮ROTATE复位; '
                       'TensorRT FP32; FUSED_DDIM=1; DDIM=4步; '
                       '不插值; 沿用软件限幅关闭配置; 完整TXT+JSONL记录; '
                       '比较纯策略跨chunk/内部跳变及含hold实际指令跳变; '
                       '三轮分别统计后等权平均; 排除初始化/首个策略动作/退出hold')
        env = os.environ.copy()
        for key in ('_DEX_REAL_LOG_ACTIVE', 'GUIDE_CKPT_PATH', 'GUIDANCE_SCALES'):
            env.pop(key, None)
        env.update(TENSORRT='1', TRT_PRECISION='fp32', FUSED_DDIM='1',
                   HOLD_DURING_INFERENCE=str(hold), SEED=str(seed),
                   MAX_STEPS='600', ACTION_CHUNK_STEPS='2', CONTROL_HZ='30',
                   SAMPLER='ddim', INFERENCE_STEPS='4', LIVE='1',
                   CHECK_ONLY='0', PREFLIGHT_ONLY='0', WAIT_FOR_ENTER='0',
                   MOVE_TO_INITIAL_POSE='1', INIT_POSE='ROTATE',
                   HAND_INTERPOLATE='0', HOLD_CURRENT_ON_EXIT='1',
                   DISABLE_STEP_CLAMP='1', DISABLE_JOINT_LIMITS='1',
                   MAX_TRACKING_ERROR='0', LOG_ACTION_STEPS='1', DEBUG_RECORDING='1',
                   CKPT_PATH=str(ROOT / 'runs/obs_4-66.ckpt'),
                   RUN_LOG=str(log), EXPERIMENT_DESCRIPTION=description)
        print(f'START {name}', flush=True)
        started = time.time()
        with (OUT / f'{name}.console.log').open('w') as stream:
            proc = subprocess.Popen(['bash', 'eval/eval_para_obs66_real.sh'],
                                    cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=180)
            except BaseException:
                os.killpg(proc.pid, signal.SIGINT)
                proc.wait(timeout=15)
                raise
        run = dict(name=name, pair=pair, seed=seed, hold=hold, log=str(log),
                   started=started, seconds=time.time()-started, returncode=code,
                   description=description)
        runs.append(run)
        (OUT / 'manifest.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2)+'\n')
        print(f'END {name} code={code} seconds={run["seconds"]:.1f}', flush=True)
        if code:
            raise SystemExit(f'Trial failed; stopping: {name}. See {log}')
print('ALL SIX TRIALS FINISHED', flush=True)
