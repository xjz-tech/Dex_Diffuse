"""Three authorized empty-hand experiments, varying only dataset initial pose."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
poses = json.loads((OUT / 'selected_poses.json').read_text())['poses']
sources = ['eval/real/robot_init.py', 'eval/real/inference_real.py',
           'eval/real/debug_recording.py', 'eval/eval_para_obs66_real.sh']
(OUT / 'source_sha256.json').write_text(json.dumps({p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, indent=2)+'\n')
runs = []
for pose in poses:
    name = pose['name'] + '_seed50'
    log = OUT / (name + '.txt')
    if log.exists() or log.with_suffix('.jsonl').exists():
        raise SystemExit(f'Refusing to overwrite existing trial: {log}')
    description = (f'空手无灯泡 数据集初始化对照; {pose["name"]}; '
                   f'来源={pose["shard"]}; row={pose["row"]}; episode={pose["episode_id"]}; step=0; '
                   '不holding; seed=50; 600动作; chunk=2; 30Hz; TensorRT FP32; fused DDIM4; '
                   '无插值; 软件步长限幅与关节限位关闭，与历史基线一致; '
                   '0.09rad是统计阈值，不是本次执行限幅; '
                   '统计相邻policy action的绝对差，分别统计逐关节与22关节最大变化; '
                   '初始化使用上述数据集qpos，不使用robot_init.py默认ROTATE数值')
    env = os.environ.copy()
    for key in ('_DEX_REAL_LOG_ACTIVE', 'GUIDE_CKPT_PATH', 'GUIDANCE_SCALES'):
        env.pop(key, None)
    env.update(MODEL_PYTHON='/home/frankagvl/anaconda3/envs/dexIL/bin/python',
        TENSORRT='1', TRT_PRECISION='fp32', FUSED_DDIM='1', HOLD_DURING_INFERENCE='0',
        SEED='50', MAX_STEPS='600', ACTION_CHUNK_STEPS='2', CONTROL_HZ='30',
        SAMPLER='ddim', INFERENCE_STEPS='4', LIVE='1', CHECK_ONLY='0', PREFLIGHT_ONLY='0',
        WAIT_FOR_ENTER='0', MOVE_TO_INITIAL_POSE='0', INIT_POSE='ROTATE',
        MODEL_WARMUP='1', START_DELAY='0', INITIAL_POSE_STEPS='50', INITIAL_POSE_SECONDS='2.0',
        INITIAL_POSE_TOLERANCE='0.1', INITIAL_POSE_TIMEOUT_SECONDS='3.0', INITIAL_POSE_POLL_SECONDS='0.1',
        JOINT_LIMIT_MARGIN='0', MAX_HAND_STEP='0.09', HAND_INTERPOLATE='0', HOLD_CURRENT_ON_EXIT='1',
        DISABLE_STEP_CLAMP='1', DISABLE_JOINT_LIMITS='1', MAX_TRACKING_ERROR='0',
        LOG_ACTION_STEPS='1', DEBUG_RECORDING='1', DEVICE='cuda:0',
        HAND_HOST='localhost', HAND_PORT='5570', HAND_TIMEOUT_MS='2000', OBSERVATION_MODE='qpos-target-residual',
        CKPT_PATH=str(ROOT/'runs/obs_4-66.ckpt'), RUN_LOG=str(log), EXPERIMENT_DESCRIPTION=description,
        EXPERIMENT_INITIAL_POSE_JSON=str(OUT/(pose['name']+'.json')))
    print('START', name, flush=True)
    started = time.time()
    with (OUT/(name+'.console.log')).open('w') as stream:
        proc = subprocess.Popen(['bash', str(OUT/'run_one.sh'), '--record'], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=180)
        except BaseException:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=15)
            raise
    runs.append(dict(name=name, seed=50, hold=0, log=str(log), pose=pose,
                     started=started, seconds=time.time()-started, returncode=code, description=description))
    (OUT/'manifest.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2)+'\n')
    print(f'END {name} code={code} seconds={runs[-1]["seconds"]:.1f}', flush=True)
    if code:
        raise SystemExit(f'Trial failed: {log}; stopping remaining trials')
