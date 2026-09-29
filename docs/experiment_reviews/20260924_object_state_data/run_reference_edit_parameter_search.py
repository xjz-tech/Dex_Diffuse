"""Staged, matched Object_state_data reference-edit parameter search."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

P = Path(__file__).resolve().parent
R = P / 'reference_turn_baseline_20260926'
O = R / 'reference_edit015_parameter_search_20260928'
EPISODES = (76, 34, 54, 2)
CHECKPOINT = '/home/carus/data_usb/10B_obs_4-66.ckpt'


def folder(config, ep):
    return O / f"episode_{ep:02d}" / f"t{config['t']:02d}_d{config['ddim']}_e{config['exec']}_seed{config.get('seed',44)}"


def existing_folder(config, ep):
    if config['t'] == 8 and config['exec'] == 2 and config.get('seed', 44) == 44:
        if config['ddim'] in (4, 6):
            return R / f"four_reference_edit_noise_ddim_sweep_20260928/episode_{ep:02d}/edit_015_ddim{config['ddim']}_seed44"
    return None


def ratios_from_trained_scheduler():
    if (O / 'noise_grid.json').exists():
        return json.loads((O / 'noise_grid.json').read_text())
    from reference_action_editor import ReferenceActionEditor
    editor = ReferenceActionEditor(CHECKPOINT, .15, steps=4, execution_steps=2)
    alphas = editor.controller.scheduler.alphas_cumprod
    ratios = ((1 - alphas) / alphas).sqrt()
    data = {str(t): float(ratios[t]) for t in range(6, 11)}
    assert editor.timesteps[0] == 8 and abs(data['8']-editor.actual_noise_ratio) < 1e-8
    O.mkdir(parents=True, exist_ok=True)
    (O / 'noise_grid.json').write_text(json.dumps(data, indent=2)+'\n')
    return data


def config(t, ddim, execution, ratios, seed=44):
    return dict(t=int(t), ddim=int(ddim), exec=int(execution), ratio=float(ratios[str(t)]), seed=int(seed))


def config_key(c):
    return (c['t'], c['ddim'], c['exec'], c.get('seed', 44))


def planned(stage, ratios):
    if stage == 1:
        return [config(t, 4, 2, ratios) for t in range(6, 11)]
    source = json.loads((O / ('stage1_scores.json' if stage == 2 else 'stage2_scores.json')).read_text())
    choices = [x['config'] for x in source['selected_for_next_stage']]
    assert len(choices) == 2
    if stage == 2:
        return [config(x['t'], 4, execution, ratios) for x in choices for execution in (1, 3, 4)]
    assert stage == 3
    return [config(x['t'], 6, x['exec'], ratios) for x in choices]


def run_one(job):
    c, ep = job
    cached = existing_folder(c, ep)
    if cached is not None:
        assert (cached/'analysis.json').exists()
        return dict(config=c, episode=ep, status='archived', folder=str(cached))
    out = folder(c, ep)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').exists() and (out/'predictions.json').exists():
        return dict(config=c, episode=ep, status='existing', folder=str(out))
    case = R / f'qualified_comparison/episode_{ep:02d}'
    simenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                  PATH='/home/carus/miniforge3/envs/decv2/bin:'+os.environ['PATH'],
                  PYTHONPATH='/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:/home/carus/Program/Dexterous_Manipulation/Dex_diffuse/eval',
                  LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    serverenv = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                     LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    sock = Path(f"/tmp/editsearch_{ep}_{c['t']}_{c['ddim']}_{c['exec']}_{c['seed']}_{os.getpid()}.sock")
    server = None
    try:
        command = ['/home/carus/miniforge3/envs/dp/bin/python', str(P/'server_reference_action_editor.py'),
                   '--socket', str(sock), '--log', str(out/'predictions.json'),
                   '--checkpoint', CHECKPOINT, '--reference', str(case/'reference_full.npz'),
                   '--reference-interpolation-threshold', '.1',
                   '--noise-ratio', repr(c['ratio']), '--ddim-steps', str(c['ddim']),
                   '--execution-steps', str(c['exec'])]
        with (out/'server.log').open('w') as log:
            server = subprocess.Popen(command, env=serverenv, stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if sock.exists():
                break
            if server.poll() is not None:
                raise RuntimeError(f'server exited: {out}')
            time.sleep(.2)
        else:
            raise TimeoutError(out)
        command = ['/home/carus/miniforge3/envs/decv2/bin/python', str(P/'compare_reference_edit_rollouts.py'),
                   '--mode', 'guided', '--out', str(out), '--reference', str(case/'reference_full.npz'),
                   '--reference-id', '0', '--source-episode', str(ep), '--front-only',
                   '--audit-recording', '--grasp-evidence', '--no-video',
                   '--reference-interpolation-threshold', '.1', '--object-mass-kg', '.044',
                   '--friction', '1.1', '--guidance-steps', '9', '--guidance-scale', '0',
                   '--execution-steps', str(c['exec']), '--prior-noise-seed', str(c['seed']),
                   '--socket', str(sock)]
        with (out/'sim.log').open('w') as log:
            subprocess.run(command, env=simenv, stdout=log, stderr=subprocess.STDOUT, check=True)
        assert server.wait(timeout=30) == 0
        print('DONE', ep, config_key(c), flush=True)
        return dict(config=c, episode=ep, status='complete', folder=str(out))
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            server.wait()
        sock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=(1, 2, 3))
    args = parser.parse_args()
    ratios = ratios_from_trained_scheduler()
    cfgs = planned(args.stage, ratios)
    manifest = dict(stage=args.stage, configs=cfgs, episodes=EPISODES,
        source_frames=dict(zip(EPISODES, (114,79,110,103))), checkpoint=CHECKPOINT,
        mass_kg=.044, friction=1.1, interpolation_threshold_rad=.1,
        source='own qualified initial state and full reference tail per episode',
        sim_seed=42, prior_seed=44, complete_tail=True, native_protocol='unchanged',
        video_recorded=False, selected_without_looking_at_future_stage_outcomes=True)
    (O/f'stage{args.stage}_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    jobs = [(c, ep) for c in cfgs for ep in EPISODES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(run_one, jobs))
    (O/f'stage{args.stage}_run_status.json').write_text(json.dumps(result, indent=2)+'\n')
    print('STAGE COMPLETE', args.stage, len(result), flush=True)


if __name__ == '__main__':
    main()
