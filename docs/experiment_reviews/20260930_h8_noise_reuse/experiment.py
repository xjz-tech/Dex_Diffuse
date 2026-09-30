"""Prepare/inspect by default. Simulation requires the explicit --execute flag."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent / '20260924_object_state_data'
ARCHIVE = HERE.parent / '20260930_h8_object_state_random10'
QUALIFIED = SOURCE / 'reference_turn_baseline_20260926/qualified_comparison'
CHECKPOINT = Path('/home/carus/data_usb/aggresive_random_ckpt/h8.ckpt')
DP_PYTHON = '/home/carus/miniforge3/envs/dp/bin/python'
SIM_PYTHON = '/home/carus/miniforge3/envs/decv2/bin/python'
ARMS = {'B': 0., 'N50': .5, 'N80': .8, 'N100': 1.}
SEEDS = [486266, 426182, 662165, 766174, 234124]
EPISODES = [76, 34, 54, 2]
PHYSICS = {'m044_mu11': [.044, 1.1], 'm170_mu20': [.17, 2.], 'm130_mu24': [.13, 2.4]}
PROTOCOL = dict(failureObjPosThres=.05, failureThumbTipPosThres=.1,
    failureIndexTipPosThres=.1, failureMiddleTipPosThres=.1,
    failurePinkyTipPosThres=.1, failureRingTipPosThres=.1,
    failureObjRotThres=180., invalidObjPosThres=.15,
    FailureToleranceScale=10000., fixedToleranceSteps=20000,
    trajStepsLimit=12000, resetOnReachGoal=False,
    enableCrossTrajectoryReset=True, crossTrajectoryGoalProb=.3)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def baseline(seed, physics, episode):
    return ARCHIVE / f'seed{seed}/h8/{physics}/episode_{episode:02d}/edit015'


def destination(seed, physics, episode, arm):
    return HERE / f'runs/seed{seed}/{physics}/episode_{episode:02d}/{arm}'


def episodes_for(physics):
    # Original direct: only 15 consecutive valid vertical-contact steps here.
    return [e for e in EPISODES if (physics, e) != ('m170_mu20', 2)]


def raw_folder(physics, episode):
    if physics == 'm044_mu11':
        return QUALIFIED / f'episode_{episode:02d}/direct_m044_mu11'
    return SOURCE / f'reference_turn_baseline_20260926/{physics}_guidance4_vs_edit015_20260928/episode_{episode:02d}/raw'


def jobs():
    return [(s, p, e, a) for s in SEEDS for p in PHYSICS for e in episodes_for(p) for a in ARMS]


def check_summary(s, seed, physics, episode):
    assert s['prior_noise_seed'] == seed and s['source_episode'] == episode
    assert [s['mass_kg'], s['friction']] == PHYSICS[physics]
    assert s['native_protocol'] == PROTOCOL
    assert s['control_hz'] == 30 and s['execution_steps'] == 2
    assert s['reference_interpolation_threshold'] == .1
    assert s['reference_mode'] == 'expanded' and s['reference_repeat'] == 1
    assert s['reference_interpolation'] == 0 and s['reference_interpolation_equal_jump'] is None
    assert s['settle_target_source'] == 'qpos' and s['object_size_multiplier'] == 1
    assert not s['stop_on_native_failure'] and not s['static_only'] and s['action_limit'] is None
    assert s['vertical_scale_switch_angle_deg'] is None
    assert s['init_order'] == 'properties; simulate/fetch/observe; root import'
    assert s['steps'] == s['intended_steps'] and s['steps']['settle'] == s['steps']['hold'] == 60
    prior = s['prior']
    assert Path(prior['prior']).resolve() == CHECKPOINT.resolve()
    assert prior['spec']['horizon'] == 8 and prior['spec']['n_obs_steps'] == 4
    assert prior['spec']['n_pred_action_steps'] == 5 and prior['ddim'] == 4
    assert prior['guidance_scale'] == 0 and prior['guidance_steps'] == 5
    editor = prior['editor']
    assert editor['requested_noise_ratio'] == .15 and editor['timesteps'] == [8, 5, 3, 0]
    assert editor['num_train_timesteps'] == 100 and editor['eta'] == 0
    assert editor['known_history_steps'] == 3 and editor['future_reference_steps'] == 5
    assert editor['weight_source'] == 'EMA model' and editor['clip_sample']


def commands(job):
    seed, physics, episode, arm = job
    out = destination(*job)
    ref = QUALIFIED / f'episode_{episode:02d}/reference_full.npz'
    sock = f'/tmp/h8_noise_reuse_{os.getpid()}_{seed}_{physics}_{episode}_{arm}.sock'
    server = [DP_PYTHON, str(HERE / 'server.py'), '--socket', sock,
              '--log', str(out / 'predictions.json'), '--reference', str(ref),
              '--baseline-trace', str(baseline(seed, physics, episode) / 'trace.json'),
              '--checkpoint', str(CHECKPOINT), '--rho', str(ARMS[arm]), '--seed', str(seed)]
    mass, friction = PHYSICS[physics]
    sim = [SIM_PYTHON, str(SOURCE / 'compare_reference_edit_rollouts.py'),
           '--mode', 'guided', '--out', str(out), '--reference', str(ref),
           '--reference-id', '0', '--source-episode', str(episode), '--front-only',
           '--audit-recording', '--grasp-evidence', '--object-mass-kg', str(mass),
           '--video-label', f'h8 SDEdit {arm} rho={ARMS[arm]:g}',
           '--friction', str(friction), '--guidance-steps', '5', '--guidance-scale', '0',
           '--execution-steps', '2', '--prior-noise-seed', str(seed),
           '--reference-interpolation-threshold', '.1', '--socket', sock]
    return server, sim, sock


def prepare():
    if (HERE / 'runs').exists():
        raise RuntimeError('Cannot overwrite the frozen plan after any simulation output exists')
    source_manifest = read(ARCHIVE / 'manifest.json')
    assert source_manifest['prior_noise_seeds'][:5] == SEEDS
    assert source_manifest['episodes'] == EPISODES and source_manifest['physics'] == PHYSICS
    # Verify explicit native thresholds in the actual simulator without importing IsaacGym.
    tree = ast.parse((SOURCE / 'compare_reference_edit_rollouts.py').read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PROTOCOL' for t in n.targets))
    assert {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords} == PROTOCOL
    evidence, rows = set(), []
    for seed in SEEDS:
        for physics in PHYSICS:
            for episode in episodes_for(physics):
                folder = baseline(seed, physics, episode)
                s, a = read(folder / 'summary.json'), read(folder / 'audit.json')
                check_summary(s, seed, physics, episode)
                assert (a['seed'], a['model'], a['physics'], a['episode'], a['method']) == (
                    seed, 'h8', physics, episode, 'edit015')
                assert a['validation']['initial_all_fields_exact'] and a['validation']['settle_motion_exact']
                assert a['validation']['settle_contact_force_max_delta_N'] < 1e-5
                assert (folder / 'trace.json').is_file() and (folder / 'initial_state.npz').is_file()
                predictions = read(folder / 'predictions.json')
                assert [p['reference_index'] for p in predictions] == list(range(0, s['steps']['action'], 2))
                assert all(p['seeds'] == [seed] for p in predictions)
                for filename in ('summary.json', 'audit.json', 'initial_state.npz', 'predictions.json', 'trace.json'):
                    evidence.add(folder / filename)
                rows.append(dict(seed=seed, physics=physics, episode=episode, arm='L_archived',
                    folder=str(folder), stable_turn=a['stable_turn'], ordinary_turn=a['ordinary_turn'],
                    retained_steps=a['first_separation_action_step'] or s['steps']['action'],
                    action_steps=s['steps']['action'], source_start_frame=s['source_start_frame'],
                    first_native_failure=a['first_native_failure']))
    for episode in EPISODES:
        evidence.add(QUALIFIED / f'episode_{episode:02d}/reference_full.npz')
        proof = QUALIFIED / f'episode_{episode:02d}/baseline_verified.json'
        assert read(proof)['baseline_verified']
        evidence.add(proof)
    qualifications = []
    for physics in PHYSICS:
        for episode in EPISODES:
            raw = raw_folder(physics, episode)
            proof = raw / ('baseline_verified.json' if physics == 'm044_mu11' else 'analysis.json')
            result = read(proof)
            longest = result['longest_vertical_contact_steps']
            eligible = longest >= 30
            assert eligible == (episode in episodes_for(physics))
            raw_summary = read(raw / 'summary.json')
            assert raw_summary['mode'] == 'direct' and raw_summary['reference_interpolation_threshold'] is None
            assert [raw_summary['mass_kg'], raw_summary['friction']] == PHYSICS[physics]
            evidence.update((proof, raw / 'summary.json', raw / 'initial_state.npz'))
            qualifications.append(dict(physics=physics, episode=episode, longest_vertical_contact_steps=longest,
                                       eligible=eligible, source=str(proof)))
    # Freeze all executable experiment and shared inference code, plus native
    # controller task sources. Current dirty workspace is allowed and hashed.
    sources = set(HERE.glob('*.py')) | {
        SOURCE / n for n in ('reference_action_editor.py', 'reference_resampling.py',
            'compare_reference_edit_rollouts.py', 'random4_geometry.py',
            'analyze_h8_cross_model_comparison.py', 'run_h8_cross_model_comparison.py')}
    sources |= set((ROOT / 'eval').glob('*.py'))
    sources |= set((ROOT / 'diffusion_policy').rglob('*.py'))
    sources |= set(Path('/home/carus/Program/dex-controller/maniptrans_envs').rglob('*.py'))
    sources |= set(Path('/home/carus/Program/dex-controller/maniptrans_envs').rglob('*.yaml'))
    sources |= {Path('/home/carus/Data/exp_data/hydra_config.yaml'),
                SOURCE / 'astra_demo_cache.py', SOURCE / 'reference/initial_state.json'}
    evidence.add(ARCHIVE / 'manifest.json')
    hashes = {str(path): sha(path) for path in sorted(sources | evidence | {CHECKPOINT})}
    manifest = dict(status='prepared_not_started', seed_selection='first five of original preselected order; no performance filtering',
        seeds=SEEDS, environment_seed=42, episodes=EPISODES, physics=PHYSICS, arms=ARMS,
        new_rollouts=len(jobs()), archived_rollouts=len(rows), direct_qualifications=qualifications, checkpoint=str(CHECKPOINT), weight_source='EMA model',
        obs=4, horizon=8, pred=5, exec=2, noise_ratio=.15, timesteps=[8, 5, 3, 0], eta=0,
        native_protocol=PROTOCOL, history_noise='fresh in all new arms',
        archived_noise='same Gaussian tensor regenerated from the same seed at each local slot; NOT rho=0',
        video=True, baseline_rows=rows, sha256=hashes,
        provenance_limit='Current checkpoint/source hashes frozen now; historical archive has no checkpoint digest, so pre-archive immutability cannot be independently proven.')
    write(HERE / 'manifest.json', manifest)
    write(HERE / 'commands.json', [dict(job=j, server=commands(j)[0], simulator=commands(j)[1]) for j in jobs()])
    print(f'PREPARED: {len(rows)} archived baselines, {len(jobs())} planned runs. No simulation started.')


def verify_manifest():
    m = read(HERE / 'manifest.json')
    assert m['seeds'] == SEEDS and m['arms'] == ARMS and m['new_rollouts'] == len(jobs())
    for path, digest in m['sha256'].items():
        if sha(path) != digest:
            raise RuntimeError(f'Frozen input changed: {path}')
    return m


def run_one(job):
    out = destination(*job)
    if out.exists():
        if (out / 'complete.json').exists():
            saved = read(out / 'complete.json')
            assert saved['job'] == list(job) and saved['manifest_sha256'] == sha(HERE / 'manifest.json')
            return
        raise RuntimeError(f'Partial output exists; inspect it before retrying: {out}')
    out.mkdir(parents=True)
    server_cmd, sim_cmd, sock = commands(job)
    sim_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
        PATH='/home/carus/miniforge3/envs/decv2/bin:' + os.environ['PATH'],
        PYTHONPATH=f'/home/carus/opt/isaacgym/python:/home/carus/Program/dex-controller:{ROOT / "eval"}',
        LD_LIBRARY_PATH='/home/carus/miniforge3/envs/decv2/lib')
    server_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                      LD_LIBRARY_PATH='/home/carus/miniforge3/envs/dp/lib')
    write(out / 'command.json', dict(job=job, server=server_cmd, simulator=sim_cmd))
    server = None
    try:
        with (out / 'server.log').open('w') as log:
            server = subprocess.Popen(server_cmd, env=server_env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 180
        while not Path(sock).exists():
            if server.poll() is not None:
                raise RuntimeError('Server exited before becoming ready')
            if time.monotonic() > deadline:
                raise TimeoutError('Server socket')
            time.sleep(.2)
        with (out / 'sim.log').open('w') as log:
            subprocess.run(sim_cmd, env=sim_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if server.wait(timeout=60) != 0:
            raise RuntimeError('Inference server failed')
        check_summary(read(out / 'summary.json'), *job[:3])
        # A completed simulation is still not an audited experimental result.
        write(out / 'complete.json', dict(job=job, manifest_sha256=sha(HERE / 'manifest.json'),
                                         status='simulation_complete_analysis_pending'))
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        Path(sock).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        with (HERE / 'batch.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_manifest()
            for job in jobs():
                print('RUN', job, flush=True)
                run_one(job)
    else:
        print('DRY RUN ONLY:', SEEDS, ARMS, '220 new rollouts + 55 archived references; one direct-unqualified condition excluded.')
        print('Use --prepare to validate/freeze inputs; --execute explicitly starts simulations.')


if __name__ == '__main__':
    main()
