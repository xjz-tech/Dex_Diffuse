"""Compare archived tracking and probe guidance on fixed recorded observations.

Offline probes do not measure closed-loop simulation success.
"""
import json
import sys
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
OUT = P / 'corrected_direct_vs_reference/tracking_diagnostic'
OLD = P.parent / '20260924_horizontal_to_vertical/native_wrist_170g_mu22/case110_direct_vs_guided'


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    OUT.mkdir(exist_ok=True)
    names = json.loads((P / 'reference/initial_state.json').read_text())['hand_joint_names']
    old_ref = np.load(OLD.parent.parent / 'reference/reference.npz')['hand_target_rad']
    old = np.load(OLD / 'guided/trajectory.npz')
    new_ref = np.load(P / 'reference/reference.npz')['hand_target_rad'][1]
    series = {'case110_g2e2': (old_ref, old['command'][60:150, 110], old['q'][60:150, 110])}
    traces = {}
    for label in ['guided', 'guided9', 'guided9_exec4', 'direct']:
        trace = json.loads((P / 'corrected_direct_vs_reference' / label / 'trace.json').read_text())
        traces[label] = trace
        rows = [x for x in trace if x['phase'] == 'action']
        series[label] = (new_ref, np.array([x['command'] for x in rows]), np.array([x['q'] for x in rows]))
    result = {'archived': {}, 'offline_fixed_observation': {}}
    for label, (ref, command, q) in series.items():
        per_joint = np.sqrt(np.mean((command - ref) ** 2, axis=0))
        top = np.argsort(per_joint)[-5:][::-1]
        result['archived'][label] = {
            'command_reference_rmse_rad': rmse(command-ref),
            'q_command_rmse_rad': rmse(q-command),
            'reference_max_step_rad': float(abs(np.diff(ref, axis=0)).max()),
            'command_max_step_rad': float(abs(np.diff(command, axis=0)).max()),
            'reference_mean_joint_range_rad': float(np.ptp(ref, axis=0).mean()),
            'command_mean_joint_range_rad': float(np.ptp(command, axis=0).mean()),
            'first_third_rmse_rad': rmse((command-ref)[:len(ref)//3]),
            'last_third_rmse_rad': rmse((command-ref)[-len(ref)//3:]),
            'top_joint_rmse': {names[k]: float(per_joint[k]) for k in top},
        }
    # Exactly reconstruct the latest run's observation histories from q and
    # executed targets; repeat the last settle observation for the first call.
    sys.path[:0] = [str(ROOT), str(ROOT / 'eval')]
    import torch
    from inference_dp_controller import GuidedDDIMController
    from policy_observation import compose_policy_observation
    torch.set_num_threads(2)
    controller = GuidedDDIMController(Path('/home/carus/data_usb/10B_obs_4-66.ckpt'),
        torch.device('cuda:0'), inference_steps=4, execution_steps=4,
        guidance_scale=25., eta=0., fixed_noise=True, seed=42, allow_salvage=True)
    trace = traces['guided9_exec4']
    histories = []
    history = np.repeat(compose_policy_observation(np.array(trace[59]['q']),
        np.array(trace[59]['executed_target']), 'qpos-target-residual')[None], 4, axis=0)
    for j in range(75):
        if j % 4 == 0:
            histories.append(history.copy())
        row = trace[60+j]
        obs = compose_policy_observation(np.array(row['q']), np.array(row['executed_target']), 'qpos-target-residual')
        history = np.concatenate([history[1:], obs[None]])
    predictions = {}
    for scale in [0., 25., 50., 112.5]:
        controller.guidance_scale = scale
        controller.set_guidance_horizon(9)
        parts = []
        for j, history in zip(range(0, 75, 4), histories):
            controller.set_fixed_noise_from_seeds([44])
            ref = new_ref[np.minimum(np.arange(j, j+9), 74)]
            command, _ = controller.predict(history[None], ref[None])
            parts.append(command[0, :min(4, 75-j)])
        pred = np.concatenate(parts)
        predictions[str(scale)] = pred
        result['offline_fixed_observation'][str(scale)] = {
            'command_reference_rmse_rad': rmse(pred-new_ref),
            'max_abs_action_rad': float(abs(pred).max()),
        }
    archived = series['guided9_exec4'][1]
    replay_error = float(abs(predictions['25.0']-archived).max())
    result['offline_reconstruction_max_error_rad'] = replay_error
    assert replay_error < 1e-5, replay_error
    result['scope'] = 'Scale probes share archived guide9/exec4 observations and seed44; no new physics rollout, no success claim.'
    (OUT/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(OUT/'offline_predictions.npz', **predictions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for ax, label in zip(axs[:, 0], ['case110_g2e2', 'guided9_exec4']):
        ref, command, q = series[label]
        k = np.argmax(np.mean((command-ref)**2, axis=0))
        x = np.arange(len(ref))/30
        ax.plot(x, ref[:, k], label='Reference')
        ax.plot(x, command[:, k], label='Guided command')
        ax.plot(x, q[:, k], label='Simulated joint', alpha=.8)
        ax.set_title(label+' / '+names[k]); ax.set_ylabel('Joint target (rad)')
    ax = axs[0, 1]
    for label in ['case110_g2e2', 'guided', 'guided9', 'guided9_exec4']:
        ref, command, _ = series[label]
        ax.plot(np.arange(len(ref))/30, np.sqrt(np.mean((command-ref)**2, axis=1)), label=label)
    ax.set_title('Command-reference error'); ax.set_ylabel('RMSE over 22 joints (rad)')
    ax = axs[1, 1]
    for scale, pred in predictions.items():
        ax.plot(np.arange(75)/30, np.sqrt(np.mean((pred-new_ref)**2, axis=1)), label='Scale '+scale)
    ax.set_title('Fixed-observation probes; no physics rollout'); ax.set_ylabel('RMSE (rad)')
    for ax in axs.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_xlabel('Reference time (s)')
    fig.tight_layout(); fig.savefig(OUT/'tracking.png', dpi=160)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
