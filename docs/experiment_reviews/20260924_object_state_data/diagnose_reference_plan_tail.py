"""Offline common-prefix diagnostics; these references never enter the policy."""
import json
import numpy as np
from run_four_reference_plan_tail_sweep import O, R, EPISODES, RATIOS, STEPS, folder
from run_four_reference_edit_noise_ddim_sweep import folder as archived_folder
from run_four_episode_reference_edit_adaptive010_video import folder as older_folder
from reference_resampling import interpolate_large_jumps


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def main():
    results = []
    for ep in EPISODES:
        ref, _ = interpolate_large_jumps(np.load(R / f'qualified_comparison/episode_{ep:02d}/reference_full.npz')['hand_target_rad'], .1)
        ref = ref[0]
        for ratio in RATIOS:
            for steps in STEPS:
                new_dir = folder(ep, ratio, steps)
                old_dir = older_folder(ep, f'edit{int(round(ratio*100)):03d}', 44) if steps == 4 and ratio in (.1, .2) else archived_folder(ep, ratio, steps)
                pred = json.loads((new_dir / 'predictions.json').read_text())
                new_trace = json.loads((new_dir / 'trace.json').read_text())
                old_trace = json.loads((old_dir / 'trace.json').read_text())
                new_cmd = np.asarray([x['command'] for x in new_trace if x['phase'] == 'action'])[:60]
                old_cmd = np.asarray([x['command'] for x in old_trace if x['phase'] == 'action'])[:60]
                rows = []
                for x in pred:
                    j = x['reference_index']
                    if j >= 60:
                        break
                    tail = np.asarray(x['initialized_tail'])[0]
                    truth = ref[np.minimum(np.arange(j+4, j+9), len(ref)-1)]
                    rows.append(dict(reference_index=j,
                        initialized_tail_vs_unused_reference_rmse_rad=rmse(tail-truth),
                        reference_tail_boundary_max_abs_rad=float(np.max(np.abs(tail[0]-ref[min(j+3,len(ref)-1)]))),
                        reference_tail_boundary_rmse_rad=rmse(tail[0]-ref[min(j+3,len(ref)-1)])))
                results.append(dict(episode=ep, noise_ratio=ratio, ddim_steps=steps,
                    common_prefix_control_steps=60,
                    new_command_reference_rmse_rad=rmse(new_cmd-ref[:60]),
                    old_command_reference_rmse_rad=rmse(old_cmd-ref[:60]),
                    initial_plan_only_command_delta_rmse_rad=rmse(new_cmd[:2]-old_cmd[:2]),
                    tail_diagnostics=rows))
    (O / 'TRACE_DIAGNOSTICS.json').write_text(json.dumps(dict(
        note='Offline comparisons only. Tail has no tracking obligation; its deviation is not itself a failure metric. First 60 control steps are fixed across methods and may include separation.',
        results=results), indent=2)+'\n')
    print('WROTE TRACE_DIAGNOSTICS.json', len(results))


if __name__ == '__main__':
    main()
