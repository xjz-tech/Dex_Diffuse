"""Validate time alignment, caching, conditioning and unchanged SDEdit calls."""
import json
import numpy as np
import torch
from reference_plan_tail_editor import ReferencePlanTailEditor, initialize_future
from reference_action_editor import ReferenceActionEditor
from server_reference_plan_tail_editor import reference_window4
from reference_resampling import interpolate_large_jumps
from run_four_reference_plan_tail_sweep import O, R

O.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
case = R / 'qualified_comparison/episode_54'
ref, _ = interpolate_large_jumps(np.load(case / 'reference_full.npz')['hand_target_rad'], .1)
a = reference_window4(ref, [0], 0)
changed = ref.copy(); changed[:, 4:] = 12345.
b = reference_window4(changed, [0], 0)
assert np.array_equal(a, b)
sentinel = np.repeat(np.arange(9, dtype=np.float32)[None, :, None], 22, axis=2)
future, indices = initialize_future(a, sentinel)
assert indices.tolist() == [6, 7, 8, 8, 8]
assert np.array_equal(future[:, 4:], sentinel[:, [6, 7, 8, 8, 8]])
trace = json.loads((case / 'direct_m044_mu11/trace.json').read_text())
q = np.asarray(trace[59]['q'], np.float32)
target = np.asarray(trace[59]['command'], np.float32)
h = np.repeat(np.concatenate([q, target, target-q])[None, None], 4, axis=1)
records = []
for ratio, steps in [(.1,4), (.1,6), (.15,4), (.15,6), (.2,4), (.2,6)]:
    editor = ReferencePlanTailEditor('/home/carus/data_usb/10B_obs_4-66.ckpt', ratio, steps, 2)
    c = editor.controller; original = c._predict_epsilon
    calls = []; max_mask = 0.
    clean = torch.as_tensor(h[:, 1:, 22:44], device=c.device, dtype=editor.policy.dtype)
    clean = editor.policy.normalizer['action'].normalize(clean)
    c.set_fixed_noise_from_seeds([44]); noise = c._noise(1, editor.policy.dtype)
    def inspect(sample, timestep, global_cond):
        global max_mask
        t = int(timestep); calls.append(t)
        alpha = c.scheduler.alphas_cumprod[t].to(device=c.device, dtype=sample.dtype)
        expected = alpha.sqrt()*clean + (1-alpha).sqrt()*noise[:, :3]
        max_mask = max(max_mask, float((sample[:, :3]-expected).abs().max()))
        return original(sample, timestep, global_cond=global_cond)
    c._predict_epsilon = inspect
    output, stats = editor.predict(h, a, [44], 0, [0])
    assert calls == editor.timesteps and max_mask == 0.
    assert stats['network_calls'] == steps and stats['tail_source'] == 'hold_bootstrap'
    full = np.asarray(stats['generated_future_plan'], np.float32)
    assert np.array_equal(output, full[:, :2])
    assert np.max(np.abs(output-a[:, :2])) > 1e-5
    calls.clear()
    next_ref = reference_window4(ref, [0], 2)
    _, next_stats = editor.predict(h, next_ref, [44], 2, [0])
    assert calls == editor.timesteps
    assert np.array_equal(np.asarray(next_stats['initialized_tail'], np.float32), full[:, [6,7,8,8,8]])
    assert next_stats['previous_plan_reference_index'] == 0
    c._predict_epsilon = original
    repeat, _ = editor.predict(h, b, [44], 0, [0])
    assert np.array_equal(output, repeat)
    # The optional full-plan return leaves the archived editor's default output unchanged.
    init, _ = initialize_future(a, None)
    default, _ = ReferenceActionEditor.predict(editor, h, init, [44])
    all9, _ = ReferenceActionEditor.predict(editor, h, init, [44], return_full_plan=True)
    assert np.array_equal(default, all9[:, :2]) and np.array_equal(output, default)
    for ref_arg, j, ids in [(ref[:, :9], 2, [0]), (next_ref, 4, [0]), (next_ref, 2, [1])]:
        try:
            editor.predict(h, ref_arg, [44], j, ids)
        except AssertionError:
            pass
        else:
            raise AssertionError('invalid reference width, cache index, or episode accepted')
    records.append(dict(ratio=ratio, steps=steps, timesteps=editor.timesteps,
        network_calls=steps, history_mask_max_error=max_mask,
        deterministic_reset=True, future_reference_leakage_check=True,
        cache_alignment_check=True, base_default_output_unchanged=True,
        first_command_rmse_rad=stats['executed_prefix_edit_rmse_rad']))
    print('VALIDATED', ratio, steps, editor.timesteps, flush=True)
    del editor, c
    torch.cuda.empty_cache()
(O / 'validation.json').write_text(json.dumps(dict(passed=True, records=records), indent=2)+'\n')
