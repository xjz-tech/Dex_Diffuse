"""Same-state/noise counterfactuals; no simulation or production code changes."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ASTRA = ROOT / '.worktrees/Astra-controller'
sys.path[:0] = [str(ASTRA), str(ASTRA / 'eval')]
from inference_dp_controller import GuidedDDIMController
from diffusion_policy.guidance.guided_ddim import guided_ddim_step

RUN = ASTRA / 'outputs/astra_scale50_20260925/no_gait_e3577_n48'
meta = json.loads((RUN / 'model.json').read_text())
torch.manual_seed(meta['seed'])
c = GuidedDDIMController(Path(meta['checkpoint']), torch.device('cuda:0'),
    inference_steps=4, execution_steps=2, guidance_scale=50., eta=0.,
    fixed_noise=False, seed=meta['noise_seed'], allow_salvage=False)
c.set_guidance_horizon(9)
variants = {'g9_s0': (9, 0), 'g9_s50': (9, 50), 'g2_s25': (2, 25),
            'g2_s50': (2, 50), 'g2_s11.111': (2, 100/9), 'g9_s112.5': (9, 112.5)}
commands = {key: [] for key in variants}
references = []
reproduction = []
# All 35 consecutive archived windows in the common first-70-step interval.
for i in range(35):
    data = np.load(RUN / f'prediction_{2*i:06d}.npz')
    before = c.generator.get_state()
    after = None
    for key, (h, scale) in variants.items():
        c.generator.set_state(before)
        c.set_guidance_horizon(h)
        c.guidance_scale = scale
        cmd, _ = c.predict(data['observation'], data['reference'])
        commands[key].append(cmd[0])
        if after is None:
            after = c.generator.get_state()
        else:
            assert torch.equal(after, c.generator.get_state())
        if key == 'g9_s50':
            error = float(np.abs(cmd - data['prediction']).max())
            reproduction.append(error)
            np.testing.assert_allclose(cmd, data['prediction'], atol=1e-6, rtol=1e-5)
    c.generator.set_state(after)
    references.append(data['reference'][0])
    if (i+1) % 5 == 0:
        print(f'{i+1}/35 paired windows', flush=True)
commands = {key: np.asarray(value, dtype=np.float64) for key, value in commands.items()}
refs = np.asarray(references, dtype=np.float64)
def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))

metrics = {}
for group, selection in [('all_first70', slice(None)), ('left_command_steps9to70', slice(4,None))]:
    metrics[group] = {}
    for key, cmd in commands.items():
        a = cmd[selection]
        pure = commands['g9_s0'][selection]
        ref = refs[selection,:2]
        metrics[group][key] = dict(reference_rmse_rad=rms(a-ref),
            change_from_unguided_rmse_rad=rms(a-pure),
            action1_change_rad=rms((a-pure)[:,0]), action2_change_rad=rms((a-pure)[:,1]),
            projected_reference_correction=float(np.sum((a-pure)*(ref-pure))/np.sum((ref-pure)**2)))

# Exact sampler coefficients and direct numerical verification with frozen epsilon.
c.scheduler.set_timesteps(4, device=c.device)
coefficients = []
max_formula_error = 0.
for t in c.scheduler.timesteps:
    ti = int(t)
    a = float(c.scheduler.alphas_cumprod[ti])
    prev = ti - c.scheduler.config.num_train_timesteps//4
    ap = float(c.scheduler.alphas_cumprod[prev] if prev>=0 else c.scheduler.final_alpha_cumprod)
    factor = np.sqrt(ap)*(1-a)/a - np.sqrt(1-ap)*np.sqrt((1-a)/a)
    row = dict(t=ti, alpha=a, alpha_prev=ap, coefficients={})
    for key, (h,s) in variants.items():
        k = 2*s/(22*h)
        row['coefficients'][key] = dict(loss_gradient_coefficient=k,
            x0_correction_coefficient=k*(1-a)/a, prev_sample_correction_coefficient=k*factor)
        xt = torch.full((1,12,22), .04, device=c.device, requires_grad=True)
        eps = torch.full_like(xt, .02)
        r = torch.full((1,h,22), .1, device=c.device)
        sl = slice(3,3+h)
        result = guided_ddim_step(c.scheduler,eps,ti,xt,r,s,sl)
        raw = (xt.detach()-np.sqrt(1-a)*eps)/np.sqrt(a)
        expected = result.base_prev_sample.clone()
        expected[:,sl] -= k*factor*(raw[:,sl]-r)
        max_formula_error = max(max_formula_error,float((expected-result.prev_sample).abs().max()))
    coefficients.append(row)

def variation(q):
    d = np.diff(q,axis=0)
    return dict(steps=len(q), mean_step_joint_rms_rad=float(np.sqrt(np.mean(d*d,axis=1)).mean()),
        global_step_rmse_rad=rms(d), step_rms_p95_rad=float(np.quantile(np.sqrt(np.mean(d*d,axis=1)),.95)),
        max_single_joint_step_rad=float(np.abs(d).max()))
real = np.load(ROOT/'docs/experiment_reviews/20260924_object_state_data/reference/reference.npz')['hand_target_rad'][1]
assert real.shape == (75,22)
expanded = np.empty((149,22)); expanded[::2]=real; expanded[1::2]=(real[:-1]+real[1:])/2
rows = [json.loads(s) for s in (RUN/'trajectory.jsonl').read_text().splitlines()][:70]
astra = np.asarray([r['astra_reference'] for r in rows])
smoothness = {'astra_first70': variation(astra), 'real_ep53_75': variation(real),
              'real_ep53_insert1_149': variation(expanded)}
np.savez_compressed(OUT/'paired_commands.npz',reference9=refs,**commands)
result = dict(source=str(RUN), windows=35, physics_advanced=False,
    scope='Counterfactual sampling at archived guide9/scale50 states, paired original fresh noise; not closed-loop task success.',
    checkpoint=meta['checkpoint'], spec=c.spec, scheduler=dict(c.scheduler.config),
    max_historical_reproduction_error_rad=max(reproduction), max_formula_error=max_formula_error,
    metrics=metrics, coefficients=coefficients, reference_variation=smoothness,
    matched_coefficient_horizon_effect_rmse_rad=rms(commands['g9_s50']-commands['g2_s11.111']))
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
