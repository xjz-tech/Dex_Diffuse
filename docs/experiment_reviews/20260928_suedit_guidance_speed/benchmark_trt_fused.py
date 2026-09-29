"""Isolated TRT FP16 + GPU DDIM comparison; no simulator or robot control.

Fused means precomputed GPU coefficients and no diagnostic host sync inside
sampling, as in eval/trt_unet.py. It does not mean a single CUDA kernel/graph.
Guidance uses the exact analytic gradient of the existing frozen-epsilon MSE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "eval"), str(ROOT)]
from benchmark import measured_input, ReferenceActionEditor, compose_policy_observation
from reference_action_editor import ddim_transition
from diffusion_policy.guidance.guided_ddim import guided_ddim_step
from trt_unet import compile_unet, TensorRTUnetAdapter

METHODS = {"sdedit_r020": (0, 0.0), "guidance_g2_s25": (2, 25.0),
           "guidance_g4_s50": (4, 50.0)}


def exclusive():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    others = [line.strip() for line in raw.splitlines() if line.strip() and line.strip() != str(os.getpid())]
    if others:
        raise RuntimeError(f"Other GPU compute PIDs {others}; timing is not exclusive")


def stats(values):
    v = np.asarray(values, dtype=np.float64)
    return dict(median_ms=float(np.median(v)), mean_ms=float(v.mean()),
                p95_ms=float(np.percentile(v, 95)), samples_ms=v.tolist())


class FastSampler:
    def __init__(self, editor):
        self.editor = editor
        self.c = editor.controller
        self.p = editor.policy
        scheduler = self.c.scheduler
        scheduler.set_timesteps(4, device=self.c.device)
        assert scheduler.config.clip_sample and scheduler.config.prediction_type == "epsilon"
        self.guidance_times = [int(t) for t in scheduler.timesteps]
        self.edit_steps = self.coeffs(editor.timesteps, editor.timesteps[1:] + [-1])
        self.guide_steps = self.coeffs(self.guidance_times, [t - 25 for t in self.guidance_times])

    def coeffs(self, times, next_times):
        alphas = self.c.scheduler.alphas_cumprod.to(self.c.device)
        rows = []
        for t, nxt in zip(times, next_times):
            at = alphas[t]
            an = alphas[nxt] if nxt >= 0 else torch.ones_like(at)
            rows.append((torch.tensor([t], device=self.c.device), at.sqrt(), (1-at).sqrt(), an.sqrt(), (1-an).sqrt()))
        return rows

    @staticmethod
    def update(x, eps, step, reference=None, scale=0.0):
        _, a, b, ap, bp = step
        raw = (x - b * eps) / a
        clean = raw.clamp(-1, 1)
        corrected = (x - a * clean) / b
        if reference is not None:
            n = reference.shape[1]
            grad = torch.zeros_like(x)
            grad[:, 3:3+n] = (2 * (raw[:, 3:3+n] - reference) / (n * 22)) / a
            corrected = corrected + b * scale * grad
            clean = (x - b * corrected) / a
        return ap * clean + bp * corrected

    @torch.no_grad()
    def predict(self, history, future, noise, method):
        h = torch.as_tensor(history, device=self.c.device, dtype=torch.float32)
        f = torch.as_tensor(future, device=self.c.device, dtype=torch.float32)
        cond = self.p.normalizer['obs'].normalize(h).reshape(1, -1)
        n, scale = METHODS[method]
        if n == 0:
            clean = self.p.normalizer['action'].normalize(torch.cat([h[:, 1:, 22:44], f], dim=1))
            _, a, b, _, _ = self.edit_steps[0]
            x = a * clean + b * noise
            for step in self.edit_steps:
                x[:, :3] = step[1] * clean[:, :3] + step[2] * noise[:, :3]
                eps = self.p.model(x, step[0], global_cond=cond)
                x = self.update(x, eps, step)
        else:
            ref = self.p.normalizer['action'].normalize(f[:, :n])
            x = noise.clone()
            for step in self.guide_steps:
                eps = self.p.model(x, step[0], global_cond=cond)
                x = self.update(x, eps, step, ref, scale)
        return self.p.normalizer['action'].unnormalize(x[:, 3:5])


def fixtures(editor):
    h, f, trace, ref_path = measured_input(editor.controller)
    values = [(h, f)]
    z = np.load(ref_path)
    q, target = z['hand_qpos_rad'][0], z['hand_target_rad'][0]
    for j in (40, 100):
        hist = compose_policy_observation(q[j-3:j+1], target[j-4:j], editor.controller.observation_mode)[None]
        values.append((hist, target[None, j:j+9]))
    result = []
    for index, (h, f) in enumerate(values):
        for seed in (44, 45, 46):
            editor.controller.set_fixed_noise_from_seeds([seed])
            result.append(dict(history=h, future=f, gpu_h=torch.as_tensor(h, device='cuda'),
                gpu_f=torch.as_tensor(f, device='cuda'), noise=editor.controller._fixed_noise.clone(),
                seed=seed, fixture=index))
    return result, dict(trace=trace, reference=ref_path, source_window_indices=["last_four_settle", 40, 100])


def original(editor, fixture, method):
    n, scale = METHODS[method]
    if not n:
        return editor.predict(fixture['history'], fixture['future'], [fixture['seed']])[0]
    c = editor.controller
    c.set_guidance_horizon(n)
    c.guidance_scale = scale
    c.set_fixed_noise_from_seeds([fixture['seed']])
    return c.predict(fixture['history'], fixture['future'][:, :n])[0]


def validate_fp32(fast, cases):
    report, baseline = {}, {}
    for name, (n, scale) in METHODS.items():
        errors, local = [], []
        baseline[name] = []
        for case in cases:
            ref = original(fast.editor, case, name)
            out = fast.predict(case['history'], case['future'], case['noise'], name).cpu().numpy()
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-3)
            errors.append(float(np.max(np.abs(out-ref))))
            baseline[name].append(out.copy())
            cond = fast.p.normalizer['obs'].normalize(case['gpu_h']).reshape(1, -1)
            target = fast.p.normalizer['action'].normalize(case['gpu_f'][:, :n]) if n else None
            for step in (fast.guide_steps if n else fast.edit_steps):
                x = case['noise'].clone()
                with torch.no_grad():
                    eps = fast.p.model(x, step[0], global_cond=cond)
                    left = fast.update(x, eps, step, target, scale)
                t = int(step[0].item())
                if n:
                    right = guided_ddim_step(fast.c.scheduler, eps, t, x.requires_grad_(True), target, scale, slice(3, 3+n)).prev_sample
                else:
                    times = fast.editor.timesteps
                    i = times.index(t)
                    at = fast.c.scheduler.alphas_cumprod[t].to('cuda')
                    an = fast.c.scheduler.alphas_cumprod[times[i+1]].to('cuda') if i+1 < len(times) else torch.ones_like(at)
                    right = ddim_transition(x, eps, at, an, True)[0]
                torch.testing.assert_close(left, right, rtol=1e-5, atol=2e-6)
                local.append(float((left-right).abs().max()))
        report[name] = dict(local_formula_max_abs=max(local), action_max_abs_rad=max(errors),
                            per_case_action_max_abs_rad=errors, passed=True)
    return report, baseline


def timing(fast, cases, repeats, warmup, rounds):
    result = {}
    rng = random.Random(20260928)
    for boundary in ('numpy_to_numpy', 'gpu_resident'):
        wall = {n: [] for n in METHODS}
        events = {n: [] for n in METHODS}
        per_round = []
        def call(case, name):
            if boundary == 'numpy_to_numpy':
                return fast.predict(case['history'], case['future'], case['noise'], name).cpu().numpy()
            return fast.predict(case['gpu_h'], case['gpu_f'], case['noise'], name)
        exclusive()
        with torch.inference_mode():
            for name in METHODS:
                for i in range(warmup):
                    call(cases[i % len(cases)], name)
            torch.cuda.synchronize()
            start_ev, end_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for rd in range(rounds):
                exclusive()
                current = {n: [] for n in METHODS}
                for i in range(repeats):
                    order = list(METHODS)
                    rng.shuffle(order)
                    case = cases[i % len(cases)]
                    for name in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        start_ev.record()
                        call(case, name)
                        end_ev.record()
                        end_ev.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                        wall[name].append(elapsed)
                        current[name].append(elapsed)
                        events[name].append(start_ev.elapsed_time(end_ev))
                exclusive()
                per_round.append({n: stats(v) for n, v in current.items()})
        result[boundary] = dict(wall={n: stats(v) for n, v in wall.items()},
            cuda_event_interval={n: stats(v) for n, v in events.items()}, rounds=per_round)
        print(boundary, {n: round(stats(v)['median_ms'], 4) for n, v in wall.items()}, flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=HERE / 'trt_fused_results.json')
    ap.add_argument('--repeats', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--rounds', type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(2)
    exclusive()
    checkpoint = '/home/carus/data_usb/10B_obs_4-66.ckpt'
    editor = ReferenceActionEditor(checkpoint, 0.2, steps=4, execution_steps=2)
    fast = FastSampler(editor)
    cases, sources = fixtures(editor)
    report = dict(status='running', hardware=torch.cuda.get_device_name(0), torch=torch.__version__,
        checkpoint=checkpoint, sources=sources, batch=1, ddim_steps=4, execution_steps=2,
        edit_metadata=editor.metadata, guidance_timesteps=fast.guidance_times,
        fixed_noise_seeds=[44,45,46], methods=METHODS, rounds=args.rounds,
        repeats_per_round=args.repeats, warmup=args.warmup,
        measurement='wall latency includes explicit CUDA synchronization; numpy-to-numpy includes input/output transfer; fixed noise cached; no simulator, IPC, model load, compilation, diagnostics or reference generation; no CUDA Graph',
        fusion='precomputed GPU coefficients; corrected epsilon after x0 clipping; analytic frozen-epsilon joint MSE guidance; PyTorch DDIM tensor arithmetic',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/'eval/trt_unet.py', ROOT/'diffusion_policy/guidance/guided_ddim.py']})
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        report['fp32_validation'], baseline = validate_fp32(fast, cases)
        print('FP32 formula and action checks passed', report['fp32_validation'], flush=True)
        report['eager_fp32_fused'] = timing(fast, cases, args.repeats, args.warmup, args.rounds)
        save()
        print('Compiling shared batch-1 TensorRT FP16 UNet', flush=True)
        started = time.perf_counter()
        compiled = compile_unet(fast.p.model, horizon=12, action_dim=22, global_cond_dim=264,
            device=torch.device('cuda:0'), fp16=True, max_batch=1)
        report['compile_seconds'] = time.perf_counter() - started
        fast.p.model = TensorRTUnetAdapter(compiled)
        import tensorrt, torch_tensorrt
        report['tensorrt'] = tensorrt.__version__
        report['torch_tensorrt'] = torch_tensorrt.__version__
        drift = {}
        with torch.inference_mode():
            for name in METHODS:
                deltas = []
                for index, case in enumerate(cases):
                    out = fast.predict(case['history'], case['future'], case['noise'], name).cpu().numpy()
                    assert np.isfinite(out).all()
                    deltas.append(np.abs(out - baseline[name][index]))
                delta = np.stack(deltas)
                drift[name] = dict(mean_abs_rad=float(delta.mean()), max_abs_rad=float(delta.max()))
        report['trt_fp16_action_drift_vs_fused_fp32'] = drift
        print('TRT drift', drift, flush=True)
        report['trt_fp16_fused'] = timing(fast, cases, args.repeats, args.warmup, args.rounds)
        report['status'] = 'complete'
        save()
        print('COMPLETE', args.output, flush=True)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
