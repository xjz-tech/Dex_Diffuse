"""CPU-only checks. No checkpoint load, physics, or policy forward pass."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from noise_stream import AlignedNoiseStream
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '20260924_object_state_data'))
from reference_action_editor import ReferenceActionEditor


class NoiseStreamTests(unittest.TestCase):
    def test_rho_zero_is_fresh_and_first_draw_is_legacy(self):
        stream = AlignedNoiseStream(486266, 0.)
        generator = torch.Generator().manual_seed(486266)
        draws = []
        for j in range(0, 8, 2):
            noise, xi = stream.draw(j)
            expected = torch.randn((1, 8, 22), generator=generator)
            self.assertTrue(torch.equal(noise, expected))
            self.assertTrue(torch.equal(noise, xi))
            draws.append(noise)
        self.assertFalse(torch.equal(draws[0], draws[1]))

    def test_alignment_formula_and_paired_innovations(self):
        streams = [AlignedNoiseStream(77, rho) for rho in (0., .5, .8, 1.)]
        previous = [None] * 4
        for j in range(0, 20, 2):
            baseline_xi = None
            for i, stream in enumerate(streams):
                noise, xi = stream.draw(j)
                if baseline_xi is None:
                    baseline_xi = xi
                self.assertTrue(torch.equal(xi, baseline_xi))
                self.assertTrue(torch.equal(noise[:, :3], xi[:, :3]))
                self.assertTrue(torch.equal(noise[:, 6:], xi[:, 6:]))
                if previous[i] is not None:
                    expected = stream.rho * previous[i][:, 5:8] + (1 - stream.rho**2)**.5 * xi[:, 3:6]
                    torch.testing.assert_close(noise[:, 3:6], expected, rtol=0, atol=0)
                previous[i] = noise.clone()
                noise.zero_()  # The returned tensor must not mutate cached noise.

    def test_reset_and_invalid_requests(self):
        for rho in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                AlignedNoiseStream(1, rho)
        s = AlignedNoiseStream(1, .8)
        first, _ = s.draw(0)
        for bad in (0, 1, 4):
            with self.assertRaises(ValueError):
                s.draw(bad)
        fresh, _ = AlignedNoiseStream(1, .8).draw(0)
        self.assertTrue(torch.equal(first, fresh))
        s.draw(2)

    def test_gaussian_marginal_and_aligned_correlation(self):
        for rho in (0., .5, .8, 1.):
            stream = AlignedNoiseStream(91234, rho)
            draws = torch.cat([stream.draw(j)[0] for j in range(0, 4000, 2)])
            values = draws[:, 3:]
            self.assertLess(abs(values.mean().item()), .02)
            self.assertLess(abs(values.var().item() - 1), .03)
            corr = np.corrcoef(draws[1:, 3:6].flatten(), draws[:-1, 5:8].flatten())[0, 1]
            self.assertLess(abs(corr - rho), .02)


class IdentityNormalizer:
    def normalize(self, value):
        return value

    def unnormalize(self, value):
        return value


def fake_editor():
    editor = ReferenceActionEditor.__new__(ReferenceActionEditor)
    editor.future_steps, editor.execution_steps, editor.noise_ratio = 5, 2, .15
    editor.timesteps = [8, 5, 3, 0]
    editor.policy = SimpleNamespace(dtype=torch.float32,
        normalizer={'action': IdentityNormalizer(), 'obs': IdentityNormalizer()})
    samples = []
    controller = SimpleNamespace(device=torch.device('cpu'),
        scheduler=SimpleNamespace(alphas_cumprod=torch.linspace(.99, .8, 10),
                                  config=SimpleNamespace(clip_sample=False)))
    def predict_epsilon(sample, timestep, global_cond):
        samples.append((int(timestep), sample.clone()))
        return torch.zeros_like(sample)
    def set_seed(seeds):
        generator = torch.Generator().manual_seed(seeds[0])
        controller.noise = torch.randn((1, 8, 22), generator=generator)
    controller.set_fixed_noise_from_seeds = set_seed
    controller._noise = lambda batch, dtype: controller.noise.clone()
    controller._predict_epsilon = predict_epsilon
    editor.controller = controller
    return editor, samples


class EditorIntegrationTests(unittest.TestCase):
    def test_legacy_path_and_external_noise_are_identical(self):
        editor, _ = fake_editor()
        history = np.zeros((1, 4, 66), dtype=np.float32)
        future = np.full((1, 5, 22), .2, dtype=np.float32)
        legacy, old_stats = editor.predict(history, future, [48], return_full_plan=True)
        noise = AlignedNoiseStream(48, .8).draw(0)[0]
        explicit, stats = editor.predict(history, future, [48], return_full_plan=True, initial_noise=noise)
        np.testing.assert_array_equal(legacy, explicit)
        self.assertEqual(old_stats, stats)
        prefix, _ = editor.predict(history, future, [48], initial_noise=noise)
        np.testing.assert_array_equal(prefix, explicit[:, :2])

    def test_rebuilds_current_reference_and_masks_history_each_timestep(self):
        editor, samples = fake_editor()
        history = np.full((1, 4, 66), .3, dtype=np.float32)
        future = np.full((1, 5, 22), .1, dtype=np.float32)
        stream = AlignedNoiseStream(50, 1.)
        for j in (0, 2):
            noise, _ = stream.draw(j)
            future += .15
            samples.clear()
            editor.predict(history, future, [50], initial_noise=noise)
            alpha = editor.controller.scheduler.alphas_cumprod[8]
            expected = alpha.sqrt() * torch.as_tensor(future) + (1-alpha).sqrt() * noise[:, 3:]
            torch.testing.assert_close(samples[0][1][:, 3:], expected)
            for timestep, sample in samples:
                a = editor.controller.scheduler.alphas_cumprod[timestep]
                known = a.sqrt() * torch.as_tensor(history[:, 1:, 22:44]) + (1-a).sqrt() * noise[:, :3]
                torch.testing.assert_close(sample[:, :3], known)

    def test_invalid_noise_and_identity_baseline(self):
        editor, _ = fake_editor()
        h, f = np.zeros((1, 4, 66), np.float32), np.zeros((1, 5, 22), np.float32)
        for invalid in (torch.zeros(1, 5, 22), torch.full((1, 8, 22), float('nan'))):
            with self.assertRaises(ValueError):
                editor.predict(h, f, [1], initial_noise=invalid)
        editor.noise_ratio = 0
        f[:] = 4.5
        plan, _ = editor.predict(h, f, [1], return_full_plan=True)
        np.testing.assert_array_equal(plan, f)


if __name__ == '__main__':
    unittest.main(verbosity=2)
