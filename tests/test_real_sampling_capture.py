"""CPU capture/replay contracts; no checkpoints, TensorRT, or hand hardware."""
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from diffusers import DDIMScheduler, DDPMScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval" / "real"))
from guided_policy import GuidedRealPolicy
from sim_hand_policy import DiffusionUnetLowdimPolicy, SingleFieldLinearNormalizer
from trt_unet import fused_conditional_sample


class ToyModel(torch.nn.Module):
    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        condition = global_cond if global_cond is not None else local_cond.flatten(1)
        return sample * 0.1 + condition.mean(1).reshape(-1, 1, 1) * 0.01


class AlternateStepDDIM(DDIMScheduler):
    """Exercise the fallback when a step has no directly interceptable draw."""
    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)


def policy(scheduler=None, local=False, scale=2.0, offset=0.25, steps=4):
    result = DiffusionUnetLowdimPolicy(
        ToyModel(), scheduler or DDIMScheduler(num_train_timesteps=16),
        horizon=8, obs_dim=3, action_dim=2, n_action_steps=3, n_obs_steps=2,
        num_inference_steps=steps, obs_as_local_cond=local,
        obs_as_global_cond=not local, oa_step_convention=True,
    )
    for key, width in (("obs", 3), ("action", 2)):
        result.normalizer[key] = SingleFieldLinearNormalizer.create_manual(
            torch.full((width,), scale), torch.full((width,), offset), {}
        )
    return result


class SamplingCaptureTest(unittest.TestCase):
    def assert_snapshot(self, value):
        if isinstance(value, torch.Tensor):
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.device.type, "cpu")
        elif isinstance(value, dict):
            for item in value.values():
                self.assert_snapshot(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.assert_snapshot(item)

    def test_plain_and_fused_capture_preserves_outputs_and_rng(self):
        # Catches extra noise draws, partial trajectories, aliases, wrong conditioning.
        obs = {"obs": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)}
        for fused in (False, True):
            for local in (False, True):
                with self.subTest(fused=fused, local=local):
                    current = policy(local=local)
                    if fused:
                        current.conditional_sample = MethodType(fused_conditional_sample, current)
                    torch.manual_seed(91)
                    expected = current.predict_action(obs)
                    expected_next = torch.randn(5)
                    torch.manual_seed(91)
                    current._debug_capture = capture = {}
                    actual = current.predict_action(obs)
                    torch.testing.assert_close(torch.randn(5), expected_next, rtol=0, atol=0)
                    torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
                    self.assertIn("normalized_obs", capture)
                    torch.testing.assert_close(capture["normalized_obs"], obs["obs"] * 2 + 0.25)
                    self.assertEqual(tuple(capture["normalized_trajectory"].shape), (2, 8, 2))
                    torch.testing.assert_close(capture["action_pred"], actual["action_pred"])
                    torch.testing.assert_close(capture["action_pred"][:, 1:4], actual["action"])
                    torch.testing.assert_close(capture["normalized_trajectory"], actual["action_pred"] * 2 + 0.25)
                    torch.testing.assert_close(capture["timesteps"], torch.tensor([12, 8, 4, 0]))
                    self.assertEqual(capture["initial_noise"].shape, (2, 8, 2))
                    self.assert_snapshot(capture)
                    if local:
                        torch.testing.assert_close(capture["local_cond"][:, :2], obs["obs"] * 2 + 0.25)
                        self.assertEqual(torch.count_nonzero(capture["local_cond"][:, 2:]), 0)
                    else:
                        torch.testing.assert_close(capture["global_cond"], (obs["obs"] * 2 + 0.25).flatten(1))
                    saved = capture["action_pred"].clone()
                    actual["action_pred"].detach().zero_()
                    torch.testing.assert_close(capture["action_pred"], saved)

    def test_actual_scheduler_draws_match_stream_and_replay_ddim(self):
        # Catches omission of DDPM variance or stochastic DDIM scheduler noise.
        for kind in ("ddpm", "ddim"):
            with self.subTest(kind=kind):
                scheduler = (DDPMScheduler if kind == "ddpm" else DDIMScheduler)(num_train_timesteps=16)
                current = policy(scheduler)
                original_step = scheduler.step.__func__
                original_randn = original_step.__globals__["randn_tensor"]
                if kind == "ddim":
                    current.scheduler_step_kwargs["eta"] = 0.5
                obs = {"obs": torch.ones(1, 2, 3)}
                torch.manual_seed(19)
                expected = current.predict_action(obs)
                expected_next = torch.randn(3)
                torch.manual_seed(19)
                current._debug_capture = capture = {}
                actual = current.predict_action(obs)
                self.assertIs(scheduler.step.__func__, original_step)
                self.assertIs(original_step.__globals__["randn_tensor"], original_randn)
                torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
                torch.testing.assert_close(torch.randn(3), expected_next, rtol=0, atol=0)
                torch.manual_seed(19)
                self.assertIn("initial_noise", capture)
                torch.testing.assert_close(capture["initial_noise"], torch.randn(1, 8, 2), rtol=0, atol=0)
                self.assertEqual(capture["scheduler_capture_method"], "randn_tensor")
                self.assertEqual(len(capture["scheduler_noise"]), 4)
                for index, draws in enumerate(capture["scheduler_noise"]):
                    count = 0 if kind == "ddpm" and index == 3 else 1
                    self.assertEqual(len(draws), count)
                    for draw in draws:
                        torch.testing.assert_close(draw, torch.randn(1, 8, 2), rtol=0, atol=0)
                if kind == "ddim":
                    trajectory = capture["initial_noise"].clone()
                    for t, draws in zip(capture["timesteps"], capture["scheduler_noise"]):
                        epsilon = current.model(trajectory, t, global_cond=capture["global_cond"])
                        trajectory = scheduler.step(epsilon, t, trajectory, eta=0.5, variance_noise=draws[0]).prev_sample
                    torch.testing.assert_close(trajectory, capture["normalized_trajectory"], rtol=0, atol=0)

    def test_fallback_states_replay_stochastic_steps_with_explicit_generator(self):
        # Catches snapshots taken after the draw or from the wrong generator.
        for explicit in (False, True):
            current = policy(AlternateStepDDIM(num_train_timesteps=16))
            generator = torch.Generator().manual_seed(65) if explicit else None
            current.scheduler_step_kwargs.update(eta=0.5, generator=generator)
            obs = {"obs": torch.ones(1, 2, 3)}
            torch.manual_seed(42)
            expected = current.predict_action(obs)
            expected_next = torch.randn(3)
            expected_generator_next = torch.randn(3, generator=generator)
            torch.manual_seed(42)
            if generator is not None:
                generator.manual_seed(65)
            current._debug_capture = capture = {}
            actual = current.predict_action(obs)
            torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
            torch.testing.assert_close(torch.randn(3), expected_next, rtol=0, atol=0)
            torch.testing.assert_close(torch.randn(3, generator=generator), expected_generator_next, rtol=0, atol=0)
            self.assertEqual(capture["scheduler_capture_method"], "pre_step_rng_state")
            self.assertEqual(len(capture["scheduler_rng_states"]), 4)
            trajectory = capture["initial_noise"].clone()
            for timestep, state in zip(capture["timesteps"], capture["scheduler_rng_states"]):
                torch.set_rng_state(state["cpu"])
                if generator is not None:
                    generator.set_state(state["generators"][0]["state"])
                epsilon = current.model(trajectory, timestep, global_cond=capture["global_cond"])
                trajectory = current.noise_scheduler.step(
                    epsilon, timestep, trajectory, eta=0.5, generator=generator).prev_sample
            torch.testing.assert_close(trajectory, capture["normalized_trajectory"], rtol=0, atol=0)

    def test_supplied_variance_noise_is_cloned_without_random_draws(self):
        current = policy()
        noise = torch.full((1, 8, 2), 0.75)
        current.scheduler_step_kwargs.update(eta=0.5, variance_noise=noise)
        current._debug_capture = capture = {}
        current.predict_action({"obs": torch.ones(1, 2, 3)})
        self.assertEqual(capture["scheduler_noise"], [[], [], [], []])
        noise.zero_()
        torch.testing.assert_close(capture["scheduler_supplied_variance_noise"], torch.full((1, 8, 2), 0.75))

    def test_guided_full_reference_and_rng_parity(self):
        # Catches guide/prior normalizer mixups and changes to fixed-noise caches.
        obs = {"obs": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)}
        for fused in (False, True):
            for fixed in (False, True):
                with self.subTest(fused=fused, fixed=fixed):
                    args = SimpleNamespace(action_chunk_steps=2, inference_steps=4,
                        guide_inference_steps=2, guidance_steps=3, guidance_scale=2.0,
                        seed=21, guide_seed=32, fixed_noise=fixed, fused_ddim=fused)
                    plain = GuidedRealPolicy(policy(), policy(scale=4., offset=-0.5, steps=2), args, "cpu")
                    captured = GuidedRealPolicy(policy(), policy(scale=4., offset=-0.5, steps=2), args, "cpu")
                    first_noise = None
                    for _ in range(2):
                        rng_before = torch.get_rng_state().clone()
                        with torch.inference_mode():
                            expected = plain.predict_action(obs)
                            captured._debug_capture = capture = {}
                            actual = captured.predict_action(obs)
                        torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
                        torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)
                        for a, b in zip(plain.generators, captured.generators):
                            torch.testing.assert_close(a.get_state(), b.get_state(), rtol=0, atol=0)
                        self.assertIn("action_pred", capture)
                        self.assertEqual(capture["action_pred"].shape, (1, 8, 2))
                        torch.testing.assert_close(actual["action_pred"], capture["action_pred"])
                        torch.testing.assert_close(capture["action_pred"][:, 1:3], actual["action"])
                        torch.testing.assert_close(capture["normalized_trajectory"], capture["action_pred"] * 2 + 0.25)
                        torch.testing.assert_close(capture["guide_normalized_trajectory"], capture["guide_action_pred"] * 4 - 0.5)
                        torch.testing.assert_close(capture["guide_reference_physical"], capture["guide_action_pred"][:, 1:4])
                        torch.testing.assert_close(capture["guide_reference_normalized"], capture["guide_reference_physical"] * 2 + 0.25)
                        torch.testing.assert_close(capture["prior_normalized_obs"], obs["obs"] * 2 + 0.25)
                        torch.testing.assert_close(capture["guide_normalized_obs"], obs["obs"] * 4 - 0.5)
                        torch.testing.assert_close(capture["prior_timesteps"], torch.tensor([12, 8, 4, 0]))
                        torch.testing.assert_close(capture["guide_timesteps"], torch.tensor([8, 0]))
                        self.assertEqual(capture["guidance_window"], {"start": 1, "stop": 4})
                        self.assert_snapshot(capture)
                        if first_noise is not None:
                            self.assertEqual(torch.equal(first_noise, capture["prior_initial_noise"]), fixed)
                        first_noise = capture["prior_initial_noise"].clone()


if __name__ == "__main__":
    unittest.main()
