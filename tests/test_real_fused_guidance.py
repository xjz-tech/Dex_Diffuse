"""Numerical parity for real-hand analytic guidance; no hardware or TRT needed."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval" / "real"))

from diffusion_policy.guidance.guided_ddim import guided_ddim_step
from guided_policy import GuidedRealPolicy
from trt_unet import ExportableGroupNorm, default_engine_path, fused_ddim_coeffs_from_scheduler, fused_guided_ddim_update


class ToyModel(torch.nn.Module):
    def forward(self, sample, timestep, global_cond=None):
        # Depend on the input so accumulated trajectory errors are exercised.
        return sample * 0.1 + global_cond.mean(dim=1).reshape(-1, 1, 1) * 0.01


class FieldNormalizer:
    def __init__(self, scale):
        self.scale = scale

    def normalize(self, value):
        return value * self.scale

    def unnormalize(self, value):
        return value / self.scale


class ToyPolicy(torch.nn.Module):
    def __init__(self, steps, scale):
        super().__init__()
        self.model = ToyModel()
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=True)
        self.normalizer = {key: FieldNormalizer(scale) for key in ("obs", "action")}
        self.n_obs_steps, self.obs_dim, self.action_dim = 4, 66, 22
        self.horizon, self.n_action_steps, self.num_inference_steps = 12, 5, steps


class FusedGuidanceTest(unittest.TestCase):
    def test_export_group_norm_preserves_learned_affine(self):
        torch.manual_seed(12)
        for affine in (False, True):
            for length in (3, 6, 12):
                with self.subTest(affine=affine, length=length):
                    original = torch.nn.GroupNorm(8, 32, eps=1e-5, affine=affine)
                    if affine:
                        with torch.no_grad():
                            original.weight.normal_()
                            original.bias.normal_()
                    sample = torch.randn(2, 32, length)
                    torch.testing.assert_close(
                        ExportableGroupNorm(original)(sample), original(sample),
                        atol=2e-6, rtol=2e-6,
                    )

    def test_trt_loader_compiles_both_components_at_batch_one(self):
        import inference_real

        with patch.object(sys, "argv", [
            "inference_real.py", "--checkpoint", __file__, "--guide-checkpoint", __file__,
            "--tensorrt", "--no-warmup",
        ]):
            args = inference_real.parse_args()
        with patch.dict(sys.modules, {"torch_tensorrt": SimpleNamespace()}):
            inference_real.validate_args(args)
        prior, guide = ToyPolicy(4, 1.2), ToyPolicy(4, 0.8)
        policy = GuidedRealPolicy(prior, guide, args, "cpu")
        loaded = SimpleNamespace(weight_source="test", global_step=0, epoch=0)
        with patch("guided_policy.load_guided_policy", return_value=(loaded, policy, {"obs_dim": 66})), \
             patch("trt_unet.accelerate_policy_unet") as compile_unet, \
             patch.object(inference_real, "normalizer_observation_mean", return_value=torch.zeros(66).numpy()):
            result = inference_real.load_inference_policy(args, torch.device("cpu"))
        self.assertIs(result, policy)
        self.assertTrue(result.fused_ddim)
        self.assertEqual(compile_unet.call_args_list, [
            call(prior, fp16=True, max_batch=1,
                 engine_path=default_engine_path(args.checkpoint), source_path=args.checkpoint),
            call(guide, fp16=True, max_batch=1,
                 engine_path=default_engine_path(args.guide_checkpoint), source_path=args.guide_checkpoint),
        ])

    def test_every_step_matches_autograd(self):
        torch.manual_seed(8)
        for clip in (False, True):
            for scale in (0.0, 0.5, 25.0):
                for batch in (1, 3):
                    for window in (slice(3, 5), slice(3, 12)):
                        with self.subTest(clip=clip, scale=scale, batch=batch, window=window):
                            scheduler = DDIMScheduler(num_train_timesteps=100, clip_sample=clip)
                            scheduler.set_timesteps(8)
                            sample = torch.randn(batch, 12, 22) * 2
                            epsilon = torch.randn_like(sample)
                            reference = torch.randn_like(sample[:, window])
                            coeffs = fused_ddim_coeffs_from_scheduler(scheduler, sample.device, sample.dtype)
                            for index, timestep in enumerate(scheduler.timesteps):
                                official = guided_ddim_step(
                                    scheduler, epsilon, timestep, sample.clone().requires_grad_(True),
                                    reference, scale, window,
                                ).prev_sample
                                with torch.inference_mode():
                                    actual = fused_guided_ddim_update(
                                        sample, epsilon, index, coeffs, reference, scale, window,
                                    )
                                torch.testing.assert_close(actual, official, atol=1e-5, rtol=1e-5)

    def test_full_policy_parity_and_noise_reset(self):
        for fixed_noise in (0, 1):
            for scale in (0.0, 25.0):
                with self.subTest(fixed_noise=fixed_noise, scale=scale):
                    args = SimpleNamespace(
                        action_chunk_steps=2, inference_steps=8, guide_inference_steps=4,
                        guidance_steps=2, guidance_scale=scale, seed=42, guide_seed=43,
                        fixed_noise=fixed_noise, fused_ddim=False,
                    )
                    eager = GuidedRealPolicy(ToyPolicy(8, 1.2), ToyPolicy(4, 0.8), args, "cpu")
                    fused = GuidedRealPolicy(ToyPolicy(8, 1.2), ToyPolicy(4, 0.8), args, "cpu")
                    fused.fused_ddim = True
                    observation = {"obs": torch.ones(1, 4, 66)}
                    for _ in range(2):
                        with torch.inference_mode():
                            expected = eager.predict_action(observation)["action"]
                            actual = fused.predict_action(observation)["action"]
                        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                    fused.reset_noise()
                    eager.reset_noise()
                    # Once cached, neither scheduler nor autograd should run.
                    with patch.object(fused.prior.noise_scheduler, "set_timesteps", side_effect=AssertionError), \
                         patch.object(fused.guide.noise_scheduler, "step", side_effect=AssertionError), \
                         patch("torch.autograd.grad", side_effect=AssertionError), torch.inference_mode():
                        actual = fused.predict_action(observation)["action"]
                    with torch.inference_mode():
                        expected = eager.predict_action(observation)["action"]
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
