"""Real-hand policy pairing using the same score-guided DDIM as XJZ eval."""

from pathlib import Path
import sys

import torch

from policy_loader import load_policy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from diffusion_policy.guidance.guided_ddim import sample_guided_trajectory
from trt_unet import _cached_fused_coeffs, fused_ddim_update, fused_guided_ddim_update


class GuidedRealPolicy:
    def __init__(self, prior, guide, args, device):
        self.prior = prior
        self.guide = guide
        self.args = args
        self.device = device
        self.normalizer = prior.normalizer
        self.n_obs_steps = prior.n_obs_steps
        self.obs_dim = prior.obs_dim
        self.action_dim = prior.action_dim
        self.n_action_steps = args.action_chunk_steps or prior.n_action_steps
        self.num_inference_steps = args.inference_steps
        self.start = self.n_obs_steps - 1
        self.window = slice(self.start, self.start + args.guidance_steps)
        self.fused_ddim = bool(getattr(args, "fused_ddim", False) or getattr(args, "tensorrt", False))
        self.reset_noise()
        for policy in (prior, guide):
            for parameter in policy.parameters():
                parameter.requires_grad_(False)

    def reset_noise(self):
        self.generators = [torch.Generator(device=self.device).manual_seed(seed)
                           for seed in (self.args.seed, self.args.guide_seed)]
        self.noises = {}

    def noise(self, index, observation):
        shape = (observation.shape[0], self.prior.horizon, self.action_dim)
        if not self.args.fixed_noise or index not in self.noises or self.noises[index].shape != shape:
            self.noises[index] = torch.randn(shape, device=self.device,
                                            dtype=observation.dtype,
                                            generator=self.generators[index])
        return self.noises[index].clone()

    def predict_action(self, obs_dict):
        observation = obs_dict["obs"]
        capture = getattr(self, "_debug_capture", None)
        with torch.no_grad():
            guide_normalized_obs = self.guide.normalizer["obs"].normalize(observation)
            guide_cond = guide_normalized_obs.flatten(1)
            reference = self.noise(1, observation)
            if capture is not None:
                capture["guide_normalized_obs"] = guide_normalized_obs.detach().clone()
                capture["guide_global_cond"] = guide_cond.detach().clone()
                capture["guide_initial_noise"] = reference.detach().clone()
            if self.fused_ddim:
                coeffs = _cached_fused_coeffs(self.guide, reference)
                if capture is not None:
                    capture["guide_timesteps"] = coeffs.timesteps.detach().clone()
                for index, timestep in enumerate(coeffs.timesteps):
                    epsilon = self.guide.model(reference, timestep, global_cond=guide_cond)
                    reference = fused_ddim_update(reference, epsilon, index, coeffs)
            else:
                scheduler = self.guide.noise_scheduler
                scheduler.set_timesteps(self.args.guide_inference_steps)
                if capture is not None:
                    capture["guide_timesteps"] = scheduler.timesteps.detach().clone()
                for timestep in scheduler.timesteps:
                    epsilon = self.guide.model(reference, timestep, global_cond=guide_cond)
                    reference = scheduler.step(epsilon, timestep, reference, eta=0.0).prev_sample
            if capture is not None:
                capture["guide_normalized_trajectory"] = reference.detach().clone()
                capture["guide_action_pred"] = self.guide.normalizer["action"].unnormalize(reference).detach().clone()
            reference = self.guide.normalizer["action"].unnormalize(reference[:, self.window])
            if capture is not None:
                capture["guide_reference_physical"] = reference.detach().clone()
            reference = self.normalizer["action"].normalize(reference)
            prior_normalized_obs = self.normalizer["obs"].normalize(observation)
            condition = prior_normalized_obs.flatten(1)
            if capture is not None:
                capture["prior_normalized_obs"] = prior_normalized_obs.detach().clone()
                capture["prior_global_cond"] = condition.detach().clone()
                capture["guide_reference_normalized"] = reference.detach().clone()
                capture["guidance_window"] = {"start": self.window.start, "stop": self.window.stop}
                capture["scheduler_capture_method"] = "deterministic_eta_zero"
        if self.fused_ddim:
            with torch.no_grad():
                trajectory = self.noise(0, observation)
                coeffs = _cached_fused_coeffs(self.prior, trajectory)
                if capture is not None:
                    capture["prior_initial_noise"] = trajectory.detach().clone()
                    capture["prior_timesteps"] = coeffs.timesteps.detach().clone()
                for index, timestep in enumerate(coeffs.timesteps):
                    epsilon = self.prior.model(trajectory, timestep, global_cond=condition)
                    trajectory = fused_guided_ddim_update(
                        trajectory, epsilon, index, coeffs, reference,
                        self.args.guidance_scale, self.window,
                    )
                action = trajectory[:, self.start:self.start + self.n_action_steps]
                output = {"action": self.normalizer["action"].unnormalize(action)}
                if capture is not None:
                    capture["normalized_trajectory"] = trajectory.detach().clone()
                    output["action_pred"] = self.normalizer["action"].unnormalize(trajectory)
                    capture["action_pred"] = output["action_pred"].detach().clone()
                return output
        initial_noise = self.noise(0, observation)
        if capture is not None:
            capture["prior_initial_noise"] = initial_noise.detach().clone()
        result = sample_guided_trajectory(
            model=self.prior.model,
            scheduler=self.prior.noise_scheduler,
            initial_noise=initial_noise,
            global_cond=condition,
            reference=reference,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.args.guidance_scale,
            guidance_slice=self.window,
            eta=0.0,
        )
        action = result.trajectory[:, self.start:self.start + self.n_action_steps]
        output = {"action": self.normalizer["action"].unnormalize(action)}
        if capture is not None:
            capture["prior_timesteps"] = self.prior.noise_scheduler.timesteps.detach().clone()
            capture["normalized_trajectory"] = result.trajectory.detach().clone()
            output["action_pred"] = self.normalizer["action"].unnormalize(result.trajectory)
            capture["action_pred"] = output["action_pred"].detach().clone()
        return output


def load_guided_policy(args, device):
    loaded, prior, spec = load_policy(args.checkpoint, device, "ddim", args.inference_steps)
    guide_loaded, guide, guide_spec = load_policy(
        args.guide_checkpoint, device, "ddim", args.guide_inference_steps)
    if spec != guide_spec:
        raise ValueError(f"prior and guide specs disagree: {spec} != {guide_spec}")
    for policy in (prior, guide):
        if policy.noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("guided DDIM requires epsilon prediction")
        if getattr(policy.noise_scheduler.config, "thresholding", False):
            raise ValueError("guided DDIM does not support dynamic thresholding")
    if args.guidance_steps > spec["horizon"] - spec["n_obs_steps"] + 1:
        raise ValueError("guidance window exceeds the prediction horizon")
    print(f"[guidance] guide={args.guide_checkpoint}; weights={guide_loaded.weight_source}; "
          f"scale={args.guidance_scale}; prior_steps={args.inference_steps}; "
          f"guide_steps={args.guide_inference_steps}; guidance_actions={args.guidance_steps}; "
          f"fixed_noise={args.fixed_noise}; seeds={args.seed}/{args.guide_seed}; "
          f"fused_ddim={bool(getattr(args, 'fused_ddim', False) or getattr(args, 'tensorrt', False))}", flush=True)
    return loaded, GuidedRealPolicy(prior, guide, args, device), spec
