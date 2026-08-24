import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.common.bulb_action_normalizer import HAND_DIM, normalize_action
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy


def _hand_policy(pred_action_steps_only, n_action_steps, horizon):
    shape_meta = {
        "obs": {"hand_joint": {"shape": [22], "type": "low_dim"}},
        "action": {"shape": [22]},
    }
    encoder = MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=None)
    scheduler = DDIMScheduler(
        num_train_timesteps=4,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )
    policy = DiffusionUnetImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        obs_encoder=encoder,
        horizon=horizon,
        n_action_steps=n_action_steps,
        n_obs_steps=2,
        num_inference_steps=2,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=32,
        # Single-scale UNet: (32, 64) down/up-samples T=1 to T=2.
        down_dims=(32,),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        pred_action_steps_only=pred_action_steps_only,
    )
    normalizer = LinearNormalizer()
    stat = {
        "min": torch.zeros(HAND_DIM),
        "max": torch.ones(HAND_DIM),
        "mean": torch.full((HAND_DIM,), 0.5),
        "std": torch.ones(HAND_DIM),
    }
    # get_range_normalizer_from_stat wants numpy
    import numpy as np
    np_stat = {k: v.numpy() for k, v in stat.items()}
    from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat as gr
    normalizer["hand_joint"] = gr(np_stat)
    policy.set_normalizer(normalizer)
    policy.num_inference_steps = 2
    return policy


def test_step_predict_action_shape():
    policy = _hand_policy(True, 1, 2)
    obs = {"hand_joint": torch.zeros(1, 2, 22)}
    out = policy.predict_action(obs)
    assert tuple(out["action"].shape) == (1, 1, 22)


def test_chunk_predict_action_shape():
    policy = _hand_policy(True, 8, 16)
    obs = {"hand_joint": torch.zeros(1, 2, 22)}
    out = policy.predict_action(obs)
    assert tuple(out["action"].shape) == (1, 8, 22)


def test_encode_obs_and_predict_eps_shapes():
    policy = _hand_policy(True, 1, 2)
    obs = {"hand_joint": torch.zeros(2, 2, 22)}
    cond = policy.encode_obs(obs)
    assert cond.shape[0] == 2
    x = torch.zeros(2, 1, 22)
    t = torch.zeros(2, dtype=torch.long)
    eps = policy.predict_eps(x, t, cond)
    assert eps.shape == x.shape


def test_compute_loss_step_runs():
    policy = _hand_policy(True, 1, 2)
    batch = {
        "obs": {"hand_joint": torch.zeros(3, 2, 22)},
        "action": torch.zeros(3, 2, 22),
    }
    loss = policy.compute_loss(batch)
    assert torch.isfinite(loss)
