import torch

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


def test_lowdim_only_concat_hand_joint():
    shape_meta = {"obs": {"hand_joint": {"shape": [22], "type": "low_dim"}}}
    encoder = MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=None)
    x = {"hand_joint": torch.zeros(4, 22)}
    out = encoder(x)
    assert tuple(out.shape) == (4, 22)
    assert tuple(encoder.output_shape()) == (22,)


def test_rgb_without_model_raises():
    shape_meta = {
        "obs": {
            "front_image": {"shape": [3, 240, 320], "type": "rgb"},
            "hand_joint": {"shape": [22], "type": "low_dim"},
        }
    }
    try:
        MultiImageObsEncoder(shape_meta=shape_meta, rgb_model=None)
    except RuntimeError as exc:
        assert "rgb_model" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
