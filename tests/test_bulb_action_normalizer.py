import warnings

import numpy as np
import pytest
import torch

from diffusion_policy.common.normalize_util import get_range_normalizer_from_stat
from diffusion_policy.common.bulb_action_normalizer import (
    EE_DIM,
    HAND_DIM,
    has_legacy_action_normalizer,
    warn_legacy_action_normalizer,
    hand_joint_normalizer_from_stat,
    load_hand_joint_stat,
    normalize_action,
    unnormalize_action,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def _stat(lo, hi):
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    return {
        "min": lo,
        "max": hi,
        "mean": (lo + hi) / 2,
        "std": np.maximum(hi - lo, 1e-6) / np.sqrt(12.0),
    }


def test_unnormalize_22_uses_only_hand_joint():
    normalizer = LinearNormalizer()
    hand_stat = _stat(np.zeros(HAND_DIM), np.ones(HAND_DIM) * 2)
    ee_stat = _stat(np.full(EE_DIM, -10.0), np.full(EE_DIM, 10.0))
    normalizer["hand_joint"] = get_range_normalizer_from_stat(hand_stat)
    normalizer["ee_pose"] = get_range_normalizer_from_stat(ee_stat)
    naction = torch.zeros(2, 4, HAND_DIM)
    out = unnormalize_action(normalizer, naction)
    assert out.shape == (2, 4, HAND_DIM)
    torch.testing.assert_close(out, torch.ones_like(out))


def test_unnormalize_31_splits_ee_and_hand():
    normalizer = LinearNormalizer()
    hand_stat = _stat(np.zeros(HAND_DIM), np.ones(HAND_DIM) * 2)
    ee_stat = _stat(np.zeros(EE_DIM), np.ones(EE_DIM) * 4)
    normalizer["hand_joint"] = get_range_normalizer_from_stat(hand_stat)
    normalizer["ee_pose"] = get_range_normalizer_from_stat(ee_stat)
    naction = torch.zeros(1, 3, EE_DIM + HAND_DIM)
    out = unnormalize_action(normalizer, naction)
    assert out.shape[-1] == 31
    torch.testing.assert_close(out[..., :EE_DIM], torch.full((1, 3, EE_DIM), 2.0))
    torch.testing.assert_close(out[..., EE_DIM:], torch.ones(1, 3, HAND_DIM))


def test_normalize_roundtrip_31():
    normalizer = LinearNormalizer()
    hand_stat = _stat(np.zeros(HAND_DIM), np.ones(HAND_DIM))
    ee_stat = _stat(-np.ones(EE_DIM), np.ones(EE_DIM))
    normalizer["hand_joint"] = get_range_normalizer_from_stat(hand_stat)
    normalizer["ee_pose"] = get_range_normalizer_from_stat(ee_stat)
    action = torch.cat(
        [torch.zeros(1, 2, EE_DIM), torch.full((1, 2, HAND_DIM), 0.25)], dim=-1
    )
    naction = normalize_action(normalizer, action)
    restored = unnormalize_action(normalizer, naction)
    torch.testing.assert_close(restored, action, atol=1e-5, rtol=1e-5)


def test_legacy_31_uses_action_key_not_split():
    import diffusion_policy.common.bulb_action_normalizer as m
    m._LEGACY_ACTION_NORMALIZER_WARNED = False
    normalizer = LinearNormalizer()
    action = np.concatenate(
        [np.zeros((8, EE_DIM)), np.ones((8, HAND_DIM))], axis=-1
    ).astype(np.float32)
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(action)
    dummy_stat = _stat(np.full(HAND_DIM, 99.0), np.full(HAND_DIM, 100.0))
    normalizer["hand_joint"] = get_range_normalizer_from_stat(dummy_stat)
    dummy_ee = _stat(np.full(EE_DIM, 99.0), np.full(EE_DIM, 100.0))
    normalizer["ee_pose"] = get_range_normalizer_from_stat(dummy_ee)
    assert has_legacy_action_normalizer(normalizer)
    naction = torch.zeros(1, 2, 31)
    with pytest.warns(UserWarning, match="legacy"):
        out = unnormalize_action(normalizer, naction)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        unnormalize_action(normalizer, naction)
        assert record == []
    split = torch.cat(
        [
            normalizer["ee_pose"].unnormalize(naction[..., :EE_DIM]),
            normalizer["hand_joint"].unnormalize(naction[..., EE_DIM:]),
        ],
        dim=-1,
    )
    assert not torch.allclose(out, split)


def test_rejects_bad_last_dim():
    normalizer = LinearNormalizer()
    with pytest.raises(ValueError, match="22 or 31"):
        unnormalize_action(normalizer, torch.zeros(1, 3, 10))


def test_load_hand_joint_stat_roundtrip(tmp_path):
    path = tmp_path / "hand.npz"
    mn = np.arange(HAND_DIM, dtype=np.float32)
    mx = mn + 1
    np.savez(path, min=mn, max=mx)
    stat = load_hand_joint_stat(str(path))
    np.testing.assert_array_equal(stat["min"], mn)
    field = hand_joint_normalizer_from_stat(stat)
    x = torch.from_numpy(mn)
    torch.testing.assert_close(field.normalize(x), torch.full((HAND_DIM,), -1.0))
