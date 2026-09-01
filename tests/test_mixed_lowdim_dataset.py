from __future__ import annotations

import pytest
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.dataset.mixed_lowdim_dataset import MixedLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class ToyDataset(BaseLowdimDataset):
    def __init__(self, values, val_values):
        self.values = torch.tensor(values, dtype=torch.float32)
        self.val_values = torch.tensor(val_values, dtype=torch.float32)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        value = self.values[index]
        return {
            "obs": value.expand(2, 66).clone(),
            "action": value.expand(2, 22).clone(),
        }

    def get_validation_dataset(self):
        return ToyDataset(self.val_values.tolist(), [])

    def get_all_actions(self):
        return self.values[:, None].expand(-1, 22).clone()

    def get_normalizer(self, mode="limits", **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            {
                "obs": self.values[:, None].expand(-1, 66),
                "action": self.values[:, None].expand(-1, 22),
            },
            mode=mode,
            last_n_dims=1,
            **kwargs,
        )
        return normalizer


def _mixed(samples_per_epoch=None):
    return MixedLowdimDataset(
        datasets={
            "expdata": ToyDataset([-2.0, -1.0, 0.0], [10.0]),
            "bulb_tac": ToyDataset([4.0, 8.0], [20.0, 30.0]),
        },
        probabilities={"expdata": 0.8, "bulb_tac": 0.2},
        samples_per_epoch=samples_per_epoch,
    )


def test_mixed_dataset_preserves_lengths_and_routes_source_metadata():
    dataset = _mixed()
    assert len(dataset) == 5
    assert dataset.source_names == ("expdata", "bulb_tac")
    assert dataset.source_lengths == (3, 2)
    assert dataset[0]["source_id"].item() == 0
    assert dataset[2]["obs"][0, 0].item() == 0.0
    assert dataset[3]["source_id"].item() == 1
    assert dataset[4]["action"][0, 0].item() == 8.0


def test_validation_is_plain_concatenation_without_custom_sampler():
    validation = _mixed().get_validation_dataset()
    assert len(validation) == 3
    assert validation[0]["obs"][0, 0].item() == 10.0
    assert validation[1]["obs"][0, 0].item() == 20.0
    assert set(validation[0]) == {"obs", "action", "source_id"}
    assert validation.get_training_sampler(seed=1) is None


def test_default_and_explicit_epoch_size_feed_custom_sampler():
    default_sampler = _mixed().get_training_sampler(seed=7)
    explicit_sampler = _mixed(samples_per_epoch=123).get_training_sampler(seed=7)
    assert len(default_sampler) == 5
    assert len(explicit_sampler) == 123


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ({"expdata": 1.0}, "exactly match"),
        (
            {"expdata": 0.8, "bulb_tac": 0.1, "extra": 0.1},
            "exactly match",
        ),
        ({"expdata": 1.0, "bulb_tac": 0.0}, "finite, positive"),
        ({"expdata": 0.8, "bulb_tac": float("nan")}, "finite, positive"),
        ({"expdata": 0.8, "bulb_tac": 0.1}, "sum to 1"),
    ],
)
def test_mixed_dataset_rejects_invalid_probability_maps(
    probabilities,
    message,
):
    with pytest.raises(ValueError, match=message):
        MixedLowdimDataset(
            datasets={
                "expdata": ToyDataset([1.0], []),
                "bulb_tac": ToyDataset([2.0], []),
            },
            probabilities=probabilities,
        )


def test_mixed_dataset_rejects_empty_sources_and_invalid_epoch_size():
    with pytest.raises(ValueError, match="non-empty"):
        MixedLowdimDataset(
            datasets={
                "expdata": ToyDataset([1.0], []),
                "bulb_tac": ToyDataset([], []),
            },
            probabilities={"expdata": 0.8, "bulb_tac": 0.2},
        )
    with pytest.raises(ValueError, match="samples_per_epoch"):
        _mixed(samples_per_epoch=0)


def test_normalizer_merges_complete_source_statistics_without_ratio_weighting():
    normalizer = _mixed().get_normalizer()
    obs_stats = normalizer["obs"].get_input_stats()
    action_stats = normalizer["action"].get_input_stats()
    expected = torch.tensor([-2.0, -1.0, 0.0, 4.0, 8.0])
    torch.testing.assert_close(obs_stats["min"], torch.full((66,), -2.0))
    torch.testing.assert_close(obs_stats["max"], torch.full((66,), 8.0))
    torch.testing.assert_close(
        obs_stats["mean"], torch.full((66,), expected.mean().item())
    )
    torch.testing.assert_close(
        obs_stats["std"], torch.full((66,), expected.std().item())
    )
    torch.testing.assert_close(action_stats["min"], torch.full((22,), -2.0))
    torch.testing.assert_close(action_stats["max"], torch.full((22,), 8.0))
