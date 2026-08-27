import pytest

from diffusion_policy.common.bounded_sampler import EpochRandomSampler, EvenlySpacedSampler


def test_epoch_random_sampler_is_exact_in_range_and_epoch_deterministic():
    sampler = EpochRandomSampler(range(17), num_samples=41, seed=9, chunk_size=5)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    second = list(sampler)
    sampler.set_epoch(4)
    third = list(sampler)

    assert len(sampler) == 41
    assert first == second
    assert first != third
    assert all(0 <= index < 17 for index in first + third)
    assert not hasattr(sampler, "indices")


def test_evenly_spaced_sampler_is_fixed_bounded_and_lazy():
    sampler = EvenlySpacedSampler(range(10), num_samples=4)
    assert len(sampler) == 4
    assert list(sampler) == [1, 3, 6, 8]
    assert list(sampler) == [1, 3, 6, 8]
    assert not hasattr(sampler, "indices")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EpochRandomSampler([], num_samples=1, seed=0),
        lambda: EpochRandomSampler(range(2), num_samples=0, seed=0),
        lambda: EpochRandomSampler(range(2), num_samples=1, seed=0, chunk_size=0),
        lambda: EvenlySpacedSampler([], num_samples=1),
        lambda: EvenlySpacedSampler(range(2), num_samples=0),
    ],
)
def test_bounded_samplers_reject_empty_sources_and_nonpositive_sizes(factory):
    with pytest.raises(ValueError):
        factory()
