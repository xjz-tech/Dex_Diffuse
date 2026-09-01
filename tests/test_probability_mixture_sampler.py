from itertools import islice

import numpy as np
import pytest

from diffusion_policy.common.probability_mixture_sampler import (
    ProbabilityMixtureSampler,
)


def _sampler(**kwargs):
    defaults = dict(
        source_lengths=[100, 20],
        probabilities=[0.8, 0.2],
        samples_per_epoch=10_000,
        seed=42,
        num_replicas=1,
        rank=0,
        chunk_size=257,
    )
    defaults.update(kwargs)
    return ProbabilityMixtureSampler(**defaults)


def test_sampler_is_reproducible_per_seed_epoch_and_rank():
    first = _sampler()
    second = _sampler()
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)
    assert list(_sampler(rank=0, num_replicas=2)) != list(
        _sampler(rank=1, num_replicas=2)
    )


def test_sampler_realized_source_ratio_matches_probability():
    indices = np.asarray(list(_sampler(samples_per_epoch=100_000)))
    exp_fraction = float(np.mean(indices < 100))
    assert exp_fraction == pytest.approx(0.8, abs=0.01)
    assert np.all((indices >= 0) & (indices < 120))


def test_sampler_uses_bounded_chunks_for_huge_sources():
    sampler = _sampler(
        source_lengths=[100_000_000, 44_142],
        samples_per_epoch=100_044_142,
        chunk_size=128,
    )
    first_thousand = list(islice(iter(sampler), 1000))
    assert len(first_thousand) == 1000
    assert len(sampler) == 100_044_142


def test_ddp_ranks_have_equal_lengths_without_dropping_epoch_exposure():
    samplers = [
        _sampler(samples_per_epoch=11, num_replicas=4, rank=rank)
        for rank in range(4)
    ]
    assert [len(sampler) for sampler in samplers] == [3, 3, 3, 3]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"source_lengths": [100, 0]}, "source lengths"),
        ({"probabilities": [0.8]}, "same length"),
        ({"probabilities": [0.8, 0.3]}, "sum to 1"),
        ({"probabilities": [1.0, 0.0]}, "strictly positive"),
        ({"samples_per_epoch": 0}, "samples_per_epoch"),
        ({"num_replicas": 0}, "num_replicas"),
        ({"num_replicas": 2, "rank": 2}, "rank"),
    ],
)
def test_sampler_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _sampler(**kwargs)
