from itertools import islice

import numpy as np
import pytest

from diffusion_policy.common.chunked_uniform_sampler import ChunkedUniformSampler


def _sampler(**kwargs):
    defaults = dict(
        dataset_length=100,
        samples_per_epoch=10_000,
        seed=42,
        num_replicas=1,
        rank=0,
        chunk_size=257,
    )
    defaults.update(kwargs)
    return ChunkedUniformSampler(**defaults)


def test_uniform_sampler_is_reproducible_per_seed_epoch_and_rank():
    first = _sampler()
    second = _sampler()
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)
    assert list(_sampler(rank=0, num_replicas=2)) != list(
        _sampler(rank=1, num_replicas=2)
    )


def test_uniform_sampler_covers_full_index_space():
    indices = np.asarray(list(_sampler(dataset_length=20, samples_per_epoch=50_000)))
    counts = np.bincount(indices, minlength=20)
    assert counts.min() > 0
    expected = 50_000 / 20
    np.testing.assert_allclose(counts, expected, rtol=0.08)


def test_uniform_sampler_uses_bounded_chunks_for_huge_index_space():
    sampler = _sampler(
        dataset_length=700_000_000,
        samples_per_epoch=100_000,
        chunk_size=128,
    )
    first_thousand = list(islice(iter(sampler), 1000))
    assert len(first_thousand) == 1000
    assert len(sampler) == 100_000
    assert all(0 <= index < 700_000_000 for index in first_thousand)


def test_ddp_ranks_have_equal_lengths():
    samplers = [
        _sampler(samples_per_epoch=11, num_replicas=4, rank=rank)
        for rank in range(4)
    ]
    assert [len(sampler) for sampler in samplers] == [3, 3, 3, 3]


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"dataset_length": 0}, "dataset_length"),
        ({"samples_per_epoch": 0}, "samples_per_epoch"),
        ({"num_replicas": 0}, "num_replicas"),
        ({"num_replicas": 2, "rank": 2}, "rank"),
    ],
)
def test_uniform_sampler_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _sampler(**kwargs)
