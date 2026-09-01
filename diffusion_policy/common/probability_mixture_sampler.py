from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class ProbabilityMixtureSampler(Sampler[int]):
    def __init__(
        self,
        source_lengths: Sequence[int],
        probabilities: Sequence[float],
        samples_per_epoch: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        chunk_size: int = 65_536,
    ):
        lengths = np.asarray(source_lengths, dtype=np.int64)
        probs = np.asarray(probabilities, dtype=np.float64)
        if lengths.ndim != 1 or len(lengths) < 2 or np.any(lengths <= 0):
            raise ValueError("source lengths must contain at least two positive values")
        if probs.shape != lengths.shape:
            raise ValueError("probabilities and source lengths must have the same length")
        if not np.isfinite(probs).all() or np.any(probs <= 0):
            raise ValueError("probabilities must be finite and strictly positive")
        if not np.isclose(probs.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("probabilities must sum to 1")
        if type(samples_per_epoch) is not int or samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be a positive integer")
        if type(num_replicas) is not int or num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer")
        if type(rank) is not int or not 0 <= rank < num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.source_lengths = lengths
        self.probabilities = probs
        self.offsets = np.r_[0, np.cumsum(lengths[:-1])]
        self.samples_per_epoch = samples_per_epoch
        self.num_samples = math.ceil(samples_per_epoch / num_replicas)
        self.seed = int(seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.chunk_size = chunk_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        seed = np.random.SeedSequence([self.seed, self.epoch, self.rank])
        rng = np.random.default_rng(seed)
        remaining = self.num_samples
        while remaining:
            size = min(remaining, self.chunk_size)
            sources = rng.choice(
                len(self.source_lengths), size=size, p=self.probabilities
            )
            indices = np.empty(size, dtype=np.int64)
            for source_id in range(len(self.source_lengths)):
                mask = sources == source_id
                count = int(mask.sum())
                if count:
                    indices[mask] = self.offsets[source_id] + rng.integers(
                        0, self.source_lengths[source_id], size=count
                    )
            yield from indices.tolist()
            remaining -= size
