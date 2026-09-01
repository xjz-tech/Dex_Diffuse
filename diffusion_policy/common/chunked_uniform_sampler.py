from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class ChunkedUniformSampler(Sampler[int]):
    def __init__(
        self,
        dataset_length: int,
        samples_per_epoch: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
        chunk_size: int = 65_536,
    ):
        if type(dataset_length) is not int or dataset_length <= 0:
            raise ValueError("dataset_length must be a positive integer")
        if type(samples_per_epoch) is not int or samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be a positive integer")
        if type(num_replicas) is not int or num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer")
        if type(rank) is not int or not 0 <= rank < num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.dataset_length = dataset_length
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
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, self.rank])
        )
        remaining = self.num_samples
        while remaining:
            size = min(remaining, self.chunk_size)
            yield from rng.integers(
                0,
                self.dataset_length,
                size=size,
                dtype=np.int64,
            ).tolist()
            remaining -= size
