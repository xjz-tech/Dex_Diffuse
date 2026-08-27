from collections.abc import Iterator, Sized

import torch


class EpochRandomSampler:
    def __init__(
        self,
        data_source: Sized,
        *,
        num_samples: int,
        seed: int,
        chunk_size: int = 65_536,
    ) -> None:
        if len(data_source) == 0:
            raise ValueError("data_source must be non-empty")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.data_source = data_source
        self.num_samples = num_samples
        self.seed = seed
        self.chunk_size = chunk_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        remaining = self.num_samples
        length = len(self.data_source)
        while remaining > 0:
            chunk = min(remaining, self.chunk_size)
            for index in torch.randint(
                low=0,
                high=length,
                size=(chunk,),
                generator=generator,
            ).tolist():
                yield index
            remaining -= chunk

    def __len__(self) -> int:
        return self.num_samples


class EvenlySpacedSampler:
    def __init__(self, data_source: Sized, *, num_samples: int) -> None:
        if len(data_source) == 0:
            raise ValueError("data_source must be non-empty")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        length = len(self.data_source)
        for i in range(self.num_samples):
            yield ((2 * i + 1) * length) // (2 * self.num_samples)

    def __len__(self) -> int:
        return self.num_samples
