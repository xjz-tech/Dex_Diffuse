"""One seeded raw-Gaussian stream per rollout; no policy or simulator imports."""
import math
import torch


class AlignedNoiseStream:
    def __init__(self, seed, rho, device='cpu', dtype=torch.float32):
        if not math.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError('rho must be in [0,1]')
        self.rho = float(rho)
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        self.device, self.dtype = device, dtype
        self.previous = None
        self.next_index = 0

    def draw(self, reference_index):
        # Disallow skipped/repeated windows and implicit episode changes. A new
        # episode/seed/reference needs a new stream, starting at index zero.
        if reference_index != self.next_index:
            raise ValueError(f'expected reference index {self.next_index}, got {reference_index}')
        # Always consume a complete horizon, even at rho=1. Thus all arms use
        # identical innovations and the first draw matches historical _noise.
        xi = torch.randn((1, 8, 22), generator=self.generator,
                         device=self.device, dtype=self.dtype)
        noise = xi.clone()
        if self.previous is not None and self.rho != 0:
            # Full horizon = 3 known history + 5 future. Shift future by exec2.
            noise[:, 3:6] = (self.rho * self.previous[:, 5:8]
                              + math.sqrt(1 - self.rho ** 2) * xi[:, 3:6])
        # History 0:3 and newly appearing future 6:8 stay fresh in every arm.
        self.previous = noise.clone()
        self.next_index += 2
        return noise, xi
